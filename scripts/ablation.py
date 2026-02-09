#!/usr/bin/env python3
"""
CogMoE Ablation Studies.

Systematic ablation experiments to evaluate the contribution of each
CogMoE component: MoE architecture, CORTEX loss terms, and individual experts.

Usage:
    python scripts/ablation.py --config configs/default.yaml --study moe
    python scripts/ablation.py --config configs/default.yaml --study cortex
    python scripts/ablation.py --config configs/default.yaml --study experts
    python scripts/ablation.py --config configs/default.yaml --study moe --fold 2
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# ---- CogMoE imports ----
from cogmoe.models.cogmoe import CogMoE, build_model
from cogmoe.losses.cortex_loss import CORTEXLoss
from cogmoe.data.dataset import CLDriveDataset, collate_fn, build_dataset
from cogmoe.data.augmentation import build_augmentation
from cogmoe.data.cross_validation import get_segment_stratified_splits
from cogmoe.utils.reproducibility import set_seed, get_device, count_parameters
from cogmoe.utils.metrics import compute_all_metrics, compute_expert_utilization
from cogmoe.utils.logging import ExperimentLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_optimizer(model, cfg):
    """Build optimizer from training config section."""
    train_cfg = cfg["training"]
    lr = train_cfg.get("lr", 3e-4)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


# ---------------------------------------------------------------------------
# Training and evaluation (mirrors train.py)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, grad_clip=1.0):
    """Run one training epoch. Returns dict of averaged losses."""
    model.train()
    running = {"total": 0.0, "task": 0.0, "noise": 0.0,
               "refinement": 0.0, "gate_reg": 0.0}
    n_batches = 0

    for batch in loader:
        inputs = {}
        for mod in model.modalities:
            if mod in batch:
                inputs[mod] = batch[mod].to(device)
        quality = batch["quality"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"].to(device)

        out = model(inputs=inputs, quality=quality, mask=mask)
        loss_dict = criterion(
            logits=out["logits"],
            targets=targets,
            gate_weights_list=out["gate_weights_list"],
            expert_outs_list=out["expert_outs_list"],
            fused_features=out["fused_features"],
            epoch=epoch,
        )

        optimizer.zero_grad()
        loss_dict["total"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key in running:
            if key in loss_dict:
                val = loss_dict[key]
                running[key] += val.item() if torch.is_tensor(val) else float(val)
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(model, loader, criterion, device, epoch=0):
    """
    Evaluate model on a data loader.

    Returns:
        metrics: dict with accuracy, f1, precision, recall, auc_roc.
        loss_avg: dict with averaged loss components.
        expert_util: dict with expert utilization percentages.
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    all_gate_weights = []
    running = {"total": 0.0, "task": 0.0, "noise": 0.0,
               "refinement": 0.0, "gate_reg": 0.0}
    n_batches = 0

    for batch in loader:
        inputs = {}
        for mod in model.modalities:
            if mod in batch:
                inputs[mod] = batch[mod].to(device)
        quality = batch["quality"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"].to(device)

        out = model(inputs=inputs, quality=quality, mask=mask)
        loss_dict = criterion(
            logits=out["logits"],
            targets=targets,
            gate_weights_list=out["gate_weights_list"],
            expert_outs_list=out["expert_outs_list"],
            fused_features=out["fused_features"],
            epoch=epoch,
        )

        probs = torch.softmax(out["logits"], dim=-1)[:, 1]
        preds = out["logits"].argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())
        all_gate_weights.extend(out["gate_weights_list"])

        for key in running:
            if key in loss_dict:
                val = loss_dict[key]
                running[key] += val.item() if torch.is_tensor(val) else float(val)
        n_batches += 1

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    metrics = compute_all_metrics(all_targets, all_preds, all_probs)
    loss_avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    expert_util = compute_expert_utilization(all_gate_weights)

    return metrics, loss_avg, expert_util


def train_ablation_variant(variant_name, cfg, df, train_idx, val_idx, device):
    """
    Train a single ablation variant and return best validation metrics.

    Args:
        variant_name: str - name of this ablation variant.
        cfg: dict - modified configuration for this variant.
        df: pd.DataFrame - full dataset.
        train_idx: array of training indices.
        val_idx: array of validation indices.
        device: torch.device.

    Returns:
        dict with accuracy, f1, expert_util, train_time, num_params, and loss components.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    aug_cfg = cfg.get("augmentation", {})

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    transform = build_augmentation(aug_cfg)
    train_ds = build_dataset(data_cfg, train_df, transform=transform)
    val_ds = build_dataset(data_cfg, val_df, transform=None)

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, drop_last=False,
    )

    # Build model, loss, optimizer
    model = build_model(cfg).to(device)
    criterion = CORTEXLoss(
        gamma=loss_cfg.get("gamma", 0.75),
        lambda_=loss_cfg.get("lambda_", 0.6),
        beta_init=loss_cfg.get("beta_init", 1.0),
        beta_max=loss_cfg.get("beta_max", 0.2),
        alpha_decay=loss_cfg.get("alpha_decay", 0.05),
        num_experts=cfg["model"].get("num_experts", 3),
    )
    optimizer = build_optimizer(model, cfg)

    epochs = train_cfg.get("epochs", 50)
    patience = train_cfg.get("early_stopping_patience", 10)
    grad_clip = train_cfg.get("grad_clip", 1.0)
    num_params = count_parameters(model)

    best_val_acc = -1.0
    best_metrics = {}
    best_expert_util = {}
    epochs_no_improve = 0

    print(f"    Training '{variant_name}' ({num_params:,} params) ...")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, grad_clip
        )
        val_metrics, val_loss, expert_util = evaluate(
            model, val_loader, criterion, device, epoch
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = {**val_metrics, "val_loss": val_loss["total"]}
            best_expert_util = expert_util
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    train_time = time.time() - t_start

    result = {
        **best_metrics,
        "expert_util": best_expert_util,
        "num_params": num_params,
        "train_time": round(train_time, 2),
        "epochs_trained": epoch,
    }

    print(
        f"      acc={best_metrics.get('accuracy', 0):.4f}, "
        f"f1={best_metrics.get('f1', 0):.4f}, "
        f"experts={best_expert_util}, "
        f"time={train_time:.1f}s"
    )

    return result


# ---------------------------------------------------------------------------
# Ablation study definitions
# ---------------------------------------------------------------------------

def define_moe_ablation(base_cfg):
    """
    Define MoE architecture ablation variants.

    Compares:
        1. FFN-only (no MoE): num_experts=1, equivalent to standard transformer FFN
        2. MoE without CORTEX: Full MoE but all loss coefficients zeroed
        3. Full CogMoE: Complete system with MoE + CORTEX loss

    Args:
        base_cfg: dict - base configuration.

    Returns:
        list of (variant_name, modified_cfg) tuples.
    """
    variants = []

    # Variant 1: FFN-only (single expert = no MoE routing)
    cfg_ffn = copy.deepcopy(base_cfg)
    cfg_ffn["model"]["num_experts"] = 1
    cfg_ffn["loss"]["gamma"] = 0.0
    cfg_ffn["loss"]["lambda_"] = 0.0
    cfg_ffn["loss"]["beta_init"] = 0.0
    cfg_ffn["loss"]["beta_max"] = 0.0
    variants.append(("FFN-only (no MoE)", cfg_ffn))

    # Variant 2: MoE without CORTEX loss (standard CE only)
    cfg_moe_no_cortex = copy.deepcopy(base_cfg)
    cfg_moe_no_cortex["loss"]["gamma"] = 0.0
    cfg_moe_no_cortex["loss"]["lambda_"] = 0.0
    cfg_moe_no_cortex["loss"]["beta_init"] = 0.0
    cfg_moe_no_cortex["loss"]["beta_max"] = 0.0
    variants.append(("MoE w/o CORTEX", cfg_moe_no_cortex))

    # Variant 3: Full CogMoE (baseline = no modification)
    cfg_full = copy.deepcopy(base_cfg)
    variants.append(("Full CogMoE", cfg_full))

    return variants


def define_cortex_ablation(base_cfg):
    """
    Define CORTEX loss component ablation variants.

    Disables each CORTEX loss component one at a time:
        1. No noise suppression (gamma=0)
        2. No refinement loss (lambda=0)
        3. No gate regularization (beta=0)
        4. No adaptive beta (fixed beta=beta_init throughout)
        5. Full CORTEX (all components active)

    Args:
        base_cfg: dict - base configuration.

    Returns:
        list of (variant_name, modified_cfg) tuples.
    """
    variants = []

    # Variant 1: No noise suppression (L_noise)
    cfg_no_noise = copy.deepcopy(base_cfg)
    cfg_no_noise["loss"]["gamma"] = 0.0
    variants.append(("No L_noise (gamma=0)", cfg_no_noise))

    # Variant 2: No refinement loss (L_refinement)
    cfg_no_refine = copy.deepcopy(base_cfg)
    cfg_no_refine["loss"]["lambda_"] = 0.0
    variants.append(("No L_refine (lambda=0)", cfg_no_refine))

    # Variant 3: No gate regularization (R_gate)
    cfg_no_gate = copy.deepcopy(base_cfg)
    cfg_no_gate["loss"]["beta_init"] = 0.0
    cfg_no_gate["loss"]["beta_max"] = 0.0
    variants.append(("No R_gate (beta=0)", cfg_no_gate))

    # Variant 4: No adaptive beta (fixed beta = beta_init, no decay)
    cfg_fixed_beta = copy.deepcopy(base_cfg)
    cfg_fixed_beta["loss"]["alpha_decay"] = 0.0
    cfg_fixed_beta["loss"]["beta_max"] = cfg_fixed_beta["loss"]["beta_init"]
    variants.append(("Fixed beta (no decay)", cfg_fixed_beta))

    # Variant 5: Full CORTEX (reference)
    cfg_full = copy.deepcopy(base_cfg)
    variants.append(("Full CORTEX", cfg_full))

    return variants


def define_experts_ablation(base_cfg):
    """
    Define individual expert ablation variants.

    Tests the system with each expert type disabled by masking its
    gate weight contribution. Implemented by setting the expert's
    dropout to 1.0 (fully dropping expert outputs).

    Experts:
        - HFE (High-Fidelity Expert, index 0)
        - NRE (Noise-Resilient Expert, index 1)
        - CRE (Context-Recovery Expert, index 2)

    Note: This ablation modifies the config to signal which expert to disable.
    The actual masking depends on model support. As a proxy, we also zero
    the corresponding loss components.

    Args:
        base_cfg: dict - base configuration.

    Returns:
        list of (variant_name, modified_cfg) tuples.
    """
    variants = []
    expert_names = ["HFE", "NRE", "CRE"]

    for expert_idx, expert_name in enumerate(expert_names):
        cfg_variant = copy.deepcopy(base_cfg)
        # Signal which expert to disable via a custom config key
        cfg_variant["model"]["disabled_expert"] = expert_idx

        # When disabling NRE, zero noise suppression loss
        if expert_name == "NRE":
            cfg_variant["loss"]["gamma"] = 0.0
        # When disabling CRE, zero refinement loss
        elif expert_name == "CRE":
            cfg_variant["loss"]["lambda_"] = 0.0

        variants.append((f"No {expert_name} (expert {expert_idx})", cfg_variant))

    # Full system with all experts (reference)
    cfg_full = copy.deepcopy(base_cfg)
    variants.append(("All experts", cfg_full))

    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STUDY_DEFINITIONS = {
    "moe": ("MoE Architecture Ablation", define_moe_ablation),
    "cortex": ("CORTEX Loss Component Ablation", define_cortex_ablation),
    "experts": ("Expert Specialization Ablation", define_experts_ablation),
}


def main():
    parser = argparse.ArgumentParser(
        description="CogMoE Ablation Studies"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--study", type=str, required=True,
        choices=list(STUDY_DEFINITIONS.keys()),
        help="Ablation study type: moe, cortex, or experts."
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Which CV fold to use for ablation (default: 0)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON. Default: outputs/ablation_{study}.json"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg)

    study_title, study_fn = STUDY_DEFINITIONS[args.study]

    # Output path
    output_path = args.output
    if output_path is None:
        save_dir = cfg.get("save_dir", "outputs/")
        output_path = os.path.join(save_dir, f"ablation_{args.study}.json")

    print(f"CogMoE Ablation Study: {study_title}")
    print(f"{'='*60}")
    print(f"  Config:  {args.config}")
    print(f"  Study:   {args.study}")
    print(f"  Fold:    {args.fold}")
    print(f"  Device:  {device}")
    print(f"  Output:  {output_path}")
    print(f"{'='*60}\n")

    # Load data
    data_cfg = cfg["data"]
    csv_path = data_cfg["csv_path"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    # Cross-validation splits (same as train.py)
    labels = df[data_cfg.get("label_col", "Label")].values
    n_folds = data_cfg.get("num_folds", 10)
    splits = get_segment_stratified_splits(labels, n_folds=n_folds, seed=seed)
    train_idx, val_idx = splits[args.fold]

    print(f"  Fold {args.fold}: Train={len(train_idx)}, Val={len(val_idx)}\n")

    # Define ablation variants
    variants = study_fn(cfg)

    print(f"  Running {len(variants)} ablation variants:")
    for i, (name, _) in enumerate(variants):
        print(f"    {i + 1}. {name}")
    print()

    # Run each variant
    all_results = {}
    t_total_start = time.time()

    for variant_idx, (variant_name, variant_cfg) in enumerate(variants):
        print(f"  [{variant_idx + 1}/{len(variants)}] {variant_name}")
        set_seed(seed)

        try:
            result = train_ablation_variant(
                variant_name, variant_cfg, df, train_idx, val_idx, device
            )
            all_results[variant_name] = result
        except Exception as e:
            print(f"    FAILED: {e}")
            all_results[variant_name] = {"error": str(e)}

    total_time = time.time() - t_total_start

    # Print results table
    print(f"\n{'='*60}")
    print(f"  {study_title} Results")
    print(f"{'='*60}")

    header = f"  {'Variant':30s} | {'Acc':>7s} | {'F1':>7s} | {'Params':>10s} | Expert Util"
    print(header)
    print(f"  {'-'*(len(header) - 2)}")

    for variant_name, result in all_results.items():
        if "error" in result:
            print(f"  {variant_name:30s} | ERROR: {result['error']}")
            continue

        acc = result.get("accuracy", 0)
        f1 = result.get("f1", 0)
        params = result.get("num_params", 0)
        expert_util = result.get("expert_util", {})
        expert_str = ", ".join(f"{k}:{v}%" for k, v in expert_util.items()) if expert_util else "N/A"

        print(f"  {variant_name:30s} | {acc:7.4f} | {f1:7.4f} | {params:>10,} | {expert_str}")

    # Compute deltas relative to the last variant (reference/full system)
    ref_name = list(all_results.keys())[-1]
    ref_result = all_results.get(ref_name, {})
    ref_acc = ref_result.get("accuracy", 0)

    if ref_acc > 0 and len(all_results) > 1:
        print(f"\n  Delta from reference ({ref_name}):")
        for variant_name, result in all_results.items():
            if variant_name == ref_name or "error" in result:
                continue
            delta_acc = result.get("accuracy", 0) - ref_acc
            delta_f1 = result.get("f1", 0) - ref_result.get("f1", 0)
            sign_acc = "+" if delta_acc >= 0 else ""
            sign_f1 = "+" if delta_f1 >= 0 else ""
            print(
                f"    {variant_name:30s}: "
                f"acc {sign_acc}{delta_acc:.4f}, "
                f"f1 {sign_f1}{delta_f1:.4f}"
            )

    # Save results
    output_data = {
        "study": args.study,
        "study_title": study_title,
        "config": args.config,
        "fold": args.fold,
        "seed": seed,
        "variants": {
            name: result for name, result in all_results.items()
        },
        "reference_variant": ref_name,
        "total_time_s": round(total_time, 2),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved to {output_path}")
    print(f"  Ablation study complete.")


if __name__ == "__main__":
    main()
