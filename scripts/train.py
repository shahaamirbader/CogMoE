#!/usr/bin/env python3
"""
CogMoE Training Pipeline.

Full 10-fold cross-validation training with CORTEX loss, early stopping,
gradient clipping, and per-fold checkpointing.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --fold 0
"""

import argparse
import os
import sys
import copy
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
    optimizer_name = train_cfg.get("optimizer", "adam").lower()

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, cfg):
    """Optionally build a learning-rate scheduler."""
    train_cfg = cfg["training"]
    scheduler_name = train_cfg.get("scheduler", None)
    if scheduler_name is None:
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=train_cfg["epochs"], eta_min=1e-6
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=train_cfg.get("scheduler_step", 10),
            gamma=train_cfg.get("scheduler_gamma", 0.5),
        )
    return None


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, grad_clip=1.0):
    """Run one training epoch.  Returns dict of averaged losses."""
    model.train()
    running = {"total": 0.0, "task": 0.0, "noise": 0.0,
               "refinement": 0.0, "gate_reg": 0.0}
    n_batches = 0

    for batch in loader:
        # Move data to device
        inputs = {}
        for mod in model.modalities:
            if mod in batch:
                inputs[mod] = batch[mod].to(device)
        quality = batch["quality"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"].to(device)

        # Forward
        out = model(inputs=inputs, quality=quality, mask=mask)
        loss_dict = criterion(
            logits=out["logits"],
            targets=targets,
            gate_weights_list=out["gate_weights_list"],
            expert_outs_list=out["expert_outs_list"],
            fused_features=out["fused_features"],
            epoch=epoch,
        )

        # Backward
        optimizer.zero_grad()
        loss_dict["total"].backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Accumulate
        for key in running:
            if key in loss_dict:
                val = loss_dict[key]
                running[key] += val.item() if torch.is_tensor(val) else float(val)
        n_batches += 1

    # Average
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


# ---------------------------------------------------------------------------
# Fold training
# ---------------------------------------------------------------------------

def train_fold(fold_idx, train_idx, val_idx, df, cfg, device, logger, save_dir):
    """
    Train a single CV fold.  Returns the best validation metrics dict.
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    aug_cfg = cfg.get("augmentation", {})

    # Split dataframes
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Build augmentation (training only)
    transform = build_augmentation(aug_cfg)

    # Build datasets
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
    scheduler = build_scheduler(optimizer, cfg)

    epochs = train_cfg.get("epochs", 50)
    patience = train_cfg.get("early_stopping_patience", 10)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    best_val_acc = -1.0
    best_metrics = {}
    epochs_no_improve = 0
    best_state = None

    if fold_idx is not None:
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx}")
        print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
        print(f"  Parameters: {count_parameters(model):,}")
        print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, grad_clip
        )
        val_metrics, val_loss, expert_util = evaluate(
            model, val_loader, criterion, device, epoch
        )
        elapsed = time.time() - t0

        if scheduler is not None:
            scheduler.step()

        # Logging
        log_entry = {
            "fold": fold_idx,
            "train_loss": train_loss["total"],
            "train_task_loss": train_loss["task"],
            "val_loss": val_loss["total"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics.get("precision", 0.0),
            "val_recall": val_metrics.get("recall", 0.0),
            "val_auc_roc": val_metrics.get("auc_roc", 0.0),
            "expert_util": expert_util,
            "epoch_time_s": round(elapsed, 2),
        }
        logger.log_epoch(epoch, log_entry)

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss {train_loss['total']:.4f} | "
            f"val_loss {val_loss['total']:.4f} | "
            f"acc {val_metrics['accuracy']:.4f} | "
            f"f1 {val_metrics['f1']:.4f} | "
            f"experts {expert_util} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping on validation accuracy
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_metrics = {**val_metrics, "expert_util": expert_util}
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    # Save best checkpoint
    if best_state is not None:
        ckpt_path = os.path.join(save_dir, f"best_fold{fold_idx}.pt")
        torch.save(
            {"model_state_dict": best_state, "config": cfg, "fold": fold_idx,
             "metrics": best_metrics},
            ckpt_path,
        )
        print(f"  Saved best checkpoint -> {ckpt_path}")

    return best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CogMoE Training Pipeline")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run a single fold (0-indexed). Default: run all folds.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg)
    print(f"Device: {device}")

    # Directories
    save_dir = cfg.get("save_dir", "outputs/")
    log_dir = os.path.join(save_dir, "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Logger
    logger = ExperimentLogger(log_dir, experiment_name="cogmoe_train")
    logger.log_config(cfg)

    # Load data
    data_cfg = cfg["data"]
    csv_path = data_cfg["csv_path"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    # Cross-validation splits
    labels = df[data_cfg.get("label_col", "Label")].values
    n_folds = data_cfg.get("num_folds", 10)
    splits = get_segment_stratified_splits(labels, n_folds=n_folds, seed=seed)

    # Determine which folds to run
    if args.fold is not None:
        fold_indices = [args.fold]
    else:
        fold_indices = list(range(n_folds))

    # Run folds
    fold_results = []
    for fold_idx in fold_indices:
        train_idx, val_idx = splits[fold_idx]
        metrics = train_fold(fold_idx, train_idx, val_idx, df, cfg, device, logger, save_dir)
        fold_results.append(metrics)
        logger.log_fold_result(fold_idx, metrics)

    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("  Cross-Validation Results")
    print(f"{'='*60}")

    metric_keys = ["accuracy", "f1", "precision", "recall", "auc_roc"]
    aggregated = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated[f"{key}_mean"] = round(float(mean_val), 4)
            aggregated[f"{key}_std"] = round(float(std_val), 4)
            print(f"  {key:12s}: {mean_val:.4f} +/- {std_val:.4f}")

    # Per-fold summary
    print(f"\n  Per-fold accuracies: "
          + ", ".join(f"{r.get('accuracy', 0):.4f}" for r in fold_results))

    logger.log_final_results(aggregated)
    log_path = logger.save()
    print(f"\n  Logs saved to {log_path}")
    print("  Training complete.")


if __name__ == "__main__":
    main()
