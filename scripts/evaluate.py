#!/usr/bin/env python3
"""
CogMoE Evaluation Script.

Load a trained checkpoint and run inference on the test (validation) split,
reporting accuracy, F1, precision, recall, AUC-ROC, and expert utilization.

Usage:
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/best_fold0.pt
    python scripts/evaluate.py --config configs/default.yaml \
        --checkpoint outputs/best_fold0.pt --fold 0
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml

from cogmoe.models.cogmoe import CogMoE, build_model
from cogmoe.losses.cortex_loss import CORTEXLoss
from cogmoe.data.dataset import CLDriveDataset, collate_fn, build_dataset
from cogmoe.data.cross_validation import get_segment_stratified_splits
from cogmoe.utils.reproducibility import set_seed, get_device, count_parameters
from cogmoe.utils.metrics import compute_all_metrics, compute_expert_utilization


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, loader, device):
    """
    Run model inference on all batches of a DataLoader.

    Returns:
        all_preds: np.ndarray of predicted class labels.
        all_targets: np.ndarray of ground-truth labels.
        all_probs: np.ndarray of predicted probability for positive class.
        all_gate_weights: list of gate weight tensors across all batches.
    """
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    all_gate_weights = []

    for batch in loader:
        inputs = {}
        for mod in model.modalities:
            if mod in batch:
                inputs[mod] = batch[mod].to(device)
        quality = batch["quality"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"]

        out = model(inputs=inputs, quality=quality, mask=mask)

        probs = torch.softmax(out["logits"], dim=-1)[:, 1]
        preds = out["logits"].argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_targets.append(targets)
        all_probs.append(probs.cpu())
        all_gate_weights.extend(out["gate_weights_list"])

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    return all_preds, all_targets, all_probs, all_gate_weights


def main():
    parser = argparse.ArgumentParser(description="CogMoE Evaluation")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index for test split. If not given, uses fold "
                             "stored in checkpoint or defaults to fold 0.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size for evaluation.")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg)
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Use config from checkpoint if available, but allow CLI config override
    ckpt_cfg = ckpt.get("config", cfg)
    # Merge: use CLI config for data paths but checkpoint config for model arch
    eval_cfg = cfg.copy()
    if "model" in ckpt_cfg:
        eval_cfg["model"] = ckpt_cfg["model"]

    # Build model and load weights
    model = build_model(eval_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Model parameters: {count_parameters(model):,}")

    # Determine fold
    fold_idx = args.fold
    if fold_idx is None:
        fold_idx = ckpt.get("fold", 0)
    print(f"  Evaluating on fold {fold_idx}")

    # Load data
    data_cfg = cfg["data"]
    csv_path = data_cfg["csv_path"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows")

    # Get CV splits and extract test set
    labels = df[data_cfg.get("label_col", "Label")].values
    n_folds = data_cfg.get("num_folds", 10)
    splits = get_segment_stratified_splits(labels, n_folds=n_folds, seed=seed)
    _, test_idx = splits[fold_idx]
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Build dataset and loader
    test_ds = build_dataset(data_cfg, test_df, transform=None)
    batch_size = args.batch_size or cfg["training"].get("batch_size", 32)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # Run inference
    print("Running inference ...")
    all_preds, all_targets, all_probs, all_gate_weights = run_inference(
        model, test_loader, device
    )

    # Compute metrics
    metrics = compute_all_metrics(all_targets, all_preds, all_probs)
    expert_util = compute_expert_utilization(all_gate_weights)

    # Report results
    print(f"\n{'='*50}")
    print(f"  Evaluation Results  (Fold {fold_idx})")
    print(f"{'='*50}")
    print(f"  Samples evaluated : {len(all_targets)}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  F1 Score          : {metrics['f1']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  AUC-ROC           : {metrics.get('auc_roc', 0.0):.4f}")
    print(f"\n  Expert Utilization:")
    for expert_name, pct in expert_util.items():
        print(f"    {expert_name}: {pct:.1f}%")

    # Also print stored checkpoint metrics for comparison if available
    ckpt_metrics = ckpt.get("metrics", {})
    if ckpt_metrics:
        print(f"\n  Checkpoint stored metrics:")
        for k, v in ckpt_metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    print(f"{'='*50}")
    print("  Evaluation complete.")


if __name__ == "__main__":
    main()
