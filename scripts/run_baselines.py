#!/usr/bin/env python3
"""
CogMoE Baseline Comparison Script.
Runs all baselines (RF, XGBoost, MLP, KNN, VGG, ResNet, BIOT) on same CV splits.

Ensures fair comparison by using identical cross-validation splits, seed,
and evaluation metrics as the main CogMoE training pipeline.

Usage:
    python scripts/run_baselines.py --config configs/default.yaml
    python scripts/run_baselines.py --config configs/default.yaml --fold 0
    python scripts/run_baselines.py --config configs/default.yaml --baselines RF,XGBoost
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
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ---- CogMoE imports ----
from cogmoe.baselines.traditional import (
    RandomForestBaseline,
    XGBoostBaseline,
    MLPBaseline,
    KNNBaseline,
)
from cogmoe.baselines.deep_learning import VGGBaseline, ResNetBaseline
from cogmoe.baselines.biot import BIOT
from cogmoe.data.cross_validation import get_segment_stratified_splits
from cogmoe.utils.reproducibility import set_seed, get_device
from cogmoe.utils.metrics import compute_all_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def extract_features_and_labels(df, data_cfg):
    """
    Extract concatenated feature matrix X and label vector y from the DataFrame.

    All 89 features (33 ECG + 16 EEG + 30 Gaze + 10 EDA) are concatenated
    into a single feature vector per sample.

    Args:
        df: pd.DataFrame - the dataset.
        data_cfg: dict - data configuration section.

    Returns:
        X: np.ndarray of shape (n_samples, n_features) - concatenated features.
        y: np.ndarray of shape (n_samples,) - binary labels.
    """
    modalities = data_cfg.get("modalities", ["ECG", "EEG", "Gaze", "EDA"])
    suffixes = data_cfg.get("feature_groups", {
        "ECG": "_ECG", "EEG": "_EEG", "Gaze": "_Gaze", "EDA": "_EDA"
    })
    label_col = data_cfg.get("label_col", "Label")

    feature_cols = []
    for mod in modalities:
        suffix = suffixes.get(mod, f"_{mod}")
        cols = [c for c in df.columns if c.endswith(suffix)]
        feature_cols.extend(cols)

    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df[label_col].values.astype(int)

    return X, y


# ---------------------------------------------------------------------------
# Traditional baseline runner
# ---------------------------------------------------------------------------

def run_traditional_baseline(name, X_train, y_train, X_test, y_test, seed=42):
    """
    Train and evaluate a traditional ML baseline.

    Args:
        name: str - baseline name (RF, XGBoost, MLP, KNN).
        X_train, y_train: training data.
        X_test, y_test: test data.
        seed: int - random seed.

    Returns:
        dict with accuracy, f1, precision, recall, auc_roc, and train_time.
    """
    baseline_map = {
        "RF": lambda: RandomForestBaseline(random_state=seed),
        "XGBoost": lambda: XGBoostBaseline(random_state=seed),
        "MLP": lambda: MLPBaseline(random_state=seed),
        "KNN": lambda: KNNBaseline(),
    }

    if name not in baseline_map:
        raise ValueError(f"Unknown traditional baseline: {name}")

    model = baseline_map[name]()

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    preds = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        proba = None

    metrics = compute_all_metrics(y_test, preds, proba)
    metrics["train_time"] = round(train_time, 3)

    return metrics


# ---------------------------------------------------------------------------
# Deep learning baseline runner
# ---------------------------------------------------------------------------

def train_dl_baseline(model, X_train, y_train, device, epochs=50, lr=1e-3,
                      batch_size=32, patience=10):
    """
    Train a PyTorch deep learning baseline model.

    Args:
        model: nn.Module - the baseline model.
        X_train: np.ndarray of shape (n, d).
        y_train: np.ndarray of shape (n,).
        device: torch.device.
        epochs: int - max training epochs.
        lr: float - learning rate.
        batch_size: int - batch size.
        patience: int - early stopping patience.

    Returns:
        model: nn.Module - trained model.
    """
    model = model.to(device)
    model.train()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return model


@torch.no_grad()
def evaluate_dl_baseline(model, X_test, y_test, device, batch_size=256):
    """
    Evaluate a PyTorch deep learning baseline model.

    Args:
        model: nn.Module - trained model.
        X_test: np.ndarray.
        y_test: np.ndarray.
        device: torch.device.
        batch_size: int.

    Returns:
        dict with accuracy, f1, precision, recall, auc_roc.
    """
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_targets, all_probs = [], [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y_batch.numpy())
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    return compute_all_metrics(all_targets, all_preds, all_probs)


def run_dl_baseline(name, X_train, y_train, X_test, y_test, device, seed=42):
    """
    Train and evaluate a deep learning baseline.

    Args:
        name: str - baseline name (VGG, ResNet, BIOT).
        X_train, y_train: training data.
        X_test, y_test: test data.
        device: torch.device.
        seed: int - random seed.

    Returns:
        dict with accuracy, f1, precision, recall, auc_roc, train_time.
    """
    in_dim = X_train.shape[1]
    set_seed(seed)

    model_map = {
        "VGG": lambda: VGGBaseline(in_dim=in_dim, num_classes=2, variant="feat"),
        "ResNet": lambda: ResNetBaseline(in_dim=in_dim, num_classes=2, variant="feat"),
        "BIOT": lambda: BIOT(in_dim=in_dim, d_model=256, nhead=4, num_layers=2, num_classes=2),
    }

    if name not in model_map:
        raise ValueError(f"Unknown DL baseline: {name}")

    model = model_map[name]()

    t0 = time.time()
    model = train_dl_baseline(model, X_train, y_train, device)
    train_time = time.time() - t0

    metrics = evaluate_dl_baseline(model, X_test, y_test, device)
    metrics["train_time"] = round(train_time, 3)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CogMoE Baseline Comparison Script"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file."
    )
    parser.add_argument(
        "--fold", type=int, default=None,
        help="Run a single fold (0-indexed). Default: run all folds."
    )
    parser.add_argument(
        "--baselines", type=str, default=None,
        help="Comma-separated list of baselines to run "
             "(default: RF,XGBoost,MLP,KNN,VGG,ResNet,BIOT)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON (default: outputs/baseline_results.json)."
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg)

    # Determine which baselines to run
    traditional_names = ["RF", "XGBoost", "MLP", "KNN"]
    dl_names = ["VGG", "ResNet", "BIOT"]
    all_names = traditional_names + dl_names

    if args.baselines:
        selected = [b.strip() for b in args.baselines.split(",")]
        for name in selected:
            if name not in all_names:
                print(f"Warning: Unknown baseline '{name}', skipping.")
        selected = [b for b in selected if b in all_names]
    else:
        selected = all_names

    # Output path
    output_path = args.output
    if output_path is None:
        save_dir = cfg.get("save_dir", "outputs/")
        output_path = os.path.join(save_dir, "baseline_results.json")

    print(f"CogMoE Baseline Comparison")
    print(f"{'='*60}")
    print(f"  Config:    {args.config}")
    print(f"  Baselines: {', '.join(selected)}")
    print(f"  Device:    {device}")
    print(f"  Output:    {output_path}")
    print(f"{'='*60}\n")

    # Load data
    data_cfg = cfg["data"]
    csv_path = data_cfg["csv_path"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    # Extract features and labels
    X, y = extract_features_and_labels(df, data_cfg)
    print(f"  Feature matrix: {X.shape}, Labels: {y.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Cross-validation splits (same as train.py)
    n_folds = data_cfg.get("num_folds", 10)
    splits = get_segment_stratified_splits(y, n_folds=n_folds, seed=seed)

    # Determine which folds to run
    if args.fold is not None:
        fold_indices = [args.fold]
    else:
        fold_indices = list(range(n_folds))

    # Results storage: {baseline_name: {fold_idx: metrics_dict}}
    results = {name: {} for name in selected}

    t_total_start = time.time()

    for fold_idx in fold_indices:
        train_idx, val_idx = splits[fold_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[val_idx], y[val_idx]

        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx}: Train={len(train_idx)}, Test={len(val_idx)}")
        print(f"{'='*60}")

        for name in selected:
            set_seed(seed)
            print(f"  Running {name:10s} ... ", end="", flush=True)

            try:
                if name in traditional_names:
                    metrics = run_traditional_baseline(
                        name, X_train, y_train, X_test, y_test, seed=seed
                    )
                else:
                    metrics = run_dl_baseline(
                        name, X_train, y_train, X_test, y_test, device, seed=seed
                    )

                results[name][fold_idx] = metrics
                print(
                    f"acc={metrics['accuracy']:.4f}, "
                    f"f1={metrics['f1']:.4f}, "
                    f"time={metrics.get('train_time', 0):.2f}s"
                )
            except ImportError as e:
                print(f"SKIPPED ({e})")
                results[name][fold_idx] = {"error": str(e)}
            except Exception as e:
                print(f"FAILED ({e})")
                results[name][fold_idx] = {"error": str(e)}

    total_time = time.time() - t_total_start

    # Aggregate results across folds
    print(f"\n{'='*60}")
    print(f"  Cross-Validation Results (mean +/- std)")
    print(f"{'='*60}")

    metric_keys = ["accuracy", "f1", "precision", "recall", "auc_roc"]
    aggregated = {}

    # Header
    header = f"  {'Model':10s}"
    for key in metric_keys:
        header += f" | {key:>18s}"
    print(header)
    print(f"  {'-'*(len(header) - 2)}")

    for name in selected:
        fold_metrics = results[name]
        valid_folds = {k: v for k, v in fold_metrics.items() if "error" not in v}

        if not valid_folds:
            print(f"  {name:10s} | All folds failed")
            aggregated[name] = {"error": "All folds failed"}
            continue

        agg = {}
        row = f"  {name:10s}"
        for key in metric_keys:
            values = [m[key] for m in valid_folds.values() if key in m]
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                agg[f"{key}_mean"] = round(float(mean_val), 4)
                agg[f"{key}_std"] = round(float(std_val), 4)
                row += f" | {mean_val:7.4f} +/- {std_val:.4f}"
            else:
                row += f" | {'N/A':>18s}"

        agg["n_folds"] = len(valid_folds)
        aggregated[name] = agg
        print(row)

    # Save results
    output_data = {
        "config": args.config,
        "seed": seed,
        "n_folds": len(fold_indices),
        "baselines": selected,
        "per_fold_results": {
            name: {str(k): v for k, v in fold_results.items()}
            for name, fold_results in results.items()
        },
        "aggregated": aggregated,
        "total_time_s": round(total_time, 2),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    print(f"\n  Total time: {total_time:.1f}s")
    print(f"  Results saved to {output_path}")
    print(f"  Baseline comparison complete.")


if __name__ == "__main__":
    main()
