#!/usr/bin/env python3
"""
CogMoE Hyperparameter Sweep with Optuna.

Performs Bayesian hyperparameter optimization using Optuna's TPE sampler
with median pruning. Trains on a single fold (fold 0) for efficiency and
returns validation accuracy as the objective.

Usage:
    python scripts/sweep.py --config configs/default.yaml --sweep configs/sweep.yaml --n-trials 100
    python scripts/sweep.py --config configs/default.yaml --sweep configs/sweep.yaml --n-trials 50 --fold 0
"""

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    print("Error: optuna is required for sweep. Install with: pip install optuna")
    sys.exit(1)

# ---- CogMoE imports ----
from cogmoe.models.cogmoe import build_model
from cogmoe.losses.cortex_loss import CORTEXLoss
from cogmoe.data.dataset import build_dataset, collate_fn
from cogmoe.data.augmentation import build_augmentation
from cogmoe.data.cross_validation import get_segment_stratified_splits
from cogmoe.utils.reproducibility import set_seed, get_device
from cogmoe.utils.metrics import compute_all_metrics, compute_expert_utilization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path):
    """Load YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(cfg, path):
    """Save configuration dict to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def sample_hyperparameters(trial, sweep_cfg):
    """
    Sample hyperparameters from the sweep search space using an Optuna trial.

    Args:
        trial: optuna.Trial - the Optuna trial object.
        sweep_cfg: dict - sweep search space configuration.

    Returns:
        dict - sampled hyperparameter values.
    """
    search_space = sweep_cfg.get("search_space", {})
    sampled = {}

    for param_name, param_cfg in search_space.items():
        param_type = param_cfg.get("type", "uniform")

        if param_type == "loguniform":
            sampled[param_name] = trial.suggest_float(
                param_name, param_cfg["low"], param_cfg["high"], log=True
            )
        elif param_type == "uniform":
            sampled[param_name] = trial.suggest_float(
                param_name, param_cfg["low"], param_cfg["high"]
            )
        elif param_type == "int":
            sampled[param_name] = trial.suggest_int(
                param_name, param_cfg["low"], param_cfg["high"]
            )
        elif param_type == "categorical":
            sampled[param_name] = trial.suggest_categorical(
                param_name, param_cfg["choices"]
            )
        else:
            raise ValueError(f"Unknown search space type: {param_type}")

    return sampled


def apply_hyperparameters(cfg, sampled):
    """
    Apply sampled hyperparameters to the base configuration.

    Maps sampled HP names to their locations in the config dict.

    Args:
        cfg: dict - base configuration (will be modified in-place).
        sampled: dict - sampled hyperparameter values.

    Returns:
        cfg: dict - modified configuration.
    """
    if "lr" in sampled:
        cfg["training"]["lr"] = sampled["lr"]
    if "batch_size" in sampled:
        cfg["training"]["batch_size"] = int(sampled["batch_size"])
    if "d_model" in sampled:
        cfg["model"]["d_model"] = int(sampled["d_model"])
    if "nhead" in sampled:
        cfg["model"]["nhead"] = int(sampled["nhead"])
    if "d_ff" in sampled:
        cfg["model"]["d_ff"] = int(sampled["d_ff"])
    if "num_layers" in sampled:
        cfg["model"]["num_layers"] = int(sampled["num_layers"])
    if "dropout" in sampled:
        # Apply a uniform dropout to all modality-specific dropouts
        drop_val = sampled["dropout"]
        cfg["model"]["dropout"] = {
            "ecg": drop_val, "eeg": drop_val,
            "eda": drop_val, "gaze": drop_val,
        }
    if "expert_dropout" in sampled:
        cfg["model"]["expert_dropout"] = sampled["expert_dropout"]

    # Loss coefficients (if included in sweep)
    if "gamma" in sampled:
        cfg["loss"]["gamma"] = sampled["gamma"]
    if "lambda_" in sampled:
        cfg["loss"]["lambda_"] = sampled["lambda_"]
    if "beta_init" in sampled:
        cfg["loss"]["beta_init"] = sampled["beta_init"]

    return cfg


# ---------------------------------------------------------------------------
# Training routine for a single trial
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, grad_clip=1.0):
    """Run one training epoch. Returns averaged total loss."""
    model.train()
    total_loss = 0.0
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

        total_loss += loss_dict["total"].item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_trial(model, loader, criterion, device, epoch=0):
    """Evaluate model. Returns accuracy and F1."""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []

    for batch in loader:
        inputs = {}
        for mod in model.modalities:
            if mod in batch:
                inputs[mod] = batch[mod].to(device)
        quality = batch["quality"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"].to(device)

        out = model(inputs=inputs, quality=quality, mask=mask)

        probs = torch.softmax(out["logits"], dim=-1)[:, 1]
        preds = out["logits"].argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_probs.append(probs.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_probs = torch.cat(all_probs).numpy()

    metrics = compute_all_metrics(all_targets, all_preds, all_probs)
    return metrics


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def create_objective(base_cfg, sweep_cfg, df, splits, fold_idx, device):
    """
    Create the Optuna objective function.

    Args:
        base_cfg: dict - base configuration.
        sweep_cfg: dict - sweep search space config.
        df: pd.DataFrame - loaded dataset.
        splits: list of (train_idx, val_idx) tuples.
        fold_idx: int - which fold to use for evaluation.
        device: torch.device.

    Returns:
        callable - objective function for Optuna.
    """
    train_idx, val_idx = splits[fold_idx]

    def objective(trial):
        # Deep copy base config and apply sampled HPs
        cfg = copy.deepcopy(base_cfg)
        sampled = sample_hyperparameters(trial, sweep_cfg)
        cfg = apply_hyperparameters(cfg, sampled)

        data_cfg = cfg["data"]
        train_cfg = cfg["training"]
        loss_cfg = cfg["loss"]
        aug_cfg = cfg.get("augmentation", {})

        # Validate d_model is divisible by nhead
        d_model = cfg["model"]["d_model"]
        nhead = cfg["model"]["nhead"]
        if d_model % nhead != 0:
            raise optuna.TrialPruned(
                f"d_model ({d_model}) not divisible by nhead ({nhead})"
            )

        # Build datasets
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
        try:
            model = build_model(cfg).to(device)
        except Exception as e:
            raise optuna.TrialPruned(f"Model build failed: {e}")

        criterion = CORTEXLoss(
            gamma=loss_cfg.get("gamma", 0.75),
            lambda_=loss_cfg.get("lambda_", 0.6),
            beta_init=loss_cfg.get("beta_init", 1.0),
            beta_max=loss_cfg.get("beta_max", 0.2),
            alpha_decay=loss_cfg.get("alpha_decay", 0.05),
            num_experts=cfg["model"].get("num_experts", 3),
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("lr", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

        epochs = train_cfg.get("epochs", 50)
        grad_clip = train_cfg.get("grad_clip", 1.0)
        patience = train_cfg.get("early_stopping_patience", 10)
        best_acc = -1.0
        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, grad_clip
            )

            # Evaluate
            val_metrics = evaluate_trial(model, val_loader, criterion, device, epoch)
            val_acc = val_metrics["accuracy"]

            # Report to Optuna for pruning
            trial.report(val_acc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping
            if val_acc > best_acc:
                best_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        return best_acc

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CogMoE Hyperparameter Sweep with Optuna"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to base YAML config file."
    )
    parser.add_argument(
        "--sweep", type=str, required=True,
        help="Path to sweep YAML config file."
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Number of Optuna trials (default: 100)."
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Which CV fold to use for evaluation (default: 0)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for best config YAML (default: outputs/best_config.yaml)."
    )
    parser.add_argument(
        "--study-name", type=str, default="cogmoe_sweep",
        help="Optuna study name."
    )
    args = parser.parse_args()

    # Load configs
    base_cfg = load_config(args.config)
    sweep_full = load_config(args.sweep)
    sweep_cfg = sweep_full.get("sweep", sweep_full)

    seed = base_cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(base_cfg)

    n_trials = args.n_trials
    if "n_trials" in sweep_cfg and args.n_trials == 100:
        # Use sweep config value if user did not override default
        n_trials = sweep_cfg.get("n_trials", 100)

    output_path = args.output
    if output_path is None:
        save_dir = base_cfg.get("save_dir", "outputs/")
        output_path = os.path.join(save_dir, "best_config.yaml")

    print(f"CogMoE Hyperparameter Sweep")
    print(f"{'='*60}")
    print(f"  Base config: {args.config}")
    print(f"  Sweep config: {args.sweep}")
    print(f"  Trials: {n_trials}")
    print(f"  Fold: {args.fold}")
    print(f"  Device: {device}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")

    # Load data
    data_cfg = base_cfg["data"]
    csv_path = data_cfg["csv_path"]
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns")

    # Generate CV splits
    labels = df[data_cfg.get("label_col", "Label")].values
    n_folds = data_cfg.get("num_folds", 10)
    splits = get_segment_stratified_splits(labels, n_folds=n_folds, seed=seed)

    # Create Optuna study
    direction = sweep_cfg.get("direction", "maximize")
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)

    study = optuna.create_study(
        study_name=args.study_name,
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )

    # Create objective function
    objective = create_objective(
        base_cfg, sweep_cfg, df, splits, args.fold, device
    )

    # Run optimization
    t_start = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - t_start

    # Report results
    print(f"\n{'='*60}")
    print(f"  Sweep Results")
    print(f"{'='*60}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"  Best value ({sweep_cfg.get('metric', 'accuracy')}): {study.best_value:.4f}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"\n  Best hyperparameters:")
    for key, val in study.best_params.items():
        if isinstance(val, float):
            print(f"    {key:20s}: {val:.6f}")
        else:
            print(f"    {key:20s}: {val}")

    # Build best config
    best_cfg = copy.deepcopy(base_cfg)
    best_cfg = apply_hyperparameters(best_cfg, study.best_params)
    best_cfg["_sweep_metadata"] = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "n_trials": n_trials,
        "fold": args.fold,
        "best_params": study.best_params,
    }

    # Save best config
    save_config(best_cfg, output_path)
    print(f"\n  Best config saved to {output_path}")

    # Print trial summary
    print(f"\n  Top 5 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1, reverse=True)
    for i, trial in enumerate(trials_sorted[:5]):
        if trial.value is not None:
            params_str = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in trial.params.items()
            )
            print(f"    #{trial.number:3d}: {trial.value:.4f} | {params_str}")

    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"\n  Trials completed: {n_complete}, pruned: {n_pruned}")
    print(f"  Sweep complete.")


if __name__ == "__main__":
    main()
