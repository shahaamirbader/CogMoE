"""
Experiment logging utilities for CogMoE training.
"""

import os
import json
import time


class ExperimentLogger:
    """
    Logs training metrics, hyperparameters, and results to JSON files.

    Args:
        log_dir: str - directory for log files.
        experiment_name: str - name for this experiment run.
    """

    def __init__(self, log_dir, experiment_name="cogmoe"):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        os.makedirs(log_dir, exist_ok=True)
        self.log_data = {
            "experiment_name": experiment_name,
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": {},
            "epochs": [],
            "fold_results": [],
            "final_results": {},
        }

    def log_config(self, cfg):
        """Log experiment configuration."""
        self.log_data["config"] = cfg

    def log_epoch(self, epoch, metrics):
        """Log metrics for one training epoch."""
        entry = {"epoch": epoch, **metrics}
        self.log_data["epochs"].append(entry)

    def log_fold_result(self, fold, metrics):
        """Log results for one CV fold."""
        entry = {"fold": fold, **metrics}
        self.log_data["fold_results"].append(entry)

    def log_final_results(self, results):
        """Log aggregated final results across all folds."""
        self.log_data["final_results"] = results

    def save(self):
        """Write log data to JSON file."""
        path = os.path.join(self.log_dir, f"{self.experiment_name}.json")
        with open(path, "w") as f:
            json.dump(self.log_data, f, indent=2, default=str)
        return path
