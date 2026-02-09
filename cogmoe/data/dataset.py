"""
CL-Drive dataset loading for CogMoE.
Loads pre-extracted features from the combined CSV file.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from cogmoe.utils.quality import compute_quality_vector


class CLDriveDataset(Dataset):
    """
    CL-Drive dataset: pre-extracted features per 10-second segment.

    Each sample consists of `num_windows` consecutive segments.
    Features are split into modality groups by column suffix.

    CSV columns:
        - *_ECG (33 features), *_EEG (16), *_Gaze (30), *_EDA (10)
        - Label (binary 0/1), participant_ID, Timestamp

    Args:
        dataframe: pd.DataFrame - the loaded CSV data (subset for this split).
        cfg: dict - data configuration section.
        transform: optional augmentation callable.
    """

    def __init__(self, dataframe, cfg, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.num_windows = cfg.get("num_windows", 1)
        self.label_col = cfg.get("label_col", "Label")
        self.modalities = cfg.get("modalities", ["ECG", "EEG", "Gaze", "EDA"])

        # Build column groups by suffix
        self.feature_cols = {}
        suffixes = cfg.get("feature_groups", {
            "ECG": "_ECG", "EEG": "_EEG", "Gaze": "_Gaze", "EDA": "_EDA"
        })
        for mod in self.modalities:
            suffix = suffixes.get(mod, f"_{mod}")
            cols = [c for c in self.df.columns if c.endswith(suffix)]
            self.feature_cols[mod] = cols

        # Precompute valid indices (enough rows for num_windows consecutive segments)
        # Group by participant to ensure sequences stay within same participant
        self.participant_col = cfg.get("participant_col", "participant_ID")
        self.indices = self._build_indices()

        # Precompute quality scores for all rows
        self._precompute_quality()

    def _build_indices(self):
        """Build valid (start_row, end_row) tuples for each sample."""
        indices = []
        if self.num_windows == 1:
            indices = list(range(len(self.df)))
        else:
            # Group by participant for temporal continuity
            for pid, group in self.df.groupby(self.participant_col):
                group_indices = group.index.tolist()
                for i in range(len(group_indices) - self.num_windows + 1):
                    indices.append(group_indices[i:i + self.num_windows])
        return indices

    def _precompute_quality(self):
        """Compute quality scores for each row."""
        self.quality_scores = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            mod_feats = {}
            for mod in self.modalities:
                vals = row[self.feature_cols[mod]].values.astype(float)
                mod_feats[mod] = vals
            q = compute_quality_vector(mod_feats)
            self.quality_scores.append(q)
        self.quality_scores = np.array(self.quality_scores)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.num_windows == 1:
            row_idx = self.indices[idx]
            rows = self.df.iloc[row_idx:row_idx + 1]
            quality = self.quality_scores[row_idx]
        else:
            row_indices = self.indices[idx]
            rows = self.df.iloc[row_indices]
            quality = self.quality_scores[row_indices[0]]

        # Extract features per modality
        sample = {}
        for mod in self.modalities:
            vals = rows[self.feature_cols[mod]].values.astype(np.float32)
            # Replace NaN/inf with 0
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            sample[mod] = torch.tensor(vals, dtype=torch.float32)

        # Label (use last row's label for multi-window sequences)
        label = int(rows.iloc[-1][self.label_col])

        # Quality vector
        sample["quality"] = torch.tensor(quality, dtype=torch.float32)

        # Modality mask (1.0 for all present modalities)
        sample["mask"] = torch.ones(len(self.modalities), dtype=torch.float32)

        sample["label"] = label

        # Apply augmentation
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def collate_fn(batch):
    """
    Custom collate function for CLDriveDataset.
    Stacks modality tensors, labels, quality, and masks.
    """
    modalities = [k for k in batch[0].keys() if k not in ("label", "quality", "mask")]

    result = {}
    for mod in modalities:
        result[mod] = torch.stack([b[mod] for b in batch])

    result["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    result["quality"] = torch.stack([b["quality"] for b in batch])
    result["mask"] = torch.stack([b["mask"] for b in batch])

    return result


def build_dataset(cfg, dataframe, transform=None):
    """
    Build CLDriveDataset from config and dataframe.

    Args:
        cfg: dict - data configuration section.
        dataframe: pd.DataFrame - CSV data for this split.
        transform: optional augmentation.

    Returns:
        CLDriveDataset.
    """
    return CLDriveDataset(dataframe, cfg, transform)
