"""
Data augmentation transforms for CogMoE training.
Paper Section A.4.3: Gaussian noise injection, random temporal shifts, channel dropout.
"""

import torch
import numpy as np


class GaussianNoiseInjection:
    """Add Gaussian noise to feature values."""

    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, sample):
        for key in list(sample.keys()):
            if isinstance(sample[key], torch.Tensor) and key not in ("label", "quality", "mask"):
                noise = torch.randn_like(sample[key]) * self.std
                sample[key] = sample[key] + noise
        return sample


class ChannelDropout:
    """Zero out entire modality features with probability p."""

    def __init__(self, prob=0.1, modalities=None):
        self.prob = prob
        self.modalities = modalities or ["ECG", "EEG", "Gaze", "EDA"]

    def __call__(self, sample):
        for i, mod in enumerate(self.modalities):
            if mod in sample and np.random.random() < self.prob:
                sample[mod] = torch.zeros_like(sample[mod])
                if "mask" in sample:
                    sample["mask"][i] = 0.0
        return sample


class TemporalJitter:
    """Add small random perturbations to simulate temporal misalignment."""

    def __init__(self, scale=0.02):
        self.scale = scale

    def __call__(self, sample):
        for key in list(sample.keys()):
            if isinstance(sample[key], torch.Tensor) and key not in ("label", "quality", "mask"):
                jitter = torch.randn_like(sample[key]) * self.scale * sample[key].abs().mean()
                sample[key] = sample[key] + jitter
        return sample


class ComposeAugmentation:
    """Chain multiple augmentation transforms."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_augmentation(cfg):
    """
    Factory function to build augmentation pipeline from config.

    Args:
        cfg: dict - augmentation configuration section.

    Returns:
        ComposeAugmentation or None if augmentation disabled.
    """
    if not cfg.get("enabled", False):
        return None

    transforms = []

    if cfg.get("gaussian_noise_std", 0) > 0:
        transforms.append(GaussianNoiseInjection(std=cfg["gaussian_noise_std"]))

    if cfg.get("channel_dropout_prob", 0) > 0:
        transforms.append(ChannelDropout(prob=cfg["channel_dropout_prob"]))

    if cfg.get("temporal_shift_max", 0) > 0:
        transforms.append(TemporalJitter(scale=0.02))

    if not transforms:
        return None

    return ComposeAugmentation(transforms)
