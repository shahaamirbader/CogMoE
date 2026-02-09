"""
Reproducibility and environment utilities for CogMoE.
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seeds for reproducibility across torch, numpy, random, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cfg=None):
    """
    Determine compute device from config or auto-detect.

    Args:
        cfg: optional dict with 'device' key ('auto', 'cuda', 'cpu', 'mps').

    Returns:
        torch.device
    """
    device_str = "auto"
    if cfg and "device" in cfg:
        device_str = cfg["device"]

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, sample_input, device, n_runs=100):
    """
    Measure average inference time in milliseconds.

    Args:
        model: nn.Module.
        sample_input: dict of input tensors.
        device: torch.device.
        n_runs: int - number of forward passes to average.

    Returns:
        float - average time in ms.
    """
    import time

    model.eval()
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(**{k: v.to(device) for k, v in sample_input.items()})

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            model(**{k: v.to(device) for k, v in sample_input.items()})
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / n_runs * 1000.0
