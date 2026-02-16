# CogMoE: Signal-Quality-Guided Multimodal MoE for Cognitive Load Prediction

Official implementation of **CogMoE**, a two-stage framework that combines quality-aware multimodal signal recovery with signal-quality-specific Mixture-of-Experts (MoE) routing for cognitive load prediction from physiological signals.

> **Paper**: *CogMoE: Leveraging Signal Quality for Cognitive Load Prediction with Multimodal Mixture-of-Experts*
> ICLR 2026


## Overview

<img width="2016" height="774" alt="Figure_3_v3" src="https://github.com/user-attachments/assets/5865d72c-bca3-4be9-b6c9-a500e5404c51" />

CogMoE addresses the challenge of predicting cognitive load from noisy, incomplete multimodal physiological signals (ECG, EEG, Gaze, EDA). The framework consists of two stages:

1. **Stage 1 -- Quality-Aware Multimodal Synchronization and Recovery**: Uses Continuous Wavelet Transform (CWT) to convert signals to time-frequency representations, aligns them via 2D cross-correlation, generates cross-modal quality masks, and recovers corrupted regions via low-rank matrix completion.

2. **Stage 2 -- Signal-Quality-Specific Expert Modeling**: Encodes recovered features per modality, fuses them via cross-modal attention, and routes tokens through three specialized experts using Dynamic Pathway Gating (DPG) based on signal quality scores:
   - **High-Fidelity Expert (HFE)**: Lightweight FFN for clean signals (SNR > 15 dB)
   - **Noise-Resilient Expert (NRE)**: Residual FFN with noise-aware normalization for noisy signals
   - **Contextual Refinement Expert (CRE)**: Cross-modal attention for recovered/masked regions

Training uses **CORTEX Loss**: cross-entropy + noise suppression + refinement + adaptive gate regularization.

## Key Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Random Forest | 77.33 | 51.52 |
| XGBoost | 78.98 | 57.22 |
| BIOT | 79.49 | 53.29 |
| **CogMoE (Ours)** | **83.19** | **62.84** |

- ~2.27M parameters, 19.9 MB model size, 12.5M FLOPs
- Expert utilization: ~35% HFE / ~33% NRE / ~32% CRE

## Installation

```bash
# Clone the repository
git clone https://github.com/<username>/CogMoE.git
cd CogMoE

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.13
- NumPy, SciPy, Pandas, scikit-learn
- XGBoost, Optuna, matplotlib, tqdm, PyYAML

## Dataset

This project uses the **CL-Drive** dataset:

- **4,224 rows** (10-second segments from 21 participants)
- **4 modalities**: ECG (33 features), EEG (16 features), Gaze (30 features), EDA (10 features)
- **Binary labels**: 0 = low cognitive load (3,025 samples), 1 = high cognitive load (1,199 samples)

Place the combined feature CSV file in the project root:
```
ECG_EEG_Gaze_EDA_resampled_features_with_labels_combined.csv
```

## Quick Start

### Training

Train CogMoE with 10-fold cross-validation:
```bash
python scripts/train.py --config configs/default.yaml
```

Train a single fold:
```bash
python scripts/train.py --config configs/default.yaml --fold 0
```

### Evaluation

Evaluate a trained checkpoint:
```bash
python scripts/evaluate.py --config configs/default.yaml \
    --checkpoint outputs/best_fold0.pt
```

### Hyperparameter Sweep

Run Optuna optimization:
```bash
python scripts/sweep.py --config configs/default.yaml \
    --sweep configs/sweep.yaml --n-trials 100
```

### Baseline Comparison

Run all baselines on the same CV splits:
```bash
python scripts/run_baselines.py --config configs/default.yaml
```

### Ablation Studies

```bash
# MoE ablation: FFN-only vs MoE w/o CORTEX vs full
python scripts/ablation.py --config configs/default.yaml --study moe

# CORTEX loss component ablation
python scripts/ablation.py --config configs/default.yaml --study cortex

# Expert ablation
python scripts/ablation.py --config configs/default.yaml --study experts
```

### Running Tests

```bash
pytest tests/ -v
```

## Configuration

All experiments are configured via YAML files. Key parameters in `configs/default.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model.d_model` | 256 | Transformer hidden dimension |
| `model.nhead` | 4 | Number of attention heads |
| `model.d_ff` | 512 | Feed-forward dimension |
| `model.num_layers` | 2 | Number of CogMoE transformer layers |
| `model.num_experts` | 3 | Number of experts (HFE, NRE, CRE) |
| `loss.gamma` | 0.75 | Noise suppression loss weight |
| `loss.lambda_` | 0.6 | Refinement loss weight |
| `loss.beta_init` | 1.0 | Initial gate regularization weight |
| `loss.beta_max` | 0.2 | Maximum beta (adaptive decay) |
| `loss.alpha_decay` | 0.05 | Beta decay rate |
| `training.lr` | 3e-4 | Learning rate |
| `training.batch_size` | 32 | Batch size |
| `training.epochs` | 50 | Maximum training epochs |

## Model Architecture

### Dynamic Pathway Gating (DPG)

The DPG module computes quality-aware routing weights:

```
q_m = SNR_m * (1 - p_missing) * r_auto    (Eq. 4)
g_k(z, q) = softmax(W_{g,k} * [z; q])     (Eq. 5)
z_hat = sum_k g_k * f_k(z)                 (Eq. 6)
```

### CORTEX Loss

```
L = L_task + gamma * L_noise + lambda * L_refinement + beta(t) * R_gate    (Eq. 8)
beta(t) = min(beta_max, beta_init / (1 + alpha * t))                       (Eq. 9)
R_gate = sum_k (mean_i(g_k) - 1/K)^2                                      (Eq. 7)
```

## Citation

```bibtex
@inproceedings{cogmoe2026,
  title={CogMoE: Leveraging Signal Quality for Cognitive Load Prediction with Multimodal Mixture-of-Experts},
  author={...},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## License

This project is released for academic research purposes.
