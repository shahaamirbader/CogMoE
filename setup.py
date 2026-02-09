from setuptools import setup, find_packages

setup(
    name="cogmoe",
    version="1.0.0",
    description="CogMoE: Signal-Quality-Guided Multimodal MoE for Cognitive Load Prediction",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "numpy>=1.21",
        "scipy>=1.7",
        "pandas>=1.4",
        "scikit-learn>=1.0",
        "xgboost>=1.6",
        "pyyaml>=6.0",
        "optuna>=3.0",
        "matplotlib>=3.5",
        "tqdm>=4.62",
    ],
)
