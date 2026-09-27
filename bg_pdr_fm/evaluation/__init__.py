"""Evaluation helpers for BG-PDR-FM."""

from .evaluate_autoencoder import run_reconstruction_evaluation
from .evaluate_bg_pdr_fm import load_evaluation_checkpoints, run_evaluation
from .benchmark_metrics import compute_freq_metrics, compute_velocity_metrics


def run_contrastive_visualization(*args, **kwargs):
    from .visualize_contrastive import run_visualization

    return run_visualization(*args, **kwargs)


def run_benchmark_evaluation(*args, **kwargs):
    from .compare_experiments import run_benchmark_evaluation as _run

    return _run(*args, **kwargs)

__all__ = [
    "load_evaluation_checkpoints",
    "compute_freq_metrics",
    "compute_velocity_metrics",
    "run_benchmark_evaluation",
    "run_contrastive_visualization",
    "run_evaluation",
    "run_reconstruction_evaluation",
]
