"""Training-free statistics used by the Figure 1 motivation analysis.

The functions in this module deliberately operate on arrays only.  They do not
import a model, checkpoint, optimizer, or learned dimensionality-reduction
method.  The command-line analysis builds model-ready observations through the
repository dataloader, then calls these deterministic statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class EffectiveDimensionResult:
    """Eigen-spectrum summary of a centered sample-by-feature matrix."""

    eigenvalues: np.ndarray
    explained_variance: np.ndarray
    effective_rank: float
    rank: int


def _as_sample_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D sample-by-feature matrix, got {array.shape}.")
    if array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least two samples and one feature.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def pairwise_squared_distances(x: np.ndarray) -> np.ndarray:
    """Return the symmetric squared-Euclidean distance matrix for ``x``."""

    matrix = _as_sample_matrix(x, "x")
    squared_norms = np.einsum("ij,ij->i", matrix, matrix)[:, None]
    distances = squared_norms + squared_norms.T - 2.0 * (matrix @ matrix.T)
    return np.maximum(distances, 0.0)


def median_heuristic_bandwidth(x: np.ndarray) -> float:
    """Return the median non-zero Euclidean distance used as RBF bandwidth."""

    distances = pairwise_squared_distances(x)
    upper = distances[np.triu_indices_from(distances, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.sqrt(np.median(positive)))
    return bandwidth if bandwidth > 0.0 and np.isfinite(bandwidth) else 1.0


def rbf_kernel(x: np.ndarray, bandwidth: float | None = None) -> tuple[np.ndarray, float]:
    """Construct an RBF Gram matrix and return its recorded bandwidth."""

    matrix = _as_sample_matrix(x, "x")
    sigma = float(bandwidth if bandwidth is not None else median_heuristic_bandwidth(matrix))
    if sigma <= 0.0 or not np.isfinite(sigma):
        raise ValueError(f"RBF bandwidth must be positive and finite, got {sigma!r}.")
    kernel = np.exp(-pairwise_squared_distances(matrix) / (2.0 * sigma * sigma))
    return kernel, sigma


def linear_kernel(x: np.ndarray) -> np.ndarray:
    matrix = _as_sample_matrix(x, "x")
    return matrix @ matrix.T


def center_kernel(kernel: np.ndarray) -> np.ndarray:
    """Center a Gram matrix with the exact double-centering identity."""

    gram = np.asarray(kernel, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError(f"kernel must be square, got {gram.shape}.")
    row_mean = gram.mean(axis=1, keepdims=True)
    column_mean = gram.mean(axis=0, keepdims=True)
    grand_mean = float(gram.mean())
    return gram - row_mean - column_mean + grand_mean


def normalized_hsic(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    """Compute centered normalized HSIC, equivalently kernel CKA."""

    centered_x = center_kernel(gram_x)
    centered_y = center_kernel(gram_y)
    numerator = float(np.sum(centered_x * centered_y))
    denominator = float(np.linalg.norm(centered_x, ord="fro") * np.linalg.norm(centered_y, ord="fro"))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(numerator / denominator, -1.0, 1.0))


def centered_normalized_cka(x: np.ndarray, y: np.ndarray, *, kernel: str = "rbf") -> float:
    """Compute linear or median-heuristic RBF CKA between two representations."""

    matrix_x = _as_sample_matrix(x, "x")
    matrix_y = _as_sample_matrix(y, "y")
    if matrix_x.shape[0] != matrix_y.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")
    kernel_name = str(kernel).lower()
    if kernel_name == "rbf":
        gram_x, _ = rbf_kernel(matrix_x)
        gram_y, _ = rbf_kernel(matrix_y)
    elif kernel_name == "linear":
        gram_x = linear_kernel(matrix_x)
        gram_y = linear_kernel(matrix_y)
    else:
        raise ValueError(f"Unknown kernel {kernel!r}; expected 'rbf' or 'linear'.")
    return normalized_hsic(gram_x, gram_y)


def cka_with_bandwidths(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return RBF CKA and the two median-heuristic bandwidths."""

    matrix_x = _as_sample_matrix(x, "x")
    matrix_y = _as_sample_matrix(y, "y")
    if matrix_x.shape[0] != matrix_y.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")
    gram_x, bandwidth_x = rbf_kernel(matrix_x)
    gram_y, bandwidth_y = rbf_kernel(matrix_y)
    return normalized_hsic(gram_x, gram_y), bandwidth_x, bandwidth_y


def effective_dimension(x: np.ndarray, *, tolerance: float = 1e-12) -> EffectiveDimensionResult:
    """Compute cumulative variance and participation-ratio effective rank.

    The covariance spectrum is obtained from ``X X^T / (N-1)`` after removing
    the feature-wise mean.  No feature-wise variance normalization or
    per-sample normalization is applied.
    """

    matrix = _as_sample_matrix(x, "x")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    gram = (centered @ centered.T) / float(matrix.shape[0] - 1)
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    scale = float(eigenvalues.max(initial=0.0))
    threshold = max(float(tolerance) * scale, np.finfo(np.float64).eps)
    eigenvalues[eigenvalues < threshold] = 0.0
    total = float(eigenvalues.sum())
    if total <= np.finfo(np.float64).eps:
        explained = np.zeros_like(eigenvalues)
        effective_rank = 0.0
    else:
        explained = np.cumsum(eigenvalues) / total
        squared_total = total * total
        effective_rank = float(squared_total / np.sum(eigenvalues * eigenvalues))
    rank = int(np.count_nonzero(eigenvalues > 0.0))
    return EffectiveDimensionResult(
        eigenvalues=eigenvalues,
        explained_variance=explained,
        effective_rank=effective_rank,
        rank=rank,
    )


def low_high_decompose(value: np.ndarray, *, kernel_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Match ``LowHighPassFilter.lowpass`` using zero-padded average pooling.

    PyTorch's implementation uses stride-one average pooling with
    ``count_include_pad=True``.  Constant zero padding and a fixed kernel
    reproduce that boundary convention without importing a model.
    """

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"value must have at least two spatial dimensions, got {array.shape}.")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    pad = kernel_size // 2
    padded = np.pad(array, [(0, 0)] * (array.ndim - 2) + [(pad, pad), (pad, pad)], mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=(kernel_size, kernel_size),
        axis=(-2, -1),
    )
    background = windows.mean(axis=(-2, -1))
    structure = array - background
    return background, structure


def flatten_dimension_normalize(value: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Flatten samples and divide by ``sqrt(feature_count)``.

    This makes squared distances comparable to a mean per-coordinate distance
    while preserving the dataloader's normalized values and avoiding learned
    projections.  The operation is intentionally not applied to the B/S
    effective-dimension analysis.
    """

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"value must have a sample axis and feature axes, got {array.shape}.")
    matrix = array.reshape(array.shape[0], -1)
    feature_count = int(matrix.shape[1])
    return matrix / np.sqrt(float(feature_count)), {
        "feature_count": feature_count,
        "scale": f"1/sqrt({feature_count})",
        "operation": "flatten_then_divide_by_sqrt_feature_count",
    }


def sample_balanced_records(
    records: Sequence[Mapping[str, Any]],
    subset_names: Iterable[str],
    *,
    samples_per_subset: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select an equal, reproducible number of global-training records per subset."""

    if samples_per_subset <= 0:
        raise ValueError("samples_per_subset must be positive.")
    names = tuple(str(name) for name in subset_names)
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in names}
    for record in records:
        split = str(record.get("split", "train")).lower()
        if split not in {"train", "global_train"}:
            continue
        name = str(record.get("dataset_name", ""))
        if name in grouped:
            grouped[name].append(record)

    rng = np.random.default_rng(int(seed))
    selected: list[dict[str, Any]] = []
    for name in names:
        candidates = grouped[name]
        order = rng.permutation(len(candidates))
        count = min(int(samples_per_subset), len(candidates))
        for index in order[:count]:
            row = dict(candidates[int(index)])
            row["split"] = "train"
            selected.append(row)
    return selected
