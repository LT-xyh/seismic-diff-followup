from __future__ import annotations

import numpy as np

from bg_pdr_fm.evaluation.visualize_condition_embeddings import (
    compute_embedding_metrics,
    knn_overlap,
    stratified_complete_indices,
)


def _synthetic_records() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    dataset_names = ["FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB"]
    dataset_ids = np.repeat(np.arange(len(dataset_names), dtype=np.int64), 12)
    sample_indices = np.arange(dataset_ids.size, dtype=np.int64) + 1000
    availability = np.ones((dataset_ids.size, 4), dtype=bool)
    availability[1, 3] = False
    availability[14, 0] = False
    return dataset_ids, sample_indices, availability, dataset_names


def test_stratified_complete_indices_are_deterministic_and_balanced() -> None:
    dataset_ids, sample_indices, availability, dataset_names = _synthetic_records()

    selected_a = stratified_complete_indices(
        dataset_ids,
        sample_indices,
        availability,
        dataset_names,
        per_subset=5,
        seed=42,
    )
    selected_b = stratified_complete_indices(
        dataset_ids,
        sample_indices,
        availability,
        dataset_names,
        per_subset=5,
        seed=42,
    )

    np.testing.assert_array_equal(selected_a, selected_b)
    assert selected_a.size == 20
    assert np.all(availability[selected_a])
    assert np.unique(dataset_ids[selected_a], return_counts=True)[1].tolist() == [5, 5, 5, 5]
    assert np.intersect1d(selected_a, np.array([1, 14])).size == 0


def test_stratified_complete_indices_reject_insufficient_subset() -> None:
    dataset_ids, sample_indices, availability, dataset_names = _synthetic_records()

    availability[dataset_ids == 3] = False
    try:
        stratified_complete_indices(
            dataset_ids,
            sample_indices,
            availability,
            dataset_names,
            per_subset=5,
            seed=42,
        )
    except ValueError as exc:
        assert "CurveVelB" in str(exc)
    else:
        raise AssertionError("Expected an insufficient-subset ValueError.")


def test_knn_overlap_and_embedding_metrics_are_finite() -> None:
    rng = np.random.default_rng(7)
    embedding = rng.normal(size=(24, 8)).astype(np.float32)
    labels = np.repeat(["background", "structural"], 12)
    coordinates = embedding[:, :2]
    shifted = coordinates + 1e-4

    overlap = knn_overlap(coordinates, shifted, k=5)
    metrics = compute_embedding_metrics(
        embedding,
        labels,
        coordinates,
        trustworthiness_neighbors=5,
    )

    assert 0.0 <= overlap <= 1.0
    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert 0.0 <= metrics["trustworthiness"] <= 1.0
    assert np.isfinite(metrics["silhouette_original"])
    assert np.isfinite(metrics["silhouette_2d"])
