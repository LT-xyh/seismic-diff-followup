from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from bg_pdr_fm.evaluation.motivation_statistics import (
    centered_normalized_cka,
    effective_dimension,
    low_high_decompose,
    sample_balanced_records,
)


def test_rbf_cka_is_one_for_identical_representations():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(48, 9))

    assert centered_normalized_cka(x, x) == pytest.approx(1.0, abs=1e-10)


def test_rbf_cka_is_invariant_to_positive_global_scaling():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(48, 9))

    assert centered_normalized_cka(x, 7.5 * x) == pytest.approx(1.0, abs=1e-8)


def test_matched_cka_exceeds_fixed_shuffled_control_on_synthetic_data():
    rng = np.random.default_rng(2027)
    x = rng.normal(size=(96, 12))
    y = x + 0.05 * rng.normal(size=x.shape)
    shuffled = y[rng.permutation(len(y))]

    matched = centered_normalized_cka(x, y)
    control = centered_normalized_cka(x, shuffled)

    assert matched > control


def test_effective_dimension_recovers_known_rank_two_matrix():
    x = np.vstack([np.eye(2), -np.eye(2)])

    result = effective_dimension(x)

    assert result.rank == 2
    assert result.effective_rank == pytest.approx(2.0, abs=1e-10)
    assert result.explained_variance[1] == pytest.approx(1.0, abs=1e-10)


def test_low_high_decomposition_reconstructs_input():
    rng = np.random.default_rng(2027)
    velocity = rng.normal(size=(3, 1, 17, 19))

    background, structure = low_high_decompose(velocity, kernel_size=5)

    np.testing.assert_allclose(background + structure, velocity, atol=1e-12, rtol=0.0)


def test_lowpass_matches_repository_zero_padded_average_pooling():
    rng = np.random.default_rng(2027)
    velocity = rng.normal(size=(2, 1, 11, 13))

    background, _ = low_high_decompose(velocity, kernel_size=5)
    reference = F.avg_pool2d(torch.from_numpy(velocity), kernel_size=5, stride=1, padding=2).numpy()

    np.testing.assert_allclose(background, reference, atol=1e-12, rtol=0.0)


def test_balanced_record_sampling_is_reproducible_and_train_only():
    subsets = (
        "FlatVelA",
        "FlatVelB",
        "CurveVelA",
        "CurveVelB",
        "FlatFaultA",
        "FlatFaultB",
        "CurveFaultA",
        "CurveFaultB",
    )
    records = [
        {"dataset_name": subset, "sample_index": index, "split": "train"}
        for subset in subsets
        for index in range(20)
    ]
    records.append({"dataset_name": "FlatVelA", "sample_index": 999, "split": "val"})

    first = sample_balanced_records(records, subsets, samples_per_subset=7, seed=2027)
    second = sample_balanced_records(records, subsets, samples_per_subset=7, seed=2027)

    assert first == second
    assert len(first) == len(subsets) * 7
    assert {row["split"] for row in first} == {"train"}
    identities = {(row["dataset_name"], row["sample_index"]) for row in first}
    assert len(identities) == len(first)
    assert {subset: sum(row["dataset_name"] == subset for row in first) for subset in subsets} == {
        subset: 7 for subset in subsets
    }
