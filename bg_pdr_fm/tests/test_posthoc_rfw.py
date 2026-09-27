from __future__ import annotations

import numpy as np


def test_identity_warp_and_jacobian_are_exact_on_a_small_field():
    from bg_pdr_fm.evaluation.visualize_posthoc_rfw import (
        jacobian_determinant,
        warp_with_displacement,
    )

    field = np.arange(25, dtype=np.float32).reshape(5, 5)
    displacement = np.zeros((2, 5, 5), dtype=np.float32)

    warped = warp_with_displacement(field, displacement)
    determinant = jacobian_determinant(displacement)

    assert np.allclose(warped, field)
    assert np.allclose(determinant, 1.0)


def test_registration_decision_requires_all_posthoc_conditions_and_respects_provenance():
    from bg_pdr_fm.evaluation.visualize_posthoc_rfw import registration_decision

    accepted = registration_decision(
        before_mae=0.30,
        after_mae=0.20,
        before_ssim=0.40,
        after_ssim=0.60,
        folding_fraction=0.001,
        folding_threshold=0.01,
        source_kind="independent_rendered_raster_proxy",
    )
    assert accepted["posthoc_registration_supported"] is True
    assert accepted["model_internal_rfw_supported"] is False

    rejected = registration_decision(
        before_mae=0.30,
        after_mae=0.20,
        before_ssim=0.40,
        after_ssim=0.60,
        folding_fraction=0.20,
        folding_threshold=0.01,
        source_kind="raw_model_array",
    )
    assert rejected["posthoc_registration_supported"] is False
    assert rejected["model_internal_rfw_supported"] is False


def test_decode_rendered_field_returns_a_finite_signed_scalar_proxy(tmp_path):
    from PIL import Image

    from bg_pdr_fm.evaluation.visualize_posthoc_rfw import decode_rendered_field

    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[..., 0] = 255
    image[..., 2] = np.array([[0, 32, 64, 96]] * 4, dtype=np.uint8)
    image[..., 3] = 255
    path = tmp_path / "field.png"
    Image.fromarray(image, mode="RGBA").save(path)

    field = decode_rendered_field(path)

    assert field.shape == (4, 4)
    assert np.isfinite(field).all()
    assert field.max() > field.min()

