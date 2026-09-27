from __future__ import annotations

import pytest

from bg_pdr_fm.evaluation.visualize_missing_modality_radar import (
    RADAR_DATASET_ORDER,
    RADAR_METHOD_ORDER,
    RADAR_MODE_ORDER,
    RADAR_PANEL_RANGES,
    RADAR_PANEL_TICKS,
    RADAR_RADIAL_RANGE,
    RADAR_RADIAL_TICKS,
    validate_radar_rows,
)


def _complete_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for method in RADAR_METHOD_ORDER:
        for mode in RADAR_MODE_ORDER:
            for dataset_name in RADAR_DATASET_ORDER:
                rows.append(
                    {
                        "method": method,
                        "missing_mode": mode,
                        "dataset_name": dataset_name,
                        "num_samples": "1",
                        "ssim": "0.8",
                    }
                )
    return rows


def test_validate_radar_rows_requires_exact_method_mode_dataset_grid():
    result = validate_radar_rows(_complete_rows())

    assert result["status"] == "passed"
    assert result["num_rows"] == 192
    assert result["num_methods"] == 6
    assert result["num_modes"] == 4
    assert result["num_datasets"] == 8


def test_validate_radar_rows_rejects_duplicate_grid_identity():
    rows = _complete_rows()
    rows[-1] = dict(rows[-2])

    with pytest.raises(ValueError, match="duplicate"):
        validate_radar_rows(rows)


def test_radar_display_uses_explicit_zoomed_raw_ssim_range():
    assert RADAR_RADIAL_RANGE == (0.50, 1.00)
    assert RADAR_RADIAL_TICKS == (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)


def test_focus_panels_use_their_own_display_range_without_changing_raw_ssim():
    assert RADAR_PANEL_RANGES["w/o well_log"] == (0.85, 1.00)
    assert RADAR_PANEL_RANGES["w/o PSTM"] == (0.85, 1.00)
    assert RADAR_PANEL_RANGES["w/o horizon"] == RADAR_RADIAL_RANGE
    assert RADAR_PANEL_RANGES["RMS only"] == RADAR_RADIAL_RANGE
    assert RADAR_PANEL_TICKS["w/o PSTM"] == (0.85, 0.90, 0.95, 1.00)
