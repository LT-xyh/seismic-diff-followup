from __future__ import annotations

import csv
import json

import numpy as np

from bg_pdr_fm.evaluation.analyze_motivation_statistics import (
    extract_v2_render_data,
    load_motivation_artifacts,
)


SUBSETS = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)
MODALITIES = ("migrated_image", "horizon", "rms_vel", "well_log_mask")
COMPONENTS = ("background", "structure")


def _write_csv(path, rows):
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_v2_render_data_preserves_alignment_rank_and_curve_assets(tmp_path):
    alignment_rows = []
    for modality_index, modality in enumerate(MODALITIES):
        for component_index, component in enumerate(COMPONENTS):
            alignment_rows.append(
                {
                    "scope": "all_subsets_summary",
                    "subset": "ALL",
                    "modality": modality,
                    "component": component,
                    "rbf_cka_median": str(0.1 * (1 + modality_index + component_index)),
                }
            )
    _write_csv(tmp_path / "alignment_by_subset.csv", alignment_rows)

    rank_rows = []
    for subset_index, subset in enumerate(SUBSETS):
        for component_index, component in enumerate(COMPONENTS):
            rank_rows.append(
                {
                    "scope": "subset",
                    "subset": subset,
                    "component": component,
                    "effective_rank": str(2 + subset_index + component_index),
                }
            )
    _write_csv(tmp_path / "effective_rank_by_subset.csv", rank_rows)

    background = np.arange(8 * 4, dtype=np.float64).reshape(8, 4)
    structure = background + 0.5
    np.savez(tmp_path / "explained_variance_curves.npz", background_curves=background, structure_curves=structure)
    (tmp_path / "analysis_config.json").write_text(json.dumps({"analysis_seed": 2027}), encoding="utf-8")

    artifacts = load_motivation_artifacts(tmp_path)
    render_data = extract_v2_render_data(artifacts)

    assert render_data["alignment_matrix"].shape == (4, 2)
    assert render_data["alignment_matrix"][0, 0] == 0.1
    assert render_data["alignment_matrix"][3, 1] == 0.5
    assert render_data["rank_pairs"].shape == (8, 2)
    np.testing.assert_allclose(render_data["rank_pairs"][0], [2.0, 3.0])
    np.testing.assert_allclose(render_data["curves"]["background"], background)
    np.testing.assert_allclose(render_data["curves"]["structure"], structure)
