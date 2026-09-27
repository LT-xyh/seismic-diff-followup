from __future__ import annotations

import csv
import json
from pathlib import Path

from bg_pdr_fm.evaluation.update_sv_inv_net_tables import OPENFWI_SUBSETS, update_tables


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_update_sv_inv_net_tables_replaces_full_and_missing_rows(tmp_path: Path):
    tex = tmp_path / "paper.tex"
    full_dir = tmp_path / "full"
    missing_dir = tmp_path / "missing"
    full_dir.mkdir()
    missing_dir.mkdir()
    tex.write_text(
        r"""
\begin{table*}
FlatVelA & Smooth-Dix & -- & -- & -- & -- & -- & 0.00M \\
 & SV\_Inv\_Net & -- & -- & -- & -- & -- & -- \\
\caption{Full}
\label{tab:openfwi_full_comparison}
\end{table*}
\begin{table*}
FlatVelA & Smooth-Dix & -- & -- & -- & -- & -- & 0.00M \\
 & SV\_Inv\_Net & -- & -- & -- & -- & -- & -- \\
\caption{Missing}
\label{tab:openfwi_missing_comparison}
\end{table*}
""",
        encoding="utf-8",
    )
    (full_dir / "summary.json").write_text(json.dumps({"num_samples": 33600}), encoding="utf-8")
    _write_csv(
        full_dir / "dataset_summary.csv",
        [
            {
                "dataset_name": subset,
                "num_samples": 1,
                "mae": 0.11111,
                "rmse": 0.22222,
                "ssim": 0.33333,
                "mae_l": 0.44444,
                "mae_h": 0.55555,
            }
            for subset in OPENFWI_SUBSETS
        ],
    )
    (missing_dir / "summary.json").write_text(json.dumps({"num_samples": 201600}), encoding="utf-8")
    missing_rows: list[dict[str, object]] = []
    for subset in OPENFWI_SUBSETS:
        for mode in ["w/o well_log", "w/o horizon", "w/o rms_vel", "w/o well+rms", "PSTM only"]:
            missing_rows.append(
                {
                    "dataset_name": subset,
                    "missing_mode": mode,
                    "num_samples": 2,
                    "mae": 0.9,
                    "rmse": 0.8,
                    "ssim": 0.7,
                    "mae_l": 0.6,
                    "mae_h": 0.5,
                }
            )
        missing_rows.append(
            {
                "dataset_name": subset,
                "missing_mode": "full",
                "num_samples": 2,
                "mae": 0.0,
                "rmse": 0.0,
                "ssim": 1.0,
                "mae_l": 0.0,
                "mae_h": 0.0,
            }
        )
    _write_csv(missing_dir / "dataset_missing_mode_summary.csv", missing_rows)

    update_tables(tex, full_dir, missing_dir)

    text = tex.read_text(encoding="utf-8")
    assert "SV\\_Inv\\_Net & 0.1111 & 0.2222 & 0.3333 & 0.4444 & 0.5555 & 2.89M" in text
    assert "SV\\_Inv\\_Net & 0.9000 & 0.8000 & 0.7000 & 0.6000 & 0.5000 & 2.89M" in text
