"""Update AAAI27 paper tables from formal SV_Inv_Net OpenFWI evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


OPENFWI_SUBSETS = [
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
]

METRIC_KEYS = ["mae", "rmse", "ssim", "mae_l", "mae_h"]
METHOD_LABEL = "SV\\_Inv\\_Net"
PARAMS = "2.89M"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: float | str) -> str:
    return f"{float(value):.4f}"


def _validate_full_eval(path: Path) -> dict[str, dict[str, str]]:
    summary_path = path / "summary.json"
    dataset_path = path / "dataset_summary.csv"
    if not summary_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"Missing full-eval outputs under {path}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("num_samples", -1)) != 33600:
        raise RuntimeError(f"{summary_path} has num_samples={summary.get('num_samples')}; expected 33600.")
    rows = {row["dataset_name"]: row for row in _read_csv(dataset_path)}
    missing = [name for name in OPENFWI_SUBSETS if name not in rows]
    if missing:
        raise RuntimeError(f"{dataset_path} missing OpenFWI subsets: {missing}.")
    return rows


def _validate_missing_eval(path: Path) -> dict[str, dict[str, str]]:
    summary_path = path / "summary.json"
    dataset_mode_path = path / "dataset_missing_mode_summary.csv"
    if not summary_path.is_file() or not dataset_mode_path.is_file():
        raise FileNotFoundError(f"Missing missing-mode outputs under {path}.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if int(summary.get("num_samples", -1)) != 201600:
        raise RuntimeError(f"{summary_path} has num_samples={summary.get('num_samples')}; expected 201600.")
    grouped: dict[str, list[dict[str, str]]] = {subset: [] for subset in OPENFWI_SUBSETS}
    for row in _read_csv(dataset_mode_path):
        subset = row["dataset_name"]
        if subset in grouped and row["missing_mode"] != "full":
            grouped[subset].append(row)
    averaged: dict[str, dict[str, str]] = {}
    for subset, rows in grouped.items():
        if len(rows) != 5:
            raise RuntimeError(f"{dataset_mode_path} subset {subset} has {len(rows)} missing rows; expected 5.")
        total = sum(int(row["num_samples"]) for row in rows)
        averaged[subset] = {
            key: str(sum(float(row[key]) * int(row["num_samples"]) for row in rows) / total)
            for key in METRIC_KEYS
        }
    return averaged


def _sv_line(values: dict[str, str]) -> str:
    metrics = " & ".join(_fmt(values[key]) for key in METRIC_KEYS)
    return f" & {METHOD_LABEL} & {metrics} & {PARAMS} \\\\"


def _replace_sv_rows_in_table(table: str, values_by_subset: dict[str, dict[str, str]]) -> str:
    current_subset = ""
    out: list[str] = []
    for line in table.splitlines():
        if line.startswith(tuple(OPENFWI_SUBSETS)):
            current_subset = line.split("&", 1)[0].strip()
        if f"& {METHOD_LABEL} &" in line:
            if current_subset not in values_by_subset:
                raise RuntimeError(f"Could not map SV_Inv_Net row to subset in line: {line}")
            out.append(_sv_line(values_by_subset[current_subset]))
        else:
            out.append(line)
    return "\n".join(out)


def _replace_table_by_label(text: str, label: str, values_by_subset: dict[str, dict[str, str]]) -> str:
    label_pos = text.index(f"\\label{{{label}}}")
    start = text.rfind("\\begin{table*}", 0, label_pos)
    end = text.index("\\end{table*}", label_pos) + len("\\end{table*}")
    table = text[start:end]
    return text[:start] + _replace_sv_rows_in_table(table, values_by_subset) + text[end:]


def update_tables(tex_path: Path, full_eval_dir: Path, missing_eval_dir: Path) -> None:
    full_values = _validate_full_eval(full_eval_dir)
    missing_values = _validate_missing_eval(missing_eval_dir)
    text = tex_path.read_text(encoding="utf-8")
    text = _replace_table_by_label(text, "tab:openfwi_full_comparison", full_values)
    text = _replace_table_by_label(text, "tab:openfwi_missing_comparison", missing_values)
    tex_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", default="docs/paper/aaai_LaTeX/seismic_diff_method_appendix.tex")
    parser.add_argument("--full-eval-dir", default="logs/bg_pdr_fm/aaai27/eval_sv_inv_net/sv_inv_net")
    parser.add_argument("--missing-eval-dir", default="logs/bg_pdr_fm/aaai27/eval_missing_openfwi_sv_inv_net_formal/sv_inv_net")
    args = parser.parse_args()
    update_tables(Path(args.tex), Path(args.full_eval_dir), Path(args.missing_eval_dir))


if __name__ == "__main__":
    main()
