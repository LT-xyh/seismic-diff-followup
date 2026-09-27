"""Run formal SV_Inv_Net evaluation and update AAAI27 tables after training."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from bg_pdr_fm.evaluation.update_sv_inv_net_tables import update_tables


DEFAULT_CHECKPOINT = Path("logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints/last.ckpt")
DEFAULT_TEX = Path("docs/paper/aaai_LaTeX/seismic_diff_method_appendix.tex")
DEFAULT_FULL_DIR = Path("logs/bg_pdr_fm/aaai27/eval_sv_inv_net/sv_inv_net")
DEFAULT_MISSING_ROOT = Path("logs/bg_pdr_fm/aaai27/eval_missing_openfwi_sv_inv_net_formal")
DEFAULT_MISSING_DIR = DEFAULT_MISSING_ROOT / "sv_inv_net"


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_pipeline(*, checkpoint: Path, tex: Path, gpu: str, num_workers: int) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Formal SV_Inv_Net checkpoint not found: {checkpoint}")
    env_prefix = f"HIP_VISIBLE_DEVICES={gpu} CUDA_VISIBLE_DEVICES={gpu}"
    _run(
        [
            "bash",
            "-lc",
            (
                "source /public/home/xuyinghao/miniconda3/etc/profile.d/conda.sh && "
                "conda activate seg && "
                "source /opt/dtk-25.04.2/env.sh && source /opt/hyhal/env.sh && "
                f"{env_prefix} python -u -m bg_pdr_fm.evaluation.compare_experiments "
                "--config bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml"
            ),
        ]
    )
    _run(
        [
            "bash",
            "-lc",
            (
                "source /public/home/xuyinghao/miniconda3/etc/profile.d/conda.sh && "
                "conda activate seg && "
                "source /opt/dtk-25.04.2/env.sh && source /opt/hyhal/env.sh && "
                f"{env_prefix} python -u -m bg_pdr_fm.evaluation.run_missing_modality_benchmark "
                f"--jobs sv_inv_net --num-workers {num_workers} "
                f"--output-root {DEFAULT_MISSING_ROOT}"
            ),
        ]
    )
    update_tables(tex, DEFAULT_FULL_DIR, DEFAULT_MISSING_DIR)
    _run(
        [
            "bash",
            "-lc",
            (
                "cd docs/paper/aaai_LaTeX && "
                "TMPDIR=/public/home/xuyinghao/tmp/codex_tmp "
                "pdflatex -interaction=nonstopmode -halt-on-error -output-directory _build "
                "seismic_diff_method_appendix.tex"
            ),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--tex", default=str(DEFAULT_TEX))
    parser.add_argument("--gpu", default="3")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(argv)
    run_pipeline(checkpoint=Path(args.checkpoint), tex=Path(args.tex), gpu=str(args.gpu), num_workers=args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
