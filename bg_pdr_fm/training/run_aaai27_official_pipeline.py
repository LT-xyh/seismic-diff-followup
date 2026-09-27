"""Run one official-capacity baseline training and its formal evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

from bg_pdr_fm.training.benchmark_config import load_benchmark_config


def build_pipeline_commands(
    *,
    train_config: Path,
    eval_config: Path,
    checkpoint: Path,
    summary: Path,
    python: str,
) -> list[list[str]]:
    commands: list[list[str]] = []
    if not checkpoint.is_file():
        commands.append(
            [python, "-m", "bg_pdr_fm.training.train_aaai27_benchmark", "--config", str(train_config)]
        )
    if not summary.is_file():
        commands.append(
            [python, "-m", "bg_pdr_fm.evaluation.compare_experiments", "--config", str(eval_config)]
        )
    return commands


def run_pipeline(train_config: Path, eval_config: Path, python: str) -> dict[str, object]:
    train_conf = load_benchmark_config(train_config)
    eval_conf = load_benchmark_config(eval_config)
    checkpoint = Path(str(OmegaConf.select(eval_conf, "evaluation.checkpoint")))
    summary = Path(str(OmegaConf.select(eval_conf, "evaluation.output_dir"))) / "summary.json"
    commands = build_pipeline_commands(
        train_config=train_config,
        eval_config=eval_config,
        checkpoint=checkpoint,
        summary=summary,
        python=python,
    )
    results: list[dict[str, object]] = []
    for command in commands:
        result = subprocess.run(command, check=False)
        results.append({"command": command, "returncode": result.returncode})
        if result.returncode != 0:
            raise RuntimeError(f"Official baseline pipeline command failed with code {result.returncode}: {command}")

    run_dir = Path(str(OmegaConf.select(train_conf, "training.checkpoint.dirpath"))).parent
    payload: dict[str, object] = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "train_config": str(train_config),
        "eval_config": str(eval_config),
        "checkpoint": str(checkpoint),
        "summary": str(summary),
        "commands": results,
        "complete": checkpoint.is_file() and summary.is_file(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pipeline_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not payload["complete"]:
        raise RuntimeError(f"Official baseline pipeline finished without required artifacts: {payload}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--python", default="/public/home/xuyinghao/miniconda3/envs/seg/bin/python")
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.train_config, args.eval_config, args.python), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
