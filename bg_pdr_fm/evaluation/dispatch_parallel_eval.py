"""GPU-aware multi-process dispatcher for BG-PDR-FM evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import _config_summary
from bg_pdr_fm.runtime import configure_checkpoint_worker_environment
from bg_pdr_fm.training.dispatch_bg_pdr_fm_parallel_upgrade_queue import (
    idle_gpus_from_stats,
    parse_rocm_smi_text,
)


DEFAULT_OPENFWI_DATASETS = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)
METRIC_FIELDS = (
    "mae",
    "rmse",
    "mse",
    "ssim",
    "mae_l",
    "mae_h",
    "bg_mae",
    "epsilon_H_B",
    "rho_B",
    "rho_hat_B",
    "residual_energy_ratio_l2",
    "transport_target_ratio",
    "alpha_hat",
)


class WorkerResult:
    def __init__(
        self,
        *,
        name: str,
        output_dir: Path,
        datasets: list[str],
        gpu: int,
        returncode: int | None,
        start_time: float,
        end_time: float | None = None,
        pid: int | None = None,
    ) -> None:
        self.name = name
        self.output_dir = output_dir
        self.datasets = datasets
        self.gpu = gpu
        self.returncode = returncode
        self.start_time = start_time
        self.end_time = end_time
        self.pid = pid


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def append_event(output_dir: Path, event: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": now_iso(), **event}, sort_keys=True, default=str) + "\n")


def parse_csv_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def best_successful_batch(results: list[dict[str, Any]]) -> int | None:
    successes = [int(item["batch_size"]) for item in results if item.get("status") == "success"]
    if not successes:
        return None
    return max(successes)


def make_dataset_shards(datasets: list[str], workers: int) -> list[list[str]]:
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if workers == 3 and datasets == list(DEFAULT_OPENFWI_DATASETS):
        return [
            ["FlatVelA", "FlatVelB", "FlatFaultA"],
            ["CurveVelA", "CurveFaultA", "FlatFaultB"],
            ["CurveVelB", "CurveFaultB"],
        ]
    shards = [[] for _ in range(workers)]
    for idx, dataset in enumerate(datasets):
        shards[idx % workers].append(dataset)
    return [shard for shard in shards if shard]


def prepare_worker_config(
    conf: Any,
    *,
    output_dir: Path,
    datasets: list[str],
    batch_size: int,
    max_batches: int | None,
    save_arrays: bool,
    save_panels: bool,
) -> Any:
    worker = OmegaConf.create(OmegaConf.to_container(conf, resolve=False))
    OmegaConf.update(worker, "data.post_split_datasets", list(datasets), merge=False)
    OmegaConf.update(worker, "training.batch_size", int(batch_size), merge=True)
    OmegaConf.update(worker, "training.devices", 1, merge=True)
    OmegaConf.update(worker, "training.strategy", None, merge=True)
    OmegaConf.update(worker, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(worker, "evaluation.max_batches", max_batches, merge=True)
    OmegaConf.update(worker, "evaluation.save_arrays", bool(save_arrays), merge=True)
    OmegaConf.update(worker, "evaluation.save_panels", bool(save_panels), merge=True)
    return worker


def _rocm_smi_stats() -> dict[int, dict[str, float]]:
    result = subprocess.run(
        ["rocm-smi", "--showuse", "--showmemuse"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return parse_rocm_smi_text(result.stdout)
    result = subprocess.run(
        ["hy-smi"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {}
    return parse_hy_smi_text(result.stdout)


def parse_hy_smi_text(text: str) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 7 or not parts[0].isdigit():
            continue
        gpu = int(parts[0])
        mem_text = parts[5].rstrip("%")
        util_text = parts[6].rstrip("%")
        try:
            stats[gpu] = {"mem_mib": float(mem_text), "mem_percent": float(mem_text), "util": float(util_text)}
        except ValueError:
            continue
    return stats


def select_gpus(gpus: list[int], *, auto_idle: bool, max_util: float, max_mem_mib: float) -> list[int]:
    if not auto_idle:
        return gpus
    stats = _rocm_smi_stats()
    if not stats:
        return gpus
    return idle_gpus_from_stats(stats, allowed_gpus=gpus, max_util=max_util, max_mem_mib=max_mem_mib)


def command_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(Path.cwd())
    env["PYTHONUNBUFFERED"] = "1"
    return configure_checkpoint_worker_environment(env)


def classify_worker_failure(worker_dir: Path, returncode: int | None) -> str:
    metrics = worker_dir / "metrics.csv"
    summary = worker_dir / "summary.json"
    if returncode == 0 and metrics.is_file() and summary.is_file():
        return ""
    log = (worker_dir / "eval.log").read_text(encoding="utf-8", errors="replace") if (worker_dir / "eval.log").exists() else ""
    lowered = log.lower()
    if "outofmemory" in lowered or "out of memory" in lowered:
        return "oom"
    if "traceback" in lowered:
        return "python_exception"
    if not metrics.exists():
        return "silent_no_output"
    if returncode not in (0, None):
        return f"returncode_{returncode}"
    return "missing_summary"


def count_metric_rows(metrics_path: Path) -> int:
    if not metrics_path.is_file():
        return 0
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def worker_state(result: WorkerResult) -> dict[str, Any]:
    elapsed = None if result.end_time is None else max(0.0, result.end_time - result.start_time)
    samples = count_metric_rows(result.output_dir / "metrics.csv")
    samples_per_sec = None if not elapsed else samples / elapsed
    if result.returncode is None:
        failure_reason = ""
        status = "running"
    else:
        failure_reason = classify_worker_failure(result.output_dir, result.returncode)
        status = "completed" if not failure_reason else "failed"
    return {
        "name": result.name,
        "status": status,
        "failure_reason": failure_reason,
        "gpu": result.gpu,
        "pid": result.pid,
        "datasets": result.datasets,
        "output_dir": str(result.output_dir),
        "returncode": result.returncode,
        "samples": samples,
        "elapsed_seconds": elapsed,
        "samples_per_sec": samples_per_sec,
    }


def write_queue_state(output_dir: Path, *, workers: list[WorkerResult], pending: list[str]) -> dict[str, Any]:
    states = [worker_state(worker) for worker in workers]
    state = {
        "time": now_iso(),
        "pending": pending,
        "running": [item for item in states if item["status"] == "running"],
        "completed": [item for item in states if item["status"] == "completed"],
        "failed": [item for item in states if item["status"] == "failed"],
        "workers": states,
    }
    write_json(output_dir / "eval_queue_state.json", state)
    return state


def read_metric_rows(metrics_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _mean(rows: list[dict[str, str]], key: str) -> float:
    vals: list[float] = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            vals.append(number)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def merge_worker_outputs(
    *,
    output_dir: Path,
    worker_results: list[WorkerResult],
    config_summary: dict[str, Any],
    expected_samples: int | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] | None = None
    all_rows: list[dict[str, str]] = []
    group_rows: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    failures = []
    for result in worker_results:
        failure_reason = classify_worker_failure(result.output_dir, result.returncode)
        if failure_reason:
            failures.append({"name": result.name, "reason": failure_reason})
            continue
        names, rows = read_metric_rows(result.output_dir / "metrics.csv")
        if fieldnames is None:
            fieldnames = names
        for row in rows:
            row = dict(row)
            row["worker"] = result.name
            row["datasets"] = ",".join(result.datasets)
            all_rows.append(row)
        summary_path = result.output_dir / "summary.json"
        if summary_path.exists() and not checkpoints:
            worker_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            checkpoints = list(worker_summary.get("checkpoints", []))
        group_rows.append(
            {
                "worker": result.name,
                "gpu": result.gpu,
                "datasets": ",".join(result.datasets),
                "num_samples": len(rows),
                "ssim": _mean(rows, "ssim"),
                "mae": _mean(rows, "mae"),
                "mae_l": _mean(rows, "mae_l"),
                "mae_h": _mean(rows, "mae_h"),
                "bg_mae": _mean(rows, "bg_mae"),
                "samples_per_sec": worker_state(result)["samples_per_sec"],
            }
        )

    merged_fields = list(fieldnames or [])
    for extra in ("worker", "datasets"):
        if extra not in merged_fields:
            merged_fields.append(extra)
    with (output_dir / "merged_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    with (output_dir / "dataset_group_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["worker", "gpu", "datasets", "num_samples", "ssim", "mae", "mae_l", "mae_h", "bg_mae", "samples_per_sec"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(group_rows)

    rows_by_dataset: dict[str, list[dict[str, str]]] = {}
    for row in all_rows:
        dataset_name = row.get("dataset_name", "")
        if dataset_name:
            rows_by_dataset.setdefault(dataset_name, []).append(row)
    dataset_rows = [
        {
            "dataset": dataset_name,
            "num_samples": len(rows),
            **{key: _mean(rows, key) for key in METRIC_FIELDS},
        }
        for dataset_name, rows in sorted(rows_by_dataset.items())
    ]
    dataset_fields = ["dataset", "num_samples", *METRIC_FIELDS]
    with (output_dir / "dataset_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=dataset_fields)
        writer.writeheader()
        writer.writerows(dataset_rows)

    manifest_fields = ["dataset_id", "dataset_name", "source_sample_index", "worker"]
    with (output_dir / "held_out_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in manifest_fields} for row in all_rows])

    protocol_failures: list[dict[str, str]] = []
    if config_summary.get("data_name") == "openfwi":
        identities = [
            (row.get("dataset_name", ""), row.get("source_sample_index", ""))
            for row in all_rows
        ]
        missing_identity = sum(not dataset_name or source_index == "" for dataset_name, source_index in identities)
        duplicate_count = len(identities) - len(set(identities))
        if missing_identity:
            protocol_failures.append({"name": "held_out_identity", "reason": f"missing_identity:{missing_identity}"})
        if duplicate_count:
            protocol_failures.append({"name": "held_out_identity", "reason": f"duplicate_identity:{duplicate_count}"})
    failures.extend(protocol_failures)

    num_samples = len(all_rows)
    status = "completed"
    if failures:
        status = "failed"
    elif expected_samples is not None and num_samples != expected_samples:
        status = "incomplete"
    summary = {
        "status": status,
        "num_samples": num_samples,
        "expected_samples": expected_samples,
        "means": {key: _mean(all_rows, key) for key in METRIC_FIELDS},
        "config": config_summary,
        "checkpoints": checkpoints,
        "worker_groups": group_rows,
        "dataset_summaries": dataset_rows,
        "failures": failures,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def run_worker(
    *,
    python: str,
    gpu: int,
    worker_dir: Path,
    datasets: list[str],
    conf: Any,
    batch_size: int,
    max_batches: int | None,
    save_arrays: bool,
    save_panels: bool,
) -> tuple[subprocess.Popen[str], WorkerResult]:
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_conf = prepare_worker_config(
        conf,
        output_dir=worker_dir,
        datasets=datasets,
        batch_size=batch_size,
        max_batches=max_batches,
        save_arrays=save_arrays,
        save_panels=save_panels,
    )
    config_path = worker_dir / "eval_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(worker_conf, resolve=False), encoding="utf-8")
    log_path = worker_dir / "eval.log"
    command = [python, "-m", "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm", "--config", str(config_path)]
    start = time.time()
    process = subprocess.Popen(
        command,
        cwd=Path.cwd(),
        env=command_env(gpu),
        stdout=log_path.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    result = WorkerResult(name=worker_dir.name, output_dir=worker_dir, datasets=datasets, gpu=gpu, returncode=None, start_time=start, pid=process.pid)
    write_json(worker_dir / "worker_state.json", worker_state(result))
    return process, result


def run_batch_ramp(
    *,
    conf: Any,
    output_dir: Path,
    python: str,
    gpu: int,
    datasets: list[str],
    batch_candidates: list[int],
) -> tuple[int, list[dict[str, Any]]]:
    ramp_dir = output_dir / "batch_ramp"
    ramp_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for batch_size in batch_candidates:
        worker_dir = ramp_dir / f"bs{batch_size}"
        process, result = run_worker(
            python=python,
            gpu=gpu,
            worker_dir=worker_dir,
            datasets=datasets,
            conf=conf,
            batch_size=batch_size,
            max_batches=1,
            save_arrays=False,
            save_panels=False,
        )
        returncode = process.wait()
        result.returncode = returncode
        result.end_time = time.time()
        state = worker_state(result)
        status = "success" if state["status"] == "completed" else "failed"
        results.append(
            {
                "batch_size": int(batch_size),
                "status": status,
                "returncode": returncode,
                "failure_reason": state["failure_reason"],
                "samples": state["samples"],
                "output_dir": str(worker_dir),
            }
        )
    selected = best_successful_batch(results)
    write_json(ramp_dir / "batch_ramp_summary.json", {"results": results, "selected_batch_size": selected})
    if selected is None:
        raise RuntimeError(f"Batch ramp failed for all candidates: {batch_candidates}")
    return selected, results


def run_dispatch(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    conf = OmegaConf.load(args.config)
    datasets = parse_csv_strings(args.datasets) if args.datasets else list(OmegaConf.select(conf, "data.openfwi_datasets", default=list(DEFAULT_OPENFWI_DATASETS)))
    requested_gpus = parse_csv_ints(args.gpus)
    gpus = select_gpus(requested_gpus, auto_idle=bool(args.auto_idle), max_util=float(args.max_util), max_mem_mib=float(args.max_mem_mib))
    if not gpus:
        raise RuntimeError("No GPUs available for parallel evaluation.")
    shards = make_dataset_shards(datasets, workers=len(gpus))
    if len(shards) > len(gpus):
        raise RuntimeError("Internal error: more shards than GPUs.")
    append_event(output_dir, {"event": "dispatch_start", "gpus": gpus, "shards": shards})
    batch_size = int(args.batch_size or OmegaConf.select(conf, "training.batch_size", default=100))
    ramp_results: list[dict[str, Any]] = []
    if args.batch_candidates:
        candidates = parse_csv_ints(args.batch_candidates)
        batch_size, ramp_results = run_batch_ramp(
            conf=conf,
            output_dir=output_dir,
            python=args.python,
            gpu=gpus[0],
            datasets=shards[0],
            batch_candidates=candidates,
        )
        append_event(output_dir, {"event": "batch_ramp_end", "selected_batch_size": batch_size, "results": ramp_results})
    workers: list[WorkerResult] = []
    running: list[tuple[subprocess.Popen[str], WorkerResult]] = []
    for idx, shard in enumerate(shards):
        worker_dir = output_dir / f"worker_{idx:02d}"
        process, result = run_worker(
            python=args.python,
            gpu=gpus[idx],
            worker_dir=worker_dir,
            datasets=shard,
            conf=conf,
            batch_size=batch_size,
            max_batches=args.max_batches,
            save_arrays=bool(args.save_arrays),
            save_panels=bool(args.save_panels),
        )
        workers.append(result)
        running.append((process, result))
        append_event(output_dir, {"event": "worker_start", "worker": result.name, "pid": result.pid, "gpu": result.gpu, "datasets": shard})
    while running:
        still_running = []
        for process, result in running:
            returncode = process.poll()
            if returncode is None:
                still_running.append((process, result))
                continue
            result.returncode = returncode
            result.end_time = time.time()
            write_json(result.output_dir / "worker_state.json", worker_state(result))
            append_event(output_dir, {"event": "worker_end", **worker_state(result)})
        running = still_running
        write_queue_state(output_dir, workers=workers, pending=[])
        if running:
            time.sleep(max(1, int(args.poll_seconds)))
    summary = merge_worker_outputs(
        output_dir=output_dir,
        worker_results=workers,
        config_summary=_config_summary(conf),
        expected_samples=args.expected_samples,
    )
    if ramp_results:
        summary["batch_ramp"] = {"selected_batch_size": batch_size, "results": ramp_results}
        write_json(output_dir / "summary.json", summary)
    append_event(output_dir, {"event": "dispatch_end", "status": summary["status"], "num_samples": summary["num_samples"]})
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="7,6,5,4,3,2,1,0")
    parser.add_argument("--datasets", default="")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-candidates", default="", help="Optional comma-separated batch sizes for max_batches=1 ramp before full eval.")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--expected-samples", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-util", type=float, default=5.0)
    parser.add_argument("--max-mem-mib", type=float, default=1024.0)
    parser.add_argument("--auto-idle", action="store_true")
    parser.add_argument("--save-arrays", action="store_true")
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    run_dispatch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
