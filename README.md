# SeisMIC-Diff / PD-BG-RFM

Research repository for multimodal seismic subsurface velocity inversion. The current paper-linked method is **Physics-Decoupled Background-Guided Residual Flow Matching (PD-BG-RFM)**.

This repository contains the canonical source, release configurations, reproducibility checks, paper sources, and research handoff notes. It does not contain OpenFWI data, LMDB stores, checkpoints, private credentials, or large training logs.

## Start here

For a new collaborator or an AI research assistant, read in this order:

1. [`AGENTS.md`](AGENTS.md) — operating rules and evidence boundaries.
2. [`docs/research/START_HERE.md`](docs/research/START_HERE.md) — project map and reading order.
3. [`docs/research/PROJECT_STATE.md`](docs/research/PROJECT_STATE.md) — current paper, rejection, code, and evidence status.
4. [`docs/research/EXPERIMENT_PROTOCOL.md`](docs/research/EXPERIMENT_PROTOCOL.md) — data, split, seed, metric, and baseline contracts.
5. [`docs/research/EVIDENCE_INDEX.md`](docs/research/EVIDENCE_INDEX.md) — claim-to-artifact mapping.
6. [`docs/research/NEXT_WORK.md`](docs/research/NEXT_WORK.md) — prioritized follow-up work.

## Canonical code

- `bg_pdr_fm/`: model, data, training, evaluation, and diagnostics implementation.
- `scripts/aaai27/`: supported preparation, training, evaluation, smoke, and validation entrypoints.
- `reproducibility/`: source manifest, dependency contract, third-party notices, and release validator.
- `tests/` and `bg_pdr_fm/tests/`: automated checks.
- `docs/paper/AAAI2027/`: editable paper sources and paper-linked release map.
- `docs/paper/aaaii/`: immutable AAAI submission snapshot; do not use it as the development source.

## Fast checks

```bash
python reproducibility/validate_release.py
python scripts/aaai27/smoke.py
pytest -q tests bg_pdr_fm/tests
```

The smoke path is CPU-oriented and does not require OpenFWI data or a checkpoint. Full training requires the documented OpenFWI layout and an appropriate PyTorch accelerator build. See [`reproducibility/README.md`](reproducibility/README.md) for the release contract.

## Research status

The AAAI submission was rejected. The current work is to make the code and evidence independently auditable, then decide whether the method needs a bounded robustness/reproducibility experiment set and a narrower paper claim. No result should be treated as replicated across seeds unless the corresponding manifest and result files are present.
