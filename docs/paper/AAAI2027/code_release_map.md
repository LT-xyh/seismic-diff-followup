# Paper-to-Code Release Map

This map identifies the project-owned implementation path for the AAAI paper.

The canonical development source is the repository-root `bg_pdr_fm/`, `scripts/aaai27/`, and `reproducibility/` tree. The directory `docs/paper/aaaii/` is an immutable submission snapshot and is not the development source.
It is an evidence index, not a claim that historical logs or checkpoints are
recomputed by the source tree alone.

| Paper material | Project-owned implementation | Checklist evidence |
| --- | --- | --- |
| Physics-decoupled condition learning, Eqs. (1)-(3) | `bg_pdr_fm/models/encoder.py`, `bg_pdr_fm/models/contrastive.py`, `bg_pdr_fm/models/adapters.py` | 4.3 experiment source; 4.5 method references |
| Deterministic background and prediction-consistent residual transport, Eqs. (4)-(6) | `bg_pdr_fm/models/generators.py`, `bg_pdr_fm/lightning/bg_pdr_fm_module.py`, `bg_pdr_fm/lightning/stage_losses.py` | 4.3 experiment source; 4.5 method references |
| OpenFWI preprocessing, deterministic split, well observations, and masks | `bg_pdr_fm/data/datasets.py`, `bg_pdr_fm/data/openfwi_lmdb.py`, `bg_pdr_fm/data/batch.py`, `scripts/aaai27/prepare_openfwi.py` | 4.2 preprocessing; 4.6 randomness |
| Training configuration and execution | `bg_pdr_fm/training/train_bg_pdr_fm.py`, `bg_pdr_fm/training/benchmark_config.py`, `bg_pdr_fm/reproducibility.py`, `bg_pdr_fm/configs/release/aaai27/train.yaml`, `scripts/aaai27/train.py` | 4.3 source; 4.6 randomness; 4.14 final parameters |
| Held-out evaluation and metrics | `bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py`, `bg_pdr_fm/evaluation/benchmark_metrics.py`, `scripts/aaai27/evaluate.py` | 4.8 metrics; 4.9 run accounting |
| Source boundaries and environment | `reproducibility/release_manifest.yaml`, `reproducibility/requirements.txt`, `reproducibility/THIRD_PARTY_NOTICES.md` | 4.3 source inventory; 4.4 availability gate; 4.7 infrastructure |

The source-derived OpenFWI baseline definitions are declared separately in
`reproducibility/release_manifest.yaml` and retain their upstream notices. The
final public license and archive scope remain publication-owner decisions.
