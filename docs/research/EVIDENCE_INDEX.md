# Evidence index

| Claim or component | Source of truth | Status boundary |
|---|---|---|
| Model and loss implementation | `bg_pdr_fm/models/`, `bg_pdr_fm/lightning/` | Static source evidence until a run manifest exists |
| Formal training profile | `bg_pdr_fm/configs/release/aaai27/train.yaml` | Config evidence |
| Data preparation and split | `scripts/aaai27/prepare_openfwi.py`, `bg_pdr_fm/data/` | Requires prepared-data manifest for empirical evidence |
| Evaluation metrics | `bg_pdr_fm/evaluation/benchmark_metrics.py`, `scripts/aaai27/evaluate.py` | Evaluator definition; results require checkpoint manifest |
| Release boundary | `reproducibility/release_manifest.yaml` | Validated by `validate_release.py` |
| Paper-to-code mapping | `docs/paper/AAAI2027/code_release_map.md` | Documentation mapping |
| AAAI submitted manuscript | `docs/paper/aaaii/` | Immutable historical snapshot |
| Historical benchmark tables | Manuscript plus retained external run artifacts | Not a new replication |

When adding a result, add a row or update the relevant row with the exact commit, run directory, config hash, and artifact path. Never use a paper table alone as evidence of a new run.
