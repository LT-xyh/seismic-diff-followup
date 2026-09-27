# Research handoff: PD-BG-RFM

This is the entrypoint for a new human collaborator or AI research assistant, including a GitHub-connected web model.

## Current objective

Turn the rejected AAAI submission into an independently auditable research package, then decide whether a bounded experiment revision is sufficient for a conference resubmission or whether a deeper journal study is justified.

## Reading order

1. `../../README.md`
2. `../../AGENTS.md`
3. `PROJECT_STATE.md`
4. `EXPERIMENT_PROTOCOL.md`
5. `EVIDENCE_INDEX.md`
6. `NEXT_WORK.md`
7. `DECISION_LOG.md`
8. `../../reproducibility/release_manifest.yaml`
9. `../../bg_pdr_fm/configs/release/aaai27/`

## First commands

```bash
python reproducibility/validate_release.py
python scripts/aaai27/smoke.py
pytest -q tests bg_pdr_fm/tests
```

These checks establish source and smoke status only. They do not establish full benchmark performance.
