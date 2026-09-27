# Project state

Updated: 2026-09-27

## Paper

- Historical submission: AAAI 2027, PD-BG-RFM.
- AAAI status: rejected by the program chairs on 2026-09-25.
- Current target: **Remote Sensing**.
- Practical target: formal acceptance notification.
- Historical submission snapshot: `docs/paper/aaaii/`.
- Historical editable manuscript: `docs/paper/AAAI2027/`.
- Remote Sensing workspace: `docs/research/remote_sensing/`.
- Evidence audit: `docs/research/REMOTE_SENSING_EVIDENCE_AUDIT.md`.

The journal revision is intentionally narrower than the AAAI narrative. The working core is role-aware multi-source conditioning, deterministic background prediction, and prediction-consistent residual Flow Matching.

## Code

The canonical development source is the repository root:

- `bg_pdr_fm/`
- `scripts/aaai27/`
- `reproducibility/`
- `tests/`

`docs/paper/aaaii/pd_bg_rfm_code/` is a historical release copy and may contain a nested Git repository. It is not the development source.

## Reproducibility contract

- Historical training seed: 2027.
- Split seed: 42.
- Well seed: 1234.
- Historical formal profile: `bg_pdr_fm/configs/release/aaai27/train.yaml`.
- Evaluation traverses held-out membership without shuffling.
- Data, LMDB stores, checkpoints, and logs remain outside the GitHub source release.

For Remote Sensing, the final quantitative package must freeze one held-out manifest, evaluator, normalization contract, and result provenance per method.

## Evidence status

- Source organization and release validator: available.
- Remote Sensing evidence audit: completed in commit `0bb320b`.
- Historical AAAI benchmark tables: research evidence, not final journal replication.
- Historical matched-input InversionNet frequency metric provenance: unresolved.
- Direct background evaluation: implementation support exists; clean journal result pending.
- Attribution controls: historical/partial artifacts exist; common-evaluator result pending.
- Perturbation robustness: clean journal result pending.
- Multi-seed stability: not established; now treated as a gated experiment after the clean comparison.
- Real-data transfer: not established and not a first-pass requirement.
- 3D validation: not established and not a first-pass requirement.

## Current blockers

1. Recover the exact external checkpoints/manifests needed for the final comparison.
2. Freeze one common held-out manifest/evaluator.
3. Supersede or resolve the historical InversionNet `MAE_L` inconsistency.

## Current execution order

`RS-A0 -> RS-E01 -> RS-E02 -> RS-E03A -> RS-E04/RS-E05 -> evidence gate -> train only what remains necessary`

See `docs/research/remote_sensing/EXPERIMENT_QUEUE.md`.
