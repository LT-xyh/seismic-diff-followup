# Project state

Updated: 2026-09-29

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

### RS-0A source-handoff status

**RS-0A = CLOSED.**

The canonical `bg_pdr_fm/data/` package was recovered from:

- workspace: `/public/home/xuyinghao/workspace/seismic-diff`
- source commit: `75dcf1c5bb539aef97d4aaeadf01b54584108b09`

The release dependency contract now includes `tensorboard==2.20.0`, matching the retained research environment. PR #1 was merged as `4c046a4a68dd4fa2d1b26a8283a68febf5fd26a2`; release validator, release tests, CPU smoke, and GitHub Actions quality all passed on `main`.

## Reproducibility contract

- Historical training seed: 2027.
- Split seed: 42.
- Well seed: 1234.
- Historical formal profile: `bg_pdr_fm/configs/release/aaai27/train.yaml`.
- Evaluation traverses held-out membership without shuffling.
- Data, LMDB stores, checkpoints, and logs remain outside the GitHub source release.
- The formal Table 2 tooling expects a canonical global held-out set of 33,600 records keyed by `dataset_id`, `dataset_name`, and `source_sample_index`.

For Remote Sensing, the final quantitative package must freeze one held-out manifest, evaluator, normalization contract, and result provenance per method.

## Evidence status

- Canonical source handoff and release CI: PASS / CLOSED.
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

Scientific submission blockers: **none known**.

Reviewer-2 M1/M2 documentation closure is complete at the manuscript-source level:
- active condition objective and zero-weight exclusions documented;
- four-stage training and checkpoint-selection protocol documented;
- deterministic 70/20/10 split contract documented;
- evaluated 0--3-well observation contract documented;
- architecture/tensor interfaces documented;
- continuous Flow-Matching time and 50-step Euler inference documented;
- Table 3 reframed as supporting formulation-level evidence rather than a strict component ablation.

Remaining work is packaging/administrative: final PDF compile/visual inspection, author/funding/conflict metadata, and journal-template packaging.

## Current execution order

`final compile -> visual/numerical scan -> targeted Reviewer-2 M1/M2 verification -> author metadata/template packaging -> submit`

No new experiment is authorized before first submission unless a later reviewer explicitly requires it.

See `docs/research/remote_sensing/EXPERIMENT_QUEUE.md`.
