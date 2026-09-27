# Project state

Updated: 2026-09-27

## Paper

- Submission: AAAI 2027, PD-BG-RFM.
- Status: rejected by the program chairs on 2026-09-25.
- Submission snapshot: `docs/paper/aaaii/`.
- Editable manuscript: `docs/paper/AAAI2027/`.
- OpenReview forum: user-provided forum URL `https://openreview.net/forum?id=FhcayzchcU`.

The visible review feedback was largely positive on readability but requested an operational definition of “physics-decoupled,” a clearer method overview, explicit filter/boundary details, and more careful contextualization of cross-protocol results. The full decision and all reviewer scores must be treated as external evidence until captured in a permitted project note.

## Code

The canonical development source is the repository root:

- `bg_pdr_fm/`
- `scripts/aaai27/`
- `reproducibility/`
- `tests/`

`docs/paper/aaaii/pd_bg_rfm_code/` is a historical release copy and may contain a nested Git repository. It is not the development source.

## Reproducibility contract

- Training seed: 2027.
- Split seed: 42.
- Well seed: 1234.
- Formal profile: `bg_pdr_fm/configs/release/aaai27/train.yaml`.
- Evaluation traverses the materialized held-out membership without shuffling.
- Data, LMDB stores, checkpoints, and logs remain outside the GitHub source release.

## Evidence status

- Source organization and release validator: available in the repository.
- CPU smoke path: must be verified on the current branch before claiming pass.
- Full eight-subset benchmark: historical result, not a newly replicated result.
- Multi-seed stability: not yet established.
- Input-noise/missing-modality robustness: not yet established.
- Real-data transfer: not established.

## Main scientific risks

1. “Physics-decoupled” may overstate a method that currently has no explicit wave-equation or forward-operator constraint.
2. PCA rank and straight-path energy diagnostics do not by themselves prove lower conditional complexity or easier learned transport.
3. The source-native comparison mixes observation protocols; matched-input results must carry the main comparison burden.
4. A single training trajectory is insufficient evidence for stable ranking.
5. Synthetic multimodal observations may not represent realistic observation noise or interpretation error.
