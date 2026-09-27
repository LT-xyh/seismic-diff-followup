# Remote Sensing Revision Workspace

Target journal: **Remote Sensing**  
Primary success criterion: **formal acceptance notification**  
Current project state: post-AAAI-27 rejection, journal revision in progress.

This directory is the durable state for the Remote Sensing revision. GitHub is the source of truth; chat history is not.

## Read order

1. `HANDOFF.md`
2. `PUBLICATION_PRINCIPLES.md`
3. `PAPER_STORY.md`
4. `CLAIM_EVIDENCE_MATRIX.md`
5. `EXPERIMENT_QUEUE.md`
6. `RESULT_LEDGER.md`
7. `DECISION_LOG.md`
8. `REVIEW_CHECKLIST.md`
9. `../REMOTE_SENSING_EVIDENCE_AUDIT.md`
10. `../REMOTE_SENSING_RESULT_MAP.csv`

## Role split

- **Main GPT work session**: research lead, paper strategy, code review, GitHub edits, experiment design, result interpretation, manuscript writing.
- **Strategy/review GPT session**: higher-level review, claim discipline, publication positioning, final red-team review.
- **Codex**: runtime executor only when real compute, private datasets/checkpoints/logs, GPU execution, or environment-specific debugging is required.

Do not spend Codex quota on literature framing, manuscript writing, static code inspection, experiment design, or repository edits that can be completed from the GitHub-connected GPT session.

## Historical boundary

- `docs/paper/aaaii/` is an immutable AAAI submission snapshot.
- `docs/paper/AAAI2027/` is editable historical development material, but is not automatically the Remote Sensing manuscript.
- Canonical implementation is at repository root under `bg_pdr_fm/`, `scripts/aaai27/`, and `reproducibility/`.
