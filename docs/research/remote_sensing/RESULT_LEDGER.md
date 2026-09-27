# Result Ledger

This file records accepted Remote Sensing evidence.

Do not copy a number into the manuscript unless it can be linked to an entry here or to an explicitly identified historical artifact.

## Status vocabulary

- PLANNED
- RUNNING
- BLOCKED
- COMPLETE-UNREVIEWED
- ACCEPTED-EVIDENCE
- REJECTED-EVIDENCE
- SUPERSEDED

## Entry template

### RS-EXXX — experiment name

- Status:
- Purpose / claim ID:
- Git commit:
- Resolved config:
- Config SHA256:
- Dataset manifest:
- Evaluated split:
- Evaluated record count:
- Training seed:
- Split seed:
- Well seed:
- Checkpoint:
- Checkpoint SHA256:
- Runtime hardware:
- Exact command:
- Output directory:
- Metrics artifact:
- Per-record artifact:
- Timing artifact:
- Deviations from protocol:
- Review decision:
- Main paper / supplement / internal:
- Notes:

## Acceptance rule

A result becomes **ACCEPTED-EVIDENCE** only after:

1. protocol matches the experiment specification;
2. record count and split are verified;
3. evaluator is the frozen Remote Sensing evaluator;
4. result artifact exists;
5. relevant checkpoint/config provenance is recorded;
6. no unresolved metric inconsistency affects the reported claim.

## Historical results

Historical AAAI values remain research context until explicitly promoted through a reviewed ledger entry. Manuscript presence alone is not sufficient.
