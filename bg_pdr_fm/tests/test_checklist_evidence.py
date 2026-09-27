from __future__ import annotations

import json
from pathlib import Path

import yaml

from reproducibility.validate_release import CHECKLIST_ITEM_IDS, validate_release


REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = REPO_ROOT / "reproducibility" / "checklist_evidence.yaml"
CHECKLIST_PATH = REPO_ROOT / "docs" / "paper" / "AAAI2027" / "ReproducibilityChecklist.tex"


def _load_evidence() -> dict[str, object]:
    payload = yaml.safe_load(EVIDENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_code_checklist_items_have_paths_and_commands() -> None:
    payload = _load_evidence()
    items = payload["items"]
    assert isinstance(items, list)
    by_id = {str(item["id"]): item for item in items}
    assert set(CHECKLIST_ITEM_IDS).issubset(by_id)

    for item_id in CHECKLIST_ITEM_IDS:
        item = by_id[item_id]
        assert item["evidence_path"], item_id
        assert item["command"], item_id
        assert item["status"] in {"yes", "partial", "no", "conditional"}, item_id


def test_evidence_distinguishes_publication_dependent_claims() -> None:
    payload = _load_evidence()
    claims = payload["publication_dependent"]
    assert isinstance(claims, list)
    claim_names = {str(claim["claim"]) for claim in claims}
    assert {"license", "public_availability", "literature_rerun"}.issubset(claim_names)
    for claim in claims:
        assert claim["status"] == "conditional"
        assert claim["evidence_path"]


def test_checklist_answer_section_has_no_instructional_placeholder() -> None:
    text = CHECKLIST_PATH.read_text(encoding="utf-8")
    answer_section = text.split("% The questions start here", 1)[1]
    assert "Type your response here" not in answer_section


def test_validator_writes_machine_readable_report(tmp_path: Path) -> None:
    output = tmp_path / "checklist_evidence.json"
    report = validate_release(output=output)
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert set(CHECKLIST_ITEM_IDS).issubset({str(item["id"]) for item in report["items"]})
