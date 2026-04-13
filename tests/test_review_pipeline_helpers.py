# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from review.collect_changed_lines import parse_unified_diff
from review.dedupe_findings import fingerprint_for
from review.publish_review import build_inline_comment
from review.schema import Finding, extract_json_document
from review.validate_findings import classify_findings


def test_parse_unified_diff_tracks_added_lines() -> None:
    diff = """diff --git a/src/example.py b/src/example.py
index 1111111..2222222 100644
--- a/src/example.py
+++ b/src/example.py
@@ -10,0 +11,2 @@
+first = 1
+second = 2
@@ -20,1 +23,1 @@
-old = 1
+new = 2
"""

    changed_lines = parse_unified_diff(diff)

    assert changed_lines == {"src/example.py": {11, 12, 23}}


def test_fingerprint_ignores_case_and_whitespace() -> None:
    first = Finding(
        source_agent="shuna",
        stage="review/design",
        severity="medium",
        category="contract",
        title="Contract drift",
        body="Return  a  structured   object.",
        path="src/example.py",
        line=18,
    )
    second = Finding(
        source_agent="shion",
        stage="review/logic_audit",
        severity="medium",
        category="contract",
        title="Contract drift",
        body="return a structured object.",
        path="src/example.py",
        line=18,
    )

    assert fingerprint_for(first) == fingerprint_for(second)


def test_classify_findings_separates_inline_summary_and_drops(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "spec.md").write_text("hello", encoding="utf-8")
    findings = [
        {
            "fingerprint": "inline",
            "stage": "review/design",
            "severity": "medium",
            "category": "contract",
            "title": "Inline",
            "body": "Anchored",
            "path": "src/example.py",
            "line": 11,
            "spec_refs": [],
            "requires_spec_ref": False,
        },
        {
            "fingerprint": "summary",
            "stage": "review/security_audit",
            "severity": "low",
            "category": "edge-case",
            "title": "Summary",
            "body": "Needs summary only",
            "path": None,
            "line": None,
            "spec_refs": [],
            "requires_spec_ref": False,
        },
        {
            "fingerprint": "unverifiable",
            "stage": "review/design",
            "severity": "medium",
            "category": "spec",
            "title": "Spec missing",
            "body": "Needs spec",
            "path": "src/example.py",
            "line": 11,
            "spec_refs": ["docs/missing.md"],
            "requires_spec_ref": True,
        },
        {
            "fingerprint": "outdated",
            "stage": "review/logic_audit",
            "severity": "medium",
            "category": "logic",
            "title": "Outdated",
            "body": "No longer changed",
            "path": "src/example.py",
            "line": 99,
            "spec_refs": [],
            "requires_spec_ref": False,
        },
    ]

    inline, summary_only, dropped = classify_findings(
        findings,
        changed_lines={"src/example.py": {11}},
        repo_root=tmp_path,
    )

    assert [item["fingerprint"] for item in inline] == ["inline"]
    assert [item["fingerprint"] for item in summary_only] == ["summary"]
    assert {item["fingerprint"] for item in dropped} == {"unverifiable", "outdated"}


def test_extract_json_document_handles_fenced_output() -> None:
    output = """```json
{"findings": []}
```"""

    parsed = extract_json_document(output)

    assert parsed == {"findings": []}


def test_build_inline_comment_embeds_review_markers() -> None:
    comment = build_inline_comment(
        {
            "fingerprint": "abc123",
            "stage": "review/design",
            "title": "Contract drift",
            "body": "Return the structured response here.",
            "path": "src/example.py",
            "line": 21,
            "evidence": "Changed in the current diff.",
        },
        "deadbeef",
    )

    assert comment["path"] == "src/example.py"
    assert comment["line"] == 21
    assert "tempest-review:fingerprint=abc123" in comment["body"]
    assert "tempest-review:head-sha=deadbeef" in comment["body"]
