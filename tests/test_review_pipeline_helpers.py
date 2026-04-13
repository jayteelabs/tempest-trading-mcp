# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".github" / "scripts"))

from review.collect_changed_lines import parse_unified_diff
from review.dedupe_findings import fingerprint_for, merge_duplicate_findings
from review.producer_common import design_reference_paths
from review.publish_review import (
    build_inline_comment,
    build_review_body,
    create_summary_review,
    format_reported_by,
    humanize_title,
    own_pr_review_blocked,
    severity_style,
)
from review.schema import Finding, extract_json_document
from review.utils import resolve_artifact_path
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


def test_merge_duplicate_findings_accepts_artifact_dicts() -> None:
    findings = [
        {
            "source_agent": "shuna",
            "stage": "review/design",
            "severity": "medium",
            "category": "contract",
            "title": "Contract drift",
            "body": "Return a structured object.",
            "path": "src/example.py",
            "line": 18,
            "evidence": "Anchored in diff.",
            "confidence": 0.6,
            "spec_refs": [],
            "requires_spec_ref": False,
        },
        {
            "source_agent": "shion",
            "stage": "review/logic_audit",
            "severity": "high",
            "category": "contract",
            "title": "Contract drift",
            "body": "Return a structured object.",
            "path": "src/example.py",
            "line": 18,
            "evidence": "Anchored in diff.",
            "confidence": 0.9,
            "spec_refs": ["docs/spec.md"],
            "requires_spec_ref": False,
        },
    ]

    merged, dropped = merge_duplicate_findings(findings)

    assert len(merged) == 1
    assert merged[0]["severity"] == "high"
    assert merged[0]["sources"] == ["shion", "shuna"]
    assert merged[0]["stages"] == ["review/design", "review/logic_audit"]
    assert merged[0]["spec_refs"] == ["docs/spec.md"]
    assert len(dropped) == 1


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
            "stages": ["review/design", "review/logic_audit"],
            "severity": "medium",
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
    assert "**🟡 MEDIUM: Contract drift**" in comment["body"]
    assert "Reported by: Shuna / Design, Shion / Logic Audit (Lane B)" in comment["body"]
    assert "tempest-review:fingerprint=abc123" in comment["body"]
    assert "tempest-review:head-sha=deadbeef" in comment["body"]


def test_format_reported_by_uses_stage_labels() -> None:
    finding = {
        "stage": "review/security_audit",
        "stages": ["review/design", "review/security_audit", "review/security_audit"],
    }

    assert format_reported_by(finding) == "Shuna / Design, Shion / Security Audit (Lane A)"


def test_humanize_title_strips_prefix_and_ticket_suffix() -> None:
    assert humanize_title("feat(ci): centralize AI PR review publishing [ENG-62]") == (
        "centralize AI PR review publishing"
    )


def test_build_review_body_uses_brief_summary_and_footer() -> None:
    body = build_review_body(
        title="feat(ci): centralize AI PR review publishing [ENG-62]",
        head_sha="deadbeef",
        diff_summary={
            "file_count": 3,
            "areas": [".github/workflows", ".github/scripts/review", "tests"],
        },
        published_inline=[{"fingerprint": "a", "severity": "medium"}],
        summary_only=[],
        drops=[
            {"decision": "drop_duplicate"},
            {"decision": "drop_outdated"},
        ],
        artifact_errors=[],
        publish_errors=[],
    )

    assert "**Purpose:** centralize AI PR review publishing" in body
    assert (
        "**Summary:** **3 files** changed; primary areas: `.github/workflows`, `.github/scripts/review`, `tests`."
        in body
    )
    assert "Validated **1** inline finding(s) were published on the diff." in body
    assert "_Duplicates: 1 | Unverifiable: 0 | Outdated: 1_" in body


def test_severity_style_uses_expected_header_metadata() -> None:
    assert severity_style("critical")[0:2] == ("🔴", "CRITICAL")
    assert severity_style("high")[0:2] == ("🟠", "HIGH")
    assert severity_style("medium")[0:2] == ("🟡", "MEDIUM")
    assert severity_style("low")[0:2] == ("🟢", "LOW")


def test_resolve_artifact_path_rejects_non_artifact_paths() -> None:
    try:
        resolve_artifact_path("../secrets.txt")
    except ValueError as exc:
        assert "Artifact path must stay within" in str(exc)
    else:
        raise AssertionError("expected resolve_artifact_path to reject traversal")


def test_design_reference_paths_include_ticket_and_prd_docs(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    specs = tmp_path / "specs"
    docs.mkdir()
    specs.mkdir()
    (docs / "ENG-62-prd.md").write_text("ticket prd", encoding="utf-8")
    (docs / "architecture.md").write_text("arch", encoding="utf-8")
    (specs / "review-design-spec.md").write_text("spec", encoding="utf-8")

    paths = design_reference_paths(tmp_path, "ENG-62")

    relative = {str(path.relative_to(tmp_path)) for path in paths}
    assert "docs/ENG-62-prd.md" in relative
    assert "specs/review-design-spec.md" in relative


def test_own_pr_review_blocked_matches_github_error() -> None:
    from review.github_api import GitHubApiError

    error = GitHubApiError(
        'GitHub API POST /pulls/27/reviews failed with 422: {"errors":["Review Can not request changes on your own pull request"]}'
    )

    assert own_pr_review_blocked(error) is True


def test_create_summary_review_falls_back_to_comment() -> None:
    from review.github_api import GitHubApiError

    class FakeApi:
        def __init__(self) -> None:
            self.events: list[str] = []

        def create_review(self, *, pull_number: int, commit_id: str, body: str, event: str) -> dict:
            self.events.append(event)
            if event == "REQUEST_CHANGES":
                raise GitHubApiError(
                    'GitHub API POST /pulls/27/reviews failed with 422: {"errors":["Review Can not request changes on your own pull request"]}'
                )
            return {"ok": True}

    api = FakeApi()

    event = create_summary_review(
        api=api,
        pull_number=27,
        commit_id="deadbeef",
        body="summary",
        requested_event="REQUEST_CHANGES",
    )

    assert event == "COMMENT"
    assert api.events == ["REQUEST_CHANGES", "COMMENT"]


def test_github_api_rejects_malformed_repository() -> None:
    from review.github_api import GitHubApi

    try:
        GitHubApi(token="token", repository="tempest-tradingview-mcp")
    except ValueError as exc:
        assert "owner/repo" in str(exc)
    else:
        raise AssertionError("expected malformed repository to be rejected")
