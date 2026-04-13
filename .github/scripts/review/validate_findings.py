from __future__ import annotations

from pathlib import Path
from typing import Any


def classify_findings(
    findings: list[dict[str, Any]],
    *,
    changed_lines: dict[str, set[int]],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inline: list[dict[str, Any]] = []
    summary_only: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for finding in findings:
        resolved_refs = resolve_spec_refs(repo_root, finding.get("spec_refs", []))
        finding["resolved_spec_refs"] = resolved_refs

        if finding.get("requires_spec_ref") and not resolved_refs:
            dropped.append(
                _decision(finding, "drop_unverifiable", "required spec refs could not be resolved")
            )
            continue

        path = finding.get("path")
        line = finding.get("line")
        if not path or not line:
            summary_only.append(
                _decision(finding, "publish_summary_only", "finding is not line-anchorable")
            )
            continue

        if path not in changed_lines or line not in changed_lines[path]:
            dropped.append(
                _decision(finding, "drop_outdated", "line is not part of the current diff")
            )
            continue

        inline.append({**finding, "decision": "publish_inline"})

    return inline, summary_only, dropped


def resolve_spec_refs(repo_root: Path, refs: list[str]) -> list[str]:
    resolved: list[str] = []
    for ref in refs:
        if ref.startswith("http://") or ref.startswith("https://"):
            continue
        candidate = (repo_root / ref).resolve()
        try:
            candidate.relative_to(repo_root.resolve())
        except ValueError:
            continue
        if candidate.is_file():
            resolved.append(ref)
    return resolved


def _decision(finding: dict[str, Any], decision: str, reason: str) -> dict[str, Any]:
    return {**finding, "decision": decision, "reason": reason}
