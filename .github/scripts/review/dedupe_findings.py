from __future__ import annotations

import hashlib
import re
from typing import Any

from review.schema import SEVERITY_ORDER, Finding


def canonicalize_body(body: str) -> str:
    return re.sub(r"\s+", " ", body.strip().lower())


def fingerprint_for(finding: Finding | dict[str, Any]) -> str:
    path = _value(finding, "path") or ""
    line = str(_value(finding, "line") or 0)
    category = _value(finding, "category") or "general"
    body = canonicalize_body(_value(finding, "body") or _value(finding, "title") or "")
    payload = "|".join([path, line, category, body])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def merge_duplicate_findings(
    findings: list[Finding],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []

    for finding in findings:
        fingerprint = fingerprint_for(finding)
        current = finding.to_dict()
        current["fingerprint"] = fingerprint
        current["sources"] = [finding.source_agent]
        current["stages"] = [finding.stage]

        existing = merged.get(fingerprint)
        if existing is None:
            merged[fingerprint] = current
            continue

        existing["sources"] = sorted({*existing["sources"], finding.source_agent})
        existing["stages"] = sorted({*existing["stages"], finding.stage})
        existing["confidence"] = (
            max(existing.get("confidence") or 0.0, finding.confidence or 0.0) or None
        )
        existing["spec_refs"] = sorted({*existing.get("spec_refs", []), *finding.spec_refs})
        existing["requires_spec_ref"] = (
            existing.get("requires_spec_ref", False) or finding.requires_spec_ref
        )

        if SEVERITY_ORDER[finding.severity] > SEVERITY_ORDER[existing["severity"]]:
            for key in [
                "source_agent",
                "stage",
                "severity",
                "category",
                "title",
                "body",
                "path",
                "line",
                "evidence",
            ]:
                existing[key] = current[key]

        dropped.append(
            {
                "fingerprint": fingerprint,
                "decision": "drop_duplicate",
                "reason": "duplicate within merged findings",
                "source_agent": finding.source_agent,
                "stage": finding.stage,
            }
        )

    return list(merged.values()), dropped


def _value(finding: Finding | dict[str, Any], key: str) -> Any:
    if isinstance(finding, Finding):
        return getattr(finding, key)
    return finding.get(key)
