from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_DEFAULTS = {
    "review/design": "shuna",
    "review/security_audit": "shion",
    "review/logic_audit": "shion",
}

SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


@dataclass
class Finding:
    source_agent: str
    stage: str
    severity: str
    category: str
    title: str
    body: str
    path: str | None = None
    line: int | None = None
    evidence: str = ""
    confidence: float | None = None
    spec_refs: list[str] = field(default_factory=list)
    requires_spec_ref: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_json_document(text: str) -> Any:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)

    decoder = json.JSONDecoder()
    for start, char in enumerate(candidate):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(candidate[start:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON document found in model output")


def normalize_path(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_line(value: Any) -> int | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        line = int(value)
    except (TypeError, ValueError):
        return None
    return line if line > 0 else None


def normalize_severity(value: Any) -> str:
    text = str(value or "medium").strip().lower()
    return text if text in SEVERITY_ORDER else "medium"


def normalize_stage(value: Any, default_stage: str) -> str:
    text = str(value or default_stage).strip().lower().replace(" ", "_")
    return text if text in STAGE_DEFAULTS else default_stage


def normalize_spec_refs(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        refs = [value]
    elif isinstance(value, list):
        refs = value
    else:
        return []
    return [str(item).strip() for item in refs if str(item).strip()]


def normalize_finding(
    raw: dict[str, Any], default_stage: str, default_agent: str
) -> Finding | None:
    title = str(raw.get("title") or "").strip()
    body = str(raw.get("body") or title).strip()
    if not body:
        return None

    return Finding(
        source_agent=str(raw.get("source_agent") or default_agent).strip().lower() or default_agent,
        stage=normalize_stage(raw.get("stage"), default_stage),
        severity=normalize_severity(raw.get("severity")),
        category=str(raw.get("category") or "general").strip().lower() or "general",
        title=title or body.splitlines()[0][:80],
        body=body,
        path=normalize_path(raw.get("path")),
        line=normalize_line(raw.get("line")),
        evidence=str(raw.get("evidence") or "").strip(),
        confidence=_normalize_confidence(raw.get("confidence")),
        spec_refs=normalize_spec_refs(raw.get("spec_refs")),
        requires_spec_ref=bool(raw.get("requires_spec_ref", False)),
    )


def normalize_findings(payload: Any, default_stage: str) -> list[Finding]:
    if isinstance(payload, dict):
        items = payload.get("findings", [])
        agent = STAGE_DEFAULTS.get(default_stage, "raphael")
    elif isinstance(payload, list):
        items = payload
        agent = STAGE_DEFAULTS.get(default_stage, "raphael")
    else:
        return []

    findings: list[Finding] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        finding = normalize_finding(item, default_stage, agent)
        if finding is not None:
            findings.append(finding)
    return findings


def _normalize_confidence(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    return min(max(confidence, 0.0), 1.0)


def artifact_document(
    *,
    stage: str,
    findings: list[Finding],
    errors: list[str],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "findings": [finding.to_dict() for finding in findings],
        "errors": errors,
        "metadata": metadata,
    }


def write_artifact(
    path: str | Path,
    *,
    stage: str,
    findings: list[Finding],
    errors: list[str],
    metadata: dict[str, Any],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = artifact_document(stage=stage, findings=findings, errors=errors, metadata=metadata)
    output_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {
            "stage": artifact_path.stem,
            "findings": [],
            "errors": [f"Missing artifact: {artifact_path}"],
            "metadata": {},
        }
    data = json.loads(artifact_path.read_text(encoding="utf-8"))
    data.setdefault("findings", [])
    data.setdefault("errors", [])
    data.setdefault("metadata", {})
    return data


def finding_to_dict(finding: Finding) -> dict[str, Any]:
    return finding.to_dict()
