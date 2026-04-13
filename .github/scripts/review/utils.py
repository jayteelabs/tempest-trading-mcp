from __future__ import annotations

import re
from pathlib import Path

SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def validate_git_sha(name: str, value: str) -> str:
    if not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"Invalid {name}: expected 40-character git SHA")
    return value


def resolve_artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        raise ValueError("Artifact path must be relative to the repository root")

    root = repo_root().resolve()
    resolved = (root / candidate).resolve()
    artifacts_root = (root / "artifacts").resolve()

    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError(f"Artifact path must stay within {artifacts_root}") from exc

    return resolved
