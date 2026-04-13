from __future__ import annotations

import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def collect_changed_lines(base_sha: str, head_sha: str) -> dict[str, set[int]]:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--unified=0", base_sha, head_sha],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    return parse_unified_diff(result.stdout)


def parse_unified_diff(diff_text: str) -> dict[str, set[int]]:
    changed_lines: dict[str, set[int]] = {}
    current_path: str | None = None
    new_line = 0

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("diff --git "):
            current_path = None
            continue

        if raw_line.startswith("+++ "):
            file_path = raw_line[4:].strip()
            if file_path == "/dev/null":
                current_path = None
                continue
            current_path = file_path.removeprefix("b/")
            changed_lines.setdefault(current_path, set())
            continue

        if raw_line.startswith("@@ "):
            hunk_header = raw_line.split("@@", 2)[1].strip()
            plus_range = hunk_header.split(" ")[1]
            start_text = plus_range[1:].split(",", 1)[0]
            new_line = int(start_text)
            continue

        if current_path is None:
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            changed_lines[current_path].add(new_line)
            new_line += 1
            continue

        if raw_line.startswith("-") and not raw_line.startswith("---"):
            continue

        new_line += 1

    return {path: lines for path, lines in changed_lines.items() if lines}
