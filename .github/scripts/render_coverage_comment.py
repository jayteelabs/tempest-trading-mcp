from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MARKER = "<!-- tempest-coverage-comment -->"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a sticky informational PR comment from coverage.py JSON output."
    )
    parser.add_argument("--coverage-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--coverage-status", default="unknown")
    parser.add_argument("--workflow-url", default="")
    parser.add_argument("--top-files", type=int, default=10)
    return parser.parse_args()


def load_coverage_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def format_percent(value: Any) -> str:
    if isinstance(value, str) and value.endswith("%"):
        return value
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def summarize_files(report: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path, details in (report.get("files") or {}).items():
        summary = details.get("summary") or {}
        missing_lines = as_int(summary.get("missing_lines"))
        if missing_lines <= 0:
            continue
        files.append(
            {
                "path": path,
                "missing_lines": missing_lines,
                "covered_lines": as_int(summary.get("covered_lines")),
                "num_statements": as_int(summary.get("num_statements")),
                "percent_covered": summary.get("percent_covered"),
            }
        )
    files.sort(key=lambda item: (-item["missing_lines"], item["percent_covered"] or 100.0, item["path"]))
    return files[:limit]


def coverage_status_line(status: str, report_available: bool) -> str:
    normalized = (status or "unknown").lower()
    icon_by_status = {
        "success": "✅",
        "failure": "⚠️",
        "cancelled": "⏹️",
        "skipped": "ℹ️",
        "unknown": "ℹ️",
    }
    if normalized == "success" and report_available:
        description = "coverage run completed"
    elif normalized == "success":
        description = "coverage command reported success, but no JSON report was found"
    elif normalized == "failure":
        description = "coverage command failed or surfaced test failures"
    elif normalized == "cancelled":
        description = "coverage command was cancelled"
    elif normalized == "skipped":
        description = "coverage command was skipped"
    else:
        description = "coverage command status is unavailable"
    return f"{icon_by_status.get(normalized, 'ℹ️')} {description}"


def build_comment_body(args: argparse.Namespace, report: dict[str, Any] | None) -> str:
    lines = [
        MARKER,
        "## Coverage report (informational)",
        "",
        "> Coverage is reported for visibility only. Merge gates remain **ruff lint** and **pytest**.",
        "",
    ]

    short_sha = args.head_sha[:7] if args.head_sha else "unknown"
    lines.append(f"- **Head SHA:** `{short_sha}`")
    lines.append(
        f"- **Status:** {coverage_status_line(args.coverage_status, report is not None)}"
    )
    if args.workflow_url:
        lines.append(f"- **Workflow run:** [View logs]({args.workflow_url})")

    if report is None:
        lines.extend(
            [
                "- **Coverage summary:** unavailable (no `coverage.json` produced)",
                "",
                "Coverage remains non-blocking; check the workflow logs only if you want to troubleshoot the reporting path.",
            ]
        )
        return "\n".join(lines) + "\n"

    totals = report.get("totals") or {}
    num_statements = as_int(totals.get("num_statements"))
    covered_lines = as_int(totals.get("covered_lines"))
    missing_lines = as_int(totals.get("missing_lines"))
    excluded_lines = as_int(totals.get("excluded_lines"))
    percent_covered = totals.get("percent_covered_display") or totals.get("percent_covered")

    lines.extend(
        [
            f"- **Total coverage:** **{format_percent(percent_covered)}**",
            f"- **Covered / total lines:** {covered_lines} / {num_statements}",
            f"- **Missing lines:** {missing_lines}",
            f"- **Excluded lines:** {excluded_lines}",
        ]
    )

    top_files = summarize_files(report, args.top_files)
    if not top_files:
        lines.extend(["", "All measured files are fully covered."])
        return "\n".join(lines) + "\n"

    lines.extend(["", f"<details><summary>Top {len(top_files)} files with uncovered lines</summary>", "", "| File | Coverage | Missing lines | Covered / total |", "| --- | ---: | ---: | ---: |"])
    for item in top_files:
        lines.append(
            "| `{path}` | {percent} | {missing} | {covered} / {total} |".format(
                path=item["path"],
                percent=format_percent(item["percent_covered"]),
                missing=item["missing_lines"],
                covered=item["covered_lines"],
                total=item["num_statements"],
            )
        )
    lines.extend(["", "</details>"])
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    report = load_coverage_report(Path(args.coverage_json))
    output_path = Path(args.output)
    output_path.write_text(build_comment_body(args, report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
