from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Any

from review.collect_changed_lines import collect_changed_lines
from review.dedupe_findings import merge_duplicate_findings
from review.github_api import GitHubApi, GitHubApiError
from review.schema import SEVERITY_ORDER, load_artifact
from review.utils import repo_root, resolve_artifact_path, validate_git_sha
from review.validate_findings import classify_findings

CHECK_RUN_NAME = "Review / Publish"
MARKER_PREFIX = "tempest-review"
STAGE_DISPLAY_NAMES = {
    "review/design": "Shuna / Design",
    "review/security_audit": "Shion / Security Audit (Lane A)",
    "review/logic_audit": "Shion / Logic Audit (Lane B)",
}
SEVERITY_STYLES = {
    "critical": (
        "🔴",
        "CRITICAL",
        "This is critical because it can break the review pipeline or produce untrustworthy review output immediately.",
    ),
    "high": (
        "🟠",
        "HIGH",
        "This is high severity because it can cause incorrect review behavior, failed automation, or meaningful operational risk.",
    ),
    "medium": (
        "🟡",
        "MEDIUM",
        "This is medium severity because it can degrade reliability or correctness and is worth addressing before the workflow spreads further.",
    ),
    "low": (
        "🟢",
        "LOW",
        "This is low severity because the impact is limited, but cleaning it up reduces future maintenance and review noise.",
    ),
    "info": (
        "🔵",
        "INFO",
        "This is informational because it documents a minor improvement rather than an immediate correctness or security problem.",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--security-audit", required=True)
    parser.add_argument("--logic-audit", required=True)
    args = parser.parse_args()

    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GH_TOKEN"]
    api = GitHubApi(token=token, repository=repository)
    root = repo_root()
    validate_git_sha("base_sha", args.base_sha)
    validate_git_sha("head_sha", args.head_sha)

    artifacts = [
        load_artifact(resolve_artifact_path(args.design)),
        load_artifact(resolve_artifact_path(args.security_audit)),
        load_artifact(resolve_artifact_path(args.logic_audit)),
    ]

    artifact_errors = [error for artifact in artifacts for error in artifact.get("errors", [])]
    all_findings = [finding for artifact in artifacts for finding in artifact.get("findings", [])]

    merged_findings, duplicate_drops = merge_duplicate_findings(all_findings)
    changed_lines = collect_changed_lines(args.base_sha, args.head_sha)
    inline_findings, summary_only, validation_drops = classify_findings(
        merged_findings,
        changed_lines=changed_lines,
        repo_root=root,
    )

    existing_comments = api.list_review_comments(int(args.pr))
    existing_reviews = api.list_reviews(int(args.pr))
    existing_fingerprints = extract_existing_fingerprints(existing_comments, args.head_sha)
    review_exists_for_head = has_publisher_review(existing_reviews, args.head_sha)

    published_inline: list[dict[str, Any]] = []
    rerun_duplicate_drops: list[dict[str, Any]] = []
    for finding in inline_findings:
        fingerprint = finding["fingerprint"]
        if fingerprint in existing_fingerprints:
            rerun_duplicate_drops.append(
                {
                    **finding,
                    "decision": "drop_duplicate",
                    "reason": "matching inline comment already exists for this head SHA",
                }
            )
            continue
        published_inline.append(finding)

    all_drops = [*duplicate_drops, *rerun_duplicate_drops, *validation_drops]
    review_event = determine_review_event([*published_inline, *summary_only])
    diff_summary = summarize_diff(args.base_sha, args.head_sha)
    summary_payload = {
        "pr_number": int(args.pr),
        "head_sha": args.head_sha,
        "review_event": review_event,
        "published": [serialize_publication(finding) for finding in published_inline],
        "summary_only": [serialize_summary_only(finding) for finding in summary_only],
        "dropped": [serialize_drop(finding) for finding in all_drops],
        "summary": {
            "published_inline": len(published_inline),
            "summary_only": len(summary_only),
            "duplicate_dropped": count_decisions(all_drops, "drop_duplicate"),
            "unverifiable_dropped": count_decisions(all_drops, "drop_unverifiable"),
            "outdated_dropped": count_decisions(all_drops, "drop_outdated"),
            "artifact_errors": len(artifact_errors),
        },
    }
    print(json.dumps(summary_payload, indent=2))

    publish_errors: list[str] = []

    review_body = build_review_body(
        title=args.title,
        head_sha=args.head_sha,
        diff_summary=diff_summary,
        published_inline=published_inline,
        summary_only=summary_only,
        drops=all_drops,
        artifact_errors=artifact_errors,
        publish_errors=publish_errors,
    )

    if published_inline or not review_exists_for_head:
        api.create_review(
            pull_number=int(args.pr),
            commit_id=args.head_sha,
            body=review_body,
            event=review_event,
        )

    for comment in published_inline:
        try:
            api.create_review_comment(
                pull_number=int(args.pr),
                commit_id=args.head_sha,
                path=comment["path"],
                line=comment["line"],
                body=build_inline_comment(comment, args.head_sha)["body"],
            )
        except GitHubApiError as exc:
            publish_errors.append(
                f"Inline comment publish failed for {comment['path']}:{comment['line']} ({comment['fingerprint']}): {exc}"
            )

    if publish_errors:
        review_body = build_review_body(
            title=args.title,
            head_sha=args.head_sha,
            diff_summary=diff_summary,
            published_inline=published_inline,
            summary_only=summary_only,
            drops=all_drops,
            artifact_errors=artifact_errors,
            publish_errors=publish_errors,
        )

    api.upsert_check_run(
        ref=args.head_sha,
        name=CHECK_RUN_NAME,
        external_id=f"review-publish:{args.pr}",
        summary=build_check_summary(summary_payload),
        text=build_check_text(
            title=args.title,
            url=args.url,
            summary_payload=summary_payload,
            artifact_errors=artifact_errors,
            publish_errors=publish_errors,
        ),
        conclusion=determine_check_conclusion(
            [*published_inline, *summary_only], [*artifact_errors, *publish_errors]
        ),
    )
    return 1 if publish_errors else 0


def extract_existing_fingerprints(comments: list[dict[str, Any]], head_sha: str) -> set[str]:
    fingerprints: set[str] = set()
    for comment in comments:
        body = comment.get("body") or ""
        markers = extract_markers(body)
        if markers.get("head-sha") == head_sha and markers.get("fingerprint"):
            fingerprints.add(markers["fingerprint"])
    return fingerprints


def has_publisher_review(reviews: list[dict[str, Any]], head_sha: str) -> bool:
    for review in reviews:
        body = review.get("body") or ""
        markers = extract_markers(body)
        if markers.get("publisher") == "review_publish" and markers.get("head-sha") == head_sha:
            return True
    return False


def extract_markers(body: str) -> dict[str, str]:
    matches = re.findall(rf"<!--\s*{MARKER_PREFIX}:([a-z\-]+)=([^>]+)\s*-->", body)
    return {key: value.strip() for key, value in matches}


def build_inline_comment(finding: dict[str, Any], head_sha: str) -> dict[str, Any]:
    marker_block = marker_lines(
        fingerprint=finding["fingerprint"],
        stage=finding["stage"],
        head_sha=head_sha,
    )
    title = finding.get("title") or finding["body"]
    emoji, label, severity_reason = severity_style(finding.get("severity", "medium"))
    body_lines = [f"**{emoji} {label}: {title}**", finding["body"], severity_reason]
    reported_by = format_reported_by(finding)
    if reported_by:
        body_lines.append(f"Reported by: {reported_by}")
    if finding.get("evidence"):
        body_lines.append(f"Evidence: {finding['evidence']}")
    body_lines.append(marker_block)
    return {
        "path": finding["path"],
        "line": finding["line"],
        "side": "RIGHT",
        "body": "\n\n".join(line for line in body_lines if line),
    }


def build_review_body(
    *,
    title: str,
    head_sha: str,
    diff_summary: dict[str, Any],
    published_inline: list[dict[str, Any]],
    summary_only: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    artifact_errors: list[str],
    publish_errors: list[str],
) -> str:
    lines = ["## Review / Publish", "", f"**Purpose:** {humanize_title(title)}", ""]

    scope_bits = [f"**{diff_summary['file_count']} files** changed"]
    if diff_summary["areas"]:
        scope_bits.append(
            "primary areas: " + ", ".join(f"`{area}`" for area in diff_summary["areas"])
        )
    lines.append("**Summary:** " + "; ".join(scope_bits) + ".")

    if published_inline:
        lines.append("")
        lines.append(
            f"Validated **{len(published_inline)}** inline finding(s) were published on the diff."
        )
    elif summary_only:
        lines.append("")
        lines.append(
            "No line-anchored findings were published; validated notes are summarized below."
        )
    else:
        lines.append("")
        lines.append("No publishable review findings were produced for this head SHA.")

    if summary_only:
        lines.append("")
        lines.append("### Summary-only findings")
        for finding in summary_only[:5]:
            lines.append(
                f"- **{finding['severity'].upper()}** {finding.get('title') or finding['body']}"
            )
    if artifact_errors:
        lines.append("")
        lines.append("### Lane errors")
        for error in artifact_errors:
            lines.append(f"- {error}")
    if publish_errors:
        lines.append("")
        lines.append("### Publish notes")
        for error in publish_errors:
            lines.append(f"- {error}")
    lines.extend(
        [
            "",
            (
                f"_Duplicates: {count_decisions(drops, 'drop_duplicate')} | "
                f"Unverifiable: {count_decisions(drops, 'drop_unverifiable')} | "
                f"Outdated: {count_decisions(drops, 'drop_outdated')}_"
            ),
            "",
            marker_lines(head_sha=head_sha),
        ]
    )
    return "\n".join(lines)


def build_check_summary(summary_payload: dict[str, Any]) -> str:
    summary = summary_payload["summary"]
    return (
        f"Published {summary['published_inline']} inline findings, "
        f"kept {summary['summary_only']} summary-only, "
        f"dropped {summary['duplicate_dropped']} duplicates, "
        f"{summary['unverifiable_dropped']} unverifiable, and {summary['outdated_dropped']} outdated findings."
    )


def build_check_text(
    *,
    title: str,
    url: str,
    summary_payload: dict[str, Any],
    artifact_errors: list[str],
    publish_errors: list[str],
) -> str:
    lines = [
        f"PR: {title}",
        f"URL: {url}",
        "",
        json.dumps(summary_payload, indent=2),
    ]
    if artifact_errors:
        lines.extend(["", "Lane errors:", *artifact_errors])
    if publish_errors:
        lines.extend(["", "Publish errors:", *publish_errors])
    return "\n".join(lines)


def marker_lines(*, head_sha: str, fingerprint: str | None = None, stage: str | None = None) -> str:
    lines = []
    if fingerprint:
        lines.append(f"<!-- {MARKER_PREFIX}:fingerprint={fingerprint} -->")
    if stage:
        lines.append(f"<!-- {MARKER_PREFIX}:stage={stage} -->")
    lines.append(f"<!-- {MARKER_PREFIX}:publisher=review_publish -->")
    lines.append(f"<!-- {MARKER_PREFIX}:head-sha={head_sha} -->")
    return "\n".join(lines)


def determine_review_event(findings: list[dict[str, Any]]) -> str:
    highest = max((SEVERITY_ORDER[finding["severity"]] for finding in findings), default=0)
    return "REQUEST_CHANGES" if highest >= SEVERITY_ORDER["high"] else "COMMENT"


def determine_check_conclusion(findings: list[dict[str, Any]], artifact_errors: list[str]) -> str:
    highest = max((SEVERITY_ORDER[finding["severity"]] for finding in findings), default=0)
    if highest >= SEVERITY_ORDER["high"]:
        return "failure"
    if artifact_errors:
        return "neutral"
    return "success"


def serialize_publication(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": finding["fingerprint"],
        "stage": finding["stage"],
        "reported_by": format_reported_by(finding),
        "decision": "publish_inline",
        "path": finding["path"],
        "line": finding["line"],
        "body": finding["body"],
    }


def serialize_summary_only(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": finding["fingerprint"],
        "stage": finding["stage"],
        "reported_by": format_reported_by(finding),
        "decision": finding["decision"],
        "body": finding["body"],
        "reason": finding["reason"],
    }


def serialize_drop(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": finding.get("fingerprint"),
        "decision": finding["decision"],
        "reason": finding["reason"],
        "stage": finding.get("stage"),
    }


def count_decisions(findings: list[dict[str, Any]], decision: str) -> int:
    return sum(1 for finding in findings if finding.get("decision") == decision)


def format_reported_by(finding: dict[str, Any]) -> str:
    stages = finding.get("stages") or [finding.get("stage")]
    labels = [STAGE_DISPLAY_NAMES.get(stage, stage) for stage in stages if stage]
    return ", ".join(dict.fromkeys(labels))


def humanize_title(title: str) -> str:
    cleaned = re.sub(r"^[a-z]+\([^)]*\):\s*", "", title, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*\[[A-Z]+-\d+\]\s*$", "", cleaned).strip()
    return cleaned or title


def summarize_diff(base_sha: str, head_sha: str) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "diff", "--name-only", base_sha, head_sha],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    areas = summarize_areas(files)
    return {"file_count": len(files), "areas": areas}


def summarize_areas(files: list[str]) -> list[str]:
    areas: list[str] = []
    for path in files:
        area = classify_area(path)
        if area not in areas:
            areas.append(area)
    return areas[:3]


def classify_area(path: str) -> str:
    if path.startswith(".github/scripts/review/"):
        return ".github/scripts/review"
    if path.startswith(".github/workflows/"):
        return ".github/workflows"
    if path.startswith("tests/"):
        return "tests"
    if "/" in path:
        return path.split("/", 1)[0]
    return path


def severity_style(severity: str) -> tuple[str, str, str]:
    return SEVERITY_STYLES.get(severity, SEVERITY_STYLES["medium"])


if __name__ == "__main__":
    raise SystemExit(main())
