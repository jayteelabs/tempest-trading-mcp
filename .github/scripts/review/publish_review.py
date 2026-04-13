from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from review.collect_changed_lines import collect_changed_lines
from review.dedupe_findings import merge_duplicate_findings
from review.github_api import GitHubApi
from review.schema import SEVERITY_ORDER, load_artifact
from review.validate_findings import classify_findings

CHECK_RUN_NAME = "Review / Publish"
MARKER_PREFIX = "tempest-review"
STAGE_DISPLAY_NAMES = {
    "review/design": "Shuna / Design",
    "review/security_audit": "Shion / Security Audit (Lane A)",
    "review/logic_audit": "Shion / Logic Audit (Lane B)",
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
    repo_root = Path(__file__).resolve().parents[3]

    artifacts = [
        load_artifact(args.design),
        load_artifact(args.security_audit),
        load_artifact(args.logic_audit),
    ]

    artifact_errors = [error for artifact in artifacts for error in artifact.get("errors", [])]
    all_findings = [finding for artifact in artifacts for finding in artifact.get("findings", [])]

    merged_findings, duplicate_drops = merge_duplicate_findings(all_findings)
    changed_lines = collect_changed_lines(args.base_sha, args.head_sha)
    inline_findings, summary_only, validation_drops = classify_findings(
        merged_findings,
        changed_lines=changed_lines,
        repo_root=repo_root,
    )

    existing_comments = api.get_review_comments(int(args.pr))
    existing_reviews = api.get_reviews(int(args.pr))
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

    review_body = build_review_body(
        head_sha=args.head_sha,
        published_inline=published_inline,
        summary_only=summary_only,
        drops=all_drops,
        artifact_errors=artifact_errors,
    )

    if published_inline or not review_exists_for_head:
        api.create_review(
            pull_number=int(args.pr),
            commit_id=args.head_sha,
            body=review_body,
            event=review_event,
            comments=[build_inline_comment(comment, args.head_sha) for comment in published_inline],
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
        ),
        conclusion=determine_check_conclusion([*published_inline, *summary_only], artifact_errors),
    )
    return 0


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
    body_lines = [f"**{title}**", finding["body"]]
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
    head_sha: str,
    published_inline: list[dict[str, Any]],
    summary_only: list[dict[str, Any]],
    drops: list[dict[str, Any]],
    artifact_errors: list[str],
) -> str:
    lines = [
        "## Review / Publish",
        f"- Inline comments published: {len(published_inline)}",
        f"- Summary-only findings: {len(summary_only)}",
        f"- Duplicate findings dropped: {count_decisions(drops, 'drop_duplicate')}",
        f"- Unverifiable findings dropped: {count_decisions(drops, 'drop_unverifiable')}",
        f"- Outdated findings dropped: {count_decisions(drops, 'drop_outdated')}",
    ]
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
    lines.extend(
        [
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
) -> str:
    lines = [
        f"PR: {title}",
        f"URL: {url}",
        "",
        json.dumps(summary_payload, indent=2),
    ]
    if artifact_errors:
        lines.extend(["", "Lane errors:", *artifact_errors])
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


if __name__ == "__main__":
    raise SystemExit(main())
