from __future__ import annotations

import argparse
import os
import re
import subprocess
import traceback
from pathlib import Path

from review.schema import extract_json_document, normalize_findings, write_artifact
from review.utils import repo_root, validate_git_sha

MODEL = "minimax/MiniMax-M2.7-highspeed"
MAX_DIFF_CHARS = 60000
MAX_REFERENCE_CHARS = 12000
OPENCODE_TIMEOUT_SECONDS = 300
SETUP_ERROR_PREFIX = "SETUP/TRANSPORT ERROR"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--out", required=True)
    return parser


def collect_diff(base_sha: str, head_sha: str) -> str:
    validate_git_sha("base_sha", base_sha)
    validate_git_sha("head_sha", head_sha)
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--unified=3", base_sha, head_sha],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=True,
    )
    diff_text = result.stdout.strip()
    if len(diff_text) <= MAX_DIFF_CHARS:
        return diff_text
    return diff_text[:MAX_DIFF_CHARS] + "\n\n[Diff truncated for prompt budget]"


def extract_ticket_id(title: str) -> str | None:
    match = re.search(r"\bENG-\d+\b", title)
    return match.group(0) if match else None


def collect_reference_documents(stage: str, ticket_id: str | None) -> tuple[list[str], str]:
    references: list[tuple[str, str]] = []
    root = repo_root()

    stack_context = root / ".agents" / "stack-context.md"
    if stack_context.exists():
        references.append(
            (str(stack_context.relative_to(root)), stack_context.read_text(encoding="utf-8"))
        )

    if stage == "review/design":
        for path in design_reference_paths(root, ticket_id):
            rel_path = str(path.relative_to(root))
            if rel_path not in {item[0] for item in references}:
                references.append((rel_path, path.read_text(encoding="utf-8")))

    snippets: list[str] = []
    total = 0
    used_paths: list[str] = []
    for relative_path, content in references:
        snippet = f"## {relative_path}\n{content.strip()}\n"
        if total + len(snippet) > MAX_REFERENCE_CHARS:
            break
        snippets.append(snippet)
        used_paths.append(relative_path)
        total += len(snippet)

    return used_paths, "\n".join(snippets).strip()


def design_reference_paths(root: Path, ticket_id: str | None) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    search_roots = [
        root / "docs",
        root / "design-outputs",
        root / "specs",
        root / "prd",
    ]

    def add(path: Path) -> None:
        if path.is_file() and path.suffix == ".md" and path not in seen:
            seen.add(path)
            candidates.append(path)

    if ticket_id:
        ticket_patterns = [
            f"**/{ticket_id}*.md",
            f"**/*{ticket_id}*.md",
        ]
        for directory in search_roots:
            if not directory.exists():
                continue
            for pattern in ticket_patterns:
                for path in sorted(directory.glob(pattern)):
                    add(path)

    keyword_patterns = ["**/*prd*.md", "**/*spec*.md", "**/*design*.md", "**/*requirements*.md"]
    for directory in search_roots:
        if not directory.exists():
            continue
        for pattern in keyword_patterns:
            for path in sorted(directory.glob(pattern)):
                add(path)

    docs_dir = root / "docs"
    if docs_dir.exists():
        for path in sorted(docs_dir.glob("*.md")):
            add(path)

    return candidates


def stage_focus(stage: str) -> str:
    if stage == "review/design":
        return (
            "Architecture consistency, contracts, data flow, layering, naming, and whether the diff matches the repo's documented design. "
            "Only use spec_refs when you can point to a repo-local markdown file included in the prompt."
        )
    if stage == "review/security_audit":
        return "Auth, sessions, input validation, secrets, filesystem access, subprocess usage, unsafe parsing, network boundaries, and privilege mistakes."
    return "Logic correctness, state transitions, edge cases, error handling, data shape drift, and multi-file interaction failures."


def build_prompt(
    *, stage: str, pr_number: str, title: str, url: str, diff_text: str, reference_text: str
) -> str:
    reference_block = (
        reference_text
        if reference_text
        else "No additional repo-local reference docs were supplied."
    )
    return f"""
You are running {stage} for a GitHub pull request.

Return JSON only. No markdown, no prose outside JSON.

Required response shape:
{{
  "findings": [
    {{
      "source_agent": "{stage.split("/")[-1]}",
      "stage": "{stage}",
      "severity": "critical|high|medium|low|info",
      "category": "short-kebab-case-category",
      "path": "repo/relative/path.py or null",
      "line": 123,
      "title": "short title",
      "body": "concise actionable reviewer comment",
      "evidence": "why this is supported by the diff",
      "confidence": 0.0,
      "spec_refs": ["docs/example.md"],
      "requires_spec_ref": false
    }}
  ]
}}

Rules:
- Emit only findings that are supported by the diff.
- Emit only actionable findings with a concrete correctness, security, or operational impact.
- Prefer path+line anchors on changed lines. If you cannot anchor safely, set path and line to null.
- If a claim depends on repo-local docs, include those files in spec_refs and set requires_spec_ref=true.
- If there are no valid findings, return {{"findings": []}}.
- Ignore naming-only suggestions, empty package markers, module-size/style opinions, intentional documented auth patterns, and speculative refactors without concrete impact.
- Do not misclassify `except Exception` as catching `KeyboardInterrupt` or `SystemExit`; in Python those derive from `BaseException` and propagate.
- Write `body` as concise GitHub-flavored markdown. Prefer 1-2 short paragraphs and bullets only when they improve readability.
- Keep body concise and ready to publish as a GitHub review comment.

Focus:
{stage_focus(stage)}

PR context:
- PR #: {pr_number}
- Title: {title}
- URL: {url}

Repo-local references:
{reference_block}

Diff:
{diff_text}
""".strip()


def run_opencode(agent: str, prompt: str) -> str:
    result = subprocess.run(
        ["/home/tempest/.opencode/bin/opencode", "run", "--agent", agent, "--model", MODEL],
        cwd=repo_root(),
        capture_output=True,
        input=prompt,
        text=True,
        check=True,
        timeout=OPENCODE_TIMEOUT_SECONDS,
    )
    return result.stdout.strip()


def format_setup_error(stage: str, detail: str) -> str:
    return f"{SETUP_ERROR_PREFIX}: {stage} - {detail}"


def execute_review(*, agent: str, stage: str) -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    ticket_id = extract_ticket_id(args.title)
    errors: list[str] = []
    findings = []
    used_references: list[str] = []

    try:
        diff_text = collect_diff(args.base_sha, args.head_sha)
        used_references, reference_text = collect_reference_documents(stage, ticket_id)
        prompt = build_prompt(
            stage=stage,
            pr_number=args.pr,
            title=args.title,
            url=args.url,
            diff_text=diff_text,
            reference_text=reference_text,
        )
        raw_output = run_opencode(agent, prompt)
        payload = extract_json_document(raw_output)
        findings = normalize_findings(payload, stage)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        errors.append(
            format_setup_error(
                stage, f"opencode exited non-zero before findings were parsed: {message}"
            )
        )
    except subprocess.TimeoutExpired as exc:
        errors.append(
            format_setup_error(
                stage,
                f"opencode timed out after {exc.timeout}s before findings were produced",
            )
        )
    except (OSError, ValueError) as exc:
        errors.append(
            format_setup_error(
                stage,
                f"review setup or transport failed before findings were produced: {exc}\n{traceback.format_exc().strip()}",
            )
        )

    write_artifact(
        args.out,
        stage=stage,
        findings=findings,
        errors=errors,
        metadata={
            "pr_number": int(args.pr),
            "title": args.title,
            "url": args.url,
            "ticket_id": ticket_id,
            "base_sha": args.base_sha,
            "head_sha": args.head_sha,
            "repository": os.environ.get("GITHUB_REPOSITORY", ""),
            "references": used_references,
        },
    )
    return 0
