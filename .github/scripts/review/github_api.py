from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any
from urllib.error import HTTPError


class GitHubApiError(RuntimeError):
    """GitHub API request failed."""


class GitHubApi:
    def __init__(self, *, token: str, repository: str) -> None:
        if repository.count("/") != 1:
            raise ValueError("Invalid repository: expected 'owner/repo'")
        owner, repo = repository.split("/", 1)
        if not owner or not repo:
            raise ValueError("Invalid repository: expected non-empty 'owner/repo'")
        self.owner = owner
        self.repo = repo
        self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
        self.token = token

    def list_review_comments(self, pull_number: int) -> list[dict[str, Any]]:
        return self._request("GET", f"/pulls/{pull_number}/comments?per_page=100")

    def list_reviews(self, pull_number: int) -> list[dict[str, Any]]:
        return self._request("GET", f"/pulls/{pull_number}/reviews?per_page=100")

    def create_review(
        self,
        *,
        pull_number: int,
        commit_id: str,
        body: str,
        event: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"commit_id": commit_id, "body": body, "event": event}
        return self._request("POST", f"/pulls/{pull_number}/reviews", payload)

    def create_review_comment(
        self,
        *,
        pull_number: int,
        commit_id: str,
        path: str,
        line: int,
        body: str,
    ) -> dict[str, Any]:
        payload = {
            "body": body,
            "commit_id": commit_id,
            "path": path,
            "line": line,
            "side": "RIGHT",
        }
        return self._request("POST", f"/pulls/{pull_number}/comments", payload)

    def list_check_runs(self, ref: str, check_name: str) -> dict[str, Any]:
        query = urllib.parse.urlencode(
            {"check_name": check_name, "filter": "latest", "per_page": 100}
        )
        return self._request("GET", f"/commits/{ref}/check-runs?{query}")

    def upsert_check_run(
        self,
        *,
        ref: str,
        name: str,
        external_id: str,
        summary: str,
        text: str,
        conclusion: str,
    ) -> dict[str, Any]:
        payload = {
            "name": name,
            "head_sha": ref,
            "status": "completed",
            "conclusion": conclusion,
            "external_id": external_id,
            "output": {"title": name, "summary": summary, "text": text},
        }
        existing = self.list_check_runs(ref, name).get("check_runs", [])
        for check_run in existing:
            if check_run.get("external_id") == external_id or check_run.get("name") == name:
                return self._request("PATCH", f"/check-runs/{check_run['id']}", payload)
        return self._request("POST", "/check-runs", payload)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request) as response:
                if response.length == 0:
                    return {}
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise GitHubApiError(
                f"GitHub API {method} {path} failed with {exc.code}: {detail}"
            ) from exc
