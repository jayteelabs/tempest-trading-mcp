"""Reddit sentiment analyzer for crypto subreddit monitoring.

This module routes Reddit listing fetches through the VPS-side reddit-adapter
service (ENG-130) rather than hitting Reddit's public JSON API directly.

**Interim workaround — operational caveats:**
- The adapter must be running at the configured base URL for Reddit sentiment to work.
- If the adapter is unavailable, Reddit sentiment degrades to ``status="error"`` rather
  than pretending Reddit is healthy.
- This path depends on Josh's local machine and Tailscale exit-node remaining available.
- This is intentionally brittle and should be replaced with authenticated Reddit access
  when available.

Runtime configuration:
  Set ``REDDIT_ADAPTER_URL`` (env var) or pass ``adapter_url`` to the constructor.
  Defaults to ``http://127.0.0.1:8080`` — the loopback-published reddit-adapter endpoint.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from tempest_mcp.logging_config import get_logger
from tempest_mcp.sentiment.text_scoring import (
    clamp_score,
    coerce_scoring_text,
    compute_keyword_boost,
    score_sentiment_text,
)

logger = get_logger(__name__)

#: Default reddit-adapter base URL (ENG-130 stack, loopback-only on VPS).
DEFAULT_REDDIT_ADAPTER_URL = "http://127.0.0.1:8080"


_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin",),
    "ETH": ("ethereum",),
    "DOGE": ("dogecoin",),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_base_token(symbol: str) -> str:
    """Extract a plausible base token from a trading symbol.

    Handles common formats:
      - USDT pairs  (BTCUSDT -> BTC)
      - USD pairs   (BTCUSD  -> BTC)
      - BTC pairs   (ETHBTC  -> ETH)
      - Slash pairs (BTC/USD -> BTC)
      - Dash pairs  (BTC-USDT -> BTC)
      - Plain symbols (BTC -> BTC, DOGE -> DOGE)

    Returns the input unchanged if no recognized quote suffix is found.
    """
    s = symbol.upper()
    # Handle slash/dash split separators first
    if "/" in s:
        return s.split("/")[0]
    if "-" in s:
        return s.split("-")[0]
    # Then handle suffix-based quote tokens
    for sep in ("USDT", "USD", "BTC", "ETH"):
        if s.endswith(sep) and len(s) > len(sep):
            return s[: -len(sep)]
    return s


def _coerce_text(value: Any) -> str:
    """Return a deterministic string for Reddit text fields."""
    return coerce_scoring_text(value)


def _default_if_none(value: Any, default: Any) -> Any:
    """Preserve legitimate falsy values while defaulting missing Reddit fields."""
    return default if value is None else value


def _symbol_matches_title(symbol: str, base_token: str, title: str) -> bool:
    """Return True if the title mentions the raw symbol, base token, or a known alias."""
    lower_title = _coerce_text(title).lower()
    symbol_l = symbol.lower()
    base_l = base_token.lower()

    def matches_term(term: str) -> bool:
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        return pattern.search(lower_title) is not None

    if matches_term(symbol_l) or matches_term(base_l):
        return True

    for alias in _SYMBOL_ALIASES.get(base_token.upper(), ()):
        if matches_term(alias):
            return True

    return False


def _post_mentions_symbol(symbol: str, base_token: str, title: str, selftext: str) -> bool:
    """Return True if either title or selftext mentions the symbol/base token."""
    return _symbol_matches_title(symbol, base_token, title) or _symbol_matches_title(
        symbol, base_token, selftext
    )


def _compute_keyword_boost(title: str) -> float:
    """Compatibility wrapper for the shared keyword boost scorer."""
    return compute_keyword_boost(title)


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return clamp_score(value, lo=lo, hi=hi)


# ---------------------------------------------------------------------------
# RedditSentimentAnalyzer
# ---------------------------------------------------------------------------


@dataclass
class RedditSentimentAnalyzer:
    """Deterministic Reddit sentiment analyzer for crypto subreddits.

    Fetches hot posts from configured subreddits through the VPS-side
    reddit-adapter (ENG-130) and scores headlines with VADER sentiment
    augmented by a deterministic keyword boost layer.

    Args:
        subreddits: Tuple of subreddit names to monitor.
        adapter_url: Base URL of the reddit-adapter service.
            Defaults to ``http://127.0.0.1:8080`` or ``REDDIT_ADAPTER_URL`` env var.
    """

    subreddits: tuple[str, ...] = (
        "CryptoCurrency",
        "Bitcoin",
        "ethereum",
        "dogecoin",
    )
    adapter_url: str = field(default="")
    _http_client: httpx.Client = field(default=None, init=False, repr=False)
    _vader: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Resolve adapter URL: explicit param > env var > default
        if self.adapter_url:
            base = self.adapter_url.rstrip("/")
        else:
            base = os.environ.get("REDDIT_ADAPTER_URL", DEFAULT_REDDIT_ADAPTER_URL).rstrip("/")
        self._adapter_base = base

        # HTTP client used for adapter fetches; raise_for_status surfaces 4xx/5xx failures.
        self._http_client = httpx.Client(
            timeout=10.0,
            headers={"User-Agent": "tempest-trading-mcp/1.0"},
        )
        # VADER scorer instantiated once per analyzer instance
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._vader = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying HTTP client resources."""
        if self._http_client is None:
            return
        self._http_client.close()
        self._http_client = None

    def __enter__(self) -> RedditSentimentAnalyzer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def analyze(self, symbol: str) -> dict[str, Any]:
        """Analyze Reddit sentiment for a trading symbol.

        Fetches hot posts from the configured subreddits via the reddit-adapter,
        filters to posts whose title or selftext mentions the symbol or its base
        token, scores each title with VADER + keyword boost, and returns a
        structured result envelope.

        Returns
        -------
        dict
            A dict conforming to the schema defined in the ENG-39 ticket:
            {
                "symbol": str,
                "fetched_at": str,          # ISO-8601 UTC
                "subreddits": list[str],
                "posts": list[dict],         # empty when status != "ok"
                "summary": dict,
                "status": "ok" | "no_results" | "error"
            }

        Notes
        -----
        If the reddit-adapter is unavailable (connection refused, timeout, etc.),
        returns ``status="error"`` with zeroed summary — truthful degradation rather
        than false-positive health.
        """
        base_token = _extract_base_token(symbol)
        fetched_at = datetime.now(timezone.utc).isoformat()
        all_posts: list[dict[str, Any]] = []

        for subreddit in self.subreddits:
            try:
                raw_posts = self._fetch_subreddit_posts(subreddit)
            except httpx.HTTPError as exc:
                logger.warning(
                    "reddit_fetch_failed",
                    subreddit=subreddit,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                # Deterministic full-error envelope: any fetch failure → "error"
                return self._error_result(symbol, fetched_at, exc)

            for raw in raw_posts:
                title = _coerce_text(raw.get("title"))
                selftext = _coerce_text(raw.get("selftext"))
                if not _post_mentions_symbol(symbol, base_token, title, selftext):
                    continue
                post_record = self._build_post_record(subreddit, raw, title)
                all_posts.append(post_record)

        if not all_posts:
            return self._no_results_result(symbol, fetched_at)

        summary = self._summarize_posts(all_posts)
        return {
            "symbol": symbol,
            "fetched_at": fetched_at,
            "subreddits": list(self.subreddits),
            "posts": all_posts,
            "summary": summary,
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Private seam (mockable in tests)
    # ------------------------------------------------------------------

    def _fetch_subreddit_posts(self, subreddit: str) -> list[dict[str, Any]]:
        """Fetch and return the ``children`` list from the reddit-adapter hot endpoint.

        The adapter preserves Reddit's native JSON shape, so the parsing logic
        is identical to the direct-Reddit path.
        """
        url = f"{self._adapter_base}/r/{subreddit}/hot.json?limit=25"
        response = self._http_client.get(url)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise httpx.DecodingError(
                f"Invalid JSON returned by adapter for r/{subreddit}",
                request=response.request,
            ) from exc
        children = data.get("data", {}).get("children", [])
        return [child.get("data", {}) for child in children]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_post_record(self, subreddit: str, raw: dict[str, Any], title: str) -> dict[str, Any]:
        """Normalize a Reddit post into the required schema shape."""
        return {
            "subreddit": subreddit,
            "title": title,
            "score": _default_if_none(raw.get("score"), 0),
            "num_comments": _default_if_none(raw.get("num_comments"), 0),
            "upvote_ratio": _default_if_none(raw.get("upvote_ratio"), 0.0),
            "flair": raw.get("link_flair_text"),
            "created_utc": _default_if_none(raw.get("created_utc"), 0.0),
            "sentiment": score_sentiment_text(title, self._vader).as_dict(),
        }

    def _summarize_posts(self, posts: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate statistics from scored posts."""
        total = len(posts)
        final_scores = [p["sentiment"]["final_score"] for p in posts]
        avg = round(sum(final_scores) / total, 4) if total else 0.0
        positive = sum(1 for s in final_scores if s > 0.05)
        negative = sum(1 for s in final_scores if s < -0.05)
        neutral = total - positive - negative
        return {
            "total_posts": total,
            "avg_sentiment": avg,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
        }

    def _no_results_result(self, symbol: str, fetched_at: str) -> dict[str, Any]:
        """Deterministic envelope returned when no matching posts were found."""
        return {
            "symbol": symbol,
            "fetched_at": fetched_at,
            "subreddits": list(self.subreddits),
            "posts": [],
            "summary": {
                "total_posts": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "status": "no_results",
        }

    def _error_result(
        self, symbol: str, fetched_at: str, reason: Exception | str
    ) -> dict[str, Any]:
        """Deterministic envelope returned on any fetch failure."""
        reason_text = str(reason)
        log_kwargs = {"symbol": symbol, "reason": reason_text}
        if isinstance(reason, Exception):
            log_kwargs["error_type"] = type(reason).__name__
        logger.warning("reddit_analyze_error", **log_kwargs)
        return {
            "symbol": symbol,
            "fetched_at": fetched_at,
            "subreddits": list(self.subreddits),
            "posts": [],
            "summary": {
                "total_posts": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "status": "error",
        }
