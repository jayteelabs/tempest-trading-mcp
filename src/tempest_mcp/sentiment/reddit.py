"""Reddit sentiment analyzer for crypto subreddit monitoring.

.. deprecated::
    The Reddit public-feed source (``https://www.reddit.com/r/{subreddit}/hot.json``)
    is no longer supported. Reddit now returns HTTP 403 on all public-feed requests
    with no authenticated workaround within this ticket's scope.

    - ENG-128 restored the combined dashboard via RSS fallback after ``vaderSentiment``
      container failure.
    - ENG-129 explicitly marks the Reddit public-feed path as a deprecated source
      rather than pursuing authenticated recovery.
    - Combined sentiment diagnostics already honestly degrade to ``single_source`` mode
      when Reddit returns ``status="deprecated"``.

    See: https://linear.app/jayteelabs/issue/ENG-129
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword boost tables
# ---------------------------------------------------------------------------
# Each term is applied at most once per post.
# Terms are ordered longest-first so phrases win over their shorter substrings.
# Matching is case-insensitive and overlap-aware.
_ALL_BOOST_TERMS: list[tuple[str, float]] = [
    ("to the moon", 0.15),
    ("alts are up", 0.15),
    ("bullish", 0.15),
    ("moon", 0.15),
    ("pump", 0.15),
    ("rugpull", -0.15),
    ("bearish", -0.15),
    ("dump", -0.15),
    ("crash", -0.15),
    ("rug", -0.15),
    ("bear", -0.15),
]

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
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


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


# Deduplicated boost map: term -> boost (first occurrence wins for determinism)
_BOOST_MAP: dict[str, float] = dict(_ALL_BOOST_TERMS)
# Ordered list for deterministic iteration
_BOOST_TERMS: list[tuple[str, float]] = list(_BOOST_MAP.items())


def _compute_keyword_boost(title: str) -> float:
    """Compute deterministic keyword modifier for a post title.

    Each known term is applied at most once. Boosts are summed and clamped
    to [-1.0, 1.0]. Matching is case-insensitive and overlap-aware.
    Longer phrases are evaluated first so they win over shorter substrings.
    """
    lower_title = _coerce_text(title).lower()
    occupied_spans: list[tuple[int, int]] = []
    boost = 0.0

    def overlaps(span: tuple[int, int]) -> bool:
        start, end = span
        for other_start, other_end in occupied_spans:
            if start < other_end and end > other_start:
                return True
        return False

    for term, value in _BOOST_TERMS:
        pattern = (
            re.compile(rf"\b{re.escape(term)}\b") if term.isalpha() else re.compile(re.escape(term))
        )
        match = pattern.search(lower_title)
        if match is None:
            continue
        span = match.span()
        if overlaps(span):
            continue
        occupied_spans.append(span)
        boost += value

    return max(-1.0, min(1.0, boost))


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Custom exceptions for Reddit source diagnostics
# ---------------------------------------------------------------------------


class RedditHTTPError(Exception):
    """Base class for Reddit-specific HTTP errors.

    Used internally to distinguish Reddit source failures from generic
    network errors. The ``is_blocked`` attribute indicates whether the
    error appears to be a 403/blocked response from Reddit's public feed.
    """

    is_blocked: bool = False


class RedditBlockedError(RedditHTTPError):
    """Raised when Reddit returns HTTP 403 on public-feed requests.

    This indicates the Reddit public-feed path is no longer accessible.
    The combined sentiment dashboard treats this as an explicit deprecated-source
    outcome per ENG-129, not a recoverable network error.
    """

    is_blocked: bool = True


# ---------------------------------------------------------------------------
# RedditSentimentAnalyzer
# ---------------------------------------------------------------------------


@dataclass
class RedditSentimentAnalyzer:
    """Deterministic Reddit sentiment analyzer for crypto subreddits.

    .. deprecated::
        The public JSON feed at ``https://www.reddit.com/r/{subreddit}/hot.json``
        is no longer supported. Reddit now returns HTTP 403 on all public-feed
        requests. This analyzer returns ``status="deprecated"`` for all requests
        per ENG-129.

        Use :class:`CombinedSentimentDashboard` which falls back to RSS when
        Reddit is unavailable.

    Monitors r/CryptoCurrency, r/Bitcoin, r/ethereum, and r/dogecoin using
    Reddit's public JSON API. Scores post headlines with VADER sentiment
    augmented by a deterministic keyword boost layer.
    """

    subreddits: tuple[str, ...] = (
        "CryptoCurrency",
        "Bitcoin",
        "ethereum",
        "dogecoin",
    )
    _http_client: httpx.Client = field(default=None, init=False, repr=False)
    _vader: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # HTTP client used for subreddit fetches; raise_for_status surfaces 4xx/5xx failures.
        self._http_client = httpx.Client(
            timeout=10.0,
            headers={"User-Agent": "tempest-tradingview-mcp/1.0"},
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

        .. deprecated::
            The Reddit public-feed path is no longer accessible (HTTP 403).
            This method always returns ``status="deprecated"`` per ENG-129.

            Use :class:`CombinedSentimentDashboard` for sentiment analysis
            with RSS fallback.

        Fetches hot posts from the configured subreddits, filters to posts
        whose title or selftext mentions the symbol or its base token, scores
        each title with VADER + keyword boost, and returns a structured result
        envelope.

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
                "status": "ok" | "no_results" | "error" | "deprecated"
            }

            ``status="deprecated"`` is returned when Reddit returns HTTP 403,
            indicating the public-feed path is no longer a supported data source.
        """
        base_token = _extract_base_token(symbol)
        fetched_at = datetime.now(timezone.utc).isoformat()
        all_posts: list[dict[str, Any]] = []

        for subreddit in self.subreddits:
            try:
                raw_posts = self._fetch_subreddit_posts(subreddit)
            except RedditBlockedError:
                logger.warning(
                    "reddit_public_feed_blocked",
                    subreddit=subreddit,
                    reason="403 Blocked - Reddit public-feed is no longer supported",
                )
                # Explicit deprecated-source outcome: 403 from Reddit public feed
                return self._deprecated_result(symbol, fetched_at)
            except httpx.HTTPError as exc:
                logger.warning(
                    "reddit_fetch_failed",
                    subreddit=subreddit,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                # Generic network error envelope
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
        """Fetch and return the `children` list from Reddit's hot endpoint.

        Raises
        ------
        RedditBlockedError
            When Reddit returns HTTP 403, indicating the public-feed path
            is no longer accessible.
        httpx.HTTPError
            For other HTTP error responses (4xx/5xx) or network failures.
        """
        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"
        response = self._http_client.get(url)

        # Check for 403 specifically to provide explicit deprecated-source outcome
        if response.status_code == 403:
            raise RedditBlockedError(
                "403 Blocked: Reddit public-feed is no longer supported (ENG-129)"
            )

        response.raise_for_status()
        try:
            data = response.json()
        except ValueError as exc:
            raise httpx.DecodingError(
                f"Invalid JSON returned by Reddit for r/{subreddit}",
                request=response.request,
            ) from exc
        children = data.get("data", {}).get("children", [])
        return [child.get("data", {}) for child in children]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_post_record(self, subreddit: str, raw: dict[str, Any], title: str) -> dict[str, Any]:
        """Normalize a Reddit post into the required schema shape."""
        vader_scores = self._vader.polarity_scores(title)
        keyword_boost = _compute_keyword_boost(title)
        final_score = _clamp(vader_scores["compound"] + keyword_boost)

        return {
            "subreddit": subreddit,
            "title": title,
            "score": _default_if_none(raw.get("score"), 0),
            "num_comments": _default_if_none(raw.get("num_comments"), 0),
            "upvote_ratio": _default_if_none(raw.get("upvote_ratio"), 0.0),
            "flair": raw.get("link_flair_text"),
            "created_utc": _default_if_none(raw.get("created_utc"), 0.0),
            "sentiment": {
                "vader_pos": round(vader_scores["pos"], 4),
                "vader_neu": round(vader_scores["neu"], 4),
                "vader_neg": round(vader_scores["neg"], 4),
                "vader_compound": round(vader_scores["compound"], 4),
                "keyword_boost": round(keyword_boost, 4),
                "final_score": round(final_score, 4),
            },
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

    def _deprecated_result(self, symbol: str, fetched_at: str) -> dict[str, Any]:
        """Deterministic envelope returned when Reddit public-feed returns 403.

        This explicitly marks the Reddit source as deprecated per ENG-129 rather
        than treating it as a generic network error. The combined sentiment
        dashboard uses ``status != "ok"`` to determine source usability, so
        ``"deprecated"`` correctly signals the source is not usable.
        """
        logger.warning(
            "reddit_source_deprecated",
            symbol=symbol,
            reason="403 Blocked - Reddit public-feed is no longer supported (ENG-129)",
        )
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
            "status": "deprecated",
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
