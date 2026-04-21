"""Combined sentiment dashboard — composes Reddit + RSS sentiment into one deterministic envelope."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from tempest_mcp.sentiment.reddit import RedditSentimentAnalyzer
from tempest_mcp.sentiment.rss import RSSSentimentAnalyzer

# Fixed weighting: Reddit 40%, News (RSS) 60%
_REDDIT_WEIGHT = 0.4
_RSS_WEIGHT = 0.6

# Polarity thresholds (derived from VADER compound conventions)
_POLARITY_BULLISH_THRESHOLD = 0.05
_POLARITY_BEARISH_THRESHOLD = -0.05


def _derive_polarity(sentiment_index: float) -> str:
    """Derive polarity label from sentiment_index value.

    Thresholds match VADER compound interpretation:
      - > 0.05  -> bullish
      - < -0.05 -> bearish
      - else    -> neutral
    """
    if sentiment_index > _POLARITY_BULLISH_THRESHOLD:
        return "bullish"
    if sentiment_index < _POLARITY_BEARISH_THRESHOLD:
        return "bearish"
    return "neutral"


def _is_usable(result: dict[str, Any]) -> bool:
    """Return True when a source result has a usable sentiment score.

    A source is usable only when:
      - status == "ok" AND
      - avg_sentiment is finite (not None, not NaN, not infinite)
    """
    if result.get("status") != "ok":
        return False
    avg = result.get("summary", {}).get("avg_sentiment")
    if avg is None:
        return False
    if not math.isfinite(avg):
        return False
    return True


def _build_source_diagnostic(result: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a source result (or absence) into a diagnostic record.

    Returns a dict with:
      - status: "ok" | "no_results" | "error" | "unavailable"
      - usable: bool
      - avg_sentiment: float | null
      - sample_count: int (total_posts for reddit, total_items for rss)
    """
    if result is None:
        return {
            "status": "unavailable",
            "usable": False,
            "avg_sentiment": None,
            "sample_count": 0,
        }

    status = result.get("status", "error")
    usable = _is_usable(result)

    # Determine sample count from source type
    summary = result.get("summary", {})
    if "total_posts" in summary:
        sample_count = summary.get("total_posts", 0)
    elif "total_items" in summary:
        sample_count = summary.get("total_items", 0)
    else:
        sample_count = 0

    return {
        "status": status,
        "usable": usable,
        "avg_sentiment": summary.get("avg_sentiment"),
        "sample_count": sample_count,
    }


def _build_cross_signal_flags(
    sentiment_polarity: str, price_bias: str
) -> list[str]:
    """Emit cross-signal flags when sentiment disagrees with caller-supplied bias.

    Cross-signal is emitted only when:
      - price_bias is explicitly bullish OR bearish (not neutral) AND
      - sentiment polarity is opposite to price_bias

    Returns a list of flag strings (empty when no disagreement).
    """
    if price_bias == "neutral":
        return []

    if price_bias == "bullish" and sentiment_polarity == "bearish":
        return ["sentiment_bearish_vs_price_bias_bullish"]
    if price_bias == "bearish" and sentiment_polarity == "bullish":
        return ["sentiment_bullish_vs_price_bias_bearish"]

    return []


# ---------------------------------------------------------------------------
# CombinedSentimentDashboard
# ---------------------------------------------------------------------------


@dataclass
class CombinedSentimentDashboard:
    """Deterministic Reddit + RSS sentiment dashboard.

    Composes RedditSentimentAnalyzer and RSSSentimentAnalyzer into a single
    weighted sentiment_index with explicit per-source diagnostics.

    Constructor accepts injectable analyzers for testability. When not
    provided, concrete instances are created lazily on first use.
    """

    reddit_analyzer: RedditSentimentAnalyzer | None = None
    rss_analyzer: RSSSentimentAnalyzer | None = None

    # Internally initialized on first analyze() call
    _reddit: RedditSentimentAnalyzer = field(init=False, repr=False)
    _rss: RSSSentimentAnalyzer = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Lazily created if not injected (keeps default constructor side-effect free)
        self._reddit = self.reddit_analyzer
        self._rss = self.rss_analyzer

    def _ensure_analyzers(self) -> None:
        """Lazily instantiate concrete analyzers if not injected."""
        if self._reddit is None:
            self._reddit = RedditSentimentAnalyzer()
        if self._rss is None:
            self._rss = RSSSentimentAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, symbol: str, price_bias: str) -> dict[str, Any]:
        """Analyze combined sentiment for a symbol.

        Fetches from both Reddit and RSS sources, applies deterministic
        weighting (40% Reddit / 60% RSS) when both are usable, falls back
        to the single usable source when only one is available, and returns
        a structured dashboard envelope.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. "BTCUSDT", "ETH/USD")
        price_bias : str
            Caller-supplied directional bias. Must be one of:
            "bullish", "bearish", "neutral"

        Returns
        -------
        dict
            Dashboard envelope conforming to ENG-41 schema:
            {
                "symbol": str,
                "fetched_at": str,          # ISO-8601 UTC
                "price_bias": str,
                "sentiment_index": float | null,
                "sentiment_polarity": str,  # "bullish" | "bearish" | "neutral"
                "combination_mode": str,    # "weighted" | "single_source" | "unavailable"
                "diagnostics": {
                    "sources": { "reddit": {...}, "rss": {...} },
                    "weights": {
                        "configured": { "reddit": 0.4, "rss": 0.6 },
                        "applied": { "reddit": float, "rss": float }
                    },
                    "fallback_reason": str | null
                },
                "cross_signal_flags": list[str]
            }
        """
        self._ensure_analyzers()

        fetched_at = datetime.now(timezone.utc).isoformat()

        # Execute both analyzers
        reddit_result = self._reddit.analyze(symbol)
        rss_result = self._rss.analyze(symbol)

        # Build per-source diagnostics
        reddit_diag = _build_source_diagnostic(reddit_result)
        rss_diag = _build_source_diagnostic(rss_result)

        # Determine usability
        reddit_usable = reddit_diag["usable"]
        rss_usable = rss_diag["usable"]

        # Compute sentiment_index using deterministic combination algorithm
        (
            sentiment_index,
            combination_mode,
            fallback_reason,
            applied_reddit_weight,
            applied_rss_weight,
        ) = self._combine(reddit_result, rss_result, reddit_usable, rss_usable)

        # Derive polarity
        if sentiment_index is not None:
            sentiment_polarity = _derive_polarity(sentiment_index)
        else:
            sentiment_polarity = "neutral"

        # Cross-signal detection
        cross_signal_flags = _build_cross_signal_flags(sentiment_polarity, price_bias)

        return {
            "symbol": symbol,
            "fetched_at": fetched_at,
            "price_bias": price_bias,
            "sentiment_index": sentiment_index,
            "sentiment_polarity": sentiment_polarity,
            "combination_mode": combination_mode,
            "diagnostics": {
                "sources": {
                    "reddit": reddit_diag,
                    "rss": rss_diag,
                },
                "weights": {
                    "configured": {"reddit": _REDDIT_WEIGHT, "rss": _RSS_WEIGHT},
                    "applied": {
                        "reddit": applied_reddit_weight,
                        "rss": applied_rss_weight,
                    },
                },
                "fallback_reason": fallback_reason,
            },
            "cross_signal_flags": cross_signal_flags,
        }

    # ------------------------------------------------------------------
    # Private seam (mockable in tests)
    # ------------------------------------------------------------------

    def _combine(
        self,
        reddit_result: dict[str, Any],
        rss_result: dict[str, Any],
        reddit_usable: bool,
        rss_usable: bool,
    ) -> tuple[float | None, str, str | None, float, float]:
        """Deterministic combination algorithm.

        Returns a 5-tuple:
          (sentiment_index, combination_mode, fallback_reason,
           applied_reddit_weight, applied_rss_weight)
        """
        both_usable = reddit_usable and rss_usable
        only_reddit_usable = reddit_usable and not rss_usable
        only_rss_usable = not reddit_usable and rss_usable
        neither_usable = not reddit_usable and not rss_usable

        if both_usable:
            reddit_avg = reddit_result["summary"]["avg_sentiment"]
            rss_avg = rss_result["summary"]["avg_sentiment"]
            sentiment_index = round(
                (reddit_avg * _REDDIT_WEIGHT) + (rss_avg * _RSS_WEIGHT), 4
            )
            return (
                sentiment_index,
                "weighted",
                None,
                _REDDIT_WEIGHT,
                _RSS_WEIGHT,
            )

        if only_reddit_usable:
            sentiment_index = round(
                reddit_result["summary"]["avg_sentiment"], 4
            )
            return (
                sentiment_index,
                "single_source",
                "rss source not usable",
                1.0,
                0.0,
            )

        if only_rss_usable:
            sentiment_index = round(rss_result["summary"]["avg_sentiment"], 4)
            return (
                sentiment_index,
                "single_source",
                "reddit source not usable",
                0.0,
                1.0,
            )

        # neither_usable: neither source returned a usable score
        if neither_usable:
            return (
                None,
                "unavailable",
                "neither source returned usable sentiment",
                0.0,
                0.0,
            )

        # Defensive: should not reach here given exhaustive conditions above
        return (
            None,
            "unavailable",
            "neither source returned usable sentiment",
            0.0,
            0.0,
        )

