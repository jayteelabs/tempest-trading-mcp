"""Unit tests for CombinedSentimentDashboard — ENG-41."""

import sys
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from tempest_mcp.sentiment.combined_sentiment import (
    CombinedSentimentDashboard,
    _build_cross_signal_flags,
    _build_source_diagnostic,
    _derive_polarity,
    _is_usable,
)

# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


def make_reddit_ok(avg_sentiment: float, total_posts: int) -> dict:
    """Create a Reddit 'ok' result envelope."""
    return {
        "symbol": "BTCUSDT",
        "fetched_at": "2026-04-21T00:00:00+00:00",
        "subreddits": ["CryptoCurrency"],
        "posts": [],
        "summary": {
            "total_posts": total_posts,
            "avg_sentiment": avg_sentiment,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        },
        "status": "ok",
    }


def make_rss_ok(avg_sentiment: float, total_items: int) -> dict:
    """Create an RSS 'ok' result envelope."""
    return {
        "symbol": "BTCUSDT",
        "fetched_at": "2026-04-21T00:00:00+00:00",
        "feeds": ["https://www.coindesk.com/arc/outboundfeeds/rss/"],
        "items": [],
        "summary": {
            "total_items": total_items,
            "avg_sentiment": avg_sentiment,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
        },
        "status": "ok",
    }


def make_no_results(source: str) -> dict:
    """Create a 'no_results' envelope for either source."""
    if source == "reddit":
        return {
            "symbol": "BTCUSDT",
            "fetched_at": "2026-04-21T00:00:00+00:00",
            "subreddits": ["CryptoCurrency"],
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
    else:
        return {
            "symbol": "BTCUSDT",
            "fetched_at": "2026-04-21T00:00:00+00:00",
            "feeds": ["https://www.coindesk.com/arc/outboundfeeds/rss/"],
            "items": [],
            "summary": {
                "total_items": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "status": "no_results",
        }


def make_error(source: str) -> dict:
    """Create an 'error' envelope for either source."""
    if source == "reddit":
        return {
            "symbol": "BTCUSDT",
            "fetched_at": "2026-04-21T00:00:00+00:00",
            "subreddits": ["CryptoCurrency"],
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
    else:
        return {
            "symbol": "BTCUSDT",
            "fetched_at": "2026-04-21T00:00:00+00:00",
            "feeds": ["https://www.coindesk.com/arc/outboundfeeds/rss/"],
            "items": [],
            "summary": {
                "total_items": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "status": "error",
        }


# ---------------------------------------------------------------------------
# _derive_polarity tests
# ---------------------------------------------------------------------------


class TestDerivePolarity:
    """Tests for _derive_polarity threshold behavior."""

    def test_above_bullish_threshold(self):
        assert _derive_polarity(0.06) == "bullish"
        assert _derive_polarity(0.5) == "bullish"

    def test_at_bullish_threshold_is_neutral(self):
        """At exactly 0.05, polarity is neutral (not bullish)."""
        assert _derive_polarity(0.05) == "neutral"

    def test_below_bearish_threshold(self):
        assert _derive_polarity(-0.06) == "bearish"
        assert _derive_polarity(-0.5) == "bearish"

    def test_at_bearish_threshold_is_neutral(self):
        """At exactly -0.05, polarity is neutral (not bearish)."""
        assert _derive_polarity(-0.05) == "neutral"

    def test_within_neutral_band(self):
        assert _derive_polarity(0.0) == "neutral"
        assert _derive_polarity(0.04) == "neutral"
        assert _derive_polarity(-0.04) == "neutral"


# ---------------------------------------------------------------------------
# _is_usable tests
# ---------------------------------------------------------------------------


class TestIsUsable:
    """Tests for _is_usable source-usability check."""

    def test_ok_with_finite_avg_is_usable(self):
        result = make_reddit_ok(avg_sentiment=0.1, total_posts=5)
        assert _is_usable(result) is True

    def test_no_results_is_not_usable(self):
        result = make_no_results("reddit")
        assert _is_usable(result) is False

    def test_error_is_not_usable(self):
        result = make_error("reddit")
        assert _is_usable(result) is False

    def test_ok_with_none_avg_is_not_usable(self):
        result = make_reddit_ok(avg_sentiment=None, total_posts=5)
        assert _is_usable(result) is False

    def test_ok_with_nan_avg_is_not_usable(self):
        import math

        result = make_reddit_ok(avg_sentiment=math.nan, total_posts=5)
        assert _is_usable(result) is False

    def test_ok_with_infinity_avg_is_not_usable(self):
        import math

        result = make_reddit_ok(avg_sentiment=math.inf, total_posts=5)
        assert _is_usable(result) is False

    def test_missing_summary_is_not_usable(self):
        result = {"status": "ok"}
        assert _is_usable(result) is False


# ---------------------------------------------------------------------------
# _build_source_diagnostic tests
# ---------------------------------------------------------------------------


class TestBuildSourceDiagnostic:
    """Tests for _build_source_diagnostic normalization."""

    def test_reddit_ok_result(self):
        result = make_reddit_ok(avg_sentiment=0.25, total_posts=12)
        diag = _build_source_diagnostic(result)
        assert diag["status"] == "ok"
        assert diag["usable"] is True
        assert diag["avg_sentiment"] == 0.25
        assert diag["sample_count"] == 12

    def test_rss_ok_result(self):
        result = make_rss_ok(avg_sentiment=0.33, total_items=9)
        diag = _build_source_diagnostic(result)
        assert diag["status"] == "ok"
        assert diag["usable"] is True
        assert diag["avg_sentiment"] == 0.33
        assert diag["sample_count"] == 9

    def test_no_results_diagnostic(self):
        result = make_no_results("reddit")
        diag = _build_source_diagnostic(result)
        assert diag["status"] == "no_results"
        assert diag["usable"] is False
        assert diag["avg_sentiment"] == 0.0
        assert diag["sample_count"] == 0

    def test_error_diagnostic(self):
        result = make_error("reddit")
        diag = _build_source_diagnostic(result)
        assert diag["status"] == "error"
        assert diag["usable"] is False

    def test_none_result_returns_unavailable(self):
        diag = _build_source_diagnostic(None)
        assert diag["status"] == "unavailable"
        assert diag["usable"] is False
        assert diag["avg_sentiment"] is None
        assert diag["sample_count"] == 0


# ---------------------------------------------------------------------------
# _build_cross_signal_flags tests
# ---------------------------------------------------------------------------


class TestBuildCrossSignalFlags:
    """Tests for _build_cross_signal_flags disagreement detection."""

    def test_bullish_bias_with_bearish_sentiment_emits_flag(self):
        flags = _build_cross_signal_flags("bearish", "bullish")
        assert flags == ["sentiment_bearish_vs_price_bias_bullish"]

    def test_bearish_bias_with_bullish_sentiment_emits_flag(self):
        flags = _build_cross_signal_flags("bullish", "bearish")
        assert flags == ["sentiment_bullish_vs_price_bias_bearish"]

    def test_neutral_bias_suppresses_flags(self):
        flags = _build_cross_signal_flags("bullish", "neutral")
        assert flags == []
        flags = _build_cross_signal_flags("bearish", "neutral")
        assert flags == []

    def test_matching_sentiment_and_bias_no_flags(self):
        flags = _build_cross_signal_flags("bullish", "bullish")
        assert flags == []
        flags = _build_cross_signal_flags("bearish", "bearish")
        assert flags == []

    def test_neutral_sentiment_no_flags(self):
        flags = _build_cross_signal_flags("neutral", "bullish")
        assert flags == []
        flags = _build_cross_signal_flags("neutral", "bearish")
        assert flags == []


# ---------------------------------------------------------------------------
# CombinedSentimentDashboard happy-path weighted mode
# ---------------------------------------------------------------------------


class TestDashboardWeightedMode:
    """Tests for the weighted combination path (both sources usable)."""

    def test_weighted_mode_applies_40_60(self):
        """Both sources usable -> exact 0.4/0.6 weighting with 4-decimal rounding."""
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.10, total_posts=10)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.40, total_items=10)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("BTCUSDT", price_bias="neutral")

        # 0.10 * 0.4 + 0.40 * 0.6 = 0.04 + 0.24 = 0.28
        assert result["sentiment_index"] == 0.28
        assert result["combination_mode"] == "weighted"
        assert result["diagnostics"]["weights"]["applied"]["reddit"] == 0.4
        assert result["diagnostics"]["weights"]["applied"]["rss"] == 0.6
        assert result["diagnostics"]["fallback_reason"] is None

    def test_weighted_mode_polarity_bullish(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.2, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.3, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("ETHUSDT", price_bias="neutral")

        # (0.2 * 0.4) + (0.3 * 0.6) = 0.26 > 0.05 -> bullish
        assert result["sentiment_polarity"] == "bullish"

    def test_weighted_mode_polarity_bearish(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=-0.2, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=-0.3, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("DOGEUSDT", price_bias="neutral")

        # (0.2 * 0.4) + (0.3 * 0.6) = 0.26 > 0.05 -> bullish
        # But wait, these are negative: -0.2 * 0.4 + -0.3 * 0.6 = -0.26 < -0.05 -> bearish
        assert result["sentiment_polarity"] == "bearish"


# ---------------------------------------------------------------------------
# CombinedSentimentDashboard single-source fallback
# ---------------------------------------------------------------------------


class TestDashboardSingleSourceFallback:
    """Tests for single-source fallback behavior."""

    def test_reddit_only_usable(self):
        """Reddit usable, RSS not usable -> return Reddit score unchanged."""
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.15, total_posts=8)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_no_results("rss")

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("BTCUSDT", price_bias="neutral")

        assert result["sentiment_index"] == 0.15
        assert result["combination_mode"] == "single_source"
        assert result["diagnostics"]["fallback_reason"] == "rss source not usable"
        assert result["diagnostics"]["sources"]["reddit"]["usable"] is True
        assert result["diagnostics"]["sources"]["rss"]["usable"] is False
        # Applied weights: reddit=1.0, rss=0.0
        assert result["diagnostics"]["weights"]["applied"]["reddit"] == 1.0
        assert result["diagnostics"]["weights"]["applied"]["rss"] == 0.0

    def test_rss_only_usable(self):
        """RSS usable, Reddit not usable -> return RSS score unchanged."""
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_error("reddit")

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.22, total_items=7)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("ETHUSDT", price_bias="neutral")

        assert result["sentiment_index"] == 0.22
        assert result["combination_mode"] == "single_source"
        assert result["diagnostics"]["fallback_reason"] == "reddit source not usable"
        assert result["diagnostics"]["sources"]["rss"]["usable"] is True
        assert result["diagnostics"]["sources"]["reddit"]["usable"] is False
        # Applied weights: reddit=0.0, rss=1.0
        assert result["diagnostics"]["weights"]["applied"]["reddit"] == 0.0
        assert result["diagnostics"]["weights"]["applied"]["rss"] == 1.0


# ---------------------------------------------------------------------------
# CombinedSentimentDashboard neither source usable
# ---------------------------------------------------------------------------


class TestDashboardNeitherUsable:
    """Tests for the unavailable combination mode."""

    def test_neither_usable_returns_null_index(self):
        """Both sources unusable -> combination_mode=unavailable, sentiment_index=null."""
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_error("reddit")

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_no_results("rss")

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("BTCUSDT", price_bias="neutral")

        assert result["sentiment_index"] is None
        assert result["combination_mode"] == "unavailable"
        assert result["sentiment_polarity"] == "neutral"
        assert "neither source returned usable sentiment" in result["diagnostics"]["fallback_reason"]
        assert result["diagnostics"]["weights"]["applied"]["reddit"] == 0.0
        assert result["diagnostics"]["weights"]["applied"]["rss"] == 0.0


# ---------------------------------------------------------------------------
# Cross-signal flag emission
# ---------------------------------------------------------------------------


class TestDashboardCrossSignalFlags:
    """Tests for cross-signal flag emission."""

    def test_bearish_sentiment_vs_bullish_bias_emits_flag(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=-0.2, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=-0.2, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("BTCUSDT", price_bias="bullish")

        assert "sentiment_bearish_vs_price_bias_bullish" in result["cross_signal_flags"]

    def test_bullish_sentiment_vs_bearish_bias_emits_flag(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.2, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.2, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("ETHUSDT", price_bias="bearish")

        assert "sentiment_bullish_vs_price_bias_bearish" in result["cross_signal_flags"]

    def test_neutral_bias_suppresses_flags(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=-0.2, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=-0.2, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("DOGEUSDT", price_bias="neutral")

        assert result["cross_signal_flags"] == []

    def test_matching_sentiment_and_bias_no_flag(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.2, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.2, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("BTCUSDT", price_bias="bullish")

        assert result["cross_signal_flags"] == []


# ---------------------------------------------------------------------------
# Dashboard envelope shape
# ---------------------------------------------------------------------------


class TestDashboardEnvelopeShape:
    """Tests that the dashboard returns all required envelope fields."""

    def test_required_fields_present(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.1, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.1, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("BTCUSDT", price_bias="neutral")

        required_keys = {
            "symbol",
            "fetched_at",
            "price_bias",
            "sentiment_index",
            "sentiment_polarity",
            "combination_mode",
            "diagnostics",
            "cross_signal_flags",
        }
        assert required_keys <= result.keys()

    def test_diagnostics_contains_required_keys(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.1, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.1, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("ETHUSDT", price_bias="bullish")

        diag = result["diagnostics"]
        assert "sources" in diag
        assert "reddit" in diag["sources"]
        assert "rss" in diag["sources"]
        assert "weights" in diag
        assert "configured" in diag["weights"]
        assert "applied" in diag["weights"]
        assert "fallback_reason" in diag

    def test_source_diagnostic_fields(self):
        mock_reddit = MagicMock()
        mock_reddit.analyze.return_value = make_reddit_ok(avg_sentiment=0.1, total_posts=5)

        mock_rss = MagicMock()
        mock_rss.analyze.return_value = make_rss_ok(avg_sentiment=0.1, total_items=5)

        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        result = dashboard.analyze("DOGEUSDT", price_bias="neutral")

        reddit_diag = result["diagnostics"]["sources"]["reddit"]
        rss_diag = result["diagnostics"]["sources"]["rss"]

        for diag in (reddit_diag, rss_diag):
            assert "status" in diag
            assert "usable" in diag
            assert "avg_sentiment" in diag
            assert "sample_count" in diag


# ---------------------------------------------------------------------------
# Package export stability
# ---------------------------------------------------------------------------


class TestPackageExportStability:
    """Verify CombinedSentimentDashboard is exported from the sentiment package."""

    def test_can_import_from_sentiment_package(self):
        from tempest_mcp.sentiment import CombinedSentimentDashboard as Imported

        assert Imported is CombinedSentimentDashboard

    def test_can_instantiate(self):
        dashboard = CombinedSentimentDashboard()
        assert hasattr(dashboard, "analyze")

    def test_injects_analyzers(self):
        mock_reddit = MagicMock()
        mock_rss = MagicMock()
        dashboard = CombinedSentimentDashboard(
            reddit_analyzer=mock_reddit,
            rss_analyzer=mock_rss,
        )
        assert dashboard._reddit is mock_reddit
        assert dashboard._rss is mock_rss
