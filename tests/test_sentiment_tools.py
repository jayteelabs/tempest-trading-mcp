"""Unit tests for sentiment_tools MCP handler envelope — ENG-41."""

import sys

sys.path.insert(0, "src")

import tempest_mcp.tools.sentiment_tools as st

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class MockDashboard:
    """Fake dashboard that returns configurable results."""

    def __init__(self, result: dict | Exception):
        self._result = result

    def analyze(self, symbol: str, price_bias: str) -> dict:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


def make_success_result(combination_mode: str = "weighted", sentiment_index: float = 0.25) -> dict:
    """Build a minimal success-style dashboard result."""
    return {
        "symbol": "BTCUSDT",
        "fetched_at": "2026-04-21T00:00:00+00:00",
        "price_bias": "bullish",
        "sentiment_index": sentiment_index,
        "sentiment_polarity": "bullish",
        "combination_mode": combination_mode,
        "diagnostics": {
            "sources": {
                "reddit": {
                    "status": "ok",
                    "usable": True,
                    "avg_sentiment": 0.15,
                    "sample_count": 10,
                },
                "rss": {
                    "status": "ok",
                    "usable": True,
                    "avg_sentiment": 0.30,
                    "sample_count": 8,
                },
            },
            "weights": {
                "configured": {"reddit": 0.4, "rss": 0.6},
                "applied": {"reddit": 0.4, "rss": 0.6},
            },
            "fallback_reason": None,
        },
        "cross_signal_flags": [],
    }


# ---------------------------------------------------------------------------
# Success envelope tests
# ---------------------------------------------------------------------------


class TestSentimentToolSuccessEnvelope:
    """Tests for the success path MCP envelope."""

    async def test_success_envelope_structure(self):
        """Success path returns success:true with full data payload."""
        fake_result = make_success_result()
        st._dashboard = MockDashboard(fake_result)

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="BTCUSDT",
                price_bias="bullish",
            )
        finally:
            st._dashboard = None

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["tool"] == "get_combined_sentiment_dashboard"
        assert result["data"]["symbol"] == "BTCUSDT"
        assert result["data"]["price_bias"] == "bullish"
        assert "sentiment_index" in result["data"]
        assert "sentiment_polarity" in result["data"]
        assert "combination_mode" in result["data"]
        assert "diagnostics" in result["data"]
        assert "cross_signal_flags" in result["data"]

    async def test_success_envelope_contains_no_error_key(self):
        """Success path does NOT include an 'error' key."""
        fake_result = make_success_result()
        st._dashboard = MockDashboard(fake_result)

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="ETHUSDT",
                price_bias="neutral",
            )
        finally:
            st._dashboard = None

        assert "error" not in result


# ---------------------------------------------------------------------------
# Unavailable / failure envelope tests
# ---------------------------------------------------------------------------


class TestSentimentToolFailureEnvelope:
    """Tests for the failure (unavailable) path MCP envelope."""

    async def test_unavailable_returns_success_false(self):
        """When combination_mode is unavailable, success must be False."""
        fake_result = make_success_result(combination_mode="unavailable", sentiment_index=None)
        st._dashboard = MockDashboard(fake_result)

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="BTCUSDT",
                price_bias="bullish",
            )
        finally:
            st._dashboard = None

        assert result["success"] is False

    async def test_unavailable_includes_error_code_and_message(self):
        """Failure envelope includes error.code and error.message."""
        fake_result = make_success_result(combination_mode="unavailable", sentiment_index=None)
        st._dashboard = MockDashboard(fake_result)

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="BTCUSDT",
                price_bias="bullish",
            )
        finally:
            st._dashboard = None

        assert "error" in result
        assert "code" in result["error"]
        assert "message" in result["error"]
        assert result["error"]["code"] == 3000  # DATA_SOURCE_ERROR

    async def test_unavailable_includes_diagnostics_payload(self):
        """Per ENG-41 contract: failure includes data.diagnostics alongside error."""
        fake_result = make_success_result(combination_mode="unavailable", sentiment_index=None)
        st._dashboard = MockDashboard(fake_result)

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="BTCUSDT",
                price_bias="bullish",
            )
        finally:
            st._dashboard = None

        # Failure envelope must include both error AND data.diagnostics
        assert "data" in result
        assert "diagnostics" in result["data"]
        assert "sources" in result["data"]["diagnostics"]


# ---------------------------------------------------------------------------
# Internal exception path tests
# ---------------------------------------------------------------------------


class TestSentimentToolExceptionEnvelope:
    """Tests for the internal error path MCP envelope."""

    async def test_internal_exception_returns_sanitized_envelope(self):
        """Raw exceptions must be sanitized; never leak to client."""
        st._dashboard = MockDashboard(RuntimeError("network timeout"))

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="BTCUSDT",
                price_bias="neutral",
            )
        finally:
            st._dashboard = None

        assert result["success"] is False
        assert "error" in result
        # Must be sanitized error code, not the raw exception
        assert result["error"]["code"] == 9000  # INTERNAL_ERROR
        assert result["error"]["message"] == "An internal error occurred"
        # No raw exception leaked
        assert "network timeout" not in str(result)

    async def test_exception_path_includes_diagnostics_payload(self):
        """Internal error still returns diagnostics payload for debuggability."""
        st._dashboard = MockDashboard(RuntimeError("surprise"))

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="ETHUSDT",
                price_bias="bearish",
            )
        finally:
            st._dashboard = None

        assert "data" in result
        assert "diagnostics" in result["data"]
        assert result["data"]["diagnostics"]["sources"]["reddit"]["status"] == "error"
        assert result["data"]["diagnostics"]["sources"]["rss"]["status"] == "error"

    async def test_exception_path_preserves_symbol_and_bias(self):
        """Error path envelope preserves input symbol and price_bias."""
        st._dashboard = MockDashboard(ValueError("bad input"))

        try:
            result = await st.get_combined_sentiment_dashboard(
                symbol="DOGEUSDT",
                price_bias="neutral",
            )
        finally:
            st._dashboard = None

        assert result["data"]["symbol"] == "DOGEUSDT"
        assert result["data"]["price_bias"] == "neutral"


# ---------------------------------------------------------------------------
# Tool registration smoke tests
# ---------------------------------------------------------------------------


class TestSentimentToolCallable:
    """Smoke tests verifying the tool handler is properly exported."""

    def test_handler_is_importable(self):
        assert callable(st.get_combined_sentiment_dashboard)

    def test_handler_is_awaitable(self):
        import asyncio

        coro = st.get_combined_sentiment_dashboard(symbol="BTCUSDT", price_bias="neutral")
        assert asyncio.iscoroutine(coro)
        # Clean up the created dashboard
        st._dashboard = None
