"""Phase 4 Kurisu-integration live suite (ENG-43).

This module exercises representative Kurisu-facing flows against the repo's
current public MCP surfaces and validates the resulting envelopes are
compatible with the existing DiscordFormatter contract.

Scope (ENG-43):
- Single repo-local integration module: tests/test_kurisu_integration.py
- Representative coverage: fetch_ticker, screener_scan,
  get_combined_sentiment_dashboard, backtest_rsi, detect_elliot_wave
- Envelope-first assertions: validate MCP envelope shape before formatter checks
- DiscordFormatter.format(...) compatibility for every representative call
- Bounded retry on transient upstream CCXT instability → skip with loud warning
- fetch_ticker stays contract/formatter smoke only (currently stubbed)

Run with:  uv run pytest --run-integration tests/test_kurisu_integration.py -v
"""

from __future__ import annotations

import asyncio
import warnings

import ccxt
import pytest

from tempest_mcp.formatters.discord import DiscordFormatter
from tempest_mcp.tools.analytical_tools import detect_elliot_wave
from tempest_mcp.tools.backtest_tools import BACKTEST_TOOLS
from tempest_mcp.tools.market_tools import fetch_ticker
from tempest_mcp.tools.screener_tools import screener_scan
from tempest_mcp.tools.sentiment_tools import get_combined_sentiment_dashboard

# =============================================================================
# Constants
# =============================================================================

CANONICAL_SYMBOL = "BTCUSDT"
MAX_RETRIES = 3
TRANSIENT_FETCH_EXCEPTIONS = (ccxt.NetworkError, ConnectionError, TimeoutError)
TRANSIENT_FAILURE_PATTERNS = {
    "screener_scan": (
        "unable to complete screener scan",
        "all symbols failed to scan",
    ),
    "backtest_rsi": ("data fetch failed",),
    "detect_elliot_wave": ("data fetch failed",),
}

# =============================================================================
# Helpers — async runner
# =============================================================================


def _run_async(coro):
    """Run an async coroutine synchronously (for use in sync test methods)."""
    return asyncio.run(coro)


# =============================================================================
# Helpers — envelope assertions (canonical contract per Shuna's design)
# =============================================================================


def assert_envelope_shape(result: dict) -> bool:
    """Assert result matches the top-level MCP envelope contract.

    Returns the success flag so callers can branch only after shape validation.
    """
    assert isinstance(result, dict), (
        f"Envelope must be a dict, got {type(result).__name__}"
    )
    assert "success" in result, "Envelope must have 'success' key"
    success = result["success"]
    assert isinstance(success, bool), "Envelope 'success' must be a bool"

    if success:
        assert "data" in result, "Success envelope must have 'data' key"
        assert isinstance(result["data"], dict), "data must be a dict"
    else:
        assert "error" in result, "Failure envelope must have 'error' key"
        error = result["error"]
        assert isinstance(error, dict), "error must be a dict"
        assert "code" in error, "error must have 'code'"
        assert "message" in error, "error must have 'message'"

    return success


def assert_success_envelope(result: dict, tool_name: str | None = None) -> dict:
    """Assert result is a deterministic success MCP envelope.

    Hard-fails on any contract violation.
    Returns the data dict for further tool-specific assertions.
    """
    assert assert_envelope_shape(result) is True, (
        f"Expected success=True envelope, got: {result.get('error')}"
    )
    data = result["data"]

    if tool_name:
        assert data.get("tool") == tool_name, (
            f"Expected tool={tool_name!r}, got {data.get('tool')!r}"
        )

    return data


def assert_failure_envelope(result: dict) -> None:
    """Assert result is a deterministic failure MCP envelope.

    Hard-fails on any contract violation.
    """
    assert assert_envelope_shape(result) is False, (
        "Expected success=False envelope, got success=True"
    )


def assert_discord_embed_shape(embed: dict) -> None:
    """Assert Discord embed dict has stable top-level structure.

    Hard-fails on any contract violation.
    """
    assert isinstance(embed, dict), (
        f"Embed must be a dict, got {type(embed).__name__}"
    )
    assert "title" in embed, "Embed must have 'title' key"
    assert "color" in embed, "Embed must have 'color' key"
    assert "fields" in embed, "Embed must have 'fields' key"
    assert isinstance(embed["title"], str), "title must be a string"
    assert embed["title"], "title must not be empty"
    assert isinstance(embed["color"], int), "color must be an int"
    assert isinstance(embed["fields"], list), "fields must be a list"


def assert_discord_embed_safe(result: dict) -> dict:
    """Assert envelope is formatter-safe and return the embed.

    Envelope-first: validates the MCP envelope shape before passing to formatter.
    Hard-fails on contract violations; returns the embed for additional checks.
    """
    formatter = DiscordFormatter()
    embed = formatter.format(result)
    assert_discord_embed_shape(embed)
    return embed


def _describe_failure(result: dict) -> str:
    """Build a concise diagnostic string for a failure envelope."""
    error = result["error"]
    return f"code={error['code']} message={error['message']}"


def _is_transient_failure_envelope(result: dict, tool_name: str) -> bool:
    """Return True when a failure envelope matches known transient upstream patterns."""
    if assert_envelope_shape(result):
        return False

    error = result["error"]
    message = str(error.get("message", "")).lower()
    patterns = TRANSIENT_FAILURE_PATTERNS.get(tool_name, ())

    if any(pattern in message for pattern in patterns):
        return True

    return False


def _run_with_bounded_retry(tool_name: str, operation):
    """Run a live tool with bounded retry on transient exceptions/envelopes."""
    last_transient: str | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = operation()
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, TRANSIENT_FETCH_EXCEPTIONS):
                raise
            last_transient = f"{type(exc).__name__}: {exc}"
        else:
            if _is_transient_failure_envelope(result, tool_name):
                last_transient = _describe_failure(result)
            else:
                return result

        if attempt < MAX_RETRIES:
            warnings.warn(
                f"[{tool_name}] Attempt {attempt}/{MAX_RETRIES} hit transient upstream failure: "
                f"{last_transient or 'unknown'}. Retrying...",
                stacklevel=2,
            )

    pytest.skip(
        f"[{tool_name}] Bounded retries ({MAX_RETRIES}) exhausted. "
        f"Skipping due to transient upstream failure: {last_transient or 'unknown'}"
    )
    raise AssertionError("unreachable: pytest.skip should raise Skipped")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def network_available():
    """Check if network is available."""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


# =============================================================================
# Tests — fetch_ticker (contract/formatter smoke only — currently stubbed)
# =============================================================================


@pytest.mark.integration
class TestFetchTickerKurisuIntegration:
    """Kurisu-integration tests for fetch_ticker.

    fetch_ticker is currently a stubbed public surface, so coverage is
    contract/formatter smoke only — no live-price assertions.
    """

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_fetch_ticker_envelope_and_formatter(self):
        """fetch_ticker returns valid stub envelope and is DiscordFormatter-safe."""
        result = _run_async(fetch_ticker(symbol="BTCUSDT", exchange="binance"))

        # Envelope assertions
        data = assert_success_envelope(result, tool_name="fetch_ticker")
        assert data.get("stub") is True, "fetch_ticker stub must have stub=True"

        # Formatter compatibility — must not crash, must return stable embed shape
        assert_discord_embed_safe(result)


# =============================================================================
# Tests — screener_scan (live CCXT data)
# =============================================================================


@pytest.mark.integration
class TestScreenerScanKurisuIntegration:
    """Kurisu-integration tests for screener_scan using live CCXT data."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def _screener_scan_with_retry(self, symbols: list[str] | None, **kwargs):
        """Invoke screener_scan with bounded retry for transient upstream failures."""
        return _run_with_bounded_retry(
            "screener_scan",
            lambda: _run_async(screener_scan(symbols=symbols, **kwargs)),
        )

    def test_screener_scan_success_envelope_and_formatter(self):
        """screener_scan returns valid success envelope and is DiscordFormatter-safe."""
        result = self._screener_scan_with_retry(
            symbols=["BTCUSDT", "ETHUSDT"],
            min_score=0.0,
        )

        # Envelope assertions — success path
        data = assert_success_envelope(result, tool_name="screener_scan")
        assert "exchange" in data
        assert "results" in data
        assert "failures" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["failures"], list)

        # Formatter compatibility
        assert_discord_embed_safe(result)

    def test_screener_scan_high_threshold_envelope_and_formatter(self):
        """screener_scan with a high threshold still returns a formatter-safe envelope."""
        # Use a high threshold to exercise a non-happy-path live scan without
        # over-asserting partial-result behavior that live market data may not guarantee.
        result = self._screener_scan_with_retry(
            symbols=["BTCUSDT"],
            min_score=95.0,
        )

        # Partial or full failure is acceptable — just needs deterministic envelope
        if assert_envelope_shape(result):
            data = assert_success_envelope(result, tool_name="screener_scan")
            assert isinstance(data.get("results"), list)
            assert isinstance(data.get("failures"), list)
        else:
            assert_failure_envelope(result)

        # Formatter must be safe regardless of partial/full failure
        assert_discord_embed_safe(result)


# =============================================================================
# Tests — get_combined_sentiment_dashboard (live data with diagnostics on failure)
# =============================================================================


@pytest.mark.integration
class TestSentimentDashboardKurisuIntegration:
    """Kurisu-integration tests for get_combined_sentiment_dashboard.

    Accepts either success envelope OR deterministic unavailable/error envelope.
    When unavailable/failure is returned, asserts diagnostics remain present
    and formatter-compatible.
    """

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def _sentiment_with_retry(self, symbol: str, price_bias: str, **kwargs):
        """Invoke sentiment dashboard with bounded retry for transient upstream failures."""
        return _run_with_bounded_retry(
            "sentiment",
            lambda: _run_async(
                get_combined_sentiment_dashboard(
                    symbol=symbol, price_bias=price_bias, **kwargs
                )
            ),
        )

    def test_sentiment_success_envelope_and_formatter(self):
        """sentiment dashboard returns success envelope and is DiscordFormatter-safe."""
        result = self._sentiment_with_retry(symbol="BTCUSDT", price_bias="bullish")

        # Accept success envelope
        if assert_envelope_shape(result):
            data = assert_success_envelope(result, tool_name="get_combined_sentiment_dashboard")
            assert "sentiment_polarity" in data
            assert "sentiment_index" in data
            assert "combination_mode" in data
        else:
            # Failure envelope is also acceptable — check diagnostics are present
            assert_failure_envelope(result)
            # Diagnostics should be in data even on failure
            assert "data" in result, "Failure envelope should still carry data with diagnostics"
            data = result["data"]
            assert isinstance(data, dict), "Diagnostics data must be a dict"
            assert "diagnostics" in data or "combination_mode" in data, (
                "Expected diagnostics or combination_mode in data"
            )

        # Formatter compatibility — must not crash, must return stable embed
        assert_discord_embed_safe(result)

    def test_sentiment_failure_diagnostics_and_formatter(self):
        """sentiment dashboard failure carries diagnostics and remains formatter-safe."""
        # Use a clearly invalid/edge-case symbol to trigger failure path
        result = self._sentiment_with_retry(symbol="INVALIDCOINXYZ123", price_bias="neutral")

        # Either success with unavailable mode OR deterministic failure is acceptable
        if assert_envelope_shape(result):
            data = assert_success_envelope(
                result,
                tool_name="get_combined_sentiment_dashboard",
            )
            # combination_mode="unavailable" is the documented failure path
            if data.get("combination_mode") == "unavailable":
                assert "diagnostics" in data, (
                    "Unavailable sentiment must carry diagnostics"
                )
        else:
            # Full failure envelope
            assert_failure_envelope(result)
            assert "data" in result, "Failure envelope should carry data with diagnostics"
            data = result["data"]
            assert isinstance(data, dict), "Diagnostics data must be a dict"

        # Formatter must be safe on both success(unavailable) and failure paths
        assert_discord_embed_safe(result)


# =============================================================================
# Tests — backtest_rsi (live CCXT data)
# =============================================================================


@pytest.mark.integration
class TestBacktestRsiKurisuIntegration:
    """Kurisu-integration tests for backtest_rsi using live CCXT data."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def _backtest_rsi_with_retry(self, **kwargs):
        """Invoke backtest_rsi with bounded retry for transient upstream failures."""
        # Get the handler from BACKTEST_TOOLS registry
        handler = BACKTEST_TOOLS.get("backtest_rsi")
        assert handler is not None, "backtest_rsi handler not found in BACKTEST_TOOLS"

        return _run_with_bounded_retry(
            "backtest_rsi",
            lambda: _run_async(handler(**kwargs)),
        )

    def test_backtest_rsi_success_envelope_and_formatter(self):
        """backtest_rsi returns valid success envelope and is DiscordFormatter-safe."""
        # Use recent window for stability
        result = self._backtest_rsi_with_retry(
            symbol="BTCUSDT",
            trade_style="day_trade",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-07T00:00:00",
            initial_capital=100000.0,
        )

        # Envelope assertions — success path
        if assert_envelope_shape(result):
            data = assert_success_envelope(result, tool_name="backtest_rsi")
            assert "strategy_id" in data
            assert "window" in data
            assert "metrics" in data
            assert "trade_count" in data
            assert "initial_capital" in data
            assert "final_equity" in data
        else:
            # Validation/infrastructure failures are acceptable with deterministic error
            assert_failure_envelope(result)

        # Formatter compatibility
        assert_discord_embed_safe(result)

    def test_backtest_rsi_failure_envelope_and_formatter(self):
        """backtest_rsi with invalid params returns deterministic error and formatter-safe."""
        result = _run_async(
            BACKTEST_TOOLS["backtest_rsi"](
                symbol="INVALIDCOINXYZ123",  # Invalid symbol to trigger error
                trade_style="day_trade",
                timeframe="1h",
                start_at="2024-01-01T00:00:00",
                end_at="2024-01-07T00:00:00",
            )
        )

        # Either validation error (failure) or success with insufficient data is acceptable
        if assert_envelope_shape(result):
            # Success is fine if data is insufficient — still valid envelope
            data = assert_success_envelope(result, tool_name="backtest_rsi")
            assert isinstance(data, dict)
        else:
            assert_failure_envelope(result)

        # Formatter must be safe on both paths
        assert_discord_embed_safe(result)


# =============================================================================
# Tests — detect_elliot_wave (live CCXT data)
# =============================================================================


@pytest.mark.integration
class TestDetectElliotWaveKurisuIntegration:
    """Kurisu-integration tests for detect_elliot_wave using live CCXT data."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def _elliot_wave_with_retry(self, **kwargs):
        """Invoke detect_elliot_wave with bounded retry for transient upstream failures."""
        return _run_with_bounded_retry(
            "detect_elliot_wave",
            lambda: _run_async(detect_elliot_wave(**kwargs)),
        )

    def test_elliot_wave_success_envelope_and_formatter(self):
        """detect_elliot_wave returns valid success envelope and is DiscordFormatter-safe."""
        result = self._elliot_wave_with_retry(
            symbol="BTCUSDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-14T00:00:00",  # ~14 days for wave detection
        )

        # Envelope assertions — success path
        if assert_envelope_shape(result):
            data = assert_success_envelope(result, tool_name="detect_elliot_wave")
            assert "window" in data
            assert "parameters" in data
            assert "wave_sequences" in data
            assert "count" in data
        else:
            # Validation/infrastructure failures are acceptable with deterministic error
            assert_failure_envelope(result)

        # Formatter compatibility
        assert_discord_embed_safe(result)

    def test_elliot_wave_4h_timeframe_and_formatter(self):
        """detect_elliot_wave on 4h timeframe returns valid envelope and formatter-safe."""
        result = self._elliot_wave_with_retry(
            symbol="BTCUSDT",
            timeframe="4h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-02-01T00:00:00",  # ~1 month on 4h
        )

        # Envelope assertions
        if assert_envelope_shape(result):
            data = assert_success_envelope(result, tool_name="detect_elliot_wave")
            assert data["timeframe"] == "4h"
            assert isinstance(data["wave_sequences"], list)
        else:
            assert_failure_envelope(result)

        # Formatter compatibility
        assert_discord_embed_safe(result)
