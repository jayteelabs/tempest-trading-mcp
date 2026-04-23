"""Unit tests for indicator tools (ENG-126).

Tests:
- indicator_rsi: validation, success, failure
- exchange-aware adapter selection
- limit semantics
- zone detection
- formatter compatibility
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tempest_mcp.indicators.momentum.rsi import OVERBOUGHT_THRESHOLD, OVERSOLD_THRESHOLD
from tempest_mcp.tools.indicator_tools import (
    _detect_zone,
    indicator_rsi,
)

# =============================================================================
# Helpers
# =============================================================================


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def assert_success_envelope(result: dict, tool_name: str | None = None) -> dict:
    """Assert result is a deterministic success MCP envelope."""
    assert isinstance(result, dict), f"Envelope must be a dict, got {type(result).__name__}"
    assert "success" in result, "Envelope must have 'success' key"
    assert result["success"] is True, f"Expected success=True, got: {result.get('error')}"
    assert "data" in result, "Success envelope must have 'data' key"
    data = result["data"]
    if tool_name:
        assert data.get("tool") == tool_name, f"Expected tool={tool_name!r}, got {data.get('tool')!r}"
    return data


def assert_failure_envelope(result: dict) -> None:
    """Assert result is a deterministic failure MCP envelope."""
    assert isinstance(result, dict), f"Envelope must be a dict, got {type(result).__name__}"
    assert "success" in result, "Envelope must have 'success' key"
    assert result["success"] is False, "Expected success=False, got success=True"
    assert "error" in result, "Failure envelope must have 'error' key"
    error = result["error"]
    assert isinstance(error, dict), "error must be a dict"
    assert "code" in error, "error must have 'code'"
    assert "message" in error, "error must have 'message'"


# =============================================================================
# Zone Detection
# =============================================================================


class TestDetectZone:
    """Tests for _detect_zone helper."""

    def test_oversold_zone(self):
        """RSI <= 30 returns 'oversold'."""
        assert _detect_zone(30.0) == "oversold"
        assert _detect_zone(20.0) == "oversold"
        assert _detect_zone(0.0) == "oversold"

    def test_overbought_zone(self):
        """RSI >= 70 returns 'overbought'."""
        assert _detect_zone(70.0) == "overbought"
        assert _detect_zone(85.0) == "overbought"
        assert _detect_zone(100.0) == "overbought"

    def test_neutral_zone(self):
        """30 < RSI < 70 returns 'neutral'."""
        assert _detect_zone(31.0) == "neutral"
        assert _detect_zone(50.0) == "neutral"
        assert _detect_zone(69.0) == "neutral"

    def test_threshold_boundaries(self):
        """Boundary values at exactly 30 and 70."""
        assert _detect_zone(OVERSOLD_THRESHOLD) == "oversold"
        assert _detect_zone(OVERBOUGHT_THRESHOLD) == "overbought"


# =============================================================================
# Tests — indicator_rsi Validation
# =============================================================================


class TestIndicatorRsiValidation:
    """Validation tests for indicator_rsi."""

    def test_invalid_symbol_format(self):
        """Invalid symbol format returns code 1001."""
        result = _run_async(indicator_rsi(symbol="INVALID@#$", exchange="binance"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1001
        assert "symbol" in result["error"]["message"].lower()

    def test_empty_symbol(self):
        """Empty symbol returns code 1001."""
        result = _run_async(indicator_rsi(symbol="", exchange="binance"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1001

    def test_invalid_exchange(self):
        """Invalid exchange returns code 1002."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", exchange="invalid_exchange"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1002
        assert "exchange" in result["error"]["message"].lower()

    def test_invalid_timeframe(self):
        """Invalid timeframe returns code 1003."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", timeframe="invalid"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1003
        assert "timeframe" in result["error"]["message"].lower()

    def test_unsupported_timeframe(self):
        """Unsupported but valid-looking timeframe returns code 1003."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", timeframe="2h"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1003

    def test_invalid_period_type(self):
        """Non-integer period returns code 1004."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", period="14"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_period_too_small(self):
        """period < 2 returns code 1004."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", period=1))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_period_zero(self):
        """period = 0 returns code 1004."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", period=0))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_period_negative(self):
        """Negative period returns code 1004."""
        result = _run_async(indicator_rsi(symbol="BTCUSDT", period=-5))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    @pytest.mark.parametrize(
        ("symbol", "expected_symbol"),
        [
            ("BTC/USDT", "BTC/USDT"),
            ("BTC-USDT", "BTC/USDT"),
            ("btcusdt", "BTC/USDT"),
            ("BTCUSDT", "BTC/USDT"),
            ("ETHUSDT", "ETH/USDT"),
            ("ethusdt", "ETH/USDT"),
        ],
    )
    def test_symbol_normalization(self, symbol, expected_symbol):
        """Symbol normalization produces canonical CCXT format."""
        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            # Mock adapter with valid data
            mock_adapter = MagicMock()
            dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
            mock_adapter.fetch_ohlcv.return_value = (
                pd.DataFrame(
                    {
                        "open": [100.0] * 50,
                        "high": [105.0] * 50,
                        "low": [95.0] * 50,
                        "close": [100.0 + i for i in range(50)],
                        "volume": [1000.0] * 50,
                    },
                    index=dates,
                ),
                "ccxt",
            )
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol=symbol, period=14, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")
            assert data["symbol"] == expected_symbol


# =============================================================================
# Tests — indicator_rsi Success
# =============================================================================


class TestIndicatorRsiSuccess:
    """Success path tests for indicator_rsi."""

    @pytest.fixture
    def mock_adapter_with_data(self):
        """Create a mock adapter that returns valid OHLCV data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.5 for i in range(100)]
        return (
            pd.DataFrame(
                {
                    "open": close_prices,
                    "high": [p + 2 for p in close_prices],
                    "low": [p - 2 for p in close_prices],
                    "close": close_prices,
                    "volume": [1000.0] * 100,
                },
                index=dates,
            ),
            "ccxt",
        )

    def test_happy_path(self, mock_adapter_with_data):
        """Basic successful RSI calculation."""
        ohlcv_data, source = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, source)
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            assert data["tool"] == "indicator_rsi"
            assert data["symbol"] == "BTC/USDT"
            assert data["exchange"] == "binance"
            assert data["period"] == 14
            assert data["source_used"] == "ccxt"
            assert isinstance(data["values"], list)
            assert len(data["values"]) > 0
            assert "latest" in data
            assert "timestamp" in data["latest"]
            assert "rsi" in data["latest"]
            assert "zone" in data["latest"]
            assert data["latest"]["zone"] in ("oversold", "neutral", "overbought")

    def test_limit_semantics(self, mock_adapter_with_data):
        """limit parameter controls returned row count."""
        ohlcv_data, source = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, source)
            mock_get_adapter.return_value = mock_adapter

            # Request only 5 rows
            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, limit=5, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            assert len(data["values"]) == 5
            assert data["limit"] == 5

    def test_limit_clamp_to_1000(self):
        """limit > 1000 is clamped to 1000."""
        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            dates = pd.date_range("2024-01-01", periods=2000, freq="h", tz="UTC")
            close_prices = [100.0 + i * 0.1 for i in range(2000)]
            mock_adapter.fetch_ohlcv.return_value = (
                pd.DataFrame(
                    {
                        "open": close_prices,
                        "high": [p + 1 for p in close_prices],
                        "low": [p - 1 for p in close_prices],
                        "close": close_prices,
                        "volume": [1000.0] * 2000,
                    },
                    index=dates,
                ),
                "ccxt",
            )
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, limit=5000, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            # limit should be clamped
            assert data["limit"] == 1000
            # values should be capped at available RSI rows, not exceed 1000
            assert len(data["values"]) <= 1000

    def test_yfinance_source_used(self, mock_adapter_with_data):
        """source_used reflects actual data source."""
        ohlcv_data, _ = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "yfinance")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            assert data["source_used"] == "yfinance"

    def test_zone_detection_oversold(self, mock_adapter_with_data):
        """Latest RSI in oversold zone returns zone='oversold'."""
        ohlcv_data, _ = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            # Create data that will produce RSI <= 30
            dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
            # Declining prices that will produce low RSI
            close_prices = [100.0 - i * 1.5 for i in range(100)]
            ohlcv_data = pd.DataFrame(
                {
                    "open": close_prices,
                    "high": [p + 0.5 for p in close_prices],
                    "low": [p - 0.5 for p in close_prices],
                    "close": close_prices,
                    "volume": [1000.0] * 100,
                },
                index=dates,
            )
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            # Latest zone should be oversold given the declining prices
            assert data["latest"]["zone"] == "oversold"

    def test_zone_detection_overbought(self, mock_adapter_with_data):
        """Latest RSI in overbought zone returns zone='overbought'."""
        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            # Create data that will produce RSI >= 70
            dates = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
            # Rising prices that will produce high RSI
            close_prices = [100.0 + i * 1.5 for i in range(100)]
            ohlcv_data = pd.DataFrame(
                {
                    "open": close_prices,
                    "high": [p + 0.5 for p in close_prices],
                    "low": [p - 0.5 for p in close_prices],
                    "close": close_prices,
                    "volume": [1000.0] * 100,
                },
                index=dates,
            )
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            # Latest zone should be overbought given the rising prices
            assert data["latest"]["zone"] == "overbought"

    def test_canonical_symbol_format(self, mock_adapter_with_data):
        """Response uses canonical CCXT symbol format BTC/USDT."""
        ohlcv_data, _ = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="btcusdt", exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            # Symbol should be in canonical BTC/USDT format
            assert data["symbol"] == "BTC/USDT"
            assert "/" in data["symbol"]

    def test_values_ordered_oldest_to_newest(self, mock_adapter_with_data):
        """values list is ordered oldest → newest."""
        ohlcv_data, _ = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, limit=10, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            values = data["values"]
            assert len(values) == 10

            # Verify ordering by comparing timestamps
            timestamps = [v["timestamp"] for v in values]
            assert timestamps == sorted(timestamps), "values should be ordered oldest to newest"


# =============================================================================
# Tests — indicator_rsi Failure
# =============================================================================


class TestIndicatorRsiFailure:
    """Failure path tests for indicator_rsi."""

    def test_empty_ohlcv_from_adapter(self):
        """Empty DataFrame from adapter returns code 2001."""
        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (pd.DataFrame(), "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            assert_failure_envelope(result)
            assert result["error"]["code"] == 2001

    def test_insufficient_candles_for_rsi(self):
        """Fewer candles than period returns code 2001."""
        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            # Only 5 candles - not enough for period=14 RSI
            dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
            mock_adapter.fetch_ohlcv.return_value = (
                pd.DataFrame(
                    {
                        "open": [100.0] * 5,
                        "high": [105.0] * 5,
                        "low": [95.0] * 5,
                        "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                        "volume": [1000.0] * 5,
                    },
                    index=dates,
                ),
                "ccxt",
            )
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            assert_failure_envelope(result)
            assert result["error"]["code"] == 2001

    def test_adapter_exception(self):
        """Exception from adapter returns code 2001."""
        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.side_effect = RuntimeError("Network error")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            assert_failure_envelope(result)
            assert result["error"]["code"] == 2001


# =============================================================================
# Tests — Formatter Compatibility
# =============================================================================


class TestIndicatorRsiFormatterCompatibility:
    """Tests for formatter compatibility (ENG-126)."""

    def test_values_is_list_like(self, mock_adapter_with_data=None):
        """Formatter compatibility: values must be list-like."""
        if mock_adapter_with_data is None:
            dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
            close_prices = [100.0 + i * 0.5 for i in range(50)]
            mock_adapter_with_data = (
                pd.DataFrame(
                    {
                        "open": close_prices,
                        "high": [p + 2 for p in close_prices],
                        "low": [p - 2 for p in close_prices],
                        "close": close_prices,
                        "volume": [1000.0] * 50,
                    },
                    index=dates,
                ),
                "ccxt",
            )

        ohlcv_data, source = mock_adapter_with_data

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, source)
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            # DiscordFormatter.format_indicator accesses data.values as list-like
            values = data["values"]
            assert isinstance(values, list), "values must be a list for formatter compatibility"
            assert len(values) > 0, "values should not be empty for happy path"

            # Each value should have timestamp and rsi
            first_value = values[0]
            assert "timestamp" in first_value
            assert "rsi" in first_value
            assert isinstance(first_value["rsi"], (int, float))


# =============================================================================
# Tests — Exchange-Aware Behavior
# =============================================================================


class TestIndicatorRsiExchangeAware:
    """Tests for exchange-aware behavior."""

    @pytest.mark.parametrize("exchange", ["binance", "bybit", "coinbase", "kraken"])
    def test_supported_exchange(self, exchange):
        """Each supported exchange is accepted."""
        dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.5 for i in range(50)]
        ohlcv_data = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 2 for p in close_prices],
                "low": [p - 2 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange=exchange))
            data = assert_success_envelope(result, "indicator_rsi")

            assert data["exchange"] == exchange.lower()

    def test_exchange_case_insensitive(self):
        """Exchange name is case-insensitive."""
        dates = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.5 for i in range(50)]
        ohlcv_data = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 2 for p in close_prices],
                "low": [p - 2 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )

        with patch("tempest_mcp.tools.indicator_tools.get_historical_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ohlcv.return_value = (ohlcv_data, "ccxt")
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(indicator_rsi(symbol="BTCUSDT", period=14, exchange="Binance"))
            data = assert_success_envelope(result, "indicator_rsi")

            assert data["exchange"] == "binance"
