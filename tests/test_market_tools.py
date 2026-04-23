"""Unit tests for market tools (ENG-122).

Tests:
- fetch_ticker: validation, success, failure
- fetch_klines: validation, success, failure
- fetch_orderbook: validation, success, failure
- exchange-aware adapter selection
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tempest_mcp.tools.market_tools import (
    fetch_klines,
    fetch_orderbook,
    fetch_ticker,
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
# Tests — fetch_ticker
# =============================================================================


class TestFetchTickerValidation:
    """Validation tests for fetch_ticker."""

    def test_invalid_exchange(self):
        """Invalid exchange returns 1004."""
        result = _run_async(fetch_ticker(symbol="BTCUSDT", exchange="invalid_exchange"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004
        assert "exchange" in result["error"]["message"].lower()

    def test_invalid_symbol_format(self):
        """Invalid symbol format returns 1004."""
        result = _run_async(fetch_ticker(symbol="", exchange="binance"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    @pytest.mark.parametrize(
        ("symbol", "expected_symbol"),
        [("BTC/USDT", "BTC/USDT"), ("BTC-USDT", "BTC/USDT"), ("ETH-USDT", "ETH/USDT")],
    )
    def test_hyphen_and_slash_symbol_normalization(self, symbol, expected_symbol):
        """Hyphen and slash market-pair symbols normalize correctly (ENG-124)."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ticker_snapshot.return_value = {
                "price": 67234.5,
                "bid": 67234.4,
                "ask": 67234.6,
                "change_pct_24h": 1.23,
                "volume_24h": 12345.67,
                "timestamp": pd.Timestamp.now(tz="UTC"),
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_ticker(symbol=symbol, exchange="binance"))
            data = assert_success_envelope(result, "fetch_ticker")
            assert data["symbol"] == expected_symbol

    def test_yfinance_style_symbol_rejected(self):
        """Hyphenated USD aliases stay rejected on the exchange-backed path."""
        result = _run_async(fetch_ticker(symbol="BTC-USD", exchange="binance"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    @pytest.mark.parametrize("symbol", ["BTC--USDT", "BTC:USDT", "BTC\\USDT"])
    def test_malformed_symbols_rejected(self, symbol):
        """Malformed symbols return 1004 (ENG-124)."""
        result = _run_async(fetch_ticker(symbol=symbol, exchange="binance"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    @pytest.mark.parametrize("exchange", ["binance", "bybit", "coinbase", "kraken"])
    def test_valid_exchange(self, exchange):
        """Valid exchanges pass validation and return success."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ticker_snapshot.return_value = {
                "price": 67234.5,
                "bid": 67234.4,
                "ask": 67234.6,
                "change_pct_24h": 1.23,
                "volume_24h": 12345.67,
                "timestamp": pd.Timestamp.now(tz="UTC"),
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_ticker(symbol="BTCUSDT", exchange=exchange))
            # Should succeed with real adapter data
            data = assert_success_envelope(result, "fetch_ticker")
            assert data["exchange"] == exchange
            assert data["price"] == 67234.5


class TestFetchTickerSuccess:
    """Success path tests for fetch_ticker."""

    def test_fetch_ticker_success_with_all_fields(self):
        """fetch_ticker returns complete success envelope."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ticker_snapshot.return_value = {
                "price": 67234.5,
                "bid": 67234.4,
                "ask": 67234.6,
                "change_pct_24h": 1.23,
                "volume_24h": 12345.67,
                "timestamp": pd.Timestamp("2026-04-23T10:00:00+00:00"),
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_ticker(symbol="BTCUSDT", exchange="binance"))
            data = assert_success_envelope(result, "fetch_ticker")

            assert data["symbol"] == "BTC/USDT"
            assert data["exchange"] == "binance"
            assert data["price"] == 67234.5
            assert data["bid"] == 67234.4
            assert data["ask"] == 67234.6
            assert data["change_pct_24h"] == 1.23
            assert data["volume_24h"] == 12345.67
            assert data["timestamp"] == "2026-04-23T10:00:00+00:00"

    def test_fetch_ticker_success_with_nullable_fields(self):
        """fetch_ticker returns success with nullable fields as None."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ticker_snapshot.return_value = {
                "price": 67234.5,
                "bid": None,
                "ask": None,
                "change_pct_24h": None,
                "volume_24h": None,
                "timestamp": pd.Timestamp("2026-04-23T10:00:00+00:00"),
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_ticker(symbol="BTCUSDT", exchange="binance"))
            data = assert_success_envelope(result, "fetch_ticker")

            assert data["price"] == 67234.5
            assert data["bid"] is None
            assert data["ask"] is None
            assert data["change_pct_24h"] is None
            assert data["volume_24h"] is None


class TestFetchTickerFailure:
    """Failure path tests for fetch_ticker."""

    def test_fetch_ticker_data_source_error(self):
        """fetch_ticker returns 3000 on adapter failure."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_ticker_snapshot.return_value = {
                "price": float("nan"),
                "bid": None,
                "ask": None,
                "change_pct_24h": None,
                "volume_24h": None,
                "timestamp": None,
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_ticker(symbol="BTCUSDT", exchange="binance"))
            assert_failure_envelope(result)
            assert result["error"]["code"] == 3000


# =============================================================================
# Tests — fetch_klines
# =============================================================================


class TestFetchKlinesValidation:
    """Validation tests for fetch_klines."""

    def test_invalid_exchange(self):
        """Invalid exchange returns 1004."""
        result = _run_async(fetch_klines(symbol="BTCUSDT", exchange="invalid"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_timeframe(self):
        """Invalid timeframe returns 1004."""
        result = _run_async(fetch_klines(symbol="BTCUSDT", timeframe="invalid"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_limit_too_low(self):
        """limit < 1 returns 1004."""
        result = _run_async(fetch_klines(symbol="BTCUSDT", limit=0))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_limit_too_high(self):
        """limit > 1000 returns 1004."""
        result = _run_async(fetch_klines(symbol="BTCUSDT", limit=1001))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_source(self):
        """source != 'ccxt' returns 1004."""
        result = _run_async(fetch_klines(symbol="BTCUSDT", source="yfinance"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004
        assert "ccxt" in result["error"]["message"]

    def test_invalid_symbol(self):
        """Invalid symbol returns 1004."""
        result = _run_async(fetch_klines(symbol=""))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_since(self):
        """Malformed since returns 1004 instead of being ignored."""
        result = _run_async(fetch_klines(symbol="BTCUSDT", since="not-an-iso"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004
        assert "since" in result["error"]["message"].lower()


class TestFetchKlinesSuccess:
    """Success path tests for fetch_klines."""

    def test_fetch_klines_success(self):
        """fetch_klines returns complete success envelope."""
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.5, 102.5],
                "volume": [1000.0, 1100.0],
            },
            index=pd.to_datetime(["2026-04-23T10:00:00+00:00", "2026-04-23T11:00:00+00:00"], utc=True),
        )

        with patch("tempest_mcp.tools.market_tools.get_historical_adapter") as mock_get_hist:
            mock_hist = MagicMock()
            mock_hist.fetch_ohlcv.return_value = (df, "ccxt")
            mock_get_hist.return_value = mock_hist

            result = _run_async(fetch_klines(symbol="BTCUSDT", exchange="binance"))
            data = assert_success_envelope(result, "fetch_klines")

            assert data["symbol"] == "BTC/USDT"
            assert data["exchange"] == "binance"
            assert data["timeframe"] == "1h"
            assert data["limit"] == 100
            assert data["source"] == "ccxt"
            assert data["source_used"] == "ccxt"
            assert len(data["rows"]) == 2
            assert data["rows"][0]["timestamp"] == "2026-04-23T10:00:00+00:00"
            assert data["rows"][0]["open"] == 100.0


# =============================================================================
# Tests — fetch_orderbook
# =============================================================================


class TestFetchOrderbookValidation:
    """Validation tests for fetch_orderbook."""

    def test_invalid_exchange(self):
        """Invalid exchange returns 1004."""
        result = _run_async(fetch_orderbook(symbol="BTCUSDT", exchange="invalid"))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_limit_too_low(self):
        """limit < 1 returns 1004."""
        result = _run_async(fetch_orderbook(symbol="BTCUSDT", limit=0))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_limit_too_high(self):
        """limit > 100 returns 1004."""
        result = _run_async(fetch_orderbook(symbol="BTCUSDT", limit=101))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004

    def test_invalid_symbol(self):
        """Invalid symbol returns 1004."""
        result = _run_async(fetch_orderbook(symbol=""))
        assert_failure_envelope(result)
        assert result["error"]["code"] == 1004


class TestFetchOrderbookSuccess:
    """Success path tests for fetch_orderbook."""

    def test_fetch_orderbook_success(self):
        """fetch_orderbook returns complete success envelope."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_orderbook_snapshot.return_value = {
                "bids": [[67234.4, 1.2], [67234.3, 0.8]],
                "asks": [[67234.6, 0.8], [67234.7, 0.5]],
                "timestamp": pd.Timestamp("2026-04-23T10:00:00+00:00"),
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_orderbook(symbol="BTCUSDT", exchange="binance"))
            data = assert_success_envelope(result, "fetch_orderbook")

            assert data["symbol"] == "BTC/USDT"
            assert data["exchange"] == "binance"
            assert data["limit"] == 20
            assert data["bids"] == [[67234.4, 1.2], [67234.3, 0.8]]
            assert data["asks"] == [[67234.6, 0.8], [67234.7, 0.5]]

    def test_fetch_orderbook_success_with_partial_depth(self):
        """One-sided orderbook snapshots are returned as success."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_orderbook_snapshot.return_value = {
                "bids": [[67234.4, 1.2]],
                "asks": [],
                "timestamp": pd.Timestamp("2026-04-23T10:00:00+00:00"),
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_orderbook(symbol="BTCUSDT", exchange="binance"))
            data = assert_success_envelope(result, "fetch_orderbook")

            assert data["bids"] == [[67234.4, 1.2]]
            assert data["asks"] == []


class TestFetchOrderbookFailure:
    """Failure path tests for fetch_orderbook."""

    def test_fetch_orderbook_data_source_error(self):
        """fetch_orderbook returns 3000 on adapter failure."""
        with patch("tempest_mcp.tools.market_tools.get_live_adapter") as mock_get_adapter:
            mock_adapter = MagicMock()
            mock_adapter.fetch_orderbook_snapshot.return_value = {
                "bids": [],
                "asks": [],
                "timestamp": None,
            }
            mock_get_adapter.return_value = mock_adapter

            result = _run_async(fetch_orderbook(symbol="BTCUSDT", exchange="binance"))
            assert_failure_envelope(result)
            assert result["error"]["code"] == 3000
