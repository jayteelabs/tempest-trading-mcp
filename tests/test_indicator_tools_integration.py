"""Integration tests for indicator_rsi (ENG-126).

These tests make real network calls and are skipped by default.
Run with: pytest --run-integration

Kurisu-style live validation of the non-stub indicator_rsi path.
"""

from __future__ import annotations

import asyncio

import pytest

from tempest_mcp.tools.indicator_tools import indicator_rsi


def _run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# =============================================================================
# Integration Tests — Live indicator_rsi
# =============================================================================


@pytest.mark.integration
class TestIndicatorRsiLive:
    """Live integration tests for indicator_rsi.

    These tests require network access and are skipped unless --run-integration is passed.
    """

    def test_live_binance_btcusdt_1h(self):
        """Live RSI calculation for BTC/USDT on Binance 1h timeframe."""
        result = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=14,
                timeframe="1h",
                limit=100,
                exchange="binance",
            )
        )

        assert result["success"] is True, f"Expected success, got: {result.get('error')}"
        data = result["data"]

        assert data["tool"] == "indicator_rsi"
        assert data["symbol"] in ("BTC/USDT", "BTCUSDT")
        assert data["exchange"] == "binance"
        assert data["timeframe"] == "1h"
        assert data["period"] == 14
        assert data["limit"] == 100
        assert data["source_used"] in ("ccxt", "yfinance")
        assert isinstance(data["values"], list)
        assert len(data["values"]) > 0
        assert "latest" in data
        assert "timestamp" in data["latest"]
        assert "rsi" in data["latest"]
        assert "zone" in data["latest"]
        assert data["latest"]["zone"] in ("oversold", "neutral", "overbought")

    def test_live_binance_ethusdt_1h(self):
        """Live RSI calculation for ETH/USDT on Binance 1h timeframe."""
        result = _run_async(
            indicator_rsi(
                symbol="ETHUSDT",
                period=14,
                timeframe="1h",
                limit=50,
                exchange="binance",
            )
        )

        assert result["success"] is True, f"Expected success, got: {result.get('error')}"
        data = result["data"]

        assert data["symbol"] in ("ETH/USDT", "ETHUSDT")
        assert data["exchange"] == "binance"
        assert len(data["values"]) > 0
        assert data["latest"]["rsi"] is not None

    def test_live_bybit_btcusdt_4h(self):
        """Live RSI calculation for BTC/USDT on Bybit 4h timeframe."""
        result = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=14,
                timeframe="4h",
                limit=50,
                exchange="bybit",
            )
        )

        assert result["success"] is True, f"Expected success, got: {result.get('error')}"
        data = result["data"]

        assert data["exchange"] == "bybit"
        assert data["timeframe"] == "4h"
        assert len(data["values"]) > 0

    def test_live_symbol_variants_normalize(self):
        """Different symbol input formats normalize to canonical."""
        # Test BTC-USDT hyphen format
        result = _run_async(
            indicator_rsi(
                symbol="BTC-USDT",
                period=14,
                timeframe="1h",
                limit=10,
                exchange="binance",
            )
        )

        assert result["success"] is True, f"Expected success, got: {result.get('error')}"
        data = result["data"]

        # Should be normalized to canonical BTC/USDT format
        assert "/" in data["symbol"] or data["symbol"] == "BTC/USDT"

    def test_live_limit_respected(self):
        """limit parameter is respected in live responses."""
        result = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=14,
                timeframe="1h",
                limit=5,
                exchange="binance",
            )
        )

        assert result["success"] is True
        data = result["data"]

        assert data["limit"] == 5
        assert len(data["values"]) <= 5

    def test_live_period_14_vs_7(self):
        """Different periods produce different RSI values."""
        result_14 = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=14,
                timeframe="1h",
                limit=10,
                exchange="binance",
            )
        )

        result_7 = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=7,
                timeframe="1h",
                limit=10,
                exchange="binance",
            )
        )

        assert result_14["success"] is True
        assert result_7["success"] is True

        # Different periods should produce different RSI values
        values_14 = result_14["data"]["values"]
        values_7 = result_7["data"]["values"]

        assert len(values_14) > 0
        assert len(values_7) > 0

        # At least one RSI value should differ
        rsi_14 = [v["rsi"] for v in values_14]
        rsi_7 = [v["rsi"] for v in values_7]
        assert rsi_14 != rsi_7

    def test_live_different_timeframes(self):
        """Different timeframes work correctly."""
        timeframes = ["1h", "4h", "1d"]

        for tf in timeframes:
            result = _run_async(
                indicator_rsi(
                    symbol="BTCUSDT",
                    period=14,
                    timeframe=tf,
                    limit=10,
                    exchange="binance",
                )
            )

            assert result["success"] is True, f"Failed for timeframe {tf}: {result.get('error')}"
            assert result["data"]["timeframe"] == tf

    def test_live_unsupported_exchange_fails(self):
        """Unsupported exchange returns proper error."""
        result = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=14,
                timeframe="1h",
                limit=100,
                exchange="fake_exchange",
            )
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1002
        assert "exchange" in result["error"]["message"].lower()

    def test_live_invalid_timeframe_fails(self):
        """Invalid timeframe returns proper error."""
        result = _run_async(
            indicator_rsi(
                symbol="BTCUSDT",
                period=14,
                timeframe="2h",  # Not a supported timeframe
                limit=100,
                exchange="binance",
            )
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1003
        assert "timeframe" in result["error"]["message"].lower()
