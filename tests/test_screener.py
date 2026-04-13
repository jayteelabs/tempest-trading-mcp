"""Tests for screener."""

import pandas as pd

from tempest_mcp.models.indicator import SessionType
from tempest_mcp.screener.scanner import ScanFilter, ScanResult, Screener


class TestScreener:
    def test_init(self):
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        assert screener.symbols == ("BTC/USDT",)

    def test_scan_result(self):
        result = ScanResult(
            symbol="BTC/USDT",
            exchange="binance",
            timestamp=1.0,
            price=50000.0,
            filters_matched=["rsi_oversold"],
            indicator_values={"rsi": 25.0},
            score=80.0,
        )
        assert result.score == 80.0


class TestFilters:
    def test_values(self):
        assert ScanFilter.RSI_OVERSOLD.value == "rsi_oversold"
        assert ScanFilter.TREND_BULLISH.value == "trend_bullish"


class TestSessionBreakoutScan:
    def test_session_breakout_scan_uses_detect_session_levels(self):
        dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 48,
                "high": [101.0 + i * 0.1 for i in range(48)],
                "low": [99.0 + i * 0.1 for i in range(48)],
                "close": [100.5 + i * 0.2 for i in range(48)],
                "volume": [1000.0] * 48,
            },
            index=dates,
        )

        class DummyAdapter:
            def fetch_ohlcv_live(self, symbol: str, timeframe: str = "1h", limit: int = 48):
                return df

        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter()

        results = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(results) == 1
        assert results[0].indicator_values["session_high"] > 0
        assert results[0].indicator_values["session_low"] > 0
