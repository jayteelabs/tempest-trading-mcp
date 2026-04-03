"""Tests for screener."""
import pytest
from tempest_mcp.screener.scanner import Screener, ScanFilter, ScanResult
from tempest_mcp.models.indicator import SessionType

class TestScreener:
    def test_init(self):
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        assert screener.symbols == ("BTC/USDT",)

    def test_scan_result(self):
        result = ScanResult(symbol="BTC/USDT", exchange="binance", timestamp=1.0, price=50000.0, filters_matched=["rsi_oversold"], indicator_values={"rsi": 25.0}, score=80.0)
        assert result.score == 80.0

class TestFilters:
    def test_values(self):
        assert ScanFilter.RSI_OVERSOLD.value == "rsi_oversold"
        assert ScanFilter.TREND_BULLISH.value == "trend_bullish"
