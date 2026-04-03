"""Tests for technical indicators."""
import pytest
import numpy as np
from tempest_mcp.indicators.ta_wrapper import calculate_ema, calculate_rsi, IndicatorError
from tempest_mcp.indicators.trend import calculate_vwap, calculate_supertrend
from tempest_mcp.indicators.momentum import calculate_rsi_result, calculate_macd_result
from tempest_mcp.indicators.volatility import calculate_atr_result

class TestTAWrapper:
    def test_ema(self, price_data):
        ema = calculate_ema(price_data, period=20)
        assert len(ema) == len(price_data)

    def test_rsi(self, price_data):
        rsi = calculate_rsi(price_data, period=14)
        assert len(rsi) == len(price_data)

    def test_insufficient_data(self):
        with pytest.raises(IndicatorError):
            calculate_rsi([100, 101], period=14)

class TestTrend:
    def test_vwap(self, ohlcv_data):
        result = calculate_vwap(ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"], ohlcv_data["volume"])
        assert "vwap" in result.values

    def test_supertrend(self, ohlcv_data):
        result = calculate_supertrend(ohlcv_data["high"], ohlcv_data["low"], ohlcv_data["close"])
        assert "supertrend" in result.values

class TestMomentum:
    def test_rsi_result(self, price_data):
        result = calculate_rsi_result(price_data)
        assert "rsi" in result.values
        assert 0 <= result.values["rsi"] <= 100

    def test_macd_result(self, price_data):
        result = calculate_macd_result(price_data)
        assert "macd" in result.values
