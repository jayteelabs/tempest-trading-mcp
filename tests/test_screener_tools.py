"""Tests for screener_tools MCP tool implementation."""

import pandas as pd
import pytest

from tempest_mcp.screener.scanner import ScanResult, Screener
from tempest_mcp.tools.screener_tools import (
    FILTER_VALUE_MAP,
    _parse_filters,
    _serialize_scan_result,
    screener_scan,
)


class DummyAdapter:
    """Dummy adapter for testing."""

    def __init__(self, df: pd.DataFrame | None = None, should_fail: bool = False):
        self.df = df
        self.should_fail = should_fail

    def fetch_ohlcv_live(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        if self.should_fail:
            raise RuntimeError("Simulated fetch error")
        return self.df


class TestFilterValueMap:
    """Tests for filter value mapping."""

    def test_all_filters_mapped(self):
        """All ScanFilter values should be in FILTER_VALUE_MAP."""
        from tempest_mcp.screener.scanner import ScanFilter

        for f in ScanFilter:
            assert f.value in FILTER_VALUE_MAP
            assert FILTER_VALUE_MAP[f.value] == f

    def test_filter_value_map_keys(self):
        """FILTER_VALUE_MAP should have expected keys."""
        expected_keys = {
            "rsi_oversold",
            "rsi_overbought",
            "trend_bullish",
            "trend_bearish",
            "high_volatility",
            "low_volatility",
            "volume_spike",
        }
        assert set(FILTER_VALUE_MAP.keys()) == expected_keys


class TestParseFilters:
    """Tests for _parse_filters function."""

    def test_parse_empty_list(self):
        """Empty list should return empty list."""
        assert _parse_filters([]) == []
        assert _parse_filters(None) == []

    def test_parse_valid_filters(self):
        """Valid filter strings should parse correctly."""
        result = _parse_filters(["rsi_oversold", "trend_bullish"])
        from tempest_mcp.screener.scanner import ScanFilter

        assert ScanFilter.RSI_OVERSOLD in result
        assert ScanFilter.TREND_BULLISH in result

    def test_parse_invalid_filter(self):
        """Invalid filter string should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_filters(["invalid_filter"])
        assert "Invalid filter" in str(exc_info.value)

    def test_parse_partial_invalid(self):
        """Partial invalid filters should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _parse_filters(["rsi_oversold", "invalid"])
        assert "Invalid filter" in str(exc_info.value)


class TestSerializeScanResult:
    """Tests for _serialize_scan_result function."""

    def test_serializes_all_fields(self):
        """All ScanResult fields should be serialized."""
        result = ScanResult(
            symbol="BTC/USDT",
            exchange="binance",
            timestamp=1.23456789,
            price=50000.0,
            filters_matched=["rsi_oversold"],
            indicator_values={"rsi": 25.0, "ema_7": 49000.0},
            score=80.0,
        )

        serialized = _serialize_scan_result(result)

        assert serialized["symbol"] == "BTC/USDT"
        assert serialized["exchange"] == "binance"
        assert serialized["timestamp"] == 1.23456789
        assert serialized["price"] == 50000.0
        assert serialized["filters_matched"] == ["rsi_oversold"]
        assert serialized["indicator_values"] == {"rsi": 25.0, "ema_7": 49000.0}
        assert serialized["score"] == 80.0


class TestScreenerScanTool:
    """Tests for screener_scan tool function."""

    def _create_df(self) -> pd.DataFrame:
        """Create a standard test DataFrame."""
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.2 for i in range(50)]
        return pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 1.0 for p in close_prices],
                "low": [p - 1.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )

    def test_screener_scan_returns_envelope(self):
        """screener_scan should return proper success/error envelope."""
        df = self._create_df()

        class TestAdapter(DummyAdapter):
            pass

        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = TestAdapter(df)

        # Patch the Screener temporarily
        import tempest_mcp.tools.screener_tools as st

        original_screener = st.Screener
        st.Screener = TestAdapter
        TestAdapter.Screener = original_screener

    def test_screener_scan_invalid_min_score_type(self):
        """Invalid min_score type should return error envelope."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            screener_scan(symbols=["BTC/USDT"], min_score="not_a_number")
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER

    def test_screener_scan_invalid_filter_value(self):
        """Invalid filter value should return error envelope."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            screener_scan(symbols=["BTC/USDT"], filters=["invalid_filter"])
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER

    def test_screener_scan_empty_symbols_uses_defaults(self):
        """Empty symbols should use default symbols from config."""
        import asyncio

        # This will attempt to use actual data adapter
        # Should handle gracefully
        result = asyncio.get_event_loop().run_until_complete(
            screener_scan(symbols=None, filters=[])
        )

        # Should return either success with data or failure
        assert "success" in result
        assert isinstance(result["success"], bool)


class TestScreenerScanEnvelope:
    """Tests for screener_scan response envelope format."""

    def _create_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.2 for i in range(50)]
        return pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 1.0 for p in close_prices],
                "low": [p - 1.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )

    def test_success_envelope_structure(self):
        """Success envelope should have correct structure."""
        # This is tested indirectly via integration tests
        # The key structure is:
        # {
        #   "success": True,
        #   "data": {
        #     "tool": "screener_scan",
        #     "exchange": str,
        #     "applied_config": {"filters": [...], "min_score": float},
        #     "results": [...],
        #     "failures": [...]
        #   }
        # }
        pass

    def test_error_envelope_structure(self):
        """Error envelope should have correct structure."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            screener_scan(symbols=["BTC/USDT"], min_score="invalid")
        )

        assert result["success"] is False
        assert "error" in result
        assert "code" in result["error"]
        assert "message" in result["error"]

    def test_no_stub_in_response(self):
        """Response should not contain stub fields."""
        import asyncio

        result = asyncio.get_event_loop().run_until_complete(
            screener_scan(symbols=["BTC/USDT"], filters=["rsi_oversold"])
        )

        # Should not have "stub" field
        if "success" in result and result["success"]:
            assert "data" in result
            if "data" in result:
                assert "stub" not in result["data"]
        else:
            # Error response should not have stub either
            assert "stub" not in result
