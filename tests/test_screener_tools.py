"""Tests for screener_tools MCP tool implementation."""

import pytest

from tempest_mcp.screener.scanner import ScanFailure, ScanResult
from tempest_mcp.tools.screener_tools import (
    FILTER_VALUE_MAP,
    MAX_SCAN_SYMBOLS,
    _parse_filters,
    _serialize_scan_result,
    screener_scan,
)


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

    def test_parse_duplicate_filters_dedupes_preserving_order(self):
        """Duplicate filter strings should not inflate later scoring."""
        from tempest_mcp.screener.scanner import ScanFilter

        result = _parse_filters(["rsi_oversold", "rsi_oversold", "trend_bullish"])

        assert result == [ScanFilter.RSI_OVERSOLD, ScanFilter.TREND_BULLISH]

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

    @pytest.mark.asyncio
    async def test_screener_scan_returns_success_envelope(self, monkeypatch):
        """screener_scan should return the deterministic success envelope."""

        class FakeScreener:
            last_init: dict | None = None

            def __init__(self, symbols, exchange, filters, min_score):
                type(self).last_init = {
                    "symbols": symbols,
                    "exchange": exchange,
                    "filters": filters,
                    "min_score": min_score,
                }

            def scan(self):
                return (
                    [
                        ScanResult(
                            symbol="BTC/USDT",
                            exchange="binance",
                            timestamp=1.0,
                            price=50000.0,
                            filters_matched=["rsi_oversold"],
                            indicator_values={"rsi": 25.0},
                            score=100.0,
                        )
                    ],
                    [
                        ScanFailure(
                            symbol="ETH/USDT",
                            exchange="binance",
                            reason="fetch_error",
                        )
                    ],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_scan(
            symbols=["BTC/USDT"],
            filters=["rsi_oversold", "rsi_oversold"],
            min_score=10.0,
            exchange="BINANCE",
        )

        assert result["success"] is True
        assert result["data"]["tool"] == "screener_scan"
        assert result["data"]["exchange"] == "binance"
        assert result["data"]["applied_config"] == {
            "filters": ["rsi_oversold"],
            "min_score": 10.0,
        }
        assert result["data"]["results"][0]["filters_matched"] == ["rsi_oversold"]
        assert result["data"]["failures"] == [
            {
                "symbol": "ETH/USDT",
                "exchange": "binance",
                "reason": "fetch_error",
            }
        ]
        assert "stub" not in result["data"]
        assert FakeScreener.last_init == {
            "symbols": ("BTC/USDT",),
            "exchange": "binance",
            "filters": [FILTER_VALUE_MAP["rsi_oversold"]],
            "min_score": 10.0,
        }

    @pytest.mark.asyncio
    async def test_screener_scan_invalid_min_score_type(self):
        """Invalid min_score type should return error envelope."""
        result = await screener_scan(symbols=["BTC/USDT"], min_score="not_a_number")

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_screener_scan_rejects_non_finite_min_score(self):
        result = await screener_scan(symbols=["BTC/USDT"], min_score=float("inf"))

        assert result == {
            "success": False,
            "error": {"code": 1004, "message": "min_score must be finite"},
        }

    @pytest.mark.asyncio
    async def test_screener_scan_rejects_out_of_range_min_score(self):
        result = await screener_scan(symbols=["BTC/USDT"], min_score=101.0)

        assert result == {
            "success": False,
            "error": {"code": 1004, "message": "min_score must be between 0 and 100"},
        }

    @pytest.mark.asyncio
    async def test_screener_scan_invalid_filter_value(self):
        """Invalid filter value should return error envelope."""
        result = await screener_scan(symbols=["BTC/USDT"], filters=["invalid_filter"])

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_screener_scan_rejects_unknown_exchange(self):
        result = await screener_scan(symbols=["BTC/USDT"], exchange="okx")

        assert result == {
            "success": False,
            "error": {
                "code": 1004,
                "message": "exchange must be one of: binance, bybit, coinbase, kraken",
            },
        }

    @pytest.mark.asyncio
    async def test_screener_scan_rejects_symbol_lists_above_cap(self):
        result = await screener_scan(symbols=["BTC/USDT"] * (MAX_SCAN_SYMBOLS + 1))

        assert result == {
            "success": False,
            "error": {
                "code": 1004,
                "message": f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries",
            },
        }

    @pytest.mark.asyncio
    async def test_screener_scan_empty_symbols_uses_defaults(self, monkeypatch):
        """Empty symbols should use default symbols from config."""

        class FakeScreener:
            last_symbols = None

            def __init__(self, symbols, exchange, filters, min_score):
                type(self).last_symbols = symbols

            def scan(self):
                return [], []

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_scan(symbols=[], filters=[])

        assert result["success"] is True
        assert result["data"]["results"] == []
        assert result["data"]["failures"] == []
        assert FakeScreener.last_symbols == ("BTC/USDT", "ETH/USDT", "DOGE/USDT")

    @pytest.mark.asyncio
    async def test_screener_scan_internal_errors_are_sanitized(self, monkeypatch):
        class ExplodingScreener:
            def __init__(self, symbols, exchange, filters, min_score):
                pass

            def scan(self):
                raise RuntimeError("sensitive network error")

        monkeypatch.setattr(
            "tempest_mcp.tools.screener_tools.Screener",
            ExplodingScreener,
        )

        result = await screener_scan(symbols=["BTC/USDT"])

        assert result == {
            "success": False,
            "error": {"code": 9000, "message": "Unable to complete screener scan"},
        }


class TestScreenerScanEnvelope:
    """Tests for screener_scan response envelope format."""

    @pytest.mark.asyncio
    async def test_error_envelope_structure(self):
        """Error envelope should have correct structure."""
        result = await screener_scan(symbols=["BTC/USDT"], min_score="invalid")

        assert result["success"] is False
        assert "error" in result
        assert "code" in result["error"]
        assert "message" in result["error"]
