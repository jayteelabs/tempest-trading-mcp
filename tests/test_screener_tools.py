"""Tests for screener_tools MCP tool implementation."""

import pytest

from tempest_mcp.screener.scanner import ScanFailure, ScanResult
from tempest_mcp.tools import screener_tools
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
    async def test_screener_scan_rejects_empty_symbols(self):
        result = await screener_scan(symbols=[], filters=[])

        assert result == {
            "success": False,
            "error": {"code": 1004, "message": "symbols must contain at least 1 entry"},
        }

    @pytest.mark.asyncio
    async def test_screener_scan_reports_effective_exchange(self, monkeypatch):
        class FakeScreener:
            def __init__(self, symbols, exchange, filters, min_score):
                self.exchange = "bybit"

            def scan(self):
                return ([], [])

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_scan(symbols=["BTC/USDT"], exchange="binance")

        assert result["success"] is True
        assert result["data"]["exchange"] == "bybit"

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


class TestSessionBreakoutScanTool:
    """Tests for session_breakout_scan tool function — ENG-35."""

    @pytest.mark.asyncio
    async def test_session_breakout_scan_returns_success_envelope(self, monkeypatch):
        """session_breakout_scan should return deterministic success envelope."""

        class FakeScreener:
            last_init: dict | None = None

            def __init__(self, symbols, exchange):
                type(self).last_init = {
                    "symbols": symbols,
                    "exchange": exchange,
                }

            def session_breakout_scan(self, session, symbols, proximity_pct, volume_multiplier):
                return (
                    [
                        ScanResult(
                            symbol="BTC/USDT",
                            exchange="binance",
                            timestamp=1.0,
                            price=51000.0,
                            filters_matched=[
                                "session_high_breakout",
                                "pdh_breakout",
                                "volume_confirmation",
                            ],
                            indicator_values={
                                "session_high": 50000.0,
                                "session_low": 49000.0,
                                "session_bars": 10,
                                "previous_day_high": 50500.0,
                                "previous_day_low": 49500.0,
                                "volume_confirmed": 1.0,
                                "volume_multiplier": 2.0,
                                "proximity_pct": 1.0,
                            },
                            score=100.0,
                        )
                    ],
                    [],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
            exchange="binance",
            proximity_pct=1.0,
            volume_multiplier=2.0,
        )

        assert result["success"] is True
        assert result["data"]["tool"] == "session_breakout_scan"
        assert result["data"]["exchange"] == "binance"
        assert result["data"]["applied_config"]["session"] == "ny"
        assert result["data"]["applied_config"]["proximity_pct"] == 1.0
        assert result["data"]["applied_config"]["volume_multiplier"] == 2.0
        assert len(result["data"]["results"]) == 1
        assert result["data"]["results"][0]["filters_matched"] == [
            "session_high_breakout",
            "pdh_breakout",
            "volume_confirmation",
        ]

    @pytest.mark.asyncio
    async def test_session_breakout_scan_invalid_session(self):
        """Invalid session type should return error envelope."""
        result = await screener_tools.session_breakout_scan(
            session="invalid_session",
            symbols=["BTC/USDT"],
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_session_breakout_scan_rejects_empty_symbols(self):
        result = await screener_tools.session_breakout_scan(session="ny", symbols=[])

        assert result == {
            "success": False,
            "error": {"code": 1004, "message": "symbols must contain at least 1 entry"},
        }

    @pytest.mark.asyncio
    async def test_session_breakout_scan_new_york_alias(self, monkeypatch):
        """'new_york' should be accepted as alias for 'ny'."""
        from tempest_mcp.models.indicator import SessionType

        captured = {}

        class FakeScreener:
            def __init__(self, symbols, exchange):
                captured["symbols"] = symbols
                captured["exchange"] = exchange

            def session_breakout_scan(self, session, symbols, proximity_pct, volume_multiplier):
                captured["session"] = session
                captured["symbols_arg"] = symbols
                captured["proximity_pct"] = proximity_pct
                captured["volume_multiplier"] = volume_multiplier
                return ([], [])

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.session_breakout_scan(
            session="new_york",
            symbols=["BTC/USDT"],
        )

        assert result["success"] is True
        assert captured["session"] == SessionType.NEW_YORK
        assert captured["symbols"] == ("BTC/USDT",)
        assert captured["symbols_arg"] == ["BTC/USDT"]
        assert captured["proximity_pct"] == 1.0
        assert captured["volume_multiplier"] == 2.0

    @pytest.mark.asyncio
    async def test_session_breakout_scan_invalid_proximity_pct(self):
        """Invalid proximity_pct should return error envelope."""
        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
            proximity_pct="not_a_number",
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 1004

    @pytest.mark.asyncio
    async def test_session_breakout_scan_negative_proximity_pct(self):
        """Negative proximity_pct should return error envelope."""
        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
            proximity_pct=-1.0,
        )

        assert result["success"] is False
        assert "proximity_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_session_breakout_scan_invalid_volume_multiplier(self):
        """Invalid volume_multiplier should return error envelope."""
        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
            volume_multiplier=float("inf"),
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_session_breakout_scan_unknown_exchange(self):
        """Unknown exchange should return error envelope."""
        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
            exchange="unknown_exchange",
        )

        assert result["success"] is False
        assert "exchange" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_session_breakout_scan_partial_success(self, monkeypatch):
        """Partial success should return results + failures."""

        class FakeScreener:
            def __init__(self, symbols, exchange):
                pass

            def session_breakout_scan(self, session, symbols, proximity_pct, volume_multiplier):
                return (
                    [
                        ScanResult(
                            symbol="BBB/USDT",
                            exchange="binance",
                            timestamp=1.0,
                            price=79.0,
                            filters_matched=["pdl_breakout", "volume_confirmation"],
                            indicator_values={},
                            score=30.0,
                        ),
                        ScanResult(
                            symbol="AAA/USDT",
                            exchange="binance",
                            timestamp=2.0,
                            price=101.0,
                            filters_matched=["session_high_breakout"],
                            indicator_values={},
                            score=30.0,
                        ),
                    ],
                    [
                        ScanFailure(symbol="AAA/USDT", exchange="binance", reason="empty_ohlcv"),
                        ScanFailure(symbol="ZZZ/USDT", exchange="binance", reason="fetch_error"),
                    ],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT", "ETH/USDT"],
        )

        assert result["success"] is True
        assert result["data"]["results"] == [
            {
                "symbol": "BBB/USDT",
                "exchange": "binance",
                "timestamp": 1.0,
                "price": 79.0,
                "filters_matched": ["pdl_breakout", "volume_confirmation"],
                "indicator_values": {},
                "score": 30.0,
            },
            {
                "symbol": "AAA/USDT",
                "exchange": "binance",
                "timestamp": 2.0,
                "price": 101.0,
                "filters_matched": ["session_high_breakout"],
                "indicator_values": {},
                "score": 30.0,
            },
        ]
        assert result["data"]["failures"] == [
            {
                "symbol": "AAA/USDT",
                "exchange": "binance",
                "reason": "empty_ohlcv",
            },
            {
                "symbol": "ZZZ/USDT",
                "exchange": "binance",
                "reason": "fetch_error",
            },
        ]

    @pytest.mark.asyncio
    async def test_session_breakout_scan_full_failure(self, monkeypatch):
        """Full failure should return deterministic error response."""

        class FakeScreener:
            def __init__(self, symbols, exchange):
                pass

            def session_breakout_scan(self, session, symbols, proximity_pct, volume_multiplier):
                return (
                    [],
                    [
                        ScanFailure(
                            symbol="ZZZ/USDT",
                            exchange="binance",
                            reason="fetch_error",
                        ),
                        ScanFailure(
                            symbol="AAA/USDT",
                            exchange="binance",
                            reason="empty_ohlcv",
                        ),
                    ],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
        )

        assert result["success"] is False
        assert "error" in result
        assert result["error"]["code"] == 3000  # DATA_SOURCE_ERROR
        assert result["data"]["results"] == []
        assert result["data"]["failures"] == [
            {
                "symbol": "AAA/USDT",
                "exchange": "binance",
                "reason": "empty_ohlcv",
            },
            {
                "symbol": "ZZZ/USDT",
                "exchange": "binance",
                "reason": "fetch_error",
            },
        ]

    @pytest.mark.asyncio
    async def test_session_breakout_scan_internal_error_sanitized(self, monkeypatch):
        """Internal errors should be sanitized."""

        class ExplodingScreener:
            def __init__(self, symbols, exchange):
                pass

            def session_breakout_scan(self, session, symbols, proximity_pct, volume_multiplier):
                raise RuntimeError("sensitive network error")

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", ExplodingScreener)

        result = await screener_tools.session_breakout_scan(
            session="ny",
            symbols=["BTC/USDT"],
        )

        assert result["success"] is False
        assert result["error"]["code"] == 9000  # INTERNAL_ERROR
        assert "sensitive" not in result["error"]["message"]


class TestSessionBreakoutScanValidation:
    """Tests for session_breakout_scan argument validation — ENG-35."""

    def test_validate_session_accepts_asia(self):
        """_validate_session should accept 'asia'."""
        from tempest_mcp.tools.screener_tools import _validate_session

        normalized, error = _validate_session("asia")
        assert normalized == "asia"
        assert error is None

    def test_validate_session_accepts_london(self):
        """_validate_session should accept 'london'."""
        from tempest_mcp.tools.screener_tools import _validate_session

        normalized, error = _validate_session("london")
        assert normalized == "london"
        assert error is None

    def test_validate_session_accepts_ny(self):
        """_validate_session should accept 'ny'."""
        from tempest_mcp.tools.screener_tools import _validate_session

        normalized, error = _validate_session("ny")
        assert normalized == "ny"
        assert error is None

    def test_validate_session_normalizes_new_york(self):
        """_validate_session should normalize 'new_york' to 'ny'."""
        from tempest_mcp.tools.screener_tools import _validate_session

        normalized, error = _validate_session("new_york")
        assert normalized == "ny"
        assert error is None

    def test_validate_session_rejects_invalid(self):
        """_validate_session should reject invalid session types."""
        from tempest_mcp.tools.screener_tools import _validate_session

        normalized, error = _validate_session("invalid")
        assert normalized is None
        assert error is not None

    def test_validate_proximity_pct_accepts_valid(self):
        """_validate_proximity_pct should accept valid values."""
        from tempest_mcp.tools.screener_tools import _validate_proximity_pct

        assert _validate_proximity_pct(0.0) is None
        assert _validate_proximity_pct(1.0) is None
        assert _validate_proximity_pct(100.0) is None

    def test_validate_proximity_pct_rejects_negative(self):
        """_validate_proximity_pct should reject negative values."""
        from tempest_mcp.tools.screener_tools import _validate_proximity_pct

        assert _validate_proximity_pct(-0.1) is not None

    def test_validate_proximity_pct_rejects_over_100(self):
        """_validate_proximity_pct should reject values over 100."""
        from tempest_mcp.tools.screener_tools import _validate_proximity_pct

        assert _validate_proximity_pct(101.0) is not None

    def test_validate_volume_multiplier_accepts_valid(self):
        """_validate_volume_multiplier should accept valid values."""
        from tempest_mcp.tools.screener_tools import _validate_volume_multiplier

        assert _validate_volume_multiplier(0.0) is None
        assert _validate_volume_multiplier(1.0) is None
        assert _validate_volume_multiplier(2.0) is None

    def test_validate_volume_multiplier_rejects_negative(self):
        """_validate_volume_multiplier should reject negative values."""
        from tempest_mcp.tools.screener_tools import _validate_volume_multiplier

        assert _validate_volume_multiplier(-1.0) is not None


class TestOrderBlockScreenerScanTool:
    """Tests for order_block_screener_scan tool function — ENG-36."""

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_returns_success_envelope(self, monkeypatch):
        """order_block_screener_scan should return deterministic success envelope."""
        from tempest_mcp.screener.scanner import OrderBlockCandidate

        class FakeScreener:
            last_init = None

            def __init__(self, symbols, exchange):
                type(self).last_init = {"symbols": symbols, "exchange": exchange}

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                return (
                    [
                        OrderBlockCandidate(
                            symbol="BTC/USDT",
                            exchange="binance",
                            timeframe="1h",
                            window_days=1,
                            timestamp=1.0,
                            price=50000.0,
                            zone_type="bullish",
                            zone_high=49500.0,
                            zone_low=49000.0,
                            freshness_candles=5,
                            score=0.75,
                        )
                    ],
                    [],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"],
            exchange="binance",
            atr_period=14,
            impulse_atr_mult=1.0,
            max_zone_age_bars=20,
        )

        assert result["success"] is True
        assert result["data"]["tool"] == "order_block_screener_scan"
        assert result["data"]["exchange"] == "binance"
        assert result["data"]["applied_config"]["atr_period"] == 14
        assert result["data"]["applied_config"]["impulse_atr_mult"] == 1.0
        assert result["data"]["applied_config"]["max_zone_age_bars"] == 20
        # Fixed horizons should be visible in applied config
        assert "horizons" in result["data"]["applied_config"]
        assert {"timeframe": "1h", "window_days": 1} in result["data"]["applied_config"]["horizons"]
        assert {"timeframe": "4h", "window_days": 7} in result["data"]["applied_config"]["horizons"]
        assert len(result["data"]["candidates"]) == 1
        assert result["data"]["candidates"][0]["zone_type"] == "bullish"
        assert result["data"]["candidates"][0]["score"] == 0.75

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_candidate_top_level_fields(self, monkeypatch):
        """Candidates expose all top-level fields as per ENG-36 contract."""
        from tempest_mcp.screener.scanner import OrderBlockCandidate

        class FakeScreener:
            def __init__(self, symbols, exchange):
                pass

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                return (
                    [
                        OrderBlockCandidate(
                            symbol="BTC/USDT",
                            exchange="binance",
                            timeframe="4h",
                            window_days=7,
                            timestamp=1.0,
                            price=50000.0,
                            zone_type="bearish",
                            zone_high=50500.0,
                            zone_low=50000.0,
                            freshness_candles=3,
                            score=0.85,
                        )
                    ],
                    [],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(symbols=["BTC/USDT"])

        c = result["data"]["candidates"][0]
        required_fields = {
            "symbol",
            "exchange",
            "timeframe",
            "window_days",
            "timestamp",
            "price",
            "zone_type",
            "zone_high",
            "zone_low",
            "freshness_candles",
            "score",
        }
        assert required_fields.issubset(c.keys()), f"Missing: {required_fields - c.keys()}"

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_failure_top_level_fields(self, monkeypatch):
        """Partial success (candidates + failures) exposes all top-level fields on failure rows."""
        from tempest_mcp.screener.scanner import OrderBlockCandidate, OrderBlockFailure

        class FakeScreener:
            def __init__(self, symbols, exchange):
                pass

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                # Return one candidate + one failure to exercise partial-success shape
                candidate = OrderBlockCandidate(
                    symbol="BTC/USDT",
                    exchange="binance",
                    timeframe="4h",
                    window_days=7,
                    timestamp=1234567890.0,
                    price=50000.0,
                    zone_type="bullish",
                    zone_high=50500.0,
                    zone_low=49500.0,
                    freshness_candles=10,
                    score=0.85,
                )
                failure = OrderBlockFailure(
                    symbol="ETH/USDT",
                    exchange="binance",
                    timeframe="1h",
                    window_days=1,
                    reason="no_active_order_blocks",
                )
                return ([candidate], [failure])

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(symbols=["BTC/USDT", "ETH/USDT"])

        # Partial success: we have candidates, so success=True
        assert result["success"] is True
        assert len(result["data"]["candidates"]) == 1
        assert len(result["data"]["failures"]) == 1

        # Failure row must expose all top-level fields as per ENG-36 contract
        f = result["data"]["failures"][0]
        required_fields = {"symbol", "exchange", "timeframe", "window_days", "reason"}
        assert required_fields.issubset(f.keys()), f"Missing: {required_fields - f.keys()}"

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_full_failure_envelope(self, monkeypatch):
        """Full failure returns success:false with DATA_SOURCE_ERROR."""
        from tempest_mcp.screener.scanner import OrderBlockFailure

        class FakeScreener:
            def __init__(self, symbols, exchange):
                pass

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                return (
                    [],
                    [
                        OrderBlockFailure(
                            symbol="BTC/USDT",
                            exchange="binance",
                            timeframe="1h",
                            window_days=1,
                            reason="data_unavailable",
                        )
                    ],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(symbols=["BTC/USDT"])

        assert result["success"] is False
        assert result["error"]["code"] == 3000  # DATA_SOURCE_ERROR
        assert result["data"]["candidates"] == []
        assert len(result["data"]["failures"]) == 1

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_invalid_atr_period(self):
        """Invalid atr_period returns error envelope."""
        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"],
            atr_period=1,  # too low, min is 2
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004  # INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_invalid_impulse_atr_mult(self):
        """Invalid impulse_atr_mult returns error envelope."""
        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"],
            impulse_atr_mult=0.0,  # must be > 0
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_invalid_max_zone_age_bars(self):
        """Invalid max_zone_age_bars returns error envelope."""
        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"],
            max_zone_age_bars=501,  # too high, max is 500
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_unknown_exchange(self):
        """Unknown exchange returns error envelope."""
        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"],
            exchange="invalid_exchange",
        )

        assert result["success"] is False
        assert "exchange" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_symbols_cap(self):
        """Exceeding symbol cap returns error envelope."""
        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"] * 30,  # over MAX_SCAN_SYMBOLS of 25
        )

        assert result["success"] is False
        assert result["error"]["code"] == 1004

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_rejects_empty_symbols(self):
        result = await screener_tools.order_block_screener_scan(symbols=[])

        assert result == {
            "success": False,
            "error": {"code": 1004, "message": "symbols must contain at least 1 entry"},
        }

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_dedupes_symbols_before_scan(self, monkeypatch):
        captured: dict[str, object] = {}

        class FakeScreener:
            def __init__(self, symbols, exchange):
                captured["symbols"] = symbols
                captured["exchange"] = exchange
                self.exchange = exchange

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                captured["symbols_arg"] = symbols
                return ([], [])

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT", "ETH/USDT", "BTC/USDT"],
        )

        assert result["success"] is True
        assert captured["symbols"] == ("BTC/USDT", "ETH/USDT")
        assert captured["symbols_arg"] == ["BTC/USDT", "ETH/USDT"]
        assert result["data"]["applied_config"]["symbols"] == ["BTC/USDT", "ETH/USDT"]

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_reports_effective_exchange(self, monkeypatch):
        class FakeScreener:
            def __init__(self, symbols, exchange):
                self.exchange = "coinbase"

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                return ([], [])

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT"],
            exchange="binance",
        )

        assert result["success"] is True
        assert result["data"]["exchange"] == "coinbase"

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_internal_error_sanitized(self, monkeypatch):
        """Internal errors are sanitized."""

        class ExplodingScreener:
            def __init__(self, symbols, exchange):
                pass

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                raise RuntimeError("sensitive network error")

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", ExplodingScreener)

        result = await screener_tools.order_block_screener_scan(symbols=["BTC/USDT"])

        assert result["success"] is False
        assert result["error"]["code"] == 9000  # INTERNAL_ERROR
        assert "sensitive" not in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_order_block_screener_scan_partial_success_includes_failures(self, monkeypatch):
        """Partial success includes both candidates and failures."""
        from tempest_mcp.screener.scanner import OrderBlockCandidate, OrderBlockFailure

        class FakeScreener:
            def __init__(self, symbols, exchange):
                pass

            def order_block_scan(self, symbols, atr_period, impulse_atr_mult, max_zone_age_bars):
                return (
                    [
                        OrderBlockCandidate(
                            symbol="BTC/USDT",
                            exchange="binance",
                            timeframe="1h",
                            window_days=1,
                            timestamp=1.0,
                            price=50000.0,
                            zone_type="bullish",
                            zone_high=49500.0,
                            zone_low=49000.0,
                            freshness_candles=5,
                            score=0.75,
                        )
                    ],
                    [
                        OrderBlockFailure(
                            symbol="ETH/USDT",
                            exchange="binance",
                            timeframe="4h",
                            window_days=7,
                            reason="no_active_order_blocks",
                        )
                    ],
                )

        monkeypatch.setattr("tempest_mcp.tools.screener_tools.Screener", FakeScreener)

        result = await screener_tools.order_block_screener_scan(
            symbols=["BTC/USDT", "ETH/USDT"],
        )

        assert result["success"] is True
        assert len(result["data"]["candidates"]) == 1
        assert len(result["data"]["failures"]) == 1


class TestOrderBlockScreenerScanValidation:
    """Tests for order_block_screener_scan argument validation — ENG-36."""

    def test_validate_atr_period_accepts_valid(self):
        """_validate_atr_period accepts valid values."""
        from tempest_mcp.tools.screener_tools import _validate_atr_period

        assert _validate_atr_period(14) is None
        assert _validate_atr_period(2) is None
        assert _validate_atr_period(200) is None

    def test_validate_atr_period_rejects_too_low(self):
        """_validate_atr_period rejects values below 2."""
        from tempest_mcp.tools.screener_tools import _validate_atr_period

        assert _validate_atr_period(1) is not None

    def test_validate_atr_period_rejects_too_high(self):
        """_validate_atr_period rejects values above 200."""
        from tempest_mcp.tools.screener_tools import _validate_atr_period

        assert _validate_atr_period(201) is not None

    def test_validate_atr_period_rejects_non_int(self):
        """_validate_atr_period rejects non-integer types."""
        from tempest_mcp.tools.screener_tools import _validate_atr_period

        assert _validate_atr_period("14") is not None
        assert _validate_atr_period(14.0) is not None

    def test_validate_impulse_atr_mult_accepts_valid(self):
        """_validate_impulse_atr_mult accepts valid values."""
        from tempest_mcp.tools.screener_tools import _validate_impulse_atr_mult

        assert _validate_impulse_atr_mult(1.0) is None
        assert _validate_impulse_atr_mult(0.1) is None
        assert _validate_impulse_atr_mult(10.0) is None

    def test_validate_impulse_atr_mult_rejects_zero_or_negative(self):
        """_validate_impulse_atr_mult rejects zero and negative."""
        from tempest_mcp.tools.screener_tools import _validate_impulse_atr_mult

        assert _validate_impulse_atr_mult(0.0) is not None
        assert _validate_impulse_atr_mult(-1.0) is not None

    def test_validate_impulse_atr_mult_rejects_above_10(self):
        """_validate_impulse_atr_mult rejects values above 10."""
        from tempest_mcp.tools.screener_tools import _validate_impulse_atr_mult

        assert _validate_impulse_atr_mult(10.1) is not None

    def test_validate_max_zone_age_bars_accepts_valid(self):
        """_validate_max_zone_age_bars accepts valid values."""
        from tempest_mcp.tools.screener_tools import _validate_max_zone_age_bars

        assert _validate_max_zone_age_bars(1) is None
        assert _validate_max_zone_age_bars(20) is None
        assert _validate_max_zone_age_bars(500) is None

    def test_validate_max_zone_age_bars_rejects_zero(self):
        """_validate_max_zone_age_bars rejects zero."""
        from tempest_mcp.tools.screener_tools import _validate_max_zone_age_bars

        assert _validate_max_zone_age_bars(0) is not None

    def test_validate_max_zone_age_bars_rejects_above_500(self):
        """_validate_max_zone_age_bars rejects values above 500."""
        from tempest_mcp.tools.screener_tools import _validate_max_zone_age_bars

        assert _validate_max_zone_age_bars(501) is not None
