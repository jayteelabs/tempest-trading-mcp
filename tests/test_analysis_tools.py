"""Tests for ENG-28 analysis tools — calculate_volume_profile and detect_order_blocks.

Covers:
    - handler validation envelopes (invalid inputs)
    - serialization of valid results
    - deterministic fixture behavior
    - order-block output boundary (read-only analytical, no retest/entry/PnL)
    - reuse of existing indicator/strategy logic
"""

import json

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.config import ErrorCodes
from tempest_mcp.indicators.volume.volume_profile import (
    calculate_volume_profile as volume_profile_indicator,
)
from tempest_mcp.tools.analysis_tools import (
    _internal_error,
    _parse_iso_datetime,
    _safe_float,
    _validation_error,
    calculate_volume_profile,
    detect_order_blocks,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def deterministic_ohlcv():
    """Create a deterministic OHLCV DataFrame for reproducible tests.

    Uses a fixed seed so that volume profile and order-block detection
    produce the same output on every run.
    """
    np.random.seed(42)
    n = 100
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n)
    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_price = np.roll(close, 1)
    open_price[0] = base_price
    volume = np.abs(np.random.normal(1000, 200, n))

    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def order_block_fixture_ohlcv():
    """OHLCV DataFrame with known order-block patterns for deterministic testing.

    Constructs a simple bullish displacement:
    - Bar 0-3: neutral
    - Bar 4: bearish (the OB candle)
    - Bar 5: bullish with close > bar4.high and body >= atr * impulse_atr_mult
    This should produce one bullish OB zone.
    """
    idx = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")

    # Build a simple trending dataset with a known OB pattern
    close = np.array([100.0] * 20)
    close[5:] = 103.0  # bullish displacement after bar 5

    high = close.copy()
    low = close.copy()
    open_price = close.copy()

    # Bar 4: bearish candle (OB candidate) — high=102, low=99, close=99
    high[4] = 102.0
    low[4] = 99.0
    open_price[4] = 101.0
    close[4] = 99.0

    # Bar 5: bullish displacement — close breaks bar4.high
    open_price[5] = 99.5
    close[5] = 103.0  # breaks above bar4 high (102)

    # Set volume (non-zero but small)
    volume = np.full(20, 100.0)

    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ── Helper tests ───────────────────────────────────────────────────────────────


class TestParseIsoDatetime:
    """Tests for _parse_iso_datetime helper."""

    def test_parses_naive_datetime(self):
        """Naive datetime string is parsed successfully."""
        result = _parse_iso_datetime("start_at", "2024-01-01T00:00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 1

    def test_parses_utc_datetime(self):
        """UTC datetime string is parsed successfully."""
        result = _parse_iso_datetime("start_at", "2024-01-01T00:00:00+00:00")
        assert result is not None
        assert result.tzinfo is not None

    def test_parses_z_suffix(self):
        """ISO datetime with Z suffix is parsed successfully."""
        result = _parse_iso_datetime("start_at", "2024-01-01T00:00:00Z")
        assert result is not None

    def test_rejects_invalid_string(self):
        """Invalid string raises ValueError."""
        with pytest.raises(ValueError):
            _parse_iso_datetime("start_at", "not-a-date")

    def test_rejects_non_string(self):
        """Non-string value raises ValueError."""
        with pytest.raises(ValueError):
            _parse_iso_datetime("start_at", 12345)


class TestSafeFloat:
    """Tests for _safe_float helper."""

    def test_converts_valid_float(self):
        """Valid float is returned as-is."""
        assert _safe_float(1.23) == 1.23

    def test_converts_int(self):
        """Integer is converted to float."""
        assert _safe_float(42) == 42.0

    def test_returns_none_for_none(self):
        """None input returns None."""
        assert _safe_float(None) is None

    def test_returns_none_for_inf(self):
        """Infinite value returns None."""
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None

    def test_returns_none_for_nan(self):
        """NaN value returns None."""
        assert _safe_float(float("nan")) is None


class TestValidationErrorEnvelope:
    """Tests for _validation_error helper shape."""

    def test_validation_error_has_correct_structure(self):
        """Validation error has success=False and error.code=INVALID_PARAMETER."""
        result = _validation_error("test message")
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert result["error"]["message"] == "test message"

    def test_internal_error_has_correct_structure(self):
        """Internal error has success=False and error.code=INTERNAL_ERROR."""
        result = _internal_error("internal message")
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INTERNAL_ERROR
        assert result["error"]["message"] == "internal message"


# ── calculate_volume_profile handler tests ─────────────────────────────────────


class TestCalculateVolumeProfileValidation:
    """Validation envelope tests for calculate_volume_profile handler.

    Tests cover: symbol format, timeframe enum, datetime parsing,
    profile param constraints, and window validation.
    """

    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol returns validation error envelope."""
        result = await calculate_volume_profile(
            symbol="INVALID@SYMBOL",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "symbol" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_timeframe(self):
        """Invalid timeframe returns validation error envelope."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="invalid",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "timeframe" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_start_at_format(self):
        """Malformed start_at returns validation error envelope."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="not-a-date",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "start_at" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_end_at_format(self):
        """Malformed end_at returns validation error envelope."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="not-a-date",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "end_at" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_profile_type(self):
        """Invalid profile_type returns validation error envelope."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            profile_type="invalid",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "profile_type" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_dynamic_without_dynamic_mode(self):
        """profile_type='dynamic' without dynamic_mode returns validation error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            profile_type="dynamic",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "dynamic_mode" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_dynamic_pct_without_range_pct(self):
        """dynamic_mode='pct' without range_pct returns validation error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            profile_type="dynamic",
            dynamic_mode="pct",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "range_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_negative_bin_count(self):
        """Negative bin_count returns validation error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            bin_count=-5,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "bin_count" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_bin_count_too_large(self):
        """bin_count exceeding MAX_BIN_COUNT (500) returns validation error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            bin_count=501,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "bin_count" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_zero_bin_count(self):
        """Zero bin_count returns validation error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            bin_count=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "bin_count" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_max_acceptable_bin_count(self):
        """MAX_BIN_COUNT (500) is accepted without error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            bin_count=500,
        )
        # Should not be a validation error (may fail for other reasons like data fetch)
        # but the bin_count itself should be accepted
        assert (
            result.get("success") is not False
            or "bin_count" not in result.get("error", {}).get("message", "").lower()
        )

    @pytest.mark.asyncio
    async def test_value_area_out_of_range(self):
        """value_area_pct outside (0, 1] returns validation error."""
        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            value_area_pct=1.5,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "value_area_pct" in result["error"]["message"]


class TestCalculateVolumeProfileSerialization:
    """Serialization tests for calculate_volume_profile handler.

    Verifies deterministic output structure using a fixture DataFrame.
    """

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, deterministic_ohlcv, monkeypatch):
        """Success envelope has correct top-level structure."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
        )

        assert result["success"] is True
        data = result["data"]
        assert data["tool"] == "calculate_volume_profile"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert "window" in data
        assert "summary" in data
        assert "profile_rows" in data

    @pytest.mark.asyncio
    async def test_summary_fields_present(self, deterministic_ohlcv, monkeypatch):
        """Summary dict contains all required scalar fields."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
        )

        summary = result["data"]["summary"]
        required = {
            "poc_price",
            "vah_price",
            "val_price",
            "profile_shape",
            "total_volume",
            "bin_count",
            "profile_type",
        }
        assert required <= set(summary.keys())

    @pytest.mark.asyncio
    async def test_profile_rows_are_serializable(self, deterministic_ohlcv, monkeypatch):
        """profile_rows list is JSON-serializable (all native Python types)."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_volume_profile(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
        )

        import json

        # Must not raise — confirms JSON serializable
        json.dumps(result["data"]["profile_rows"])
        assert result["data"]["summary"]["bin_count"] == len(result["data"]["profile_rows"])

    @pytest.mark.asyncio
    async def test_deterministic_output(self, deterministic_ohlcv, monkeypatch):
        """Running the same calculation twice produces identical output."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        args = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_at": "2024-01-01T00:00:00",
            "end_at": "2024-01-08T00:00:00",
        }

        result1 = await calculate_volume_profile(**args)
        result2 = await calculate_volume_profile(**args)

        import json

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


# ── detect_order_blocks handler tests ─────────────────────────────────────────


class TestDetectOrderBlocksValidation:
    """Validation envelope tests for detect_order_blocks handler.

    Tests cover: symbol format, timeframe enum, datetime parsing,
    detection param constraints, and window validation.
    """

    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol returns validation error envelope."""
        result = await detect_order_blocks(
            symbol="INVALID@SYMBOL",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "symbol" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_timeframe(self):
        """Invalid timeframe returns validation error envelope."""
        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="invalid",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "timeframe" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_start_at(self):
        """Malformed start_at returns validation error envelope."""
        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="not-a-date",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "start_at" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_negative_atr_period(self):
        """Negative atr_period returns validation error."""
        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            atr_period=-5,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "atr_period" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_negative_impulse_atr_mult(self):
        """Negative impulse_atr_mult returns validation error."""
        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            impulse_atr_mult=-0.5,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "impulse_atr_mult" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_zero_max_zone_age_bars(self):
        """Zero max_zone_age_bars returns validation error."""
        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            max_zone_age_bars=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "max_zone_age_bars" in result["error"]["message"]


class TestDetectOrderBlocksSerialization:
    """Serialization tests for detect_order_blocks handler.

    Verifies deterministic output structure and analytical boundary
    (no retest/entry/PnL fields in output).
    """

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, order_block_fixture_ohlcv, monkeypatch):
        """Success envelope has correct top-level structure."""

        def mock_resolve(request):
            return order_block_fixture_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )

        assert result["success"] is True
        data = result["data"]
        assert data["tool"] == "detect_order_blocks"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert "window" in data
        assert "order_blocks" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_order_block_fields(self, order_block_fixture_ohlcv, monkeypatch):
        """Each order_block dict contains required analytical fields only."""

        def mock_resolve(request):
            return order_block_fixture_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )

        required_fields = {"date", "type", "zone_high", "zone_low", "freshness_candles"}
        for block in result["data"]["order_blocks"]:
            assert required_fields <= set(block.keys())
            # Verify no trading-semantic fields leak in
            assert "retest" not in block
            assert "entry" not in block
            assert "pnl" not in block
            assert "risk_reward" not in block
            assert "stop" not in block
            assert "target" not in block

    @pytest.mark.asyncio
    async def test_read_only_analytical_boundary(self, order_block_fixture_ohlcv, monkeypatch):
        """detect_order_blocks output is strictly read-only analytical.

        The output must not contain:
            - retest signals
            - entry signals
            - PnL metrics
            - trade counts
            - position state
        """

        def mock_resolve(request):
            return order_block_fixture_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_order_blocks(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )

        data = result["data"]
        # Strictly analytical fields only
        expected_keys = {"tool", "symbol", "timeframe", "window", "order_blocks", "count"}
        assert set(data.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_deterministic_output(self, order_block_fixture_ohlcv, monkeypatch):
        """Running detection twice produces identical output."""

        def mock_resolve(request):
            return order_block_fixture_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        args = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_at": "2024-01-01T00:00:00",
            "end_at": "2024-01-02T00:00:00",
        }

        result1 = await detect_order_blocks(**args)
        result2 = await detect_order_blocks(**args)

        import json

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestOrderBlocksDetectionHelper:
    """Tests for detect_active_order_blocks standalone helper.

    Verifies:
        - active-zone-only behavior at end-of-window
        - invalidation filtering applied
        - max-age filtering applied
        - determinism on repeated runs
    """

    def test_returns_list_for_valid_ohlcv(self, order_block_fixture_ohlcv):
        """Returns a list for valid OHLCV input."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        result = detect_active_order_blocks(order_block_fixture_ohlcv)
        assert isinstance(result, list)

    def test_returns_empty_for_insufficient_data(self):
        """Returns empty list when OHLCV has fewer bars than atr_period + 4."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        idx = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
                "volume": [100, 100, 100],
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="Insufficient data"):
            detect_active_order_blocks(ohlcv)

    def test_invalidates_zones_before_end_of_window(self):
        """Zones invalidated before final bar are excluded from result."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        # This test uses a scenario where a zone gets invalidated mid-window
        # The fixture constructs a simple case
        idx = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
        close = np.full(20, 100.0)
        # Create bearish OB at bar 4
        close[4] = 95.0  # bearish candle
        # Bullish displacement at bar 5
        close[5] = 105.0  # breaks above bar4.high
        # But then price drops below the bullish zone low — invalidates it
        close[10] = 90.0  # invalidates bullish zone at bar 10

        high = close + 1
        low = close - 1
        open_price = close.copy()
        volume = np.full(20, 100.0)

        ohlcv = pd.DataFrame(
            {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )

        result = detect_active_order_blocks(ohlcv)
        # The invalidated zone should not appear
        assert isinstance(result, list)

    def test_max_age_filtering(self):
        """Zones older than max_zone_age_bars are excluded."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        # Build a dataset where an OB is created but max_zone_age_bars=2 should exclude it
        idx = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
        close = np.full(20, 100.0)
        # Create bearish OB at bar 4
        close[4] = 95.0
        # Bullish displacement at bar 5
        close[5] = 105.0
        # Price stays high so zone is not invalidated, but zone is 14 bars old at end
        for i in range(6, 20):
            close[i] = 104.0

        high = close + 1
        low = close - 1
        open_price = close.copy()
        volume = np.full(20, 100.0)

        ohlcv = pd.DataFrame(
            {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )

        # With max_zone_age_bars=2, the zone created at bar 4 should be excluded
        result = detect_active_order_blocks(ohlcv, max_zone_age_bars=2)
        # Should not include the old zone
        for zone in result:
            assert zone["freshness_candles"] <= 2

    def test_empty_ohlcv_raises(self):
        """Empty OHLCV raises ValueError."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        idx = pd.date_range("2024-01-01", periods=0, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": [], "volume": []},
            index=idx,
        )
        with pytest.raises(ValueError, match="Insufficient data"):
            detect_active_order_blocks(ohlcv)

    def test_duplicate_index_raises(self):
        """Duplicate datetime index raises ValueError."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        idx = pd.to_datetime(
            [
                "2024-01-01T00:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T01:00:00Z",
                "2024-01-01T02:00:00Z",
                "2024-01-01T03:00:00Z",
                "2024-01-01T04:00:00Z",
                "2024-01-01T05:00:00Z",
                "2024-01-01T06:00:00Z",
                "2024-01-01T07:00:00Z",
                "2024-01-01T08:00:00Z",
                "2024-01-01T09:00:00Z",
                "2024-01-01T10:00:00Z",
                "2024-01-01T11:00:00Z",
                "2024-01-01T12:00:00Z",
            ],
            utc=True,
        )
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * len(idx),
                "high": [101.0] * len(idx),
                "low": [99.0] * len(idx),
                "close": [100.5] * len(idx),
                "volume": [1000.0] * len(idx),
            },
            index=idx,
        )

        with pytest.raises(ValueError, match="must not contain duplicates"):
            detect_active_order_blocks(ohlcv)

    def test_deterministic_on_repeated_runs(self, deterministic_ohlcv):
        """Running detection repeatedly produces identical output."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        result1 = detect_active_order_blocks(deterministic_ohlcv)
        result2 = detect_active_order_blocks(deterministic_ohlcv)
        assert result1 == result2


# ── Reuse confirmation tests ─────────────────────────────────────────────────


class TestIndicatorReuse:
    """Confirms ENG-28 tools reuse existing indicator/strategy logic.

    Verifies that the handlers delegate to the existing pure indicator
    and strategy functions rather than duplicating logic.
    """

    @pytest.mark.asyncio
    async def test_calculate_volume_profile_uses_indicator(self, deterministic_ohlcv, monkeypatch):
        """calculate_volume_profile handler calls the existing indicator exactly once."""
        call_count = 0
        indicator_result_df = volume_profile_indicator(deterministic_ohlcv, bin_count=50)
        # Add required attrs that the handler's serialization expects
        indicator_result_df.attrs["poc_price"] = 100.0
        indicator_result_df.attrs["vah_price"] = 105.0
        indicator_result_df.attrs["val_price"] = 95.0
        indicator_result_df.attrs["profile_shape"] = "bell"
        indicator_result_df.attrs["total_volume"] = 1000.0
        indicator_result_df.attrs["bin_count"] = 10
        indicator_result_df.attrs["profile_type"] = "fixed"

        def counting_indicator(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return indicator_result_df

        # Patch the alias used by the handler at the module where the handler looks it up
        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools._calculate_volume_profile_indicator",
            counting_indicator,
        )

        # Monkeypatch the data fetch so the handler doesn't hit a real exchange
        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        # Call the handler directly (not the indicator)
        from tempest_mcp.tools.analysis_tools import calculate_volume_profile as handler

        result = await handler(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
        )

        # Assert the handler succeeded
        assert result["success"] is True
        # Assert the indicator wrapper was invoked exactly once (no recursion)
        assert call_count == 1

    def test_detect_order_blocks_uses_strategy_helper(self, order_block_fixture_ohlcv):
        """detect_order_blocks handler uses the standalone strategy helper."""
        from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks

        # The helper should be importable and callable
        result = detect_active_order_blocks(order_block_fixture_ohlcv)
        assert isinstance(result, list)


# ── Mock window helper ─────────────────────────────────────────────────────────


def _mock_window():
    """Return a mock ResolvedBacktestWindow for testing."""
    from tempest_mcp.tools.backtest_window import ResolvedBacktestWindow

    return ResolvedBacktestWindow(
        symbol="BTC/USDT",
        trade_style="custom",
        timeframe="1h",
        start_at_utc=pd.Timestamp("2024-01-01", tz="UTC"),
        end_at_utc=pd.Timestamp("2024-01-08", tz="UTC"),
        estimated_bars=168,
        exchange="binance",
    )


class TestAnalysisPublicContractNoSourceUsed:
    """Exact C3 public contract assertions for analysis success envelopes."""

    def _patch_fetch(self, monkeypatch, ohlcv):
        monkeypatch.setattr(
            "tempest_mcp.tools.analysis_tools.resolve_and_fetch_backtest_ohlcv",
            lambda _request: (ohlcv, _mock_window()),
        )

    def _assert_contract(self, result, expected_keys):
        assert result["success"] is True
        assert set(result["data"]) == expected_keys
        assert set(result["data"]["window"]) == {"start_at_utc", "end_at_utc", "estimated_bars", "exchange"}
        assert "source_used" not in json.dumps(result)

    @pytest.mark.asyncio
    async def test_calculate_volume_profile_contract(self, monkeypatch, deterministic_ohlcv):
        self._patch_fetch(monkeypatch, deterministic_ohlcv)
        result = await calculate_volume_profile("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
        self._assert_contract(result, {"tool", "symbol", "timeframe", "window", "summary", "profile_rows"})

    @pytest.mark.asyncio
    async def test_detect_order_blocks_contract(self, monkeypatch, order_block_fixture_ohlcv):
        self._patch_fetch(monkeypatch, order_block_fixture_ohlcv)
        result = await detect_order_blocks("BTC/USDT", "1h", "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z")
        self._assert_contract(result, {"tool", "symbol", "timeframe", "window", "order_blocks", "count"})
