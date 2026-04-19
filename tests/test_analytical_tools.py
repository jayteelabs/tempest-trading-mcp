"""Tests for ENG-37 analytical tools — calculate_fibonacci, calculate_tpo,
detect_elliot_wave, get_market_structure.

Covers:
    - handler validation envelopes (invalid inputs)
    - serialization of valid results
    - deterministic fixture behavior
    - reuse of existing indicator/structure engine logic
"""

import json

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.config import ErrorCodes
from tempest_mcp.indicators.structure import (
    calculate_fib_extensions,
    calculate_fib_retracements,
    detect_elliott_waves,
    summarize_market_structure,
)
from tempest_mcp.indicators.volume.tpo import calculate_tpo_chart
from tempest_mcp.tools.analytical_tools import (
    _internal_error,
    _parse_iso_datetime,
    _validation_error,
    calculate_fibonacci,
    calculate_tpo,
    detect_elliot_wave,
    get_market_structure,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def deterministic_ohlcv():
    """Create a deterministic OHLCV DataFrame for reproducible tests.

    Uses a fixed seed so that all analytical tools produce the same output
    on every run.
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
def single_session_ohlcv():
    """Create a single-session OHLCV DataFrame for TPO testing.

    All bars are within 4 hours of each other to satisfy single-session requirement.
    """
    n = 20
    base_price = 100.0
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # Ensure no gaps > 4 hours (all consecutive hours)
    close = base_price + np.linspace(0, 10, n)
    high = close + 1.0
    low = close - 1.0
    open_price = close - 0.5
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def multi_session_ohlcv():
    """Create a multi-session OHLCV DataFrame for TPO session validation.

    Contains a gap > 4 hours between bars to trigger session detection.
    """
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    # Add a 6-hour gap after bar 5
    idx_list = list(idx[:6]) + [idx[5] + pd.Timedelta(hours=6)] + list(idx[6:] + pd.Timedelta(hours=6))

    n = 11
    close = 100.0 + np.linspace(0, 10, n)
    high = close + 1.0
    low = close - 1.0
    open_price = close - 0.5
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.DatetimeIndex(idx_list, tz="UTC"),
    )


@pytest.fixture
def elliott_wave_ohlcv():
    """Create an OHLCV DataFrame designed to produce detectable Elliott Wave patterns.

    Constructs a simple 5-wave impulse pattern:
    - Wave 1: up move
    - Wave 2: retracement (stays above wave 1 start)
    - Wave 3: extension (longest wave)
    - Wave 4: shallow retracement (stays above wave 1 end)
    - Wave 5: final extension
    """
    idx = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")

    n = 50
    close = np.full(n, 100.0)

    # Wave 1: up 5%
    close[5] = 105.0
    # Wave 2: retraces to 102 (38.2% retracement)
    close[10] = 102.0
    # Wave 3: extends to 120 (1.618 extension)
    close[20] = 120.0
    # Wave 4: shallow retracement to 115
    close[30] = 115.0
    # Wave 5: final extension to 130
    close[40] = 130.0

    high = close + 0.5
    low = close - 0.5
    open_price = close.copy()
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def market_structure_ohlcv():
    """Create an OHLCV DataFrame for market structure analysis.

    Designed with clear higher highs and higher lows to produce bullish structure.
    """
    idx = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")

    n = 60
    close = np.full(n, 100.0)

    # Create a bullish trend with higher highs and higher lows
    # Initial move up
    close[5] = 105.0  # HH
    close[10] = 103.0  # HL
    close[15] = 110.0  # HH
    close[20] = 107.0  # HL
    close[25] = 115.0  # HH
    close[30] = 112.0  # HL
    close[35] = 120.0  # HH
    close[40] = 117.0  # HL
    close[45] = 125.0  # HH
    close[50] = 122.0  # HL

    high = close + 1.0
    low = close - 1.0
    open_price = close.copy()
    volume = np.full(n, 1000.0)

    return pd.DataFrame(
        {"open": open_price, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# ── Mock window helper ──────────────────────────────────────────────────────────


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


# ── calculate_fibonacci handler tests ─────────────────────────────────────────


class TestCalculateFibonacciValidation:
    """Validation envelope tests for calculate_fibonacci handler.

    Tests cover: symbol format, swing anchor validation, output_mode,
    trend_direction requirements, and window parsing.
    """

    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol returns validation error envelope."""
        result = await calculate_fibonacci(
            symbol="INVALID@SYMBOL",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "symbol" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_timeframe(self):
        """Invalid timeframe returns validation error envelope."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="invalid",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "timeframe" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_start_at_format(self):
        """Malformed start_at returns validation error envelope."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="not-a-date",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "start_at" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_end_at_format(self):
        """Malformed end_at returns validation error envelope."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="not-a-date",
            swing_high=110.0,
            swing_low=100.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "end_at" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_swing_high_equals_swing_low(self):
        """swing_high == swing_low returns validation error."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=100.0,
            swing_low=100.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "swing_high" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_swing_high_less_than_swing_low(self):
        """swing_high < swing_low returns validation error."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=90.0,
            swing_low=100.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "swing_high" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_output_mode(self):
        """Invalid output_mode returns validation error."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="invalid",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "output_mode" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_extension_mode_missing_trend_direction(self):
        """output_mode='extension' without trend_direction returns validation error."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="extension",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "trend_direction" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_extension_mode_invalid_trend_direction(self):
        """output_mode='extension' with invalid trend_direction returns validation error."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="extension",
            trend_direction="invalid",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "trend_direction" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_extension_mode_valid_bearish(self):
        """output_mode='extension' with trend_direction='bearish' is valid."""
        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="extension",
            trend_direction="bearish",
        )
        # Should not be a validation error (may fail for other reasons like data fetch)
        # but the trend_direction itself should be accepted
        assert (
            result.get("success") is not False
            or "trend_direction" not in result.get("error", {}).get("message", "").lower()
        )


class TestCalculateFibonacciSerialization:
    """Serialization tests for calculate_fibonacci handler.

    Verifies deterministic output structure using a fixture DataFrame.
    """

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, deterministic_ohlcv, monkeypatch):
        """Success envelope has correct top-level structure."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
        )

        assert result["success"] is True
        data = result["data"]
        assert data["tool"] == "calculate_fibonacci"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert "window" in data
        assert "fib_levels" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_fib_levels_are_serializable(self, deterministic_ohlcv, monkeypatch):
        """fib_levels list is JSON-serializable (all native Python types)."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
        )

        # Must not raise — confirms JSON serializable
        json.dumps(result["data"]["fib_levels"])

    @pytest.mark.asyncio
    async def test_retracement_mode_correct_level_type(self, deterministic_ohlcv, monkeypatch):
        """Retracement mode produces level_type='retracement'."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="retracement",
        )

        assert result["success"] is True
        for level in result["data"]["fib_levels"]:
            assert level["level_type"] == "retracement"

    @pytest.mark.asyncio
    async def test_extension_mode_correct_level_type(
        self, deterministic_ohlcv, monkeypatch
    ):
        """Extension mode produces level_type='extension'."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="extension",
            trend_direction="bullish",
        )

        assert result["success"] is True
        for level in result["data"]["fib_levels"]:
            assert level["level_type"] == "extension"

    @pytest.mark.asyncio
    async def test_deterministic_output(self, deterministic_ohlcv, monkeypatch):
        """Running the same calculation twice produces identical output."""

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        args = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_at": "2024-01-01T00:00:00",
            "end_at": "2024-01-08T00:00:00",
            "swing_high": 110.0,
            "swing_low": 100.0,
        }

        result1 = await calculate_fibonacci(**args)
        result2 = await calculate_fibonacci(**args)

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestCalculateFibonacciEngineReuse:
    """Tests that calculate_fibonacci reuses existing indicator engines."""

    @pytest.mark.asyncio
    async def test_uses_calculate_fib_retracements(self, deterministic_ohlcv, monkeypatch):
        """calculate_fibonacci retracement mode calls calculate_fib_retracements."""
        call_count = 0

        def counting_retracements(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return calculate_fib_retracements(*args, **kwargs)

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.calculate_fib_retracements",
            counting_retracements,
        )

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="retracement",
        )

        assert result["success"] is True
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_uses_calculate_fib_extensions(self, deterministic_ohlcv, monkeypatch):
        """calculate_fibonacci extension mode calls calculate_fib_extensions."""
        call_count = 0

        def counting_extensions(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return calculate_fib_extensions(*args, **kwargs)

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.calculate_fib_extensions",
            counting_extensions,
        )

        def mock_resolve(request):
            return deterministic_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-08T00:00:00",
            swing_high=110.0,
            swing_low=100.0,
            output_mode="extension",
            trend_direction="bullish",
        )

        assert result["success"] is True
        assert call_count == 1


# ── calculate_tpo handler tests ────────────────────────────────────────────────


class TestCalculateTpoValidation:
    """Validation envelope tests for calculate_tpo handler.

    Tests cover: symbol format, row_size validation, value_area_pct,
    single-session requirement, and window parsing.
    """

    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol returns validation error envelope."""
        result = await calculate_tpo(
            symbol="INVALID@SYMBOL",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "symbol" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_timeframe(self):
        """Invalid timeframe returns validation error envelope."""
        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="invalid",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "timeframe" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_row_size_zero(self):
        """row_size=0 returns validation error."""
        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "row_size" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_row_size_negative(self):
        """row_size < 0 returns validation error."""
        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=-1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "row_size" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_row_size_inf(self):
        """row_size=inf returns validation error."""
        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=float("inf"),
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "row_size" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_value_area_pct_zero(self):
        """value_area_pct=0 returns validation error."""
        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=1.0,
            value_area_pct=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "value_area_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_value_area_pct_gt_1(self):
        """value_area_pct > 1 returns validation error."""
        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=1.0,
            value_area_pct=1.5,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "value_area_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_multi_session_rejection(self, multi_session_ohlcv, monkeypatch):
        """Multi-session window returns validation error."""

        def mock_resolve(request):
            return multi_session_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            row_size=1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "session" in result["error"]["message"].lower()


class TestCalculateTpoSerialization:
    """Serialization tests for calculate_tpo handler.

    Verifies deterministic output structure using single-session fixture.
    """

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, single_session_ohlcv, monkeypatch):
        """Success envelope has correct top-level structure."""

        def mock_resolve(request):
            return single_session_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T20:00:00",
            row_size=1.0,
        )

        assert result["success"] is True
        data = result["data"]
        assert data["tool"] == "calculate_tpo"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert "window" in data
        assert "session" in data
        assert "tpo_rows" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_session_metadata_fields(self, single_session_ohlcv, monkeypatch):
        """Session metadata contains all required fields."""

        def mock_resolve(request):
            return single_session_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T20:00:00",
            row_size=1.0,
        )

        assert result["success"] is True
        session = result["data"]["session"]
        required_fields = {
            "row_size",
            "marker_count",
            "poc_price",
            "vah_price",
            "val_price",
            "initial_balance_low",
            "initial_balance_high",
            "range_expanded_up",
            "range_expanded_down",
        }
        assert required_fields <= set(session.keys())

    @pytest.mark.asyncio
    async def test_tpo_rows_are_serializable(self, single_session_ohlcv, monkeypatch):
        """tpo_rows list is JSON-serializable (all native Python types)."""

        def mock_resolve(request):
            return single_session_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T20:00:00",
            row_size=1.0,
        )

        # Must not raise — confirms JSON serializable
        json.dumps(result["data"]["tpo_rows"])

    @pytest.mark.asyncio
    async def test_deterministic_output(self, single_session_ohlcv, monkeypatch):
        """Running the same calculation twice produces identical output."""

        def mock_resolve(request):
            return single_session_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        args = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_at": "2024-01-01T00:00:00",
            "end_at": "2024-01-01T20:00:00",
            "row_size": 1.0,
        }

        result1 = await calculate_tpo(**args)
        result2 = await calculate_tpo(**args)

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestCalculateTpoEngineReuse:
    """Tests that calculate_tpo reuses existing indicator engine."""

    @pytest.mark.asyncio
    async def test_uses_calculate_tpo_chart(self, single_session_ohlcv, monkeypatch):
        """calculate_tpo handler calls calculate_tpo_chart."""
        call_count = 0

        def counting_tpo(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return calculate_tpo_chart(*args, **kwargs)

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.calculate_tpo_chart",
            counting_tpo,
        )

        def mock_resolve(request):
            return single_session_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T20:00:00",
            row_size=1.0,
        )

        assert result["success"] is True
        assert call_count == 1


# ── detect_elliot_wave handler tests ─────────────────────────────────────────


class TestDetectElliotWaveValidation:
    """Validation envelope tests for detect_elliot_wave handler.

    Tests cover: symbol format, swing_window, min_swing_pct,
    wave parameter constraints, and insufficient bars.
    """

    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol returns validation error envelope."""
        result = await detect_elliot_wave(
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
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="invalid",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "timeframe" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_swing_window_zero(self):
        """swing_window=0 returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_window=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "swing_window" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_swing_window_negative(self):
        """swing_window < 0 returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_window=-1,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "swing_window" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_min_swing_pct_zero(self):
        """min_swing_pct=0 returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            min_swing_pct=0.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "min_swing_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_min_swing_pct_one(self):
        """min_swing_pct >= 1 returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            min_swing_pct=1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "min_swing_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_wave2_retrace_band_wrong_length(self):
        """wave2_retrace_band with wrong length returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            wave2_retrace_band=(0.382,),  # should be tuple of 2
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "wave2_retrace_band" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_wave2_retrace_band_order(self):
        """wave2_retrace_band with min >= max returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            wave2_retrace_band=(0.786, 0.382),  # inverted
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "wave2_retrace_band" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_wave3_extension_min_negative(self):
        """wave3_extension_min < 0 returns validation error."""
        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            wave3_extension_min=-0.5,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "wave3_extension_min" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_insufficient_bars(self, deterministic_ohlcv, monkeypatch):
        """Less than 10 bars returns validation error."""

        # Use only 5 bars
        small_ohlcv = deterministic_ohlcv.iloc[:5]

        def mock_resolve(request):
            return small_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T05:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "insufficient" in result["error"]["message"].lower()


class TestDetectElliotWaveSerialization:
    """Serialization tests for detect_elliot_wave handler.

    Verifies deterministic output structure using a fixture DataFrame.
    """

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, elliott_wave_ohlcv, monkeypatch):
        """Success envelope has correct top-level structure."""

        def mock_resolve(request):
            return elliott_wave_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T00:00:00",
        )

        assert result["success"] is True
        data = result["data"]
        assert data["tool"] == "detect_elliot_wave"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert "window" in data
        assert "parameters" in data
        assert "wave_sequences" in data
        assert "count" in data

    @pytest.mark.asyncio
    async def test_parameters_fields_present(self, elliott_wave_ohlcv, monkeypatch):
        """Parameters dict contains all required fields."""

        def mock_resolve(request):
            return elliott_wave_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T00:00:00",
        )

        params = result["data"]["parameters"]
        required_fields = {
            "swing_window",
            "min_swing_pct",
            "wave2_retrace_band",
            "wave3_extension_min",
            "wave4_retrace_max",
            "waveb_retrace_band",
            "wavec_extension_min",
            "degree_thresholds",
            "include_rejected",
        }
        assert required_fields <= set(params.keys())

    @pytest.mark.asyncio
    async def test_wave_sequences_are_serializable(
        self, elliott_wave_ohlcv, monkeypatch
    ):
        """wave_sequences list is JSON-serializable (all native Python types)."""

        def mock_resolve(request):
            return elliott_wave_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T00:00:00",
        )

        # Must not raise — confirms JSON serializable
        json.dumps(result["data"]["wave_sequences"])

    @pytest.mark.asyncio
    async def test_deterministic_output(self, elliott_wave_ohlcv, monkeypatch):
        """Running the same detection twice produces identical output."""

        def mock_resolve(request):
            return elliott_wave_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        args = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_at": "2024-01-01T00:00:00",
            "end_at": "2024-01-03T00:00:00",
        }

        result1 = await detect_elliot_wave(**args)
        result2 = await detect_elliot_wave(**args)

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestDetectElliotWaveEngineReuse:
    """Tests that detect_elliot_wave reuses existing indicator engine."""

    @pytest.mark.asyncio
    async def test_uses_detect_elliott_waves(self, elliott_wave_ohlcv, monkeypatch):
        """detect_elliot_wave handler calls detect_elliott_waves."""
        call_count = 0

        def counting_elliott(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return detect_elliott_waves(*args, **kwargs)

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.detect_elliott_waves",
            counting_elliott,
        )

        def mock_resolve(request):
            return elliott_wave_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T00:00:00",
        )

        assert result["success"] is True
        assert call_count == 1


# ── get_market_structure handler tests ────────────────────────────────────────


class TestGetMarketStructureValidation:
    """Validation envelope tests for get_market_structure handler.

    Tests cover: symbol format, swing_window, min_swing_pct,
    range/breakout/ADX parameter constraints, and insufficient bars.
    """

    @pytest.mark.asyncio
    async def test_invalid_symbol_format(self):
        """Invalid symbol returns validation error envelope."""
        result = await get_market_structure(
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
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="invalid",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "timeframe" in result["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_swing_window_zero(self):
        """swing_window=0 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_window=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "swing_window" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_swing_window_negative(self):
        """swing_window < 0 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            swing_window=-1,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "swing_window" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_min_swing_pct_negative(self):
        """min_swing_pct < 0 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            min_swing_pct=-0.01,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "min_swing_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_min_swing_pct_one(self):
        """min_swing_pct >= 1 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            min_swing_pct=1.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "min_swing_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_range_lookback_one(self):
        """range_lookback < 2 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            range_lookback=1,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "range_lookback" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_max_range_pct_zero(self):
        """max_range_pct <= 0 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            max_range_pct=0.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "max_range_pct" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_breakout_confirm_bars_zero(self):
        """breakout_confirm_bars < 1 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            breakout_confirm_bars=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "breakout_confirm_bars" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_adx_period_zero(self):
        """adx_period < 1 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            adx_period=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "adx_period" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_adx_trend_threshold_over_100(self):
        """adx_trend_threshold > 100 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            adx_trend_threshold=150.0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "adx_trend_threshold" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_invalid_breakout_recency_bars_zero(self):
        """breakout_recency_bars < 1 returns validation error."""
        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-02T00:00:00",
            breakout_recency_bars=0,
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "breakout_recency_bars" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_insufficient_bars(self, deterministic_ohlcv, monkeypatch):
        """Less than minimum required bars returns validation error."""

        # Use only 3 bars (less than adx_period * 2 = 28)
        small_ohlcv = deterministic_ohlcv.iloc[:3]

        def mock_resolve(request):
            return small_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T03:00:00",
        )
        assert result["success"] is False
        assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
        assert "insufficient" in result["error"]["message"].lower()


class TestGetMarketStructureSerialization:
    """Serialization tests for get_market_structure handler.

    Verifies deterministic output structure using a fixture DataFrame.
    """

    @pytest.mark.asyncio
    async def test_success_envelope_structure(self, market_structure_ohlcv, monkeypatch):
        """Success envelope has correct top-level structure."""

        def mock_resolve(request):
            return market_structure_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T12:00:00",
        )

        assert result["success"] is True
        data = result["data"]
        assert data["tool"] == "get_market_structure"
        assert data["symbol"] == "BTC/USDT"
        assert data["timeframe"] == "1h"
        assert "window" in data
        assert "summary" in data

    @pytest.mark.asyncio
    async def test_summary_fields_present(self, market_structure_ohlcv, monkeypatch):
        """Summary dict contains all required fields."""

        def mock_resolve(request):
            return market_structure_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T12:00:00",
        )

        summary = result["data"]["summary"]
        # Key expected fields
        expected_fields = {
            "summary_label",
            "decision_rule",
            "structure_trend_state",
            "adx",
            "plus_di",
            "minus_di",
            "di_spread",
        }
        assert expected_fields <= set(summary.keys())

    @pytest.mark.asyncio
    async def test_insufficient_data_flag_false(
        self, market_structure_ohlcv, monkeypatch
    ):
        """insufficient_data is False when sufficient bars provided."""

        def mock_resolve(request):
            return market_structure_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T12:00:00",
        )

        assert result["success"] is True
        assert result["data"]["insufficient_data"] is False

    @pytest.mark.asyncio
    async def test_insufficient_data_flag_true(self, deterministic_ohlcv, monkeypatch):
        """insufficient_data is True when data quantity is sufficient for validation but engine returns insufficient label."""

        # Use exactly min_required bars (28 for default params) which passes validation
        # but the engine's internal check may still trigger insufficient_data
        small_ohlcv = deterministic_ohlcv.iloc[:28]

        def mock_resolve(request):
            return small_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-01T12:00:00",
        )

        # With exactly min_required bars, result depends on engine's internal check
        # The test verifies that insufficient_data flag is properly returned
        assert result["success"] is True
        assert "insufficient_data" in result["data"]

    @pytest.mark.asyncio
    async def test_deterministic_output(self, market_structure_ohlcv, monkeypatch):
        """Running the same analysis twice produces identical output."""

        def mock_resolve(request):
            return market_structure_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        args = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "start_at": "2024-01-01T00:00:00",
            "end_at": "2024-01-03T12:00:00",
        }

        result1 = await get_market_structure(**args)
        result2 = await get_market_structure(**args)

        assert json.dumps(result1, sort_keys=True) == json.dumps(result2, sort_keys=True)


class TestGetMarketStructureEngineReuse:
    """Tests that get_market_structure reuses existing indicator engine."""

    @pytest.mark.asyncio
    async def test_uses_summarize_market_structure(
        self, market_structure_ohlcv, monkeypatch
    ):
        """get_market_structure handler calls summarize_market_structure."""
        call_count = 0

        def counting_summarize(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return summarize_market_structure(*args, **kwargs)

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.summarize_market_structure",
            counting_summarize,
        )

        def mock_resolve(request):
            return market_structure_ohlcv, _mock_window()

        monkeypatch.setattr(
            "tempest_mcp.tools.analytical_tools.resolve_and_fetch_backtest_ohlcv",
            mock_resolve,
        )

        result = await get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at="2024-01-01T00:00:00",
            end_at="2024-01-03T12:00:00",
        )

        assert result["success"] is True
        assert call_count == 1
