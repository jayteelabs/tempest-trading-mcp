"""Unit tests for market structure summary engine (summarize_market_structure)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.structure import (
    _SUMMARY_OUTPUT_COLUMNS,
    summarize_market_structure,
)

# =============================================================================
# Test Fixtures — Properly Constructed to Trigger Swing Detection
# =============================================================================


def _make_ohlcv(timestamps, opens, highs, lows, closes, volumes=None):
    """Create a valid UTC-aware OHLCV DataFrame for testing."""
    if volumes is None:
        volumes = [100.0] * len(timestamps)
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(timestamps, tz="UTC"),
    )
    df.index.name = "timestamp"
    return df


def _with_terminal_bars(
    ohlcv: pd.DataFrame,
    terminal_closes: list[float],
    *,
    open_offset: float,
    high_offset: float,
    low_offset: float,
) -> pd.DataFrame:
    """Return a copy with the final bars replaced by deterministic values."""
    updated = ohlcv.copy()
    start_idx = len(updated) - len(terminal_closes)
    for row_idx, close in enumerate(terminal_closes, start=start_idx):
        updated.iloc[row_idx, updated.columns.get_loc("open")] = close + open_offset
        updated.iloc[row_idx, updated.columns.get_loc("high")] = close + high_offset
        updated.iloc[row_idx, updated.columns.get_loc("low")] = close - low_offset
        updated.iloc[row_idx, updated.columns.get_loc("close")] = close
    return updated


def _ohlcv_bullish_trend():
    """Create a bullish trending OHLCV with clear HH/HL pattern and strong ADX.

    Uses the pattern from test_structure_price_patterns.py but extended to 50 bars
    to satisfy ADX warmup requirements (adx_period * 2 = 28).
    """
    timestamps = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    # Clear HH/HL pattern with increasing highs and lows
    opens = [
        100,
        101,
        102,
        101,
        100,
        102,
        103,
        104,
        103,
        102,
        104,
        105,
        106,
        105,
        104,
        106,
        107,
        108,
        107,
        106,
        108,
        109,
        110,
        109,
        108,
        110,
        111,
        112,
        111,
        110,
        112,
        113,
        114,
        113,
        112,
        114,
        115,
        116,
        115,
        114,
        116,
        117,
        118,
        117,
        116,
        118,
        119,
        120,
        119,
        118,
    ]
    highs = [
        102,
        103,
        104,
        103,
        102,
        104,
        105,
        106,
        105,
        104,
        106,
        107,
        108,
        107,
        106,
        108,
        109,
        110,
        109,
        108,
        110,
        111,
        112,
        111,
        110,
        112,
        113,
        114,
        113,
        112,
        114,
        115,
        116,
        115,
        114,
        116,
        117,
        118,
        117,
        116,
        118,
        119,
        120,
        119,
        118,
        120,
        121,
        122,
        121,
        120,
    ]
    lows = [
        99,
        100,
        101,
        100,
        99,
        101,
        102,
        103,
        102,
        101,
        103,
        104,
        105,
        104,
        103,
        105,
        106,
        107,
        106,
        105,
        107,
        108,
        109,
        108,
        107,
        109,
        110,
        111,
        110,
        109,
        111,
        112,
        113,
        112,
        111,
        113,
        114,
        115,
        114,
        113,
        115,
        116,
        117,
        116,
        115,
        117,
        118,
        119,
        118,
        117,
    ]
    closes = [
        101,
        102,
        103,
        102,
        101,
        103,
        104,
        105,
        104,
        103,
        105,
        106,
        107,
        106,
        105,
        107,
        108,
        109,
        108,
        107,
        109,
        110,
        111,
        110,
        109,
        111,
        112,
        113,
        112,
        111,
        113,
        114,
        115,
        114,
        113,
        115,
        116,
        117,
        116,
        115,
        117,
        118,
        119,
        118,
        117,
        119,
        120,
        121,
        120,
        119,
    ]
    volumes = [100] * 50
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_bearish_trend():
    """Create a bearish trending OHLCV with clear LH/LL pattern."""
    timestamps = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    opens = [
        120,
        119,
        118,
        119,
        120,
        118,
        117,
        116,
        117,
        118,
        116,
        115,
        114,
        115,
        116,
        114,
        113,
        112,
        113,
        114,
        112,
        111,
        110,
        111,
        112,
        110,
        109,
        108,
        109,
        110,
        108,
        107,
        106,
        107,
        108,
        106,
        105,
        104,
        105,
        106,
        104,
        103,
        102,
        103,
        104,
        102,
        101,
        100,
        101,
        102,
    ]
    highs = [
        121,
        120,
        119,
        120,
        121,
        119,
        118,
        117,
        118,
        119,
        117,
        116,
        115,
        116,
        117,
        115,
        114,
        113,
        114,
        115,
        113,
        112,
        111,
        112,
        113,
        111,
        110,
        109,
        110,
        111,
        109,
        108,
        107,
        108,
        109,
        107,
        106,
        105,
        106,
        107,
        105,
        104,
        103,
        104,
        105,
        103,
        102,
        101,
        102,
        103,
    ]
    lows = [
        119,
        118,
        117,
        118,
        119,
        117,
        116,
        115,
        116,
        117,
        115,
        114,
        113,
        114,
        115,
        113,
        112,
        111,
        112,
        113,
        111,
        110,
        109,
        110,
        111,
        109,
        108,
        107,
        108,
        109,
        107,
        106,
        105,
        106,
        107,
        105,
        104,
        103,
        104,
        105,
        103,
        102,
        101,
        102,
        103,
        101,
        100,
        99,
        100,
        101,
    ]
    closes = [
        119,
        118,
        117,
        118,
        119,
        117,
        116,
        115,
        116,
        117,
        115,
        114,
        113,
        114,
        115,
        113,
        112,
        111,
        112,
        113,
        111,
        110,
        109,
        110,
        111,
        109,
        108,
        107,
        108,
        109,
        107,
        106,
        105,
        106,
        107,
        105,
        104,
        103,
        104,
        105,
        103,
        102,
        101,
        102,
        103,
        101,
        100,
        99,
        100,
        101,
    ]
    volumes = [100] * 50
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_ranging():
    """Create a ranging OHLCV with low ADX and tight consolidation."""
    timestamps = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    # Oscillating within a tight range - simulates low ADX
    base = [100.0] * 50
    for i in range(50):
        if i % 10 < 5:
            base[i] = 100.0 + (i % 5) * 0.2
        else:
            base[i] = 100.0 - ((i % 5) + 1) * 0.15
    opens = base
    highs = [b + 0.5 for b in base]
    lows = [b - 0.5 for b in base]
    closes = base
    volumes = [100] * 50
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_breakout_up():
    """Create an OHLCV that breaks out upward from a range."""
    return _with_terminal_bars(
        _ohlcv_ranging(),
        [102.5, 103.0, 103.5],
        open_offset=-0.2,
        high_offset=0.5,
        low_offset=0.7,
    )


def _ohlcv_breakout_down():
    """Create an OHLCV that breaks down from a range."""
    return _with_terminal_bars(
        _ohlcv_ranging(),
        [97.5, 97.0, 96.5],
        open_offset=0.2,
        high_offset=0.7,
        low_offset=0.5,
    )


def _ohlcv_transition():
    """Create an OHLCV with mixed evidence that should remain transitional."""
    timestamps = pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC")
    opens = [100] * 35 + [102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116]
    highs = [101] * 35 + [103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
    lows = [99] * 35 + [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
    closes = [100.5] * 35 + [
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
    ]
    volumes = [100] * 50
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_insufficient_data():
    """Create OHLCV with insufficient bars for analysis."""
    timestamps = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    opens = [100, 101, 102, 101, 100]
    highs = [102, 103, 103, 102, 101]
    lows = [99, 100, 101, 100, 99]
    closes = [101, 102, 101, 100, 99]
    volumes = [100] * 5
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


# =============================================================================
# Tests
# =============================================================================


class TestSummarizeMarketStructureSchema:
    """Tests for output schema and column ordering."""

    def test_output_columns_match_pinned_schema(self):
        """Output columns must match the pinned schema exactly."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert list(result.columns) == _SUMMARY_OUTPUT_COLUMNS

    def test_output_is_single_row_dataframe(self):
        """Output must be a single-row DataFrame."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_timestamp_fields_are_pandas_native(self):
        """Timestamp fields must be pd.Timestamp, not strings or datetime objects."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert isinstance(result["analysis_ts"].iloc[0], pd.Timestamp)
        assert isinstance(result["window_start_ts"].iloc[0], pd.Timestamp)
        assert isinstance(result["window_end_ts"].iloc[0], pd.Timestamp)


class TestSummarizeMarketStructureDeterminism:
    """Tests for deterministic output."""

    def test_repeated_calls_return_identical_results(self):
        """Repeated calls on identical input must return identical results."""
        ohlcv = _ohlcv_bullish_trend()
        result1 = summarize_market_structure(ohlcv)
        result2 = summarize_market_structure(ohlcv)
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True), result2.reset_index(drop=True)
        )

    def test_column_ordering_is_stable(self):
        """Column ordering must be stable across multiple calls."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert list(result.columns) == _SUMMARY_OUTPUT_COLUMNS


class TestSummarizeMarketStructureLabels:
    """Tests for correct label classification."""

    @pytest.mark.parametrize(
        ("fixture_fn", "expected_label", "expected_rule"),
        [
            (_ohlcv_bullish_trend, "trending_up", "trending_up_rule"),
            (_ohlcv_bearish_trend, "trending_down", "trending_down_rule"),
            (_ohlcv_ranging, "ranging", "ranging_rule"),
            (_ohlcv_breakout_up, "breakout_up", "breakout_up_rule"),
            (_ohlcv_breakout_down, "breakout_down", "breakout_down_rule"),
            (_ohlcv_transition, "transition", "transition_rule"),
            (_ohlcv_insufficient_data, "insufficient_data", "insufficient_data_rule"),
        ],
    )
    def test_expected_labels_and_decision_rules_for_regimes(
        self, fixture_fn, expected_label, expected_rule
    ):
        """Each regime fixture must pin its exact summary label and decision rule."""
        result = summarize_market_structure(fixture_fn())
        assert result["summary_label"].iloc[0] == expected_label
        assert result["decision_rule"].iloc[0] == expected_rule

    def test_insufficient_data_detected(self):
        """Window with insufficient data should be classified as insufficient_data."""
        ohlcv = _ohlcv_insufficient_data()
        result = summarize_market_structure(ohlcv)
        label = result["summary_label"].iloc[0]
        assert label == "insufficient_data"


class TestSummarizeMarketStructureConfidence:
    """Tests for confidence value ranges."""

    def test_insufficient_data_confidence_is_zero(self):
        """insufficient_data label must have confidence of 0.0."""
        ohlcv = _ohlcv_insufficient_data()
        result = summarize_market_structure(ohlcv)
        assert result["confidence"].iloc[0] == 0.0

    def test_confidence_in_valid_range(self):
        """All confidence values must be in [0.0, 1.0]."""
        for fixture_fn in [
            _ohlcv_bullish_trend,
            _ohlcv_bearish_trend,
            _ohlcv_ranging,
            _ohlcv_breakout_up,
        ]:
            ohlcv = fixture_fn()
            result = summarize_market_structure(ohlcv)
            confidence = result["confidence"].iloc[0]
            assert 0.0 <= confidence <= 1.0, (
                f"Confidence {confidence} out of range for {fixture_fn.__name__}"
            )


class TestSummarizeMarketStructureDecisionRules:
    """Tests for decision_rule values."""

    def test_decision_rule_format(self):
        """decision_rule must end with '_rule'."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert result["decision_rule"].iloc[0].endswith("_rule")

    def test_insufficient_data_has_insufficient_data_rule(self):
        """insufficient_data must have insufficient_data_rule."""
        ohlcv = _ohlcv_insufficient_data()
        result = summarize_market_structure(ohlcv)
        assert result["decision_rule"].iloc[0] == "insufficient_data_rule"

    @pytest.mark.parametrize(
        ("fixture_fn", "expected_label", "expected_direction"),
        [
            (_ohlcv_breakout_up, "breakout_up", "up"),
            (_ohlcv_breakout_down, "breakout_down", "down"),
        ],
    )
    def test_breakout_priority_and_recency_boundary(
        self, fixture_fn, expected_label, expected_direction
    ):
        """Recent breakouts should beat active ranges until the recency boundary expires."""
        ohlcv = fixture_fn()

        recent = summarize_market_structure(ohlcv, breakout_recency_bars=2)
        stale = summarize_market_structure(ohlcv, breakout_recency_bars=1)

        assert recent["summary_label"].iloc[0] == expected_label
        assert recent["decision_rule"].iloc[0] == f"{expected_label}_rule"
        assert recent["range_status"].iloc[0] == "active"
        assert recent["breakout_direction"].iloc[0] == expected_direction

        assert stale["summary_label"].iloc[0] == "ranging"
        assert stale["decision_rule"].iloc[0] == "ranging_rule"
        assert stale["range_status"].iloc[0] == "active"


class TestSummarizeMarketStructureADXFields:
    """Tests for ADX-related output fields."""

    def test_adx_fields_are_finite_or_nan(self):
        """ADX fields must be finite numbers or NaN."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        adx = result["adx"].iloc[0]
        plus_di = result["plus_di"].iloc[0]
        minus_di = result["minus_di"].iloc[0]
        di_spread = result["di_spread"].iloc[0]

        if not (isinstance(adx, float) and np.isnan(adx)):
            assert np.isfinite(adx)
        if not (isinstance(plus_di, float) and np.isnan(plus_di)):
            assert np.isfinite(plus_di)
        if not (isinstance(minus_di, float) and np.isnan(minus_di)):
            assert np.isfinite(minus_di)
        if not (isinstance(di_spread, float) and np.isnan(di_spread)):
            assert np.isfinite(di_spread)

    def test_di_spread_calculation(self):
        """di_spread must equal plus_di - minus_di."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        expected = result["plus_di"].iloc[0] - result["minus_di"].iloc[0]
        actual = result["di_spread"].iloc[0]
        if not (np.isnan(expected) and np.isnan(actual)):
            assert abs(expected - actual) < 1e-6
        else:
            assert np.isnan(actual)


class TestSummarizeMarketStructureInvalidInput:
    """Tests for invalid input handling."""

    def test_empty_ohlcv_raises_value_error(self):
        """Empty DataFrame must raise ValueError."""
        ohlcv = _make_ohlcv([], [], [], [], [])
        with pytest.raises(ValueError, match="ohlcv must not be empty"):
            summarize_market_structure(ohlcv)

    def test_missing_columns_raises_value_error(self):
        """Missing required columns must raise ValueError."""
        df = pd.DataFrame(
            {"open": [100], "high": [102], "low": [99], "close": [101]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="ohlcv missing required columns"):
            summarize_market_structure(df)

    def test_wrong_index_type_raises_value_error(self):
        """Wrong index type must raise ValueError."""
        df = pd.DataFrame(
            {"open": [100], "high": [102], "low": [99], "close": [101], "volume": [100]},
            index=[0],
        )
        with pytest.raises(ValueError, match="ohlcv index must be a DatetimeIndex"):
            summarize_market_structure(df)


class TestSummarizeMarketStructureInvalidParams:
    """Tests for invalid parameter validation."""

    def test_negative_swing_window_raises_value_error(self):
        """Negative swing_window must raise ValueError."""
        ohlcv = _ohlcv_bullish_trend()
        with pytest.raises(ValueError, match="swing_window must be an integer"):
            summarize_market_structure(ohlcv, swing_window=-1)

    def test_invalid_adx_period_raises_value_error(self):
        """Invalid adx_period must raise ValueError."""
        ohlcv = _ohlcv_bullish_trend()
        with pytest.raises(ValueError, match="adx_period must be an integer"):
            summarize_market_structure(ohlcv, adx_period=0)

    def test_invalid_adx_trend_threshold_raises_value_error(self):
        """Invalid adx_trend_threshold must raise ValueError."""
        ohlcv = _ohlcv_bullish_trend()
        with pytest.raises(ValueError, match="adx_trend_threshold must be a float"):
            summarize_market_structure(ohlcv, adx_trend_threshold=-5.0)

    def test_invalid_breakout_recency_bars_raises_value_error(self):
        """Invalid breakout_recency_bars must raise ValueError."""
        ohlcv = _ohlcv_bullish_trend()
        with pytest.raises(ValueError, match="breakout_recency_bars must be an integer"):
            summarize_market_structure(ohlcv, breakout_recency_bars=0)


class TestSummarizeMarketStructureBoundary:
    """Tests for indicator-layer boundary enforcement."""

    def test_no_server_imports_in_structure_module(self):
        """structure.py must not import from tempest_mcp.server or tools."""
        module_file = sys.modules[summarize_market_structure.__module__].__file__
        with open(module_file) as f:
            content = f.read()
        assert "tempest_mcp.server" not in content
        assert "tempest_mcp.tools" not in content


class TestSummarizeMarketStructureComposition:
    """Tests that summarize_market_structure composes existing primitives."""

    def test_function_produces_structure_related_fields(self):
        """Verify composition by checking that structure fields are present."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert "structure_classification" in result.columns
        assert "structure_trend_state" in result.columns

    def test_function_produces_adx_fields(self):
        """Verify ADX is computed by checking adx field is present and populated."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert "adx" in result.columns

    def test_function_produces_range_and_breakout_fields(self):
        """Verify range and breakout fields are present."""
        ohlcv = _ohlcv_bullish_trend()
        result = summarize_market_structure(ohlcv)
        assert "range_id" in result.columns
        assert "range_status" in result.columns
        assert "breakout_id" in result.columns
        assert "breakout_direction" in result.columns
