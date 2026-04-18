"""Unit tests for price pattern engine (swing points, market structure, range, breakout)."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.structure import (
    classify_market_structure,
    detect_price_ranges,
    detect_range_breakouts,
    detect_swing_points,
)

# =============================================================================
# Test Fixtures
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


def _make_swings(rows):
    """Create a swing DataFrame with the pinned public schema."""
    return pd.DataFrame(rows)[
        [
            "swing_id",
            "pivot_index",
            "swing_type",
            "pivot_ts",
            "pivot_price",
            "leg_start_ts",
            "leg_start_price",
            "leg_end_ts",
            "leg_end_price",
            "price_delta",
            "pct_delta",
        ]
    ]


def _ohlcv_bullish_sequence():
    """Create a simple bullish OHLCV sequence with clear swings (HH/HL pattern)."""
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    # Higher highs: 102, 104, 106, 108 (increasing)
    # Higher lows: 99, 100, 101, 102 (increasing)
    opens = [
        100,
        101,
        102,
        101,
        100,
        102,
        103,
        102,
        101,
        103,
        104,
        103,
        102,
        104,
        105,
        104,
        103,
        105,
        106,
        105,
    ]
    highs = [
        102,
        103,
        104,
        103,
        102,
        104,
        105,
        104,
        103,
        105,
        106,
        105,
        104,
        106,
        107,
        106,
        105,
        107,
        108,
        107,
    ]
    lows = [
        99,
        100,
        101,
        100,
        99,
        101,
        102,
        101,
        100,
        102,
        103,
        102,
        101,
        103,
        104,
        103,
        102,
        104,
        105,
        104,
    ]
    closes = [
        101,
        102,
        103,
        102,
        101,
        103,
        104,
        103,
        102,
        104,
        105,
        104,
        103,
        105,
        106,
        105,
        104,
        106,
        107,
        106,
    ]
    volumes = [100] * 20
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_bearish_sequence():
    """Create a simple bearish OHLCV sequence with LH/LL pattern."""
    timestamps = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    # Lower highs: 108, 106, 104, 102 (decreasing)
    # Lower lows: 102, 101, 100, 99 (decreasing)
    opens = [
        106,
        105,
        104,
        105,
        106,
        104,
        103,
        104,
        105,
        103,
        102,
        103,
        104,
        102,
        101,
        102,
        103,
        101,
        100,
        101,
    ]
    highs = [
        108,
        107,
        106,
        107,
        108,
        106,
        105,
        106,
        107,
        105,
        104,
        105,
        106,
        104,
        103,
        104,
        105,
        103,
        102,
        103,
    ]
    lows = [102, 101, 100, 101, 102, 100, 99, 100, 101, 99, 98, 99, 100, 98, 97, 98, 99, 97, 96, 97]
    closes = [
        105,
        104,
        103,
        104,
        105,
        103,
        102,
        103,
        104,
        102,
        101,
        102,
        103,
        101,
        100,
        101,
        102,
        100,
        99,
        100,
    ]
    volumes = [100] * 20
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_equal_pivots():
    """Create OHLCV with equal highs and lows (EH/EL test)."""
    timestamps = pd.date_range("2024-01-01", periods=15, freq="h", tz="UTC")
    # Highs: 103, 103, 102 (two equal highs at start)
    # Lows: 97, 97, 98 (two equal lows at start)
    opens = [100, 101, 102, 102, 101, 100, 101, 102, 102, 101, 100, 101, 102, 102, 101]
    highs = [103, 103, 102, 103, 102, 101, 102, 103, 103, 102, 101, 102, 103, 102, 101]
    lows = [97, 97, 98, 97, 98, 99, 98, 97, 97, 98, 99, 98, 97, 98, 99]
    closes = [102, 103, 102, 102, 101, 100, 101, 102, 103, 102, 101, 100, 101, 102, 101]
    volumes = [100] * 15
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


def _ohlcv_range_bound():
    """Create OHLCV that stays in a range then breaks out."""
    timestamps = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    # First 20 bars: range between 99-101
    # Last 10 bars: breaks above 101
    opens = []
    highs = []
    lows = []
    closes = []
    for i in range(30):
        if i < 20:
            opens.append(100)
            highs.append(101)
            lows.append(99)
            closes.append(100 + (i % 3 - 1) * 0.5)
        else:
            opens.append(101 + (i - 20) * 0.2)
            highs.append(102 + (i - 20) * 0.3)
            lows.append(100 + (i - 20) * 0.1)
            closes.append(102 + (i - 20) * 0.25)
    volumes = [100] * 30
    return _make_ohlcv(timestamps, opens, highs, lows, closes, volumes)


# =============================================================================
# detect_swing_points Tests
# =============================================================================


class TestDetectSwingPoints:
    """Tests for detect_swing_points function."""

    def test_return_type(self):
        """Test swing detection returns a DataFrame."""
        ohlcv = _ohlcv_bullish_sequence()
        result = detect_swing_points(ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_schema_columns(self):
        """Test swing DataFrame has correct columns in exact order."""
        ohlcv = _ohlcv_bullish_sequence()
        result = detect_swing_points(ohlcv)
        expected_columns = [
            "swing_id",
            "pivot_index",
            "swing_type",
            "pivot_ts",
            "pivot_price",
            "leg_start_ts",
            "leg_start_price",
            "leg_end_ts",
            "leg_end_price",
            "price_delta",
            "pct_delta",
        ]
        assert list(result.columns) == expected_columns

    def test_empty_ohlcv_raises(self):
        """Test ValueError when OHLCV is empty."""
        ohlcv = _make_ohlcv([], [], [], [], [])
        with pytest.raises(ValueError, match="ohlcv must not be empty"):
            detect_swing_points(ohlcv)

    def test_missing_columns_raises(self):
        """Test ValueError when OHLCV is missing required columns."""
        ohlcv = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100]},
            index=pd.DatetimeIndex(["2024-01-01"], tz="UTC"),
        )
        with pytest.raises(ValueError, match="ohlcv missing required columns"):
            detect_swing_points(ohlcv)

    def test_naive_index_raises(self):
        """Test ValueError when OHLCV index is tz-naive."""
        ohlcv = pd.DataFrame(
            {"open": [100], "high": [101], "low": [99], "close": [100], "volume": [100]},
            index=pd.DatetimeIndex(["2024-01-01"]),
        )
        with pytest.raises(ValueError, match="ohlcv index must be UTC-aware"):
            detect_swing_points(ohlcv)

    def test_duplicate_index_raises(self):
        """Test ValueError when OHLCV has duplicate timestamps."""
        idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01"], tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
                "volume": [100, 100],
            },
            index=idx,
        )
        with pytest.raises(ValueError, match="ohlcv index must not have duplicates"):
            detect_swing_points(ohlcv)

    def test_invalid_swing_window_raises(self):
        """Test ValueError when swing_window is invalid."""
        ohlcv = _ohlcv_bullish_sequence()
        with pytest.raises(ValueError, match="swing_window must be an integer"):
            detect_swing_points(ohlcv, swing_window=0)
        with pytest.raises(ValueError, match="swing_window must be an integer"):
            detect_swing_points(ohlcv, swing_window=-1)

    def test_invalid_min_swing_pct_raises(self):
        """Test ValueError when min_swing_pct is invalid."""
        ohlcv = _ohlcv_bullish_sequence()
        with pytest.raises(ValueError, match="min_swing_pct must be a float"):
            detect_swing_points(ohlcv, min_swing_pct=-0.1)
        with pytest.raises(ValueError, match="min_swing_pct must be a float"):
            detect_swing_points(ohlcv, min_swing_pct=1.5)

    def test_deterministic_ordering(self):
        """Test swings are ordered by pivot_index ASC."""
        ohlcv = _ohlcv_bullish_sequence()
        result = detect_swing_points(ohlcv)
        if len(result) > 1:
            pivot_indices = result["pivot_index"].tolist()
            assert pivot_indices == sorted(pivot_indices)

    def test_swing_type_values(self):
        """Test swing_type contains only 'high' or 'low'."""
        ohlcv = _ohlcv_bullish_sequence()
        result = detect_swing_points(ohlcv)
        assert set(result["swing_type"].unique()).issubset({"high", "low"})

    def test_timestamps_are_pandas_native(self):
        """Test timestamp fields remain pandas-native (not stringified)."""
        ohlcv = _ohlcv_bullish_sequence()
        result = detect_swing_points(ohlcv)
        if len(result) > 0:
            assert isinstance(result["pivot_ts"].iloc[0], pd.Timestamp)
            assert isinstance(result["leg_start_ts"].iloc[0], pd.Timestamp)


# =============================================================================
# classify_market_structure Tests
# =============================================================================


class TestClassifyMarketStructure:
    """Tests for classify_market_structure function."""

    def test_return_type(self):
        """Test classification returns a DataFrame."""
        ohlcv = _ohlcv_bullish_sequence()
        swings = detect_swing_points(ohlcv)
        result = classify_market_structure(swings)
        assert isinstance(result, pd.DataFrame)

    def test_schema_columns(self):
        """Test classification DataFrame has correct columns in exact order."""
        ohlcv = _ohlcv_bullish_sequence()
        swings = detect_swing_points(ohlcv)
        result = classify_market_structure(swings)
        expected_columns = [
            "event_id",
            "event_ts",
            "swing_id",
            "swing_type",
            "classification",
            "reference_swing_id",
            "reference_price",
            "current_price",
            "price_delta",
            "pct_delta",
            "trend_state",
        ]
        assert list(result.columns) == expected_columns

    def test_empty_swings_returns_empty_df(self):
        """Test empty swings returns DataFrame with pinned columns."""
        swings = detect_swing_points(_ohlcv_bullish_sequence().head(3))
        result = classify_market_structure(swings)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_invalid_swings_df_raises(self):
        """Test ValueError when swings DataFrame is invalid."""
        with pytest.raises(ValueError, match="swings must be a pandas DataFrame"):
            classify_market_structure("not a dataframe")

    def test_classification_values(self):
        """Test classification contains only valid values."""
        ohlcv = _ohlcv_bullish_sequence()
        swings = detect_swing_points(ohlcv)
        result = classify_market_structure(swings)
        valid_classes = {"HH", "HL", "LH", "LL", "EH", "EL"}
        if len(result) > 0:
            assert set(result["classification"].unique()).issubset(valid_classes)

    def test_trend_state_values(self):
        """Test trend_state contains only valid values."""
        ohlcv = _ohlcv_bullish_sequence()
        swings = detect_swing_points(ohlcv)
        result = classify_market_structure(swings)
        valid_states = {"bullish", "bearish", "transition", "range"}
        if len(result) > 0:
            assert set(result["trend_state"].unique()).issubset(valid_states)

    def test_trend_state_tracks_structure_per_event(self):
        """Test trend_state reflects the latest classified high/low at each event."""
        swings = _make_swings(
            [
                {
                    "swing_id": 1,
                    "pivot_index": 1,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "pivot_price": 100.0,
                    "leg_start_ts": pd.Timestamp("2023-12-31T23:00:00Z"),
                    "leg_start_price": 101.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "leg_end_price": 100.0,
                    "price_delta": -1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 2,
                    "pivot_index": 2,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "pivot_price": 110.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "leg_start_price": 100.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "leg_end_price": 110.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.1,
                },
                {
                    "swing_id": 3,
                    "pivot_index": 3,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "pivot_price": 101.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "leg_start_price": 110.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "leg_end_price": 101.0,
                    "price_delta": -9.0,
                    "pct_delta": 0.0818,
                },
                {
                    "swing_id": 4,
                    "pivot_index": 4,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "pivot_price": 111.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "leg_start_price": 101.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "leg_end_price": 111.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.099,
                },
                {
                    "swing_id": 5,
                    "pivot_index": 5,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "pivot_price": 99.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "leg_start_price": 111.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "leg_end_price": 99.0,
                    "price_delta": -12.0,
                    "pct_delta": 0.1081,
                },
                {
                    "swing_id": 6,
                    "pivot_index": 6,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T05:00:00Z"),
                    "pivot_price": 109.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "leg_start_price": 99.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T05:00:00Z"),
                    "leg_end_price": 109.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.101,
                },
            ]
        )

        result = classify_market_structure(swings)

        assert result["classification"].tolist() == ["HL", "HH", "LL", "LH"]
        assert result["trend_state"].tolist() == [
            "transition",
            "bullish",
            "range",
            "bearish",
        ]

    def test_trend_state_marks_lh_plus_hl_as_transition(self):
        """Test mixed lower-high/higher-low structure is treated as transition."""
        swings = _make_swings(
            [
                {
                    "swing_id": 1,
                    "pivot_index": 1,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "pivot_price": 100.0,
                    "leg_start_ts": pd.Timestamp("2023-12-31T23:00:00Z"),
                    "leg_start_price": 101.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "leg_end_price": 100.0,
                    "price_delta": -1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 2,
                    "pivot_index": 2,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "pivot_price": 110.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "leg_start_price": 100.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "leg_end_price": 110.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.1,
                },
                {
                    "swing_id": 3,
                    "pivot_index": 3,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "pivot_price": 101.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "leg_start_price": 110.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "leg_end_price": 101.0,
                    "price_delta": -9.0,
                    "pct_delta": 0.0818,
                },
                {
                    "swing_id": 4,
                    "pivot_index": 4,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "pivot_price": 109.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "leg_start_price": 101.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "leg_end_price": 109.0,
                    "price_delta": 8.0,
                    "pct_delta": 0.0792,
                },
            ]
        )

        result = classify_market_structure(swings)

        assert result["classification"].tolist() == ["HL", "LH"]
        assert result["trend_state"].tolist() == ["transition", "transition"]

    def test_equal_pivots_are_eh_el(self):
        """Test equal pivots are classified as EH/EL (not HH/LL)."""
        ohlcv = _ohlcv_equal_pivots()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.001)
        result = classify_market_structure(swings, equal_epsilon=1e-6)

        equal_highs = result[result["classification"] == "EH"]
        equal_lows = result[result["classification"] == "EL"]

        assert len(equal_highs) > 0
        assert len(equal_lows) > 0

    def test_invalid_equal_epsilon_raises(self):
        """Test equal_epsilon must be finite and non-negative."""
        ohlcv = _ohlcv_equal_pivots()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.001)

        with pytest.raises(ValueError, match="equal_epsilon must be a finite float >= 0.0"):
            classify_market_structure(swings, equal_epsilon=-1)

        with pytest.raises(ValueError, match="equal_epsilon must be a finite float >= 0.0"):
            classify_market_structure(swings, equal_epsilon=float("nan"))

    def test_same_index_low_precedes_high(self):
        """Test same-index swings keep low-before-high ordering in classifications."""
        swings = _make_swings(
            [
                {
                    "swing_id": 1,
                    "pivot_index": 1,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "pivot_price": 100.0,
                    "leg_start_ts": pd.Timestamp("2023-12-31T23:00:00Z"),
                    "leg_start_price": 101.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "leg_end_price": 100.0,
                    "price_delta": -1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 2,
                    "pivot_index": 2,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "pivot_price": 110.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T00:00:00Z"),
                    "leg_start_price": 100.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "leg_end_price": 110.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.1,
                },
                {
                    "swing_id": 3,
                    "pivot_index": 3,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "pivot_price": 101.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T01:00:00Z"),
                    "leg_start_price": 110.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "leg_end_price": 101.0,
                    "price_delta": -9.0,
                    "pct_delta": 0.0818,
                },
                {
                    "swing_id": 4,
                    "pivot_index": 4,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "pivot_price": 111.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T02:00:00Z"),
                    "leg_start_price": 101.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "leg_end_price": 111.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.099,
                },
                {
                    "swing_id": 5,
                    "pivot_index": 5,
                    "swing_type": "low",
                    "pivot_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "pivot_price": 102.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T03:00:00Z"),
                    "leg_start_price": 111.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "leg_end_price": 102.0,
                    "price_delta": -9.0,
                    "pct_delta": 0.0811,
                },
                {
                    "swing_id": 6,
                    "pivot_index": 5,
                    "swing_type": "high",
                    "pivot_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "pivot_price": 112.0,
                    "leg_start_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "leg_start_price": 102.0,
                    "leg_end_ts": pd.Timestamp("2024-01-01T04:00:00Z"),
                    "leg_end_price": 112.0,
                    "price_delta": 10.0,
                    "pct_delta": 0.098,
                },
            ]
        )

        result = classify_market_structure(swings)
        same_pivot_rows = result[result["event_ts"] == pd.Timestamp("2024-01-01T04:00:00Z")]

        assert same_pivot_rows["swing_type"].tolist() == ["low", "high"]
        assert same_pivot_rows["classification"].tolist() == ["HL", "HH"]

    def test_bullish_sequence_has_hh_hl(self):
        """Test bullish sequence produces HH and HL classifications."""
        ohlcv = _ohlcv_bullish_sequence()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        result = classify_market_structure(swings)
        if len(result) > 0:
            # In a bullish sequence, we should see some HH and HL
            assert (
                "HH" in result["classification"].values or "HL" in result["classification"].values
            )

    def test_deterministic_output(self):
        """Test repeated runs on identical input produce identical output."""
        ohlcv = _ohlcv_bullish_sequence()
        swings1 = detect_swing_points(ohlcv)
        swings2 = detect_swing_points(ohlcv)
        result1 = classify_market_structure(swings1)
        result2 = classify_market_structure(swings2)
        pd.testing.assert_frame_equal(
            result1.reset_index(drop=True), result2.reset_index(drop=True)
        )


# =============================================================================
# detect_price_ranges Tests
# =============================================================================


class TestDetectPriceRanges:
    """Tests for detect_price_ranges function."""

    def test_return_type(self):
        """Test range detection returns a DataFrame."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        result = detect_price_ranges(ohlcv, swings)
        assert isinstance(result, pd.DataFrame)

    def test_schema_columns(self):
        """Test range DataFrame has correct columns in exact order."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        result = detect_price_ranges(ohlcv, swings)
        expected_columns = [
            "range_id",
            "start_ts",
            "end_ts",
            "range_high",
            "range_low",
            "range_mid",
            "range_width",
            "range_width_pct",
            "bars_evaluated",
            "containment_ratio",
            "status",
        ]
        assert list(result.columns) == expected_columns

    def test_empty_swings_returns_empty_df(self):
        """Test insufficient swings returns empty DataFrame with pinned columns."""
        ohlcv = _ohlcv_bullish_sequence().head(5)
        swings = detect_swing_points(ohlcv)
        result = detect_price_ranges(ohlcv, swings)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_ohlcv_raises(self):
        """Test ValueError when OHLCV fails validation."""
        swings = detect_swing_points(_ohlcv_bullish_sequence())
        empty_ohlcv = _make_ohlcv([], [], [], [], [])
        with pytest.raises(ValueError):
            detect_price_ranges(empty_ohlcv, swings)

    def test_invalid_params_raises(self):
        """Test ValueError when range parameters are invalid."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        with pytest.raises(ValueError, match="range_lookback must be"):
            detect_price_ranges(ohlcv, swings, range_lookback=1)
        with pytest.raises(ValueError, match="max_range_pct must be"):
            detect_price_ranges(ohlcv, swings, max_range_pct=-0.01)
        with pytest.raises(ValueError, match="containment_ratio must be"):
            detect_price_ranges(ohlcv, swings, containment_ratio=1.5)

    def test_containment_ratio_one_accepts_perfectly_contained_range(self):
        """Test containment_ratio=1.0 accepts a perfectly contained range."""
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 100.0, 100.0, 100.0],
                "high": [101.0, 100.5, 100.5, 101.0],
                "low": [99.0, 99.0, 99.0, 99.0],
                "close": [100.0, 100.0, 100.0, 100.0],
                "volume": [100.0, 100.0, 100.0, 100.0],
            },
            index=idx,
        )
        swings = _make_swings(
            [
                {
                    "swing_id": 1,
                    "pivot_index": 1,
                    "swing_type": "high",
                    "pivot_ts": idx[0],
                    "pivot_price": 101.0,
                    "leg_start_ts": idx[0],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[0],
                    "leg_end_price": 101.0,
                    "price_delta": 1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 2,
                    "pivot_index": 2,
                    "swing_type": "low",
                    "pivot_ts": idx[1],
                    "pivot_price": 99.0,
                    "leg_start_ts": idx[1],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[1],
                    "leg_end_price": 99.0,
                    "price_delta": -1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 3,
                    "pivot_index": 3,
                    "swing_type": "low",
                    "pivot_ts": idx[2],
                    "pivot_price": 99.2,
                    "leg_start_ts": idx[2],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[2],
                    "leg_end_price": 99.2,
                    "price_delta": -0.8,
                    "pct_delta": 0.008,
                },
                {
                    "swing_id": 4,
                    "pivot_index": 4,
                    "swing_type": "high",
                    "pivot_ts": idx[3],
                    "pivot_price": 101.0,
                    "leg_start_ts": idx[3],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[3],
                    "leg_end_price": 101.0,
                    "price_delta": 1.0,
                    "pct_delta": 0.01,
                },
            ]
        )

        result = detect_price_ranges(
            ohlcv,
            swings,
            containment_ratio=1.0,
            boundary_buffer_pct=0.0,
        )

        assert len(result) == 1
        assert result.iloc[0]["containment_ratio"] == 1.0

    def test_ranges_with_same_boundaries_keep_distinct_endpoints(self):
        """Test later ranges are preserved when boundaries match but end_ts differs."""
        idx = pd.date_range("2024-01-01", periods=7, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 7,
                "high": [101.0] * 7,
                "low": [99.0] * 7,
                "close": [100.0] * 7,
                "volume": [100.0] * 7,
            },
            index=idx,
        )
        swings = _make_swings(
            [
                {
                    "swing_id": 1,
                    "pivot_index": 1,
                    "swing_type": "high",
                    "pivot_ts": idx[0],
                    "pivot_price": 101.0,
                    "leg_start_ts": idx[0],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[0],
                    "leg_end_price": 101.0,
                    "price_delta": 1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 2,
                    "pivot_index": 2,
                    "swing_type": "low",
                    "pivot_ts": idx[1],
                    "pivot_price": 99.0,
                    "leg_start_ts": idx[1],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[1],
                    "leg_end_price": 99.0,
                    "price_delta": -1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 3,
                    "pivot_index": 3,
                    "swing_type": "low",
                    "pivot_ts": idx[2],
                    "pivot_price": 99.2,
                    "leg_start_ts": idx[2],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[2],
                    "leg_end_price": 99.2,
                    "price_delta": -0.8,
                    "pct_delta": 0.008,
                },
                {
                    "swing_id": 4,
                    "pivot_index": 4,
                    "swing_type": "high",
                    "pivot_ts": idx[3],
                    "pivot_price": 101.0,
                    "leg_start_ts": idx[3],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[3],
                    "leg_end_price": 101.0,
                    "price_delta": 1.0,
                    "pct_delta": 0.01,
                },
                {
                    "swing_id": 5,
                    "pivot_index": 5,
                    "swing_type": "low",
                    "pivot_ts": idx[4],
                    "pivot_price": 99.1,
                    "leg_start_ts": idx[4],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[4],
                    "leg_end_price": 99.1,
                    "price_delta": -0.9,
                    "pct_delta": 0.009,
                },
                {
                    "swing_id": 6,
                    "pivot_index": 6,
                    "swing_type": "low",
                    "pivot_ts": idx[5],
                    "pivot_price": 99.3,
                    "leg_start_ts": idx[5],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[5],
                    "leg_end_price": 99.3,
                    "price_delta": -0.7,
                    "pct_delta": 0.007,
                },
                {
                    "swing_id": 7,
                    "pivot_index": 7,
                    "swing_type": "high",
                    "pivot_ts": idx[6],
                    "pivot_price": 100.8,
                    "leg_start_ts": idx[6],
                    "leg_start_price": 100.0,
                    "leg_end_ts": idx[6],
                    "leg_end_price": 100.8,
                    "price_delta": 0.8,
                    "pct_delta": 0.008,
                },
            ]
        )

        result = detect_price_ranges(
            ohlcv,
            swings,
            containment_ratio=1.0,
            boundary_buffer_pct=0.0,
        )

        same_boundary_ranges = result[
            (result["range_high"] == 101.0) & (result["range_low"] == 99.0)
        ].sort_values("end_ts")

        assert same_boundary_ranges["end_ts"].tolist() == [idx[3], idx[6]]
        assert same_boundary_ranges["range_id"].tolist() == sorted(
            same_boundary_ranges["range_id"].tolist()
        )

    def test_status_values(self):
        """Test status contains only valid values."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        result = detect_price_ranges(ohlcv, swings)
        if len(result) > 0:
            valid_statuses = {"active", "broken_up", "broken_down"}
            assert set(result["status"].unique()).issubset(valid_statuses)

    def test_range_high_greater_than_low(self):
        """Test range_high is always greater than range_low."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        result = detect_price_ranges(ohlcv, swings)
        if len(result) > 0:
            assert (result["range_high"] > result["range_low"]).all()


# =============================================================================
# detect_range_breakouts Tests
# =============================================================================


class TestDetectRangeBreakouts:
    """Tests for detect_range_breakouts function."""

    def test_return_type(self):
        """Test breakout detection returns a DataFrame."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        ranges = detect_price_ranges(ohlcv, swings)
        result = detect_range_breakouts(ohlcv, ranges)
        assert isinstance(result, pd.DataFrame)

    def test_schema_columns(self):
        """Test breakout DataFrame has correct columns in exact order."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        ranges = detect_price_ranges(ohlcv, swings)
        result = detect_range_breakouts(ohlcv, ranges)
        expected_columns = [
            "breakout_id",
            "range_id",
            "breakout_ts",
            "direction",
            "breakout_price",
            "boundary_price",
            "distance",
            "distance_pct",
            "confirm_bars",
        ]
        assert list(result.columns) == expected_columns

    def test_empty_ranges_returns_empty_df(self):
        """Test empty ranges returns DataFrame with pinned columns."""
        ohlcv = _ohlcv_range_bound()
        empty_ranges = pd.DataFrame(
            columns=[
                "range_id",
                "start_ts",
                "end_ts",
                "range_high",
                "range_low",
                "range_mid",
                "range_width",
                "range_width_pct",
                "bars_evaluated",
                "containment_ratio",
                "status",
            ]
        )
        result = detect_range_breakouts(ohlcv, empty_ranges)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_invalid_ohlcv_raises(self):
        """Test ValueError when OHLCV fails validation."""
        swings = detect_swing_points(_ohlcv_range_bound())
        ranges = detect_price_ranges(_ohlcv_range_bound(), swings)
        empty_ohlcv = _make_ohlcv([], [], [], [], [])
        with pytest.raises(ValueError):
            detect_range_breakouts(empty_ohlcv, ranges)

    def test_invalid_params_raises(self):
        """Test ValueError when breakout parameters are invalid."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        ranges = detect_price_ranges(ohlcv, swings)
        with pytest.raises(ValueError, match="breakout_confirm_bars must be"):
            detect_range_breakouts(ohlcv, ranges, breakout_confirm_bars=0)
        with pytest.raises(ValueError, match="breakout_buffer_pct must be"):
            detect_range_breakouts(ohlcv, ranges, breakout_buffer_pct=-0.01)

    def test_invalid_ranges_type_raises(self):
        """Test non-DataFrame ranges fail fast."""
        ohlcv = _ohlcv_range_bound()

        with pytest.raises(ValueError, match="ranges must be a pandas DataFrame"):
            detect_range_breakouts(ohlcv, "not a dataframe")

    def test_breakout_at_end_ts_bar_is_detected(self):
        """Test breakout detection includes the range end bar when it breaks the boundary."""
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100, 100, 100, 100],
                "high": [101, 101, 101, 101],
                "low": [99, 99, 99, 99],
                "close": [100, 100, 101.5, 100],
                "volume": [100, 100, 100, 100],
            },
            index=idx,
        )
        ranges = pd.DataFrame(
            [
                {
                    "range_id": 1,
                    "start_ts": idx[0],
                    "end_ts": idx[2],
                    "range_high": 101.0,
                    "range_low": 99.0,
                    "range_mid": 100.0,
                    "range_width": 2.0,
                    "range_width_pct": 0.02,
                    "bars_evaluated": 3,
                    "containment_ratio": 1.0,
                    "status": "broken_up",
                }
            ]
        )

        result = detect_range_breakouts(
            ohlcv, ranges, breakout_confirm_bars=1, breakout_buffer_pct=0.0
        )

        assert len(result) == 1
        assert result.iloc[0]["range_id"] == 1
        assert result.iloc[0]["direction"] == "up"
        assert result.iloc[0]["breakout_ts"] == idx[2]

    def test_direction_values(self):
        """Test direction contains only 'up' or 'down'."""
        ohlcv = _ohlcv_range_bound()
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        ranges = detect_price_ranges(ohlcv, swings)
        result = detect_range_breakouts(ohlcv, ranges)
        if len(result) > 0:
            assert set(result["direction"].unique()).issubset({"up", "down"})


# =============================================================================
# Indicator Layer Boundary Tests
# =============================================================================


class TestIndicatorLayerBoundary:
    """Tests confirming ENG-32 remains an indicator-layer ticket."""

    def test_structure_module_has_no_server_imports(self):
        """Test that structure.py does not import server or tools modules."""
        module_file = detect_swing_points.__code__.co_filename
        with open(module_file) as f:
            content = f.read()
        # Should not have imports from these modules
        assert "from tempest_mcp.tools" not in content
        assert "from tempest_mcp.server" not in content
        assert "import tempest_mcp.tools" not in content
        assert "import tempest_mcp.server" not in content

    def test_new_functions_are_exported(self):
        """Test new functions are exported from indicators package."""
        from tempest_mcp import indicators

        assert hasattr(indicators, "detect_swing_points")
        assert hasattr(indicators, "classify_market_structure")
        assert hasattr(indicators, "detect_price_ranges")
        assert hasattr(indicators, "detect_range_breakouts")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for price pattern engine workflow."""

    def test_full_pipeline_bullish(self):
        """Test complete price pattern analysis on bullish sequence."""
        ohlcv = _ohlcv_bullish_sequence()

        # Step 1: Detect swings
        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        assert isinstance(swings, pd.DataFrame)

        # Step 2: Classify market structure
        structure = classify_market_structure(swings)
        assert isinstance(structure, pd.DataFrame)

        # Step 3: Detect ranges
        ranges = detect_price_ranges(ohlcv, swings)
        assert isinstance(ranges, pd.DataFrame)

        # Step 4: Detect breakouts
        breakouts = detect_range_breakouts(ohlcv, ranges)
        assert isinstance(breakouts, pd.DataFrame)

    def test_full_pipeline_bearish(self):
        """Test complete price pattern analysis on bearish sequence."""
        ohlcv = _ohlcv_bearish_sequence()

        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        structure = classify_market_structure(swings)
        ranges = detect_price_ranges(ohlcv, swings)
        breakouts = detect_range_breakouts(ohlcv, ranges)

        assert isinstance(swings, pd.DataFrame)
        assert isinstance(structure, pd.DataFrame)
        assert isinstance(ranges, pd.DataFrame)
        assert isinstance(breakouts, pd.DataFrame)

    def test_full_pipeline_with_range_and_breakout(self):
        """Test complete pipeline with range detection and breakout."""
        ohlcv = _ohlcv_range_bound()

        swings = detect_swing_points(ohlcv, min_swing_pct=0.01)
        classify_market_structure(swings)
        ranges = detect_price_ranges(ohlcv, swings)
        breakouts = detect_range_breakouts(ohlcv, ranges)

        # Should have detected some ranges
        if len(ranges) > 0:
            assert "range_id" in ranges.columns
            assert "status" in ranges.columns

        # Breakouts should reference valid range_ids
        if len(breakouts) > 0:
            assert "range_id" in breakouts.columns
            assert all(breakouts["range_id"].isin(ranges["range_id"]))

    def test_determinism_on_identical_input(self):
        """Test that repeated runs on identical input produce identical results."""
        ohlcv = _ohlcv_bullish_sequence()

        swings1 = detect_swing_points(ohlcv, min_swing_pct=0.01)
        swings2 = detect_swing_points(ohlcv, min_swing_pct=0.01)

        structure1 = classify_market_structure(swings1)
        structure2 = classify_market_structure(swings2)

        ranges1 = detect_price_ranges(ohlcv, swings1)
        ranges2 = detect_price_ranges(ohlcv, swings2)

        breakouts1 = detect_range_breakouts(ohlcv, ranges1)
        breakouts2 = detect_range_breakouts(ohlcv, ranges2)

        pd.testing.assert_frame_equal(
            swings1.reset_index(drop=True), swings2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            structure1.reset_index(drop=True), structure2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            ranges1.reset_index(drop=True), ranges2.reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(
            breakouts1.reset_index(drop=True), breakouts2.reset_index(drop=True)
        )
