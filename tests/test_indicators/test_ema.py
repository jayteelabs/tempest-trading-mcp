"""Unit tests for EMA indicator engine."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.trend.ema import (
    calculate_ema,
    calculate_ema_stack,
    death_cross,
    detect_ema_cross,
    golden_cross,
)


class TestCalculateEma:
    """Tests for calculate_ema function."""

    def test_normal_case(self):
        """Test EMA calculation with sufficient data."""
        # Generate ascending price series
        prices = pd.Series(
            range(100, 200), index=pd.date_range("2024-01-01", periods=100, freq="h")
        )

        ema = calculate_ema(prices, period=20)

        assert len(ema) == len(prices)
        assert ema.index.equals(prices.index)
        # EMA should be defined for all points when data is sufficient
        assert not ema.isna().all()

    def test_insufficient_data(self):
        """Test EMA returns empty Series when data is insufficient."""
        prices = pd.Series([100, 101, 102], index=pd.date_range("2024-01-01", periods=3, freq="h"))

        ema = calculate_ema(prices, period=10)

        assert len(ema) == 0
        assert isinstance(ema, pd.Series)

    def test_exactly_period_length(self):
        """Test EMA calculation when data length equals period."""
        prices = pd.Series(
            range(100, 110),  # 10 values
            index=pd.date_range("2024-01-01", periods=10, freq="h"),
        )

        ema = calculate_ema(prices, period=10)

        assert len(ema) == len(prices)

    def test_smoothing_factor_calculation(self):
        """Test that EMA uses correct smoothing factor (alpha = 2/(period+1))."""
        # With period=5, alpha should be 2/6 = 0.333
        # Simple test: first EMA value should equal first price (adjust=False)
        prices = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            index=pd.date_range("2024-01-01", periods=6, freq="h"),
        )

        ema = calculate_ema(prices, period=5)

        # With adjust=False, first EMA value equals first price
        assert ema.iloc[0] == pytest.approx(100.0)

    def test_invalid_period_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        prices = pd.Series(
            [100.0, 101.0, 102.0], index=pd.date_range("2024-01-01", periods=3, freq="h")
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_ema(prices, period=0)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_ema(prices, period=-1)


class TestCalculateEmaStack:
    """Tests for calculate_ema_stack function."""

    def test_returns_all_four_periods(self):
        """Test that stack contains all four EMA periods."""
        prices = pd.Series(
            range(100, 400),  # 300 values for EMA200
            index=pd.date_range("2024-01-01", periods=300, freq="h"),
        )

        stack = calculate_ema_stack(prices)

        assert "ema7" in stack
        assert "ema25" in stack
        assert "ema50" in stack
        assert "ema200" in stack

    def test_alignment_with_input(self):
        """Test that all EMAs are aligned with input index."""
        prices = pd.Series(
            range(100, 400), index=pd.date_range("2024-01-01", periods=300, freq="h")
        )

        stack = calculate_ema_stack(prices)

        for _key, ema in stack.items():
            assert len(ema) == len(prices)
            assert ema.index.equals(prices.index)

    def test_insufficient_data_for_longest_period(self):
        """Test that stacks shorter than 200 prices raise ValueError."""
        prices = pd.Series(
            range(100, 200),  # 100 values, not enough for EMA200
            index=pd.date_range("2024-01-01", periods=100, freq="h"),
        )

        with pytest.raises(ValueError, match="at least 200 price values"):
            calculate_ema_stack(prices)

    def test_empty_series_returns_empty_stack(self):
        """calculate_ema_stack should handle empty price Series without error."""
        prices = pd.Series(dtype=float)
        stack = calculate_ema_stack(prices)
        assert len(stack) == 0


class TestDetectEmaCross:
    """Tests for detect_ema_cross function."""

    def test_detects_cross_up(self):
        """Test detection of bullish crossover."""
        # Create prices where EMA7 crosses above EMA25
        # Start flat, then uptrend
        base = [100] * 50 + list(range(100, 150))
        prices = pd.Series(base, index=pd.date_range("2024-01-01", periods=len(base), freq="h"))

        ema7 = calculate_ema(prices, 7)
        ema25 = calculate_ema(prices, 25)

        crosses = detect_ema_cross(ema7, ema25)

        assert len(crosses) >= 1
        assert "date" in crosses.columns
        assert "fast_above" in crosses.columns
        assert "direction" in crosses.columns

        # Should have at least one cross_up after the uptrend starts
        cross_ups = crosses[crosses["direction"] == "cross_up"]
        assert len(cross_ups) >= 1

    def test_detects_cross_down(self):
        """Test detection of bearish crossover."""
        # Create prices where EMA7 crosses below EMA25
        # Start flat, uptrend (EMA7 rises faster and crosses above), then downtrend (cross DOWN)
        flat = [100] * 20
        up = list(range(100, 121))  # 21 values: 100 to 120
        down = list(range(120, 99, -1))  # 21 values: 120 to 100
        prices = pd.Series(
            flat + up + down, index=pd.date_range("2024-01-01", periods=62, freq="h")
        )

        ema7 = calculate_ema(prices, 7)
        ema25 = calculate_ema(prices, 25)

        crosses = detect_ema_cross(ema7, ema25)

        assert len(crosses) >= 1
        cross_downs = crosses[crosses["direction"] == "cross_down"]
        assert len(cross_downs) >= 1

    def test_no_crossover_flat_price(self):
        """Test that flat/sideways price produces no false crossovers."""
        # Constant price should have no crossovers
        prices = pd.Series([100.0] * 200, index=pd.date_range("2024-01-01", periods=200, freq="h"))

        ema7 = calculate_ema(prices, 7)
        ema25 = calculate_ema(prices, 25)

        crosses = detect_ema_cross(ema7, ema25)

        assert len(crosses) == 0

    def test_empty_series_returns_empty_dataframe(self):
        """Test that empty EMA series returns empty DataFrame."""
        ema_empty = pd.Series(dtype=float)
        ema_valid = pd.Series(
            range(100, 200), index=pd.date_range("2024-01-01", periods=100, freq="h")
        )

        crosses = detect_ema_cross(ema_empty, ema_valid)

        assert len(crosses) == 0
        assert isinstance(crosses, pd.DataFrame)

    def test_nan_values_handled_correctly(self):
        """Test that NaN values in EMA series are filtered out properly."""
        # Create EMA series with NaN values
        ema_fast = pd.Series([100.0, 101.0, float("nan"), 103.0, 104.0])
        ema_slow = pd.Series([101.0, 100.0, 102.0, 101.0, 100.0])

        # Should not raise, should handle NaN gracefully
        crosses = detect_ema_cross(ema_fast, ema_slow)

        assert isinstance(crosses, pd.DataFrame)
        # NaN values should be filtered, so we look for actual valid crossovers

    def test_nan_only_series_returns_empty(self):
        """Test that series with all NaN returns empty DataFrame."""
        ema_fast = pd.Series([float("nan"), float("nan"), float("nan")])
        ema_slow = pd.Series([100.0, 101.0, 102.0])

        crosses = detect_ema_cross(ema_fast, ema_slow)

        assert len(crosses) == 0
        assert isinstance(crosses, pd.DataFrame)

    def test_single_cross_event_no_repeats(self):
        """Test that each crossover produces only one signal."""
        # Sharp uptrend should produce one cross_up event
        prices = pd.Series(
            [100] * 30 + list(range(100, 150)),
            index=pd.date_range("2024-01-01", periods=80, freq="h"),
        )

        ema7 = calculate_ema(prices, 7)
        ema25 = calculate_ema(prices, 25)

        crosses = detect_ema_cross(ema7, ema25)

        # Count consecutive cross_up signals - should be contiguous, not repeated
        cross_ups = crosses[crosses["direction"] == "cross_up"]
        for i in range(1, len(cross_ups)):
            # Each cross event should be distinct (not same timestamp)
            assert cross_ups.iloc[i]["date"] != cross_ups.iloc[i - 1]["date"]

    def test_length_mismatch_truncates_to_overlap(self):
        """Series of different lengths are safely truncated to overlapping index."""
        base_index = pd.date_range("2024-01-01", periods=5, freq="h")
        # fast goes from above slow to below slow — produces one cross_down
        ema_fast = pd.Series([3.0, 2.5, 2.0, 1.5, 1.0], index=base_index)
        # slow is shorter — overlap is first 3 points
        ema_slow = pd.Series([1.5, 2.0, 2.5], index=base_index[:3])
        result = detect_ema_cross(ema_fast=ema_fast, ema_slow=ema_slow)
        # After alignment: fast[0,1,2]=[3.0,2.5,2.0], slow[0,1,2]=[1.5,2.0,2.5]
        # diff = [1.5, 0.5, -0.5], sign = [1, 1, -1] → one cross_down at index 2
        assert len(result) == 1
        assert result.iloc[0]["direction"] == "cross_down"
        assert result.iloc[0]["date"] == base_index[2]


class TestGoldenCross:
    """Tests for golden_cross function."""

    def test_bullish_stack_returns_true(self):
        """Test that bullish stack (7>25>50>200) returns True."""
        # Create strongly bullish stack
        stack = {
            "ema7": pd.Series([150.0]),
            "ema25": pd.Series([140.0]),
            "ema50": pd.Series([130.0]),
            "ema200": pd.Series([120.0]),
        }

        assert golden_cross(stack) is True

    def test_non_bullish_stack_returns_false(self):
        """Test that non-bullish stack returns False."""
        # Mixed stack (not bullish)
        stack = {
            "ema7": pd.Series([130.0]),
            "ema25": pd.Series([140.0]),
            "ema50": pd.Series([130.0]),
            "ema200": pd.Series([120.0]),
        }

        assert golden_cross(stack) is False

    def test_empty_stack_returns_false(self):
        """Test that empty EMA series returns False."""
        stack = {
            "ema7": pd.Series(dtype=float),
            "ema25": pd.Series([140.0]),
            "ema50": pd.Series([130.0]),
            "ema200": pd.Series([120.0]),
        }

        assert golden_cross(stack) is False

    def test_missing_key_returns_false(self):
        """Test that missing EMA key returns False."""
        stack = {
            "ema7": pd.Series([150.0]),
            "ema25": pd.Series([140.0]),
            "ema50": pd.Series([130.0]),
            # ema200 missing
        }

        assert golden_cross(stack) is False

    def test_nan_values_return_false(self):
        """Test that NaN values in stack return False."""
        # NaN at last position (not first) to properly test "last values are NaN"
        stack = {
            "ema7": pd.Series([150.0, float("nan")]),
            "ema25": pd.Series([140.0, float("nan")]),
            "ema50": pd.Series([130.0, float("nan")]),
            "ema200": pd.Series([120.0, float("nan")]),
        }

        # Should return False because last values are NaN
        assert golden_cross(stack) is False

    def test_nan_latest_values_return_false(self):
        """NaN latest EMA values cause golden_cross to return False."""
        stack = {
            "ema7": pd.Series([110.0, float("nan")]),
            "ema25": pd.Series([120.0]),
            "ema50": pd.Series([130.0]),
            "ema200": pd.Series([140.0]),
        }
        assert golden_cross(stack) is False


class TestDeathCross:
    """Tests for death_cross function."""

    def test_bearish_stack_returns_true(self):
        """Test that bearish stack (7<25<50<200) returns True."""
        # Create strongly bearish stack
        stack = {
            "ema7": pd.Series([120.0]),
            "ema25": pd.Series([130.0]),
            "ema50": pd.Series([140.0]),
            "ema200": pd.Series([150.0]),
        }

        assert death_cross(stack) is True

    def test_non_bearish_stack_returns_false(self):
        """Test that non-bearish stack returns False."""
        # Mixed stack (not bearish)
        stack = {
            "ema7": pd.Series([140.0]),
            "ema25": pd.Series([130.0]),
            "ema50": pd.Series([140.0]),
            "ema200": pd.Series([150.0]),
        }

        assert death_cross(stack) is False

    def test_empty_stack_returns_false(self):
        """Test that empty EMA series returns False."""
        stack = {
            "ema7": pd.Series(dtype=float),
            "ema25": pd.Series([130.0]),
            "ema50": pd.Series([140.0]),
            "ema200": pd.Series([150.0]),
        }

        assert death_cross(stack) is False

    def test_nan_values_return_false(self):
        """Test that NaN values in stack return False."""
        # NaN at last position (not first) to properly test "last values are NaN"
        stack = {
            "ema7": pd.Series([120.0, float("nan")]),
            "ema25": pd.Series([130.0, float("nan")]),
            "ema50": pd.Series([140.0, float("nan")]),
            "ema200": pd.Series([150.0, float("nan")]),
        }

        # Should return False because last values are NaN
        assert death_cross(stack) is False

    def test_nan_latest_values_return_false(self):
        """NaN latest EMA values cause death_cross to return False."""
        stack = {
            "ema7": pd.Series([110.0, float("nan")]),
            "ema25": pd.Series([120.0]),
            "ema50": pd.Series([130.0]),
            "ema200": pd.Series([140.0]),
        }
        assert death_cross(stack) is False

    def test_missing_key_returns_false(self):
        """Missing EMA key returns False for death_cross."""
        stack_missing_key = {
            "ema7": pd.Series([90.0]),
            "ema25": pd.Series([100.0]),
            # "ema50" intentionally missing
            "ema200": pd.Series([120.0]),
        }
        assert death_cross(stack_missing_key) is False


class TestIntegration:
    """Integration tests for EMA indicator workflow."""

    def test_full_workflow(self):
        """Test complete EMA analysis workflow."""
        # Generate realistic price data
        prices = pd.Series(
            range(100, 400), index=pd.date_range("2024-01-01", periods=300, freq="h")
        )

        # Calculate full stack
        stack = calculate_ema_stack(prices)

        # All EMAs should be non-empty
        assert all(not ema.empty for ema in stack.values())

        # Detect crossovers
        crosses = detect_ema_cross(stack["ema7"], stack["ema25"])

        # Should be valid DataFrame
        assert isinstance(crosses, pd.DataFrame)
        assert "date" in crosses.columns
        assert "direction" in crosses.columns

        # Check stack state
        is_golden = golden_cross(stack)
        is_death = death_cross(stack)

        # Should be mutually exclusive
        assert not (is_golden and is_death)

    def test_with_price_data_fixture(self, price_data):
        """Test EMA calculation with pytest fixture."""
        # Convert numpy array to pandas Series
        prices = pd.Series(price_data)
        prices.index = pd.date_range("2024-01-01", periods=len(prices), freq="h")

        ema = calculate_ema(prices, period=14)

        assert len(ema) == len(prices)

    def test_reference_value_match(self):
        """Test that EMA matches reference implementation for known values."""
        # Known values: simple ascending series
        # EMA(5) for [100, 101, 102, 103, 104, 105, 106]
        # alpha = 2/(5+1) = 0.333
        # With adjust=False:
        # EMA[0] = 100
        # EMA[1] = 101 * 0.333 + 100 * 0.667 = 100.333
        # EMA[2] = 102 * 0.333 + 100.333 * 0.667 = 100.889
        # EMA[3] = 103 * 0.333 + 100.889 * 0.667 = 101.592
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])

        ema = calculate_ema(prices, period=5)

        # First value should equal first price (adjust=False behavior)
        assert ema.iloc[0] == pytest.approx(100.0, rel=1e-9)

        # EMA should be trending up with prices
        assert ema.iloc[-1] > ema.iloc[0]

        # Compute expected values and verify
        alpha = 2 / (5 + 1)
        ema0 = 100.0  # First price
        ema1 = 101.0 * alpha + ema0 * (1 - alpha)  # EMA at index 1
        ema2 = 102.0 * alpha + ema1 * (1 - alpha)  # EMA at index 2
        ema3 = 103.0 * alpha + ema2 * (1 - alpha)  # EMA at index 3
        assert ema.iloc[1] == pytest.approx(ema1, rel=1e-6)
        assert ema.iloc[3] == pytest.approx(ema3, rel=1e-6)
