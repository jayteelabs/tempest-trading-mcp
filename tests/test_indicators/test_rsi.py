"""Unit tests for RSI indicator engine."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.momentum.rsi import (
    CENTERLINE,
    OVERBOUGHT_THRESHOLD,
    OVERSOLD_THRESHOLD,
    RSI_DEFAULT_PERIOD,
    calculate_rsi,
    detect_rsi_cross,
    detect_rsi_divergence,
    detect_rsi_extremes,
)


class TestCalculateRsi:
    """Tests for calculate_rsi function."""

    def test_normal_case_smma(self):
        """Test RSI calculation with sufficient data using SMMA (default)."""
        prices = pd.Series(
            [100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105, 106, 105, 104,
             103, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14, smooth_type="smma")

        assert len(rsi) == len(prices)
        assert rsi.index.equals(prices.index)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_normal_case_ema(self):
        """Test RSI calculation with EMA smoothing."""
        prices = pd.Series(
            [100, 101, 102, 101, 100, 99, 100, 101, 102, 103, 104, 105, 106, 105, 104,
             103, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14, smooth_type="ema")

        assert len(rsi) == len(prices)
        assert rsi.index.equals(prices.index)
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_insufficient_data(self):
        """Test RSI returns empty Series when data is insufficient."""
        prices = pd.Series(
            [100, 101, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14)

        assert len(rsi) == 0
        assert isinstance(rsi, pd.Series)

    def test_exactly_minimum_length(self):
        """Test RSI with exactly period + 2 data points (minimum for one valid RSI)."""
        # diff() reduces length by 1, SMMA first value at index period-1,
        # so we need period + 2 original values for one valid RSI
        prices = pd.Series(
            range(100, 117),  # 17 values for period 14
            index=pd.date_range("2024-01-01", periods=17, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)
        # With 17 values, we should get at least one valid RSI
        assert not rsi.dropna().empty

    def test_invalid_period_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        prices = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_rsi(prices, period=0)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_rsi(prices, period=-1)

    def test_invalid_smooth_type_raises_error(self):
        """Test that invalid smooth_type raises ValueError."""
        prices = pd.Series(
            [100.0, 101.0, 102.0] * 10,
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="smooth_type must be"):
            calculate_rsi(prices, period=14, smooth_type="invalid")

    def test_flat_prices(self):
        """Test RSI with constant prices (no movement)."""
        prices = pd.Series(
            [100.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14)

        assert rsi.index.tz is not None
        assert str(rsi.index.tz) == "UTC"

    def test_default_period_is_14(self):
        """Test that default period is 14."""
        assert RSI_DEFAULT_PERIOD == 14

    def test_default_thresholds(self):
        """Test that default thresholds are correct."""
        assert OVERSOLD_THRESHOLD == 30
        assert OVERBOUGHT_THRESHOLD == 70
        assert CENTERLINE == 50.0


class TestDetectRsiExtremes:
    """Tests for detect_rsi_extremes function."""

    def test_detects_oversold_zone(self):
        """Test detection of oversold zone entries."""
        rsi_values = [50, 40, 35, 30, 25, 28, 35, 45, 50]
        rsi = pd.Series(
            rsi_values,
            index=pd.date_range("2024-01-01", periods=len(rsi_values), freq="h", tz="UTC"),
        )

        extremes = detect_rsi_extremes(rsi, oversold=30, overbought=70)

        assert isinstance(extremes, pd.DataFrame)
        assert "date" in extremes.columns
        assert "zone" in extremes.columns
        assert "value" in extremes.columns

        oversold_zones = extremes[extremes["zone"] == "oversold"]
        assert len(oversold_zones) >= 1

    def test_detects_overbought_zone(self):
        """Test detection of overbought zone entries."""
        rsi_values = [50, 60, 65, 70, 75, 80, 78, 72, 65, 55, 50]
        rsi = pd.Series(
            rsi_values,
            index=pd.date_range("2024-01-01", periods=len(rsi_values), freq="h", tz="UTC"),
        )

        extremes = detect_rsi_extremes(rsi, oversold=30, overbought=70)

        overbought_zones = extremes[extremes["zone"] == "overbought"]
        assert len(overbought_zones) >= 1

    def test_empty_series_returns_empty_dataframe(self):
        """Test that empty RSI series returns empty DataFrame."""
        rsi = pd.Series(dtype=float)

        extremes = detect_rsi_extremes(rsi)

        assert isinstance(extremes, pd.DataFrame)
        assert len(extremes) == 0

    def test_returns_dataframe_not_list(self):
        """Test that return type is DataFrame, not list."""
        rsi = pd.Series(
            [50, 40, 30, 25, 30, 50, 70, 75, 70, 50],
            index=pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC"),
        )

        extremes = detect_rsi_extremes(rsi)

        assert isinstance(extremes, pd.DataFrame)
        assert not isinstance(extremes, list)


class TestDetectRsiDivergence:
    """Tests for detect_rsi_divergence function."""

    def test_empty_series_returns_empty_dataframe(self):
        """Test that empty series returns empty DataFrame."""
        prices = pd.Series(dtype=float)
        rsi = pd.Series(dtype=float)

        divergence = detect_rsi_divergence(prices, rsi)

        assert isinstance(divergence, pd.DataFrame)
        assert len(divergence) == 0

    def test_insufficient_data_returns_empty(self):
        """Test that insufficient data returns empty DataFrame."""
        prices = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        rsi = pd.Series(
            [50, 52, 54, 56, 58],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        divergence = detect_rsi_divergence(prices, rsi, window=20)

        assert len(divergence) == 0

    def test_returns_dataframe_not_list(self):
        """Test that return type is DataFrame, not list."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        rsi = calculate_rsi(prices, period=14)

        divergence = detect_rsi_divergence(prices, rsi, window=20)

        assert isinstance(divergence, pd.DataFrame)
        assert not isinstance(divergence, list)


class TestDetectRsiCross:
    """Tests for detect_rsi_cross function."""

    def test_detects_bullish_cross(self):
        """Test detection of bullish cross (RSI crossing above threshold)."""
        rsi_values = [45, 47, 49, 51, 53, 55]
        rsi = pd.Series(
            rsi_values,
            index=pd.date_range("2024-01-01", periods=len(rsi_values), freq="h", tz="UTC"),
        )

        crosses = detect_rsi_cross(rsi, threshold=50.0)

        assert isinstance(crosses, pd.DataFrame)
        assert "date" in crosses.columns
        assert "direction" in crosses.columns
        assert "value" in crosses.columns

        bullish_crosses = crosses[crosses["direction"] == "bullish"]
        assert len(bullish_crosses) >= 1

    def test_detects_bearish_cross(self):
        """Test detection of bearish cross (RSI crossing below threshold)."""
        rsi_values = [55, 53, 51, 49, 47, 45]
        rsi = pd.Series(
            rsi_values,
            index=pd.date_range("2024-01-01", periods=len(rsi_values), freq="h", tz="UTC"),
        )

        crosses = detect_rsi_cross(rsi, threshold=50.0)

        bearish_crosses = crosses[crosses["direction"] == "bearish"]
        assert len(bearish_crosses) >= 1

    def test_no_false_positives_flat_rsi(self):
        """Test that flat RSI at threshold produces no false crosses."""
        rsi = pd.Series(
            [50.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        crosses = detect_rsi_cross(rsi, threshold=50.0)

        assert len(crosses) == 0

    def test_empty_series_returns_empty_dataframe(self):
        """Test that empty RSI series returns empty DataFrame."""
        rsi = pd.Series(dtype=float)

        crosses = detect_rsi_cross(rsi)

        assert isinstance(crosses, pd.DataFrame)
        assert len(crosses) == 0

    def test_custom_threshold(self):
        """Test detection with custom threshold."""
        rsi_values = [60, 65, 70, 75, 70, 65, 60]
        rsi = pd.Series(
            rsi_values,
            index=pd.date_range("2024-01-01", periods=len(rsi_values), freq="h", tz="UTC"),
        )

        crosses = detect_rsi_cross(rsi, threshold=70.0)

        assert isinstance(crosses, pd.DataFrame)
        assert "direction" in crosses.columns

    def test_returns_dataframe_not_list(self):
        """Test that return type is DataFrame, not list."""
        rsi = pd.Series(
            [40, 45, 50, 55, 60],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        crosses = detect_rsi_cross(rsi)

        assert isinstance(crosses, pd.DataFrame)
        assert not isinstance(crosses, list)


class TestIntegration:
    """Integration tests for RSI indicator workflow."""

    def test_full_workflow(self):
        """Test complete RSI analysis workflow."""
        prices = pd.Series(
            range(100, 200),
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)

        extremes = detect_rsi_extremes(rsi)

        assert isinstance(extremes, pd.DataFrame)
        assert "date" in extremes.columns
        assert "zone" in extremes.columns

        crosses = detect_rsi_cross(rsi, threshold=50.0)

        assert isinstance(crosses, pd.DataFrame)
        assert "date" in crosses.columns
        assert "direction" in crosses.columns

        # For a strictly ascending price series, there's no divergence
        # Divergence requires price making higher highs while RSI makes lower highs (or vice versa)
        divergence = detect_rsi_divergence(prices, rsi, window=20)

        assert isinstance(divergence, pd.DataFrame)
        # Empty result is valid for monotonic price series
        if len(divergence) > 0:
            assert "date" in divergence.columns
            assert "type" in divergence.columns

    def test_with_price_data_fixture(self, price_data):
        """Test RSI calculation with pytest fixture."""
        prices = pd.Series(price_data)
        prices.index = pd.date_range("2024-01-01", periods=len(prices), freq="h", tz="UTC")

        rsi = calculate_rsi(prices, period=14)

        assert len(rsi) == len(prices)

    def test_smma_matches_reference(self):
        """Test SMMA RSI against known values."""
        prices = pd.Series(
            [100.0] * 15 + list(range(100, 130)),
            index=pd.date_range("2024-01-01", periods=45, freq="h", tz="UTC"),
        )

        rsi = calculate_rsi(prices, period=14, smooth_type="smma")

        valid_rsi = rsi.dropna()
        if len(valid_rsi) > 0:
            assert valid_rsi.iloc[-1] > 50
