"""Unit tests for MACD, ADX, Stochastic indicator engine."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.momentum.macd_adx_stoch import (
    ADX_DEFAULT_PERIOD,
    MACD_DEFAULT_FAST,
    MACD_DEFAULT_SIGNAL,
    MACD_DEFAULT_SLOW,
    STOCH_DEFAULT_D_PERIOD,
    STOCH_DEFAULT_K_PERIOD,
    STOCH_DEFAULT_SMOOTH_K,
    calculate_adx,
    calculate_macd,
    calculate_stochastic,
)


class TestCalculateMacd:
    """Tests for calculate_macd function."""

    def test_normal_case(self):
        """Test MACD calculation with sufficient data."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        macd_data = calculate_macd(prices, fast=12, slow=26, signal=9)

        assert "macd" in macd_data
        assert "signal" in macd_data
        assert "histogram" in macd_data
        assert len(macd_data["macd"]) == len(prices)
        assert len(macd_data["signal"]) == len(prices)
        assert len(macd_data["histogram"]) == len(prices)
        assert macd_data["macd"].index.equals(prices.index)

    def test_insufficient_data(self):
        """Test MACD returns empty Series when data is insufficient."""
        prices = pd.Series(
            [100, 101, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        macd_data = calculate_macd(prices, fast=12, slow=26, signal=9)

        assert len(macd_data["macd"]) == 0
        assert len(macd_data["signal"]) == 0
        assert len(macd_data["histogram"]) == 0

    def test_invalid_periods_raises_error(self):
        """Test that invalid periods raise ValueError."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Fast period must be a positive integer"):
            calculate_macd(prices, fast=0, slow=26, signal=9)

        with pytest.raises(ValueError, match="Slow period must be a positive integer"):
            calculate_macd(prices, fast=12, slow=-1, signal=9)

        with pytest.raises(ValueError, match="Signal period must be a positive integer"):
            calculate_macd(prices, fast=12, slow=26, signal=0)

        with pytest.raises(ValueError, match="Slow period must be greater than fast period"):
            calculate_macd(prices, fast=26, slow=12, signal=9)

    def test_flat_prices(self):
        """Test MACD with constant prices."""
        prices = pd.Series(
            [100.0] * 100,
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )

        macd_data = calculate_macd(prices, fast=12, slow=26, signal=9)

        assert len(macd_data["macd"]) == len(prices)
        # For constant prices, MACD line and signal should be 0
        valid_macd = macd_data["macd"].dropna()
        assert (valid_macd == 0).all()

    def test_histogram_calculation(self):
        """Test that histogram = MACD - signal."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        macd_data = calculate_macd(prices, fast=12, slow=26, signal=9)

        # Verify histogram is macd - signal
        expected_histogram = macd_data["macd"] - macd_data["signal"]
        pd.testing.assert_series_equal(
            macd_data["histogram"].dropna(),
            expected_histogram.dropna(),
            check_names=False,
        )

    def test_default_values(self):
        """Test that default MACD parameters are correct."""
        assert MACD_DEFAULT_FAST == 12
        assert MACD_DEFAULT_SLOW == 26
        assert MACD_DEFAULT_SIGNAL == 9


class TestCalculateAdx:
    """Tests for calculate_adx function."""

    def test_normal_case(self):
        """Test ADX calculation with sufficient data."""
        high = pd.Series(
            range(105, 155),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(103, 153),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        adx_data = calculate_adx(high, low, close, period=14)

        assert "adx" in adx_data
        assert "plus_di" in adx_data
        assert "minus_di" in adx_data
        assert len(adx_data["adx"]) == len(close)
        assert adx_data["adx"].index.equals(close.index)

    def test_insufficient_data(self):
        """Test ADX returns empty Series when data is insufficient."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100, 102, 104],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [103, 108, 106],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        adx_data = calculate_adx(high, low, close, period=14)

        assert len(adx_data["adx"]) == 0
        assert len(adx_data["plus_di"]) == 0
        assert len(adx_data["minus_di"]) == 0

    def test_invalid_period_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        high = pd.Series(
            [105.0, 110.0, 108.0],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100.0, 102.0, 104.0],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [103.0, 108.0, 106.0],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_adx(high, low, close, period=0)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_adx(high, low, close, period=-1)

    def test_flat_prices(self):
        """Test ADX with constant prices (no trend)."""
        high = pd.Series(
            [100.0] * 100,
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100.0] * 100,
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100.0] * 100,
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )

        adx_data = calculate_adx(high, low, close, period=14)

        assert len(adx_data["adx"]) == len(close)
        # For flat prices, ADX should be very low (weak/no trend)
        valid_adx = adx_data["adx"].dropna()
        assert (valid_adx < 20).any() or len(valid_adx) == 0

    def test_default_period_is_14(self):
        """Test that default ADX period is 14."""
        assert ADX_DEFAULT_PERIOD == 14

    def test_di_values_in_range(self):
        """Test that +DI and -DI values are in valid range [0, 100]."""
        high = pd.Series(
            range(105, 155),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(103, 153),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        adx_data = calculate_adx(high, low, close, period=14)

        valid_plus_di = adx_data["plus_di"].dropna()
        valid_minus_di = adx_data["minus_di"].dropna()

        assert (valid_plus_di >= 0).all()
        assert (valid_plus_di <= 100).all()
        assert (valid_minus_di >= 0).all()
        assert (valid_minus_di <= 100).all()


class TestCalculateStochastic:
    """Tests for calculate_stochastic function."""

    def test_normal_case(self):
        """Test Stochastic calculation with sufficient data."""
        high = pd.Series(
            range(105, 155),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(103, 153),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        stoch_data = calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3)

        assert "percent_k" in stoch_data
        assert "percent_d" in stoch_data
        assert len(stoch_data["percent_k"]) == len(close)
        assert len(stoch_data["percent_d"]) == len(close)
        assert stoch_data["percent_k"].index.equals(close.index)

    def test_insufficient_data(self):
        """Test Stochastic returns empty Series when data is insufficient."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100, 102, 104],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [103, 108, 106],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        stoch_data = calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3)

        assert len(stoch_data["percent_k"]) == 0
        assert len(stoch_data["percent_d"]) == 0

    def test_invalid_periods_raises_error(self):
        """Test that invalid periods raise ValueError."""
        high = pd.Series(
            range(105, 155),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(103, 153),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="K period must be a positive integer"):
            calculate_stochastic(high, low, close, k_period=0, d_period=3, smooth_k=3)

        with pytest.raises(ValueError, match="D period must be a positive integer"):
            calculate_stochastic(high, low, close, k_period=14, d_period=0, smooth_k=3)

        with pytest.raises(ValueError, match="Smooth K period must be a positive integer"):
            calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=0)

    def test_output_clamped_to_range(self):
        """Test that Stochastic output is clamped to [0, 100]."""
        # Create extreme price moves to potentially exceed bounds
        high = pd.Series(
            [100, 200, 100, 200, 100, 200, 100, 200, 100, 200,
             100, 200, 100, 200, 100, 200, 100, 200, 100, 200,
             100, 200, 100, 200, 100, 200, 100, 200, 100, 200],
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [90, 95, 90, 95, 90, 95, 90, 95, 90, 95,
             90, 95, 90, 95, 90, 95, 90, 95, 90, 95,
             90, 95, 90, 95, 90, 95, 90, 95, 90, 95],
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [95, 195, 95, 195, 95, 195, 95, 195, 95, 195,
             95, 195, 95, 195, 95, 195, 95, 195, 95, 195,
             95, 195, 95, 195, 95, 195, 95, 195, 95, 195],
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        stoch_data = calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3)

        valid_k = stoch_data["percent_k"].dropna()
        valid_d = stoch_data["percent_d"].dropna()

        assert (valid_k >= 0).all()
        assert (valid_k <= 100).all()
        assert (valid_d >= 0).all()
        assert (valid_d <= 100).all()

    def test_flat_prices(self):
        """Test Stochastic with constant prices."""
        high = pd.Series(
            [100.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        stoch_data = calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3)

        assert len(stoch_data["percent_k"]) == len(close)
        # For flat prices, all highs/lows/close are the same, %K should be 50 (neutral)
        valid_k = stoch_data["percent_k"].dropna()
        # When range is 0, we use 50 as neutral value
        assert (valid_k == 50.0).any() or len(valid_k) == 0

    def test_default_periods(self):
        """Test that default Stochastic parameters are correct."""
        assert STOCH_DEFAULT_K_PERIOD == 14
        assert STOCH_DEFAULT_D_PERIOD == 3
        assert STOCH_DEFAULT_SMOOTH_K == 3

    def test_no_smoothing(self):
        """Test Stochastic with smooth_k=1 (no smoothing)."""
        high = pd.Series(
            range(105, 155),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(103, 153),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        stoch_data = calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=1)

        assert len(stoch_data["percent_k"]) == len(close)


class TestIntegration:
    """Integration tests for MACD/ADX/Stochastic workflow."""

    def test_full_workflow(self, ohlcv_data):
        """Test complete analysis workflow."""
        high = pd.Series(
            ohlcv_data["high"],
            index=pd.date_range("2024-01-01", periods=len(ohlcv_data["high"]), freq="h", tz="UTC"),
        )
        low = pd.Series(
            ohlcv_data["low"],
            index=pd.date_range("2024-01-01", periods=len(ohlcv_data["low"]), freq="h", tz="UTC"),
        )
        close = pd.Series(
            ohlcv_data["close"],
            index=pd.date_range("2024-01-01", periods=len(ohlcv_data["close"]), freq="h", tz="UTC"),
        )

        # MACD
        macd_data = calculate_macd(close)
        assert len(macd_data["macd"]) == len(close)

        # ADX
        adx_data = calculate_adx(high, low, close)
        assert len(adx_data["adx"]) == len(close)

        # Stochastic
        stoch_data = calculate_stochastic(high, low, close)
        assert len(stoch_data["percent_k"]) == len(close)
