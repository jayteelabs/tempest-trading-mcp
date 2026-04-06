"""Unit tests for secondary momentum indicator engine (CCI, Williams %R, ROC)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.momentum.secondary_momentum import (
    CCI_DEFAULT_PERIOD,
    ROC_DEFAULT_PERIOD,
    WILLIAMS_R_DEFAULT_PERIOD,
    calculate_cci,
    calculate_roc,
    calculate_williams_r,
)


class TestCalculateCci:
    """Tests for calculate_cci function."""

    def test_normal_case(self):
        """Test CCI calculation with sufficient data."""
        high = pd.Series(
            [105, 110, 108, 112, 115, 113, 118, 120, 119, 122,
             125, 123, 128, 130, 127, 132, 135, 133, 138, 140],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96, 100, 103, 101, 106, 108, 107, 110,
             113, 111, 116, 118, 115, 120, 123, 121, 126, 128],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102, 107, 110, 108, 113, 115, 114, 117,
             120, 118, 123, 125, 122, 127, 130, 128, 133, 135],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        cci = calculate_cci(high, low, close, period=20)

        assert len(cci) == len(close)
        assert cci.index.equals(close.index)
        # CCI should be NaN for first (period - 1) values
        assert cci.iloc[:19].isna().all()
        # Last value should be valid
        assert not pd.isna(cci.iloc[-1])

    def test_insufficient_data(self):
        """Test CCI returns Series of NaN when data is insufficient."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        cci = calculate_cci(high, low, close, period=20)

        # Insufficient data returns Series of NaN aligned with input index
        assert len(cci) == len(close)
        assert cci.index.equals(close.index)
        assert cci.isna().all()

    def test_empty_input(self):
        """Test CCI with empty input Series returns empty Series."""
        high = pd.Series(dtype=float)
        low = pd.Series(dtype=float)
        close = pd.Series(dtype=float)

        cci = calculate_cci(high, low, close, period=20)

        assert len(cci) == 0
        assert isinstance(cci, pd.Series)

    def test_period_zero_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_cci(high, low, close, period=0)

    def test_period_negative_raises_error(self):
        """Test that negative period raises ValueError."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_cci(high, low, close, period=-5)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        high = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(90, 140),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(95, 145),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        cci = calculate_cci(high, low, close, period=20)

        assert cci.index.tz is not None
        assert str(cci.index.tz) == "UTC"

    def test_default_period_is_20(self):
        """Test that default period is 20."""
        assert CCI_DEFAULT_PERIOD == 20

    def test_cci_oscillates_around_zero(self):
        """Test that CCI oscillates around zero (overbought above +100, oversold below -100)."""
        # Create a trending series
        high = pd.Series(
            list(range(100, 200)) + list(range(200, 100, -1)),
            index=pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC"),
        )
        low = high - 5
        close = high - 2

        cci = calculate_cci(high, low, close, period=20)

        valid_cci = cci.dropna()
        # CCI should have values exceeding ±100 for strong trends
        assert (valid_cci > 100).any() or (valid_cci < -100).any()

    def test_exactly_period_length(self):
        """Test CCI with exactly period data points returns valid result at end."""
        # Use varying values so MAD is non-zero
        high = pd.Series(
            [105 + i * 0.5 for i in range(20)],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95 + i * 0.5 for i in range(20)],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100 + i * 0.5 for i in range(20)],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        cci = calculate_cci(high, low, close, period=20)

        # With varying data, CCI should be valid at the last index
        assert not pd.isna(cci.iloc[-1])

    def test_with_nan_values(self):
        """Test CCI handles NaN values in input by dropping them."""
        high = pd.Series(
            [105, np.nan, 108, 112, 115, 113, 118, 120, 119, 122,
             125, 123, 128, 130, 127, 132, 135, 133, 138, 140],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, np.nan, 100, 103, 101, 106, 108, 107, 110,
             113, 111, 116, 118, 115, 120, 123, 121, 126, 128],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102, np.nan, 110, 108, 113, 115, 114, 117,
             120, 118, 123, 125, 122, 127, 130, 128, 133, 135],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        cci = calculate_cci(high, low, close, period=14)

        # Should return valid series (NaN rows dropped during alignment)
        assert isinstance(cci, pd.Series)


class TestCalculateWilliamsR:
    """Tests for calculate_williams_r function."""

    def test_normal_case(self):
        """Test Williams %R calculation with sufficient data."""
        high = pd.Series(
            [105, 110, 108, 112, 115, 113, 118, 120, 119, 122,
             125, 123, 128, 130, 127, 132, 135, 133, 138, 140],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96, 100, 103, 101, 106, 108, 107, 110,
             113, 111, 116, 118, 115, 120, 123, 121, 126, 128],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102, 107, 110, 108, 113, 115, 114, 117,
             120, 118, 123, 125, 122, 127, 130, 128, 133, 135],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=14)

        assert len(williams_r) == len(close)
        assert williams_r.index.equals(close.index)
        # Williams %R should be NaN for first (period - 1) values
        assert williams_r.iloc[:13].isna().all()
        # Last value should be valid
        assert not pd.isna(williams_r.iloc[-1])

    def test_insufficient_data(self):
        """Test Williams %R returns Series of NaN when data is insufficient."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=14)

        # Insufficient data returns Series of NaN aligned with input index
        assert len(williams_r) == len(close)
        assert williams_r.index.equals(close.index)
        assert williams_r.isna().all()

    def test_empty_input(self):
        """Test Williams %R with empty input Series returns empty Series."""
        high = pd.Series(dtype=float)
        low = pd.Series(dtype=float)
        close = pd.Series(dtype=float)

        williams_r = calculate_williams_r(high, low, close, period=14)

        assert len(williams_r) == 0
        assert isinstance(williams_r, pd.Series)

    def test_period_zero_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_williams_r(high, low, close, period=0)

    def test_period_negative_raises_error(self):
        """Test that negative period raises ValueError."""
        high = pd.Series(
            [105, 110, 108],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 98, 96],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100, 105, 102],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_williams_r(high, low, close, period=-5)

    def test_flat_market_returns_negative_50(self):
        """Test Williams %R returns -50.0 when highest equals lowest (flat market)."""
        # Create flat market where high = low = close for all bars
        high = pd.Series(
            [100.0] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100.0] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100.0] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=14)

        # For flat market, should return -50.0 for valid bars
        valid_vals = williams_r.dropna()
        assert (valid_vals == -50.0).all()

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        high = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(90, 140),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(95, 145),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=14)

        assert williams_r.index.tz is not None
        assert str(williams_r.index.tz) == "UTC"

    def test_default_period_is_14(self):
        """Test that default period is 14."""
        assert WILLIAMS_R_DEFAULT_PERIOD == 14

    def test_range_0_to_minus_100(self):
        """Test that Williams %R ranges between 0 and -100."""
        # Create trending data
        high = pd.Series(
            list(range(100, 200)),
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )
        low = high - 5
        close = high - 2

        williams_r = calculate_williams_r(high, low, close, period=14)

        valid_williams_r = williams_r.dropna()
        # Williams %R should be between -100 and 0
        assert (valid_williams_r >= -100).all()
        assert (valid_williams_r <= 0).all()

    def test_exactly_period_length(self):
        """Test Williams %R with exactly period data points."""
        high = pd.Series(
            [105.0] * 14,
            index=pd.date_range("2024-01-01", periods=14, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95.0] * 14,
            index=pd.date_range("2024-01-01", periods=14, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [100.0] * 14,
            index=pd.date_range("2024-01-01", periods=14, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=14)

        # Last value should be valid (and should be -50 for flat market)
        assert not pd.isna(williams_r.iloc[-1])
        assert williams_r.iloc[-1] == -50.0

    def test_close_at_high_gives_zero(self):
        """Test Williams %R close to 0 when close equals highest high."""
        high = pd.Series(
            [100, 105, 110, 115, 120],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [90, 95, 100, 105, 110],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        # Close at the highest high (120)
        close = pd.Series(
            [95, 100, 105, 110, 120],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=5)

        # Last bar: HH=120, LL=110, C=120 => %R = -100 * (120-120)/(120-110) = 0
        assert abs(williams_r.iloc[-1]) < 0.001

    def test_close_at_low_gives_minus_100(self):
        """Test Williams %R close to -100 when close equals lowest low."""
        high = pd.Series(
            [100, 105, 110, 115, 120],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [90, 95, 100, 105, 110],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )
        # Close equals lowest low (90) only at the last bar
        # Lowest low over period 0-4 is 90 (at index 0), but close at index 4 is also 90
        close = pd.Series(
            [95, 100, 105, 110, 90],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        williams_r = calculate_williams_r(high, low, close, period=5)

        # Last bar: HH=120, LL=90, C=90 => %R = -100 * (120-90)/(120-90) = -100
        assert abs(williams_r.iloc[-1] - (-100)) < 0.001


class TestCalculateRoc:
    """Tests for calculate_roc function."""

    def test_normal_case(self):
        """Test ROC calculation with sufficient data."""
        prices = pd.Series(
            [100, 102, 101, 105, 103, 107, 110, 108, 112, 115,
             113, 117, 120, 118, 122, 125, 123, 127, 130, 128],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        assert len(roc) == len(prices)
        assert roc.index.equals(prices.index)
        # First 12 values should be NaN (no lookback available)
        assert roc.iloc[:12].isna().all()
        # Last value should be valid
        assert not pd.isna(roc.iloc[-1])

    def test_insufficient_data(self):
        """Test ROC returns Series of NaN when data is insufficient."""
        prices = pd.Series(
            [100, 102, 101],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        # Insufficient data returns Series of NaN aligned with input index
        assert len(roc) == len(prices)
        assert roc.index.equals(prices.index)
        assert roc.isna().all()

    def test_empty_input(self):
        """Test ROC with empty input Series returns empty Series."""
        prices = pd.Series(dtype=float)

        roc = calculate_roc(prices, period=12)

        assert len(roc) == 0
        assert isinstance(roc, pd.Series)

    def test_period_zero_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        prices = pd.Series(
            [100, 102, 101],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_roc(prices, period=0)

    def test_period_negative_raises_error(self):
        """Test that negative period raises ValueError."""
        prices = pd.Series(
            [100, 102, 101],
            index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_roc(prices, period=-5)

    def test_positive_momentum(self):
        """Test ROC with upward trending prices gives positive values."""
        # Price doubles over period
        prices = pd.Series(
            [100] * 15 + [200],
            index=pd.date_range("2024-01-01", periods=16, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        valid_roc = roc.dropna()
        # ROC should be positive (100% gain)
        assert valid_roc.iloc[-1] == 100.0

    def test_negative_momentum(self):
        """Test ROC with downward trending prices gives negative values."""
        # Price halves over period
        prices = pd.Series(
            [200] * 15 + [100],
            index=pd.date_range("2024-01-01", periods=16, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        valid_roc = roc.dropna()
        # ROC should be negative (50% loss)
        assert valid_roc.iloc[-1] == -50.0

    def test_no_momentum_flat_prices(self):
        """Test ROC with flat prices gives zero."""
        prices = pd.Series(
            [100.0] * 20,
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        valid_roc = roc.dropna()
        # ROC should be 0 for flat prices
        assert (valid_roc == 0.0).all()

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        assert roc.index.tz is not None
        assert str(roc.index.tz) == "UTC"

    def test_default_period_is_12(self):
        """Test that default period is 12."""
        assert ROC_DEFAULT_PERIOD == 12

    def test_exactly_period_plus_one_length(self):
        """Test ROC with exactly period + 1 data points returns one valid value."""
        prices = pd.Series(
            [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160],
            index=pd.date_range("2024-01-01", periods=13, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        # First 12 values should be NaN, last should be valid
        assert roc.iloc[:12].isna().all()
        assert not pd.isna(roc.iloc[-1])

    def test_period_boundary_at_1(self):
        """Test ROC at period boundary (period=1)."""
        prices = pd.Series(
            [100, 105, 110, 115, 120],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=1)

        # Period 1 means compare to previous bar
        assert len(roc) == len(prices)
        # First value should be NaN (no previous)
        assert pd.isna(roc.iloc[0])
        # Remaining should be valid
        assert not roc.iloc[1:].isna().all()

    def test_with_nan_values(self):
        """Test ROC handles NaN values in input by dropping them."""
        prices = pd.Series(
            [100, np.nan, 105, 110, np.nan, 115, 120, 125, 130, 135,
             140, 145, 150, 155, 160, 165, 170, 175, 180, 185],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        roc = calculate_roc(prices, period=12)

        # Should return valid series (NaN rows dropped)
        assert isinstance(roc, pd.Series)


class TestIntegration:
    """Integration tests for secondary momentum indicators."""

    def test_all_indicators_workflow(self):
        """Test complete secondary momentum analysis workflow."""
        high = pd.Series(
            range(100, 120),
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = high - 5
        close = high - 2
        prices = close

        cci = calculate_cci(high, low, close, period=14)
        williams_r = calculate_williams_r(high, low, close, period=14)
        roc = calculate_roc(prices, period=12)

        assert len(cci) == len(high)
        assert len(williams_r) == len(high)
        assert len(roc) == len(prices)

        # All should have UTC-aware index
        assert cci.index.tz is not None
        assert williams_r.index.tz is not None
        assert roc.index.tz is not None

    def test_default_periods_exported(self):
        """Test that all default periods are exported correctly."""
        assert CCI_DEFAULT_PERIOD == 20
        assert WILLIAMS_R_DEFAULT_PERIOD == 14
        assert ROC_DEFAULT_PERIOD == 12

    def test_consistent_index_across_indicators(self):
        """Test that all indicators preserve the same index."""
        high = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )
        low = high - 5
        close = high - 2
        prices = close

        cci = calculate_cci(high, low, close, period=20)
        williams_r = calculate_williams_r(high, low, close, period=14)
        roc = calculate_roc(prices, period=12)

        # All should share the same index
        assert cci.index.equals(williams_r.index)
        assert williams_r.index.equals(roc.index)
