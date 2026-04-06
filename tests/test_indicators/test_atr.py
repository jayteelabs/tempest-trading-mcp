"""Unit tests for ATR indicator engine."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.volatility.atr import (
    ATR_DEFAULT_PERIOD,
    calculate_atr,
)


class TestCalculateAtr:
    """Tests for calculate_atr function."""

    def test_normal_case(self):
        """Test ATR calculation with sufficient data."""
        # Generate price series with known volatility
        high = pd.Series(
            [105, 110, 115, 112, 118, 120, 117, 122, 125, 128,
             130, 127, 133, 135, 132, 138, 140, 137, 142, 145],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [100, 102, 108, 105, 110, 115, 112, 118, 120, 123,
             125, 122, 128, 130, 127, 133, 135, 132, 137, 140],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [103, 108, 112, 110, 115, 118, 115, 120, 123, 126,
             128, 125, 131, 133, 130, 136, 138, 135, 140, 143],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == len(close)
        assert atr.index.equals(close.index)
        # ATR should be positive for volatile data
        valid_atr = atr.dropna()
        assert (valid_atr > 0).all()

    def test_insufficient_data(self):
        """Test ATR returns empty Series when data is insufficient."""
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

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == 0
        assert isinstance(atr, pd.Series)

    def test_exactly_period_length(self):
        """Test ATR calculation when data length equals period."""
        period = 10
        high = pd.Series(
            range(100, 110),
            index=pd.date_range("2024-01-01", periods=period, freq="h", tz="UTC"),
        )
        low = pd.Series(
            range(95, 105),
            index=pd.date_range("2024-01-01", periods=period, freq="h", tz="UTC"),
        )
        close = pd.Series(
            range(98, 108),
            index=pd.date_range("2024-01-01", periods=period, freq="h", tz="UTC"),
        )

        atr = calculate_atr(high, low, close, period=period)

        assert len(atr) == period

    def test_flat_prices(self):
        """Test ATR with constant prices (no volatility)."""
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

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == len(close)
        # ATR should be 0 for flat prices
        valid_atr = atr.dropna()
        assert (valid_atr == 0).all()

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
            calculate_atr(high, low, close, period=0)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_atr(high, low, close, period=-1)

    def test_empty_series(self):
        """Test ATR returns empty Series for empty input."""
        high = pd.Series(dtype=float)
        low = pd.Series(dtype=float)
        close = pd.Series(dtype=float)

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == 0
        assert isinstance(atr, pd.Series)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
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

        atr = calculate_atr(high, low, close, period=14)

        assert atr.index.tz is not None
        assert str(atr.index.tz) == "UTC"

    def test_default_period_is_14(self):
        """Test that default period is 14."""
        assert ATR_DEFAULT_PERIOD == 14

    def test_large_gaps(self):
        """Test ATR handles large price gaps correctly."""
        # Create series with large gaps
        high = pd.Series(
            [100, 150, 105, 160, 110, 170, 115, 180, 120, 190,
             125, 200, 130, 210, 135, 220, 140, 230, 145, 240],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        low = pd.Series(
            [95, 100, 100, 105, 105, 110, 110, 115, 115, 120,
             120, 125, 125, 130, 130, 135, 135, 140, 140, 145],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )
        close = pd.Series(
            [98, 145, 102, 155, 107, 165, 112, 175, 117, 185,
             122, 195, 127, 205, 132, 215, 137, 225, 142, 235],
            index=pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC"),
        )

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == len(close)
        # ATR should be large due to gaps
        valid_atr = atr.dropna()
        assert (valid_atr > 10).any()  # Significant volatility


class TestIntegration:
    """Integration tests for ATR indicator workflow."""

    def test_full_workflow(self, ohlcv_data):
        """Test complete ATR analysis workflow."""
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

        atr = calculate_atr(high, low, close, period=14)

        assert len(atr) == len(close)
        assert atr.index.tz is not None
