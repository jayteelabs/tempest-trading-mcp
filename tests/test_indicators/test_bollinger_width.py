"""Unit tests for Bollinger Width indicator engine."""

import pandas as pd
import pytest
import numpy as np

from tempest_mcp.indicators.volatility import calculate_bollinger_width


class TestCalculateBollingerWidth:
    """Tests for calculate_bollinger_width function."""

    def test_normal_case(self):
        """Test Bollinger Width calculation with sufficient data."""
        # Generate price series with known volatility
        np.random.seed(42)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(50)),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=20, std_dev=2.0)

        assert len(bw) > 0
        assert bw.index.tz is not None
        assert str(bw.index.tz) == "UTC"
        # Width should be positive
        assert (bw > 0).all()

    def test_insufficient_data(self):
        """Test returns empty Series when data is insufficient."""
        prices = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=20)

        assert len(bw) == 0
        assert isinstance(bw, pd.Series)

    def test_exactly_period_length(self):
        """Test calculation when data length equals period."""
        period = 20
        prices = pd.Series(
            range(100, 100 + period),
            index=pd.date_range("2024-01-01", periods=period, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=period)

        assert len(bw) == 1  # Only the last value is valid

    def test_flat_prices(self):
        """Test Bollinger Width with constant prices."""
        prices = pd.Series(
            [100.0] * 50,
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=20)

        assert len(bw) > 0
        # Width should be 0 for flat prices
        assert (bw == 0).all()

    def test_high_volatility(self):
        """Test with highly volatile prices."""
        np.random.seed(42)
        # High volatility: random walk with large steps
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(100) * 5),
            index=pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=20)

        assert len(bw) > 0
        # Width should be larger for volatile data
        assert bw.mean() > 0.05  # Some meaningful width

    def test_invalid_period_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        prices = pd.Series(
            [100.0] * 30,
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_bollinger_width(prices, period=0)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_bollinger_width(prices, period=-1)

    def test_invalid_std_dev_raises_error(self):
        """Test that std_dev <= 0 raises ValueError."""
        prices = pd.Series(
            [100.0] * 30,
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="std_dev must be a positive number"):
            calculate_bollinger_width(prices, period=20, std_dev=0)

        with pytest.raises(ValueError, match="std_dev must be a positive number"):
            calculate_bollinger_width(prices, period=20, std_dev=-1)

    def test_empty_series(self):
        """Test returns empty Series for empty input."""
        prices = pd.Series(dtype=float)

        bw = calculate_bollinger_width(prices, period=20)

        assert len(bw) == 0
        assert isinstance(bw, pd.Series)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        prices = pd.Series(
            range(100, 150),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=20)

        assert bw.index.tz is not None
        assert str(bw.index.tz) == "UTC"

    def test_dimensionless_width(self):
        """Test that width is dimensionless (normalized by middle band)."""
        np.random.seed(42)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(50)),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(prices, period=20)

        # Width should be a small decimal since it's normalized
        assert (bw < 1).all()

    def test_different_std_dev_multiplier(self):
        """Test that different std_dev values produce different widths."""
        np.random.seed(42)
        prices = pd.Series(
            100 + np.cumsum(np.random.randn(50)),
            index=pd.date_range("2024-01-01", periods=50, freq="h", tz="UTC"),
        )

        bw_1 = calculate_bollinger_width(prices, period=20, std_dev=1.0)
        bw_2 = calculate_bollinger_width(prices, period=20, std_dev=2.0)

        # Width should be larger with higher std_dev
        assert bw_2.mean() > bw_1.mean()


class TestIntegration:
    """Integration tests for Bollinger Width indicator workflow."""

    def test_full_workflow(self, ohlcv_data):
        """Test complete Bollinger Width analysis workflow."""
        close = pd.Series(
            ohlcv_data["close"],
            index=pd.date_range("2024-01-01", periods=len(ohlcv_data["close"]), freq="h", tz="UTC"),
        )

        bw = calculate_bollinger_width(close, period=20)

        assert len(bw) > 0
        assert bw.index.tz is not None
        assert (bw >= 0).all()
