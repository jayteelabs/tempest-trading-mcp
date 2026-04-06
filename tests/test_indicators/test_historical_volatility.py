"""Unit tests for Historical Volatility indicator engine."""

import pandas as pd
import pytest
import numpy as np

from tempest_mcp.indicators.volatility import calculate_historical_volatility


class TestCalculateHistoricalVolatility:
    """Tests for calculate_historical_volatility function."""

    def test_normal_case_annualized(self):
        """Test Historical Volatility calculation with annualization."""
        np.random.seed(42)
        # Generate price series with known volatility
        returns = np.random.normal(0.0005, 0.02, 300)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=252, annualize=True)

        assert len(hv) > 0
        assert hv.index.tz is not None
        assert str(hv.index.tz) == "UTC"
        # Annualized HV should be positive and reasonable
        assert (hv > 0).all()
        assert (hv < 5).all()  # Should be less than 500% annualized vol

    def test_normal_case_not_annualized(self):
        """Test Historical Volatility without annualization."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 300)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=20, annualize=False)

        assert len(hv) > 0
        assert (hv > 0).all()

    def test_insufficient_data(self):
        """Test returns empty Series when data is insufficient."""
        prices = pd.Series(
            [100, 101, 102, 103, 104],
            index=pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=20)

        assert len(hv) == 0
        assert isinstance(hv, pd.Series)

    def test_exactly_period_plus_one_length(self):
        """Test calculation when data length equals period + 1."""
        period = 20
        prices = pd.Series(
            range(100, 100 + period + 1),
            index=pd.date_range("2024-01-01", periods=period + 1, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=period)

        # Should have exactly 1 value (the first complete window)
        assert len(hv) == 1

    def test_flat_prices(self):
        """Test Historical Volatility with constant prices."""
        prices = pd.Series(
            [100.0] * 300,
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=252)

        assert len(hv) > 0
        # HV should be 0 for flat prices
        assert (hv == 0).all()

    def test_low_volatility(self):
        """Test with low volatility prices."""
        np.random.seed(42)
        # Low volatility: small random steps
        returns = np.random.normal(0, 0.001, 300)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=252, annualize=True)

        assert len(hv) > 0
        assert hv.mean() < 0.5  # Low annualized volatility

    def test_high_volatility(self):
        """Test with high volatility prices."""
        np.random.seed(42)
        # High volatility: large random steps
        returns = np.random.normal(0, 0.05, 300)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=252, annualize=True)

        assert len(hv) > 0
        assert hv.mean() > 0.5  # High annualized volatility

    def test_invalid_period_raises_error(self):
        """Test that period < 2 raises ValueError."""
        prices = pd.Series(
            [100.0] * 30,
            index=pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="Period must be at least 2"):
            calculate_historical_volatility(prices, period=1)

        with pytest.raises(ValueError, match="Period must be at least 2"):
            calculate_historical_volatility(prices, period=0)

    def test_empty_series(self):
        """Test returns empty Series for empty input."""
        prices = pd.Series(dtype=float)

        hv = calculate_historical_volatility(prices, period=20)

        assert len(hv) == 0
        assert isinstance(hv, pd.Series)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, 300)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(prices, period=252)

        assert hv.index.tz is not None
        assert str(hv.index.tz) == "UTC"

    def test_annualization_factor(self):
        """Test that annualize=True multiplies by sqrt(252)."""
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 300)
        prices = pd.Series(
            100 * np.exp(np.cumsum(returns)),
            index=pd.date_range("2024-01-01", periods=300, freq="h", tz="UTC"),
        )

        hv_annualized = calculate_historical_volatility(prices, period=20, annualize=True)
        hv_not_annualized = calculate_historical_volatility(prices, period=20, annualize=False)

        # The annualized version should be larger by approximately sqrt(252)
        ratio = hv_annualized.mean() / hv_not_annualized.mean()
        assert 15 < ratio < 20  # sqrt(252) ≈ 15.87


class TestIntegration:
    """Integration tests for Historical Volatility indicator workflow."""

    def test_full_workflow(self, price_data):
        """Test complete Historical Volatility analysis workflow."""
        close = pd.Series(
            price_data,
            index=pd.date_range("2024-01-01", periods=len(price_data), freq="h", tz="UTC"),
        )

        hv = calculate_historical_volatility(close, period=20)

        assert len(hv) > 0
        assert hv.index.tz is not None
        assert (hv >= 0).all()
