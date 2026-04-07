"""Unit tests for MFI indicator engine."""

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.indicators.volume import calculate_mfi


class TestCalculateMFI:
    """Tests for calculate_mfi function."""

    def test_normal_case(self):
        """Test MFI calculation with sufficient data."""
        np.random.seed(42)
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=14)

        assert len(mfi) > 0
        assert mfi.index.tz is not None
        assert str(mfi.index.tz) == "UTC"
        assert (mfi >= 0).all()
        assert (mfi <= 100).all()

    def test_insufficient_data(self):
        """Test returns empty Series when data is insufficient."""
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=14)

        assert len(mfi) == 0
        assert isinstance(mfi, pd.Series)

    def test_exactly_period_plus_one_length(self):
        """Test calculation when data length equals period + 1."""
        period = 14
        n = period + 1
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=period)

        assert len(mfi) == 1

    def test_flat_sideways_price(self):
        """Test MFI with flat/sideways price."""
        period = 14
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        typical = 100.0
        high = pd.Series([typical + 1] * n, index=idx)
        low = pd.Series([typical - 1] * n, index=idx)
        close = pd.Series([typical] * n, index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=period)

        assert len(mfi) > 0

    def test_extreme_volume_spike(self):
        """Test MFI with extreme volume spike."""
        period = 14
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)
        volume.iloc[15] = 1000000

        mfi = calculate_mfi(high, low, close, volume, period=period)

        assert len(mfi) > 0
        assert (mfi >= 0).all()
        assert (mfi <= 100).all()

    def test_invalid_period_raises_error(self):
        """Test that period <= 0 raises ValueError."""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_mfi(high, low, close, volume, period=0)

        with pytest.raises(ValueError, match="Period must be a positive integer"):
            calculate_mfi(high, low, close, volume, period=-1)

    def test_empty_series(self):
        """Test returns empty Series for empty input."""
        high = pd.Series(dtype=float)
        low = pd.Series(dtype=float)
        close = pd.Series(dtype=float)
        volume = pd.Series(dtype=float)

        mfi = calculate_mfi(high, low, close, volume, period=14)

        assert len(mfi) == 0
        assert isinstance(mfi, pd.Series)

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raises ValueError."""
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n - 5), index=idx[: n - 5])
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        with pytest.raises(ValueError, match="same length"):
            calculate_mfi(high, low, close, volume, period=14)

    def test_extreme_volume_returns_nan_or_valid(self):
        """Test MFI handles extreme volume without overflow."""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        # Extreme volume that could cause infinity in money flow
        volume = pd.Series([1e308] * n, index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=14)

        # Should either return valid MFI values or NaN for affected periods
        # No overflow should occur
        assert isinstance(mfi, pd.Series)

    def test_utc_aware_index_preserved(self):
        """Test that UTC-aware index is preserved in output."""
        n = 30
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=14)

        assert mfi.index.tz is not None
        assert str(mfi.index.tz) == "UTC"

    def test_output_range_clamped(self):
        """Test that MFI output is clamped to [0, 100]."""
        period = 14
        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=period)

        assert (mfi >= 0).all()
        assert (mfi <= 100).all()

    def test_edge_case_sum_negative_zero_mfi_100(self):
        """Test that MFI = 100 when sum_negative == 0 (no selling pressure)."""
        period = 14
        n = period + 1
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        base = 100.0
        high = pd.Series([base + i + 2 for i in range(n)], index=idx)
        low = pd.Series([base + i for i in range(n)], index=idx)
        close = pd.Series([base + i + 1 for i in range(n)], index=idx)
        volume = pd.Series([1000] * n, index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=period)

        assert len(mfi) == 1
        assert mfi.iloc[0] == 100.0

    def test_edge_case_sum_positive_zero_mfi_0(self):
        """Test that MFI = 0 when sum_positive == 0 (no buying pressure)."""
        period = 14
        n = period + 1
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        base = 100.0
        high = pd.Series([base - i + 2 for i in range(n)], index=idx)
        low = pd.Series([base - i for i in range(n)], index=idx)
        close = pd.Series([base - i + 1 for i in range(n)], index=idx)
        volume = pd.Series([1000] * n, index=idx)

        mfi = calculate_mfi(high, low, close, volume, period=period)

        assert len(mfi) == 1
        assert mfi.iloc[0] == 0.0


class TestIntegration:
    """Integration tests for MFI indicator workflow."""

    def test_full_workflow(self, ohlcv_data):
        """Test complete MFI analysis workflow."""
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
        volume = pd.Series(
            ohlcv_data["volume"],
            index=pd.date_range(
                "2024-01-01", periods=len(ohlcv_data["volume"]), freq="h", tz="UTC"
            ),
        )

        mfi = calculate_mfi(high, low, close, volume, period=14)

        assert len(mfi) > 0
        assert mfi.index.tz is not None
        assert (mfi >= 0).all()
        assert (mfi <= 100).all()
