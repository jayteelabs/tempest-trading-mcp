"""Unit tests for Volume Profile indicator."""

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.indicators.volume.volume_profile import (
    COL_BIN_CANDLE_COUNT,
    COL_BIN_HIGH,
    COL_BIN_LOW,
    COL_BIN_MID,
    COL_BIN_VOLUME,
    COL_IS_HVN,
    COL_IS_LVN,
    _build_dynamic_bin_edges,
    _calculate_value_area,
    calculate_volume_profile,
)


def _create_ohlcv(
    n: int = 100,
    start_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic OHLCV data for testing.

    Args:
        n: Number of candles.
        start_price: Starting price.
        volatility: Price volatility factor.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with OHLCV columns and UTC-aware DatetimeIndex.
    """
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    # Generate price series with some trend and noise
    returns = np.random.normal(0.0005, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    # Generate high/low with random spreads
    high_spread = np.random.uniform(0.001, 0.015, n) * close
    low_spread = np.random.uniform(0.001, 0.015, n) * close

    high = close + high_spread
    low = close - low_spread
    open_prices = np.random.uniform(low, high)

    # Ensure OHLC relationships hold
    high = np.maximum.reduce([open_prices, high, low, close])
    low = np.minimum.reduce([open_prices, high, low, close])

    # Generate volume with some pattern
    volume = np.random.uniform(1000, 5000, n) * (1 + np.abs(returns) * 10)

    return pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


class TestCalculateVolumeProfileFixed:
    """Tests for calculate_volume_profile with fixed mode."""

    def test_basic_fixed_profile(self):
        """Test basic fixed-range volume profile calculation."""
        ohlcv = _create_ohlcv(n=100)

        profile = calculate_volume_profile(ohlcv, bin_count=50, profile_type="fixed")

        # Should return a DataFrame
        assert isinstance(profile, pd.DataFrame)
        assert len(profile) > 0

        # Check required columns exist
        assert COL_BIN_LOW in profile.columns
        assert COL_BIN_HIGH in profile.columns
        assert COL_BIN_MID in profile.columns
        assert COL_BIN_VOLUME in profile.columns
        assert COL_BIN_CANDLE_COUNT in profile.columns
        assert COL_IS_HVN in profile.columns
        assert COL_IS_LVN in profile.columns
        assert "in_value_area" in profile.columns

        # Check bin boundaries are valid
        assert (profile[COL_BIN_HIGH] > profile[COL_BIN_LOW]).all()

        # Check bin mids are between low and high
        assert (profile[COL_BIN_MID] >= profile[COL_BIN_LOW]).all()
        assert (profile[COL_BIN_MID] <= profile[COL_BIN_HIGH]).all()

        # Check volumes are non-negative
        assert (profile[COL_BIN_VOLUME] >= 0).all()

        # Check HVN and LVN are mutually exclusive (mostly)
        # A bin can technically be both if threshold boundaries overlap
        hvn_count = profile[COL_IS_HVN].sum()
        lvn_count = profile[COL_IS_LVN].sum()
        assert hvn_count >= 1, "Should have at least one HVN"
        assert lvn_count >= 1, "Should have at least one LVN"

    def test_fixed_profile_with_known_fixture(self):
        """Test deterministic fixed profile on known OHLCV fixture."""
        # Create simple fixture with known price range
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")

        # Price range: low=100, high=110
        ohlcv = pd.DataFrame(
            {
                "open": [105, 105, 105, 105, 105, 105, 105, 105, 105, 105],
                "high": [110, 110, 110, 110, 110, 110, 110, 110, 110, 110],
                "low": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
                "close": [105, 105, 105, 105, 105, 105, 105, 105, 105, 105],
                "volume": [1000] * 10,
            },
            index=dates,
        )

        profile = calculate_volume_profile(ohlcv, bin_count=10, profile_type="fixed")

        # All volume should be in bins that overlap [100, 110]
        # With uniform price at 105, all volume goes to middle bins
        assert profile[COL_BIN_VOLUME].sum() == 10000  # Total volume

        # POC should be in the middle of the price range
        poc_price = profile.attrs["poc_price"]
        assert 100 <= poc_price <= 110

    def test_single_price_value(self):
        """Test profile with single price value (flat market)."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [100.0] * 10,
                "low": [100.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        profile = calculate_volume_profile(ohlcv, bin_count=10, profile_type="fixed")

        # Should still produce valid profile
        assert isinstance(profile, pd.DataFrame)
        assert len(profile) >= 1
        # All volume should be in the single price bin
        assert profile[COL_BIN_VOLUME].sum() == 10000

    def test_empty_ohlcv_raises(self):
        """Test that empty OHLCV raises ValueError."""
        ohlcv = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.date_range("2024-01-01", periods=0, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="must not be empty"):
            calculate_volume_profile(ohlcv)

    def test_missing_columns_raises(self):
        """Test that missing OHLCV columns raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "close": [105.0] * 10,
                # missing 'low' and 'volume'
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="Missing:"):
            calculate_volume_profile(ohlcv)

    def test_tz_naive_index_raises(self):
        """Test that UTC-naive DatetimeIndex raises TypeError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h")  # No tz
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="UTC-aware"):
            calculate_volume_profile(ohlcv)

    def test_duplicate_index_raises(self):
        """Test that duplicate DatetimeIndex raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        dates = dates.append(dates[:3])  # Add duplicates

        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 13,
                "high": [110.0] * 13,
                "low": [90.0] * 13,
                "close": [105.0] * 13,
                "volume": [1000] * 13,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="duplicate"):
            calculate_volume_profile(ohlcv)

    def test_non_monotonic_index_raises(self):
        """Test that non-monotonic DatetimeIndex raises ValueError."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        dates = dates[::-1]  # Reverse

        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [110.0] * 10,
                "low": [90.0] * 10,
                "close": [105.0] * 10,
                "volume": [1000] * 10,
            },
            index=dates,
        )

        with pytest.raises(ValueError, match="monotonic"):
            calculate_volume_profile(ohlcv)

    def test_invalid_bin_count_raises(self):
        """Test that bin_count <= 0 raises ValueError."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match="bin_count"):
            calculate_volume_profile(ohlcv, bin_count=0)

        with pytest.raises(ValueError, match="bin_count"):
            calculate_volume_profile(ohlcv, bin_count=-1)

    def test_invalid_value_area_pct_raises(self):
        """Test that value_area_pct outside (0, 1] raises ValueError."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match="value_area_pct"):
            calculate_volume_profile(ohlcv, value_area_pct=0)

        with pytest.raises(ValueError, match="value_area_pct"):
            calculate_volume_profile(ohlcv, value_area_pct=1.01)

        with pytest.raises(ValueError, match="value_area_pct"):
            calculate_volume_profile(ohlcv, value_area_pct=-0.5)

    def test_invalid_profile_type_raises(self):
        """Test that unsupported profile types fail deterministically."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match="Invalid profile_type"):
            calculate_volume_profile(ohlcv, profile_type="range")

    def test_vah_uses_upper_edge_of_last_value_area_bin(self):
        """Test that VAH resolves to the upper edge of the last included bin."""
        dates = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [105.0] * 10,
                "high": [105.0] * 9 + [110.0],
                "low": [105.0] * 9 + [100.0],
                "close": [105.0] * 10,
                "volume": [1000.0] * 9 + [0.0],
            },
            index=dates,
        )

        profile = calculate_volume_profile(ohlcv, bin_count=10, profile_type="fixed")
        poc_bin_idx = profile.attrs["poc_bin_idx"]

        assert profile.attrs["val_price"] == pytest.approx(profile.iloc[poc_bin_idx][COL_BIN_LOW])
        assert profile.attrs["vah_price"] == pytest.approx(profile.iloc[poc_bin_idx][COL_BIN_HIGH])


class TestCalculateVolumeProfileDynamic:
    """Tests for calculate_volume_profile with dynamic mode."""

    def test_dynamic_atr_mode(self):
        """Test dynamic ATR-based profile calculation."""
        ohlcv = _create_ohlcv(n=50, volatility=0.02)

        profile = calculate_volume_profile(
            ohlcv,
            profile_type="dynamic",
            dynamic_mode="atr",
            atr_period=14,
            atr_mult=1.0,
        )

        assert isinstance(profile, pd.DataFrame)
        assert len(profile) > 0
        assert profile.attrs["profile_type"] == "dynamic"

    def test_dynamic_pct_mode(self):
        """Test dynamic percentage-of-price profile calculation."""
        ohlcv = _create_ohlcv(n=50, volatility=0.02)

        profile = calculate_volume_profile(
            ohlcv,
            profile_type="dynamic",
            dynamic_mode="pct",
            range_pct=0.02,
        )

        assert isinstance(profile, pd.DataFrame)
        assert len(profile) > 0
        assert profile.attrs["profile_type"] == "dynamic"

    def test_dynamic_mode_requires_dynamic_mode_param(self):
        """Test that dynamic profile_type requires dynamic_mode parameter."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match="dynamic_mode is required"):
            calculate_volume_profile(ohlcv, profile_type="dynamic")

    def test_dynamic_pct_mode_requires_range_pct(self):
        """Test that dynamic pct mode requires range_pct parameter."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match="range_pct is required"):
            calculate_volume_profile(
                ohlcv,
                profile_type="dynamic",
                dynamic_mode="pct",
            )

    def test_invalid_dynamic_mode_raises(self):
        """Test that invalid dynamic modes raise ValueError."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match="Invalid dynamic_mode"):
            calculate_volume_profile(ohlcv, profile_type="dynamic", dynamic_mode="volatility")

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"profile_type": "dynamic", "dynamic_mode": "atr", "atr_period": 0}, "atr_period"),
            ({"profile_type": "dynamic", "dynamic_mode": "atr", "atr_mult": 0}, "atr_mult"),
            ({"profile_type": "dynamic", "dynamic_mode": "pct", "range_pct": 0}, "range_pct"),
            ({"profile_type": "dynamic", "dynamic_mode": "pct", "range_pct": -0.01}, "range_pct"),
        ],
    )
    def test_non_positive_dynamic_thresholds_raise(self, kwargs, message):
        """Test that dynamic threshold inputs must be positive."""
        ohlcv = _create_ohlcv(n=20)

        with pytest.raises(ValueError, match=message):
            calculate_volume_profile(ohlcv, **kwargs)

    def test_dynamic_pct_uses_requested_bin_width(self):
        """Test that dynamic pct mode uses the derived bin width directly."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
        ohlcv = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [110.0] * 5,
                "low": [100.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )

        profile = calculate_volume_profile(
            ohlcv,
            profile_type="dynamic",
            dynamic_mode="pct",
            range_pct=0.02,
        )

        widths = profile[COL_BIN_HIGH] - profile[COL_BIN_LOW]
        np.testing.assert_allclose(widths.to_numpy(), np.full(len(widths), 2.0))

    def test_dynamic_bin_edges_anchor_decimal_widths_to_range_bounds(self):
        """Test decimal widths keep exact range bounds and deterministic edges."""
        price_min = 100.1
        price_max = 100.56
        bin_width = 0.1

        first_edges = _build_dynamic_bin_edges(price_min, price_max, bin_width)
        second_edges = _build_dynamic_bin_edges(price_min, price_max, bin_width)

        assert first_edges[0] == price_min
        assert first_edges[-1] == price_max
        pd.testing.assert_index_equal(first_edges, second_edges)
        np.testing.assert_allclose(
            np.diff(first_edges[:-1]), np.full(len(first_edges) - 2, bin_width)
        )


class TestVolumeProfileMetadata:
    """Tests for volume profile metadata and attributes."""

    def test_profile_attrs_exist(self):
        """Test that profile contains expected metadata in attrs."""
        ohlcv = _create_ohlcv(n=50)

        profile = calculate_volume_profile(ohlcv, bin_count=30)

        # Check attrs exist
        assert "poc_price" in profile.attrs
        assert "poc_bin_idx" in profile.attrs
        assert "vah_price" in profile.attrs
        assert "val_price" in profile.attrs
        assert "value_area_pct" in profile.attrs
        assert "profile_shape" in profile.attrs
        assert "profile_type" in profile.attrs
        assert "bin_count" in profile.attrs
        assert "total_volume" in profile.attrs

    def test_poc_within_price_range(self):
        """Test that POC price is within observed price range."""
        ohlcv = _create_ohlcv(n=50)

        profile = calculate_volume_profile(ohlcv, bin_count=30)

        min_price = float(ohlcv["low"].min())
        max_price = float(ohlcv["high"].max())

        assert profile.attrs["poc_price"] >= min_price
        assert profile.attrs["poc_price"] <= max_price

    def test_val_less_than_vah(self):
        """Test that VAL <= VAH."""
        ohlcv = _create_ohlcv(n=50)

        profile = calculate_volume_profile(ohlcv, bin_count=30)

        assert profile.attrs["val_price"] <= profile.attrs["vah_price"]

    def test_value_area_contains_poc(self):
        """Test that Value Area contains the POC."""
        ohlcv = _create_ohlcv(n=50)

        profile = calculate_volume_profile(ohlcv, bin_count=30, value_area_pct=0.70)

        poc_price = profile.attrs["poc_price"]
        val_price = profile.attrs["val_price"]
        vah_price = profile.attrs["vah_price"]

        assert val_price <= poc_price <= vah_price

    def test_profile_shape_valid(self):
        """Test that profile shape is a valid classification."""
        ohlcv = _create_ohlcv(n=100)

        profile = calculate_volume_profile(ohlcv, bin_count=50)

        valid_shapes = {"bell", "bimodal", "directional", "flat", "single"}
        assert profile.attrs["profile_shape"] in valid_shapes

    def test_value_area_reaches_target_across_zero_volume_gaps(self):
        """Test that value-area expansion does not stop at zero-volume gaps."""
        bin_volumes = pd.Series([50.0, 0.0, 0.0, 30.0, 20.0])

        va_low_idx, va_high_idx = _calculate_value_area(bin_volumes, poc_idx=0, value_area_pct=0.70)

        covered_volume = float(bin_volumes.iloc[va_low_idx : va_high_idx + 1].sum())
        assert (va_low_idx, va_high_idx) == (0, 3)
        assert covered_volume >= bin_volumes.sum() * 0.70
        assert covered_volume == pytest.approx(80.0)


class TestVolumeProfileIntegration:
    """Integration tests for volume profile indicator."""

    def test_full_workflow(self, ohlcv_data):
        """Test complete volume profile analysis workflow."""
        # ohlcv_data fixture should provide sufficient rows
        assert len(ohlcv_data["close"]) >= 20

        # Create DataFrame
        n = len(ohlcv_data["close"])
        dates = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

        ohlcv = pd.DataFrame(
            {
                "open": ohlcv_data["open"],
                "high": ohlcv_data["high"],
                "low": ohlcv_data["low"],
                "close": ohlcv_data["close"],
                "volume": ohlcv_data["volume"],
            },
            index=dates,
        )

        # Fixed profile
        profile_fixed = calculate_volume_profile(ohlcv, bin_count=50, profile_type="fixed")
        assert isinstance(profile_fixed, pd.DataFrame)
        assert len(profile_fixed) > 0

        # Dynamic ATR profile
        profile_atr = calculate_volume_profile(
            ohlcv,
            profile_type="dynamic",
            dynamic_mode="atr",
            atr_period=14,
            atr_mult=1.5,
        )
        assert isinstance(profile_atr, pd.DataFrame)

        # Dynamic PCT profile
        profile_pct = calculate_volume_profile(
            ohlcv,
            profile_type="dynamic",
            dynamic_mode="pct",
            range_pct=0.01,
        )
        assert isinstance(profile_pct, pd.DataFrame)

    def test_deterministic_reproducibility(self):
        """Test that same inputs produce same outputs (determinism)."""
        ohlcv = _create_ohlcv(n=50, seed=42)

        profile1 = calculate_volume_profile(ohlcv, bin_count=30)
        profile2 = calculate_volume_profile(ohlcv, bin_count=30)

        # Volumes should be very close (allowing for float precision)
        np.testing.assert_allclose(
            profile1[COL_BIN_VOLUME].values,
            profile2[COL_BIN_VOLUME].values,
            rtol=1e-10,
        )

        # Metadata should be identical
        assert profile1.attrs["poc_price"] == profile2.attrs["poc_price"]
        assert profile1.attrs["profile_shape"] == profile2.attrs["profile_shape"]
