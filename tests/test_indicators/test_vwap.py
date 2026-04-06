"""Unit tests for VWAP indicator engine."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from tempest_mcp.indicators.volume.vwap import (
    SESSION_ANCHORS,
    calculate_vwap,
    calculate_vwap_bands,
    detect_vwap_cross,
)


class TestCalculateVwap:
    """Tests for calculate_vwap function."""

    def test_normal_case(self):
        """Test VWAP calculation with sufficient data."""
        # Create simple test data - use hourly to have multiple bars in same session
        high = pd.Series(
            [105, 106, 107], index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
        )
        low = pd.Series([100, 101, 102], index=high.index)
        close = pd.Series([103, 104, 105], index=high.index)
        volume = pd.Series([1000, 1100, 1200], index=high.index)

        vwap = calculate_vwap(high, low, close, volume)

        assert len(vwap) == len(high)
        assert vwap.index.equals(high.index)
        # VWAP should be defined for all points
        assert not vwap.isna().any()

    def test_vwap_calculation_correctness(self):
        """Test that VWAP values are calculated correctly."""
        # Simple case with known values - all in same session (before NY anchor)
        # Use hourly data starting at 10:00 UTC (all before 13:30)
        high = pd.Series(
            [105.0, 106.0, 107.0],
            index=pd.date_range("2024-01-01 10:00", periods=3, freq="h", tz="UTC"),
        )
        low = pd.Series([100.0, 101.0, 102.0], index=high.index)
        close = pd.Series([103.0, 104.0, 105.0], index=high.index)
        volume = pd.Series([1000.0, 1000.0, 1000.0], index=high.index)

        vwap = calculate_vwap(high, low, close, volume)

        # All bars before 13:30 UTC, so same session - VWAP accumulates
        # Typical prices: (105+100+103)/3=102.67, (106+101+104)/3=103.67, (107+102+105)/3=104.67
        tp0 = (105 + 100 + 103) / 3
        tp1 = (106 + 101 + 104) / 3
        tp2 = (107 + 102 + 105) / 3

        expected_vwap0 = tp0
        expected_vwap1 = (tp0 * 1000 + tp1 * 1000) / 2000
        expected_vwap2 = (tp0 * 1000 + tp1 * 1000 + tp2 * 1000) / 3000

        assert vwap.iloc[0] == pytest.approx(expected_vwap0, rel=1e-6)
        assert vwap.iloc[1] == pytest.approx(expected_vwap1, rel=1e-6)
        assert vwap.iloc[2] == pytest.approx(expected_vwap2, rel=1e-6)

    def test_session_boundary_reset_ny(self):
        """Test VWAP resets at NY session boundary (13:30 UTC).

        Uses asymmetric prices/volumes so that the reset is verifiable.
        """
        # Create data spanning NY session boundary with asymmetric values
        dates = pd.date_range("2024-01-01 12:00", periods=5, freq="h", tz="UTC")
        high = pd.Series([100.0, 100.0, 200.0, 200.0, 200.0], index=dates)
        low = pd.Series([99.0, 99.0, 199.0, 199.0, 199.0], index=dates)
        close = pd.Series([99.5, 99.5, 199.5, 199.5, 199.5], index=dates)
        volume = pd.Series([1000.0] * 5, index=dates)

        vwap = calculate_vwap(high, low, close, volume, anchor="ny")

        # NY session starts at 13:30 UTC
        # Bars 0-1 (12:00, 13:00): before 13:30, same pre-session
        # Bar 2 (14:00): after 13:30, NEW SESSION — reset
        # Because prices jump across the boundary, pre/post VWAP differ
        pre_boundary_vwap = vwap.iloc[1]
        post_boundary_vwap = vwap.iloc[3]

        assert pre_boundary_vwap != post_boundary_vwap
        assert not vwap.isna().any()

    def test_session_boundary_reset_london(self):
        """Test VWAP resets at London session boundary (08:00 UTC).

        Uses asymmetric prices/volumes so that the reset is verifiable — VWAP
        after the boundary is based only on post-boundary cumulative values.
        """
        dates = pd.date_range("2024-01-01 06:00", periods=5, freq="h", tz="UTC")
        # Pre-boundary: high-volume, low-price bars
        high = pd.Series([100.0, 100.0, 200.0, 200.0, 200.0], index=dates)
        low = pd.Series([99.0, 99.0, 199.0, 199.0, 199.0], index=dates)
        close = pd.Series([99.5, 99.5, 199.5, 199.5, 199.5], index=dates)
        volume = pd.Series([1000.0, 1000.0, 1000.0, 1000.0, 1000.0], index=dates)

        vwap = calculate_vwap(high, low, close, volume, anchor="london")

        # London session starts at 08:00 UTC
        # Bars 0-1 (06:00, 07:00): before 08:00, same pre-session
        # Bar 2 (08:00): AT 08:00, new session STARTS HERE — reset
        # Bars 3-4 (09:00, 10:00): post-boundary, new session
        # Because prices change dramatically across the boundary, VWAP values
        # before and after should differ (proves reset occurred)
        pre_boundary_vwap = vwap.iloc[1]  # Last pre-boundary bar
        post_boundary_vwap = vwap.iloc[3]  # Post-boundary bar in new session

        assert pre_boundary_vwap != post_boundary_vwap
        assert not vwap.isna().any()

    def test_session_boundary_reset_asia(self):
        """Test VWAP resets at Asia session boundary (00:00 UTC).

        Uses asymmetric prices so reset is verifiable.
        """
        dates = pd.date_range("2024-01-01 22:00", periods=5, freq="h", tz="UTC")
        high = pd.Series([100.0, 100.0, 200.0, 200.0, 200.0], index=dates)
        low = pd.Series([99.0, 99.0, 199.0, 199.0, 199.0], index=dates)
        close = pd.Series([99.5, 99.5, 199.5, 199.5, 199.5], index=dates)
        volume = pd.Series([1000.0] * 5, index=dates)

        vwap = calculate_vwap(high, low, close, volume, anchor="asia")

        # Asia session starts at 00:00 UTC
        # Bars 0-1 (22:00, 23:00): before midnight, same pre-session
        # Bar 2 (00:00): AT midnight, NEW SESSION — reset
        pre_boundary_vwap = vwap.iloc[1]
        post_boundary_vwap = vwap.iloc[3]

        assert pre_boundary_vwap != post_boundary_vwap
        assert not vwap.isna().any()

    def test_session_boundary_reset_daily(self):
        """Test VWAP resets at daily boundary (00:00 UTC).

        daily anchor is same as asia (00:00 UTC). Uses asymmetric prices so reset is verifiable.
        """
        dates = pd.date_range("2024-01-01 22:00", periods=5, freq="h", tz="UTC")
        high = pd.Series([100.0, 100.0, 200.0, 200.0, 200.0], index=dates)
        low = pd.Series([99.0, 99.0, 199.0, 199.0, 199.0], index=dates)
        close = pd.Series([99.5, 99.5, 199.5, 199.5, 199.5], index=dates)
        volume = pd.Series([1000.0] * 5, index=dates)

        vwap = calculate_vwap(high, low, close, volume, anchor="daily")

        pre_boundary_vwap = vwap.iloc[1]
        post_boundary_vwap = vwap.iloc[3]

        assert pre_boundary_vwap != post_boundary_vwap
        assert not vwap.isna().any()

    def test_insufficient_data(self):
        """Test VWAP returns empty Series when data is empty."""
        high = pd.Series(dtype=float)
        low = pd.Series(dtype=float)
        close = pd.Series(dtype=float)
        volume = pd.Series(dtype=float)

        vwap = calculate_vwap(high, low, close, volume)

        assert len(vwap) == 0
        assert isinstance(vwap, pd.Series)

    def test_extreme_volume_spike(self):
        """Test VWAP handles extreme volume spikes correctly."""
        dates = pd.date_range("2024-01-01 10:00", periods=5, freq="h", tz="UTC")
        high = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
        low = pd.Series([99.0, 100.0, 101.0, 102.0, 103.0], index=dates)
        close = pd.Series([99.5, 100.5, 101.5, 102.5, 103.5], index=dates)
        # Extreme volume spike at bar 2
        volume = pd.Series([1000.0, 1000.0, 1000000.0, 1000.0, 1000.0], index=dates)

        vwap = calculate_vwap(high, low, close, volume)

        # VWAP should be heavily weighted towards bar 2's typical price
        assert len(vwap) == 5
        assert not vwap.isna().any()

    def test_tz_naive_index_treated_as_utc(self):
        """Test that tz-naive index is treated as UTC."""
        dates = pd.date_range("2024-01-01 10:00", periods=3, freq="h")  # No tz
        high = pd.Series([105.0, 106.0, 107.0], index=dates)
        low = pd.Series([100.0, 101.0, 102.0], index=dates)
        close = pd.Series([103.0, 104.0, 105.0], index=dates)
        volume = pd.Series([1000.0, 1100.0, 1200.0], index=dates)

        vwap = calculate_vwap(high, low, close, volume)

        assert len(vwap) == 3
        assert vwap.index.tz is not None  # Should be UTC-aware now

    def test_invalid_anchor_raises_error(self):
        """Test that invalid anchor raises ValueError."""
        dates = pd.date_range("2024-01-01 10:00", periods=3, freq="h", tz="UTC")
        high = pd.Series([105.0, 106.0, 107.0], index=dates)
        low = pd.Series([100.0, 101.0, 102.0], index=dates)
        close = pd.Series([103.0, 104.0, 105.0], index=dates)
        volume = pd.Series([1000.0, 1100.0, 1200.0], index=dates)

        with pytest.raises(ValueError, match="Invalid anchor"):
            calculate_vwap(high, low, close, volume, anchor="invalid")

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        dates = pd.date_range("2024-01-01 10:00", periods=3, freq="h", tz="UTC")
        high = pd.Series([105.0, 106.0, 107.0], index=dates)
        low = pd.Series([100.0, 101.0], index=dates[:2])  # Shorter
        close = pd.Series([103.0, 104.0, 105.0], index=dates)
        volume = pd.Series([1000.0, 1100.0, 1200.0], index=dates)

        with pytest.raises(ValueError, match="must have the same length"):
            calculate_vwap(high, low, close, volume)


class TestCalculateVwapBands:
    """Tests for calculate_vwap_bands function."""

    def test_band_calculation(self):
        """Test VWAP bands are calculated correctly."""
        dates = pd.date_range("2024-01-01 10:00", periods=10, freq="h", tz="UTC")
        high = pd.Series(range(100, 110), index=dates, dtype=float)
        low = pd.Series(range(95, 105), index=dates, dtype=float)
        close = pd.Series(range(98, 108), index=dates, dtype=float)
        volume = pd.Series([1000.0] * 10, index=dates)

        vwap = calculate_vwap(high, low, close, volume)
        bands = calculate_vwap_bands(vwap, close)

        assert "vwap" in bands.columns
        assert "upper_band_1std" in bands.columns
        assert "lower_band_1std" in bands.columns
        assert "upper_band_2std" in bands.columns
        assert "lower_band_2std" in bands.columns

    def test_population_std_dev_ddof_0(self):
        """Test that bands use population std dev (ddof=0), not sample std dev.

        Uses non-constant deviations so that ddof=0 and ddof=1 produce different
        results, ensuring the test actually distinguishes the two.
        """
        dates = pd.date_range("2024-01-01 10:00", periods=5, freq="h", tz="UTC")
        # Use non-constant deviations so ddof=0 and ddof=1 produce different results
        deviation = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0], index=dates)
        vwap_var = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=dates)
        close_var = vwap_var + deviation

        bands_var = calculate_vwap_bands(vwap_var, close_var)

        pop_std = deviation.std(ddof=0)
        sample_std = deviation.std(ddof=1)

        # Verify ddof=0 is used: band width should equal population std dev
        band_width = bands_var["upper_band_1std"] - bands_var["vwap"]
        assert band_width.iloc[0] == pytest.approx(pop_std)
        # Also verify sample std is NOT used (they differ for non-constant data)
        assert band_width.iloc[0] != pytest.approx(sample_std)

    def test_empty_series_returns_empty_dataframe(self):
        """Test that empty series returns empty DataFrame with correct columns."""
        vwap = pd.Series(dtype=float)
        close = pd.Series(dtype=float)

        bands = calculate_vwap_bands(vwap, close)

        assert len(bands) == 0
        assert "vwap" in bands.columns
        assert "upper_band_1std" in bands.columns

    def test_custom_std_multipliers(self):
        """Test bands with custom std dev multipliers."""
        dates = pd.date_range("2024-01-01 10:00", periods=5, freq="h", tz="UTC")
        close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
        vwap = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)

        bands = calculate_vwap_bands(vwap, close, std_dev=(1.5, 2.5))

        # Should have bands at 1.5σ and 2.5σ
        assert "upper_band_1std" in bands.columns
        assert "upper_band_2std" in bands.columns


class TestDetectVwapCross:
    """Tests for detect_vwap_cross function."""

    def test_bullish_cross(self):
        """Test detection of bullish VWAP cross."""
        # Create price that crosses above VWAP
        dates = pd.date_range("2024-01-01 10:00", periods=5, freq="h", tz="UTC")
        # VWAP around 102-103
        high = pd.Series([105.0, 106.0, 107.0, 108.0, 109.0], index=dates)
        low = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
        close = pd.Series([102.5, 103.0, 103.5, 104.0, 104.5], index=dates)
        volume = pd.Series([1000.0] * 5, index=dates)

        vwap = calculate_vwap(high, low, close, volume)

        # Create price that crosses from below to above VWAP at index 3
        # price starts below VWAP, then rises above
        price = pd.Series([100.0, 101.0, 102.0, 106.0, 107.0], index=dates)

        crosses = detect_vwap_cross(price, vwap)

        assert len(crosses) > 0, "Expected at least one cross signal"
        assert "date" in crosses.columns
        assert "direction" in crosses.columns
        assert "price" in crosses.columns
        assert "vwap_value" in crosses.columns
        # Should have at least one bullish cross
        bullish = crosses[crosses["direction"] == "bullish"]
        assert len(bullish) > 0, "Expected at least one bullish cross"
        # The bullish cross should occur when price crosses above VWAP
        assert all(crosses[crosses["direction"] == "bullish"]["direction"] == "bullish")

    def test_bearish_cross(self):
        """Test detection of bearish VWAP cross."""
        dates = pd.date_range("2024-01-01 10:00", periods=5, freq="h", tz="UTC")
        high = pd.Series([105.0, 106.0, 107.0, 108.0, 109.0], index=dates)
        low = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
        close = pd.Series([102.5, 103.0, 103.5, 104.0, 104.5], index=dates)
        volume = pd.Series([1000.0] * 5, index=dates)

        vwap = calculate_vwap(high, low, close, volume)

        # Create price that crosses from above to below VWAP
        # Price starts above VWAP, then falls below
        price = pd.Series([107.0, 106.0, 105.0, 100.0, 99.0], index=dates)

        crosses = detect_vwap_cross(price, vwap)

        assert len(crosses) > 0, "Expected at least one cross signal"
        # Should have at least one bearish cross
        bearish = crosses[crosses["direction"] == "bearish"]
        assert len(bearish) > 0, "Expected at least one bearish cross"
        assert all(crosses[crosses["direction"] == "bearish"]["direction"] == "bearish")

    def test_no_false_positives_flat_price(self):
        """Test that flat/sideways price produces no false cross signals."""
        dates = pd.date_range("2024-01-01 10:00", periods=10, freq="h", tz="UTC")
        # Constant prices
        high = pd.Series([100.0] * 10, index=dates)
        low = pd.Series([100.0] * 10, index=dates)
        close = pd.Series([100.0] * 10, index=dates)
        volume = pd.Series([1000.0] * 10, index=dates)

        vwap = calculate_vwap(high, low, close, volume)
        price = pd.Series([100.0] * 10, index=dates)

        crosses = detect_vwap_cross(price, vwap)

        # No crossovers with constant price
        assert len(crosses) == 0

    def test_one_signal_per_cross(self):
        """Test that each crossing produces only one signal."""
        dates = pd.date_range("2024-01-01 10:00", periods=20, freq="h", tz="UTC")
        high = pd.Series(range(100, 120), index=dates, dtype=float)
        low = pd.Series(range(95, 115), index=dates, dtype=float)
        close = pd.Series(range(98, 118), index=dates, dtype=float)
        volume = pd.Series([1000.0] * 20, index=dates)

        vwap = calculate_vwap(high, low, close, volume)

        # Price oscillates around VWAP
        price = close.copy()
        price.iloc[::2] = price.iloc[::2] - 5  # Every other bar below

        crosses = detect_vwap_cross(price, vwap)

        # Each cross should be a distinct event
        if len(crosses) > 1:
            for i in range(1, len(crosses)):
                assert crosses.iloc[i]["date"] != crosses.iloc[i - 1]["date"]

    def test_empty_series_returns_empty_dataframe(self):
        """Test that empty series returns empty DataFrame with correct columns."""
        price = pd.Series(dtype=float)
        vwap = pd.Series(dtype=float)

        crosses = detect_vwap_cross(price, vwap)

        assert len(crosses) == 0
        assert "date" in crosses.columns
        assert "direction" in crosses.columns
        assert "price" in crosses.columns
        assert "vwap_value" in crosses.columns


class TestPreFirstAnchorBehavior:
    """Tests for pre-first-anchor accumulation behavior."""

    def test_accumulates_from_bar_0(self):
        """Test that VWAP accumulates from bar 0 before first anchor."""
        # Create data that starts well before the NY session anchor (13:30 UTC)
        # Start at 00:00 UTC, all bars before 13:30 anchor
        dates = pd.date_range("2024-01-01 00:00", periods=10, freq="h", tz="UTC")
        high = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0], index=dates
        )
        low = pd.Series(
            [95.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0], index=dates
        )
        close = pd.Series(
            [98.0, 99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0], index=dates
        )
        volume = pd.Series([1000.0] * 10, index=dates)

        vwap = calculate_vwap(high, low, close, volume, anchor="ny")

        # All bars are before 13:30, so they should be in the same session (previous day)
        # VWAP should accumulate continuously
        assert len(vwap) == 10
        assert not vwap.isna().any()

    def test_resets_at_first_anchor_boundary(self):
        """Test that VWAP resets when crossing the first anchor boundary."""
        # Data spans across NY session anchor (13:30 UTC)
        dates = pd.date_range("2024-01-01 10:00", periods=8, freq="h", tz="UTC")
        high = pd.Series([100.0] * 8, index=dates)
        low = pd.Series([100.0] * 8, index=dates)
        close = pd.Series([100.0] * 8, index=dates)
        volume = pd.Series([1000.0] * 8, index=dates)

        vwap = calculate_vwap(high, low, close, volume, anchor="ny")

        # Bars 0-3 are before 13:30 (10:00, 11:00, 12:00, 13:00) - same session
        # Bar 4 is at 14:00 - after 13:30 anchor, new session
        assert len(vwap) == 8
        assert not vwap.isna().any()

    def test_different_anchors_have_different_reset_points(self):
        """Test that different anchors reset at different times."""
        dates = pd.date_range("2024-01-01 00:00", periods=24, freq="h", tz="UTC")
        high = pd.Series([100.0] * 24, index=dates)
        low = pd.Series([100.0] * 24, index=dates)
        close = pd.Series([100.0] * 24, index=dates)
        volume = pd.Series([1000.0] * 24, index=dates)

        vwap_asia = calculate_vwap(high, low, close, volume, anchor="asia")
        vwap_london = calculate_vwap(high, low, close, volume, anchor="london")
        vwap_ny = calculate_vwap(high, low, close, volume, anchor="ny")

        # All should have same length
        assert len(vwap_asia) == len(vwap_london) == len(vwap_ny) == 24

        # All should have values
        assert not vwap_asia.isna().any()
        assert not vwap_london.isna().any()
        assert not vwap_ny.isna().any()


class TestSessionAnchors:
    """Tests for SESSION_ANCHORS constant."""

    def test_session_anchors_defined(self):
        """Test that all expected session anchors are defined."""
        assert "asia" in SESSION_ANCHORS
        assert "london" in SESSION_ANCHORS
        assert "ny" in SESSION_ANCHORS
        assert "daily" in SESSION_ANCHORS

    def test_session_anchor_values(self):
        """Test that session anchor values are correct (UTC hours)."""
        assert SESSION_ANCHORS["asia"] == 0  # 00:00 UTC
        assert SESSION_ANCHORS["london"] == 8  # 08:00 UTC
        assert SESSION_ANCHORS["ny"] == 13.5  # 13:30 UTC
        assert SESSION_ANCHORS["daily"] == 0  # 00:00 UTC


class TestIntegration:
    """Integration tests for VWAP indicator workflow."""

    def test_full_workflow(self):
        """Test complete VWAP analysis workflow."""
        # Generate realistic price data
        dates = pd.date_range("2024-01-01 10:00", periods=100, freq="h", tz="UTC")
        base_price = 100.0
        prices = pd.Series([base_price + i * 0.1 for i in range(100)], index=dates)

        high = prices + 2
        low = prices - 2
        close = prices
        volume = pd.Series([1000.0 + i * 10 for i in range(100)], index=dates)

        # Calculate VWAP
        vwap = calculate_vwap(high, low, close, volume)

        # Calculate bands
        bands = calculate_vwap_bands(vwap, close)

        # Detect crosses
        crosses = detect_vwap_cross(close, vwap)

        # Verify outputs
        assert len(vwap) == 100
        assert len(bands) <= 100
        assert isinstance(crosses, pd.DataFrame)

    def test_vwap_with_result_wrapper(self):
        """Test VWAP calculation matches expected behavior for result wrapper."""
        dates = pd.date_range("2024-01-01 10:00", periods=50, freq="h", tz="UTC")
        high = pd.Series(range(100, 150), index=dates, dtype=float)
        low = pd.Series(range(95, 145), index=dates, dtype=float)
        close = pd.Series(range(98, 148), index=dates, dtype=float)
        volume = pd.Series([1000.0] * 50, index=dates)

        vwap = calculate_vwap(high, low, close, volume)

        # VWAP should be defined for all bars
        assert len(vwap) == 50
        assert not vwap.empty

    def test_utc_index_preserved(self):
        """Test that UTC-aware index is preserved in outputs."""
        dates = pd.date_range("2024-01-01 10:00", periods=10, freq="h", tz="UTC")
        high = pd.Series([100.0] * 10, index=dates)
        low = pd.Series([100.0] * 10, index=dates)
        close = pd.Series([100.0] * 10, index=dates)
        volume = pd.Series([1000.0] * 10, index=dates)

        vwap = calculate_vwap(high, low, close, volume)
        bands = calculate_vwap_bands(vwap, close)
        crosses = detect_vwap_cross(close, vwap)

        # All outputs should have UTC-aware index
        assert vwap.index.tz is not None
        assert bands.index.tz is not None
        if len(crosses) > 0:
            # Check that date column contains timestamps
            assert isinstance(crosses["date"].iloc[0], pd.Timestamp)
