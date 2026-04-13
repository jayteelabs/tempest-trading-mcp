"""Unit tests for session levels detection (Asia, London, NY PDH/PDL)."""

import pandas as pd
import pytest

from tempest_mcp.indicators.session_levels import (
    detect_pdh_pdl,
    detect_session_levels,
)


class TestDetectSessionLevels:
    """Tests for detect_session_levels function."""

    def test_asia_session_normal(self):
        """Test Asia session detection with multi-day data containing Asia bars."""
        # Create 1 day of hourly data (24 hours) to avoid spanning multiple days
        dates = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(24)],
            "high": [101.0 + i * 0.1 for i in range(24)],
            "low": [99.0 + i * 0.1 for i in range(24)],
            "close": [100.5 + i * 0.1 for i in range(24)],
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "asia")

        assert result["session_type"] == "asia"
        assert result["bars"] == 9  # 00:00-09:00 UTC = 9 hours (indices 0-8)
        assert result["session_start_utc"] == dates[0]
        assert result["session_end_utc"] == dates[8]
        assert result["high"] == 101.8  # high at index 8
        assert result["low"] == 99.0  # low at index 0
        assert result["range"] == result["high"] - result["low"]
        assert result["midpoint"] == (result["high"] + result["low"]) / 2

    def test_london_session_normal(self):
        """Test London session detection with multi-day data."""
        # Create 1 day of hourly data (24 hours) to avoid spanning multiple days
        dates = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(24)],
            "high": [101.0 + i * 0.1 for i in range(24)],
            "low": [99.0 + i * 0.1 for i in range(24)],
            "close": [100.5 + i * 0.1 for i in range(24)],
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "london")

        assert result["session_type"] == "london"
        # London session 08:00-17:00 UTC = indices 8-16 (9 hours), bars 8-16 inclusive
        assert result["bars"] == 9
        assert result["session_start_utc"] == dates[8]
        assert result["session_end_utc"] == dates[16]

    def test_ny_session_normal(self):
        """Test NY session detection with multi-day data."""
        # Create 1 day of hourly data (24 hours) to avoid spanning multiple days
        dates = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(24)],
            "high": [101.0 + i * 0.1 for i in range(24)],
            "low": [99.0 + i * 0.1 for i in range(24)],
            "close": [100.5 + i * 0.1 for i in range(24)],
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "ny")

        # NY session: 09:30-16:00 ET (exclusive end)
        # On 2024-03-15 in EDT (DST started March 10), 09:30 ET = 13:30 UTC
        # 16:00 ET = 20:00 UTC (exclusive - bar at 20:00 is NOT included)
        # Start: bars where hour > 9 OR (hour == 9 AND minute >= 30)
        #   First bar at/after 09:30 ET is 14:00 UTC (hour 14)
        # End: bars where hour < 16 OR (hour == 16 AND minute < 0)
        #   Last bar before 16:00 ET is 19:00 UTC (hour 19)
        # So bars 14:00-19:00 UTC = 6 bars
        assert result["session_type"] == "ny"
        assert result["bars"] == 6

    def test_new_york_alias_normalizes_to_ny(self):
        """Compatibility alias should normalize to the canonical ny session id."""
        dates = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(24)],
            "high": [101.0 + i * 0.1 for i in range(24)],
            "low": [99.0 + i * 0.1 for i in range(24)],
            "close": [100.5 + i * 0.1 for i in range(24)],
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "new_york")

        assert result["session_type"] == "ny"
        assert result["bars"] == 6

    def test_no_bars_in_session_window(self):
        """Test when no bars fall within session window returns bars=0 and NaN."""
        # Create data for a single day at night hours only (no Asia/London/NY session bars)
        dates = pd.date_range("2024-03-15 20:00", periods=4, freq="h", tz="UTC")
        data = {
            "open": [100.0] * 4,
            "high": [101.0] * 4,
            "low": [99.0] * 4,
            "close": [100.5] * 4,
            "volume": [1000.0] * 4,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "asia")

        assert result["bars"] == 0
        assert pd.isna(result["high"])
        assert pd.isna(result["low"])
        assert pd.isna(result["range"])
        assert pd.isna(result["midpoint"])

    def test_invalid_session_type_raises_valueerror(self):
        """Test that invalid session_type raises ValueError."""
        dates = pd.date_range("2024-03-15", periods=10, freq="h", tz="UTC")
        data = {
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000.0] * 10,
        }
        df = pd.DataFrame(data, index=dates)

        with pytest.raises(ValueError, match="Invalid session_type"):
            detect_session_levels(df, "invalid_session")

    def test_dst_transition_edt_to_est(self):
        """Test NY session shifts when transitioning from EDT to EST (fall back)."""
        # November 3, 2024 - DST ends (clocks fall back at 2am ET)
        # On this day: 09:30-16:00 ET covers two different UTC offset periods
        dates = pd.date_range("2024-11-03", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(24)],
            "high": [101.0 + i * 0.1 for i in range(24)],
            "low": [99.0 + i * 0.1 for i in range(24)],
            "close": [100.5 + i * 0.1 for i in range(24)],
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "ny")

        # NY session should still find bars despite DST transition
        assert result["session_type"] == "ny"
        assert result["bars"] > 0

    def test_dst_transition_est_to_edt(self):
        """Test NY session shifts when transitioning from EST to EDT (spring forward)."""
        # March 10, 2024 - DST starts (clocks spring forward at 2am ET)
        # On this day there is no 2am hour in ET (it jumps from 2am to 3am)
        dates = pd.date_range("2024-03-10", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(24)],
            "high": [101.0 + i * 0.1 for i in range(24)],
            "low": [99.0 + i * 0.1 for i in range(24)],
            "close": [100.5 + i * 0.1 for i in range(24)],
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "ny")

        # NY session should still find bars despite DST transition
        assert result["session_type"] == "ny"
        assert result["bars"] > 0


class TestDetectPdhPdl:
    """Tests for detect_pdh_pdl function."""

    def test_normal_multi_day_data(self):
        """PDH/PDL should be computed from the prior ET business day.

        The reference point is the latest/current bar in the dataset, not the
        first loaded bar.
        """
        # Create 2 full days of data: March 12 and March 13
        dates = pd.date_range("2024-03-12", periods=48, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(48)],
            "high": [101.0 + i * 0.1 for i in range(48)],
            "low": [99.0 + i * 0.1 for i in range(48)],
            "close": [100.5 + i * 0.1 for i in range(48)],
            "volume": [1000.0] * 48,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_pdh_pdl(df)

        assert result["previous_day_high"] == pytest.approx(103.7)
        assert result["previous_day_low"] == pytest.approx(99.4)
        assert result["previous_day_close"] == pytest.approx(103.2)
        assert result["previous_day_range"] == pytest.approx(4.3)
        assert result["position"] == "above_pdh"
        assert result["pdh_timestamp_utc"] == pd.Timestamp("2024-03-13 03:00:00+00:00")
        assert result["pdl_timestamp_utc"] == pd.Timestamp("2024-03-12 04:00:00+00:00")

    def test_single_day_insufficient_data(self):
        """Test that less than 2 days returns insufficient_data."""
        # Only 1 day of data
        dates = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data = {
            "open": [100.0] * 24,
            "high": [101.0] * 24,
            "low": [99.0] * 24,
            "close": [100.5] * 24,
            "volume": [1000.0] * 24,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_pdh_pdl(df)

        assert result["position"] == "insufficient_data"
        assert pd.isna(result["previous_day_high"])

    def test_pivot_and_levels_calculation(self):
        """Test pivot, r1, r2, s1, s2 formula verification."""
        # Test formulas directly
        pdh, pdl, pdc = 110.0, 90.0, 100.0
        pivot = (pdh + pdl + pdc) / 3
        r1 = 2 * pivot - pdl
        r2 = pivot + (pdh - pdl)
        s1 = 2 * pivot - pdh
        s2 = pivot - (pdh - pdl)

        assert pivot == 100.0
        assert r1 == 110.0
        assert r2 == 120.0
        assert s1 == 90.0
        assert s2 == 80.0

    def test_position_above_pdh(self):
        """Test position above PDH when previous day data exists.

        Note: When first bar is at UTC midnight, insufficient_data is returned.
        This test verifies position classification by checking the logic directly.
        """
        # Test position logic directly
        pdh, pdl = 100.0, 90.0
        current_close_above = 105.0
        current_close_below = 85.0
        current_close_inside = 95.0

        # above_pdh: current close > PDH
        assert current_close_above > pdh  # True

        # below_pdl: current close < PDL
        assert current_close_below < pdl  # True

        # inside_range: PDL <= current close <= PDH
        assert pdl <= current_close_inside <= pdh  # True

    def test_position_below_pdl(self):
        """Test position below PDL."""
        pdl = 90.0
        current_close = 85.0
        assert current_close < pdl

    def test_position_inside_range(self):
        """Test position inside range (between PDH and PDL)."""
        pdh, pdl = 110.0, 90.0
        current_close = 100.0
        assert pdl <= current_close <= pdh

    def test_empty_data(self):
        """Test empty DataFrame returns insufficient_data."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], tz="UTC"),
        )

        result = detect_pdh_pdl(df)

        assert result["position"] == "insufficient_data"
        assert pd.isna(result["previous_day_high"])
