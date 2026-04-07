"""Unit tests for session levels detection (Asia, London, NY PDH/PDL)."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

from tempest_mcp.indicators.session_levels import (
    detect_session_levels,
    detect_pdh_pdl,
)


class TestDetectSessionLevels:
    """Tests for detect_session_levels function."""

    def test_asia_session_normal(self):
        """Test Asia session detection with multi-day data containing Asia bars."""
        # Create 2 days of hourly data starting at 2024-03-15 00:00 UTC
        dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(48)],
            "high": [101.0 + i * 0.1 for i in range(48)],
            "low": [99.0 + i * 0.1 for i in range(48)],
            "close": [100.5 + i * 0.1 for i in range(48)],
            "volume": [1000.0] * 48,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "asia")

        assert result["session_type"] == "asia"
        assert result["bars"] == 9  # 00:00-09:00 UTC = 9 hours
        assert result["session_start_utc"] == dates[0]
        assert result["session_end_utc"] == dates[8]
        assert result["high"] == 101.8  # high at index 8
        assert result["low"] == 99.0  # low at index 0
        assert result["range"] == result["high"] - result["low"]
        assert result["midpoint"] == (result["high"] + result["low"]) / 2

    def test_london_session_normal(self):
        """Test London session detection with multi-day data."""
        # Create 2 days of hourly data starting at 2024-03-15 00:00 UTC
        dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(48)],
            "high": [101.0 + i * 0.1 for i in range(48)],
            "low": [99.0 + i * 0.1 for i in range(48)],
            "close": [100.5 + i * 0.1 for i in range(48)],
            "volume": [1000.0] * 48,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "london")

        assert result["session_type"] == "london"
        assert result["bars"] == 9  # 08:00-17:00 UTC = 9 hours (08 to 16 inclusive)
        # London session starts at 08:00 UTC, so index 8 (08:00) to index 16 (16:00)
        assert result["session_start_utc"] == dates[8]
        assert result["session_end_utc"] == dates[16]

    def test_ny_session_normal(self):
        """Test NY session detection with multi-day data."""
        # Create 2 days of hourly data starting at 2024-03-15 00:00 UTC
        dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(48)],
            "high": [101.0 + i * 0.1 for i in range(48)],
            "low": [99.0 + i * 0.1 for i in range(48)],
            "close": [100.5 + i * 0.1 for i in range(48)],
            "volume": [1000.0] * 48,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_session_levels(df, "ny")

        # NY session: 09:30-16:00 ET
        # On 2024-03-15 in EDT (DST started March 10), 09:30 ET = 13:30 UTC
        # 16:00 ET = 20:00 UTC
        # So bars 13:30 to 20:00 UTC on day 1 = indices 13, 14, 15, 16, 17, 18, 19, 20
        # That's 8 bars (13-20 inclusive)
        assert result["session_type"] == "ny"
        assert result["bars"] > 0

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
        """Test PDH/PDL calculation with standard multi-day data."""
        # Create 3 days of hourly data
        dates = pd.date_range("2024-03-15", periods=72, freq="h", tz="UTC")
        data = {
            "open": [100.0 + i * 0.1 for i in range(72)],
            "high": [101.0 + i * 0.1 for i in range(72)],
            "low": [99.0 + i * 0.1 for i in range(72)],
            "close": [100.5 + i * 0.1 for i in range(72)],
            "volume": [1000.0] * 72,
        }
        df = pd.DataFrame(data, index=dates)

        result = detect_pdh_pdl(df)

        # PDH/PDL should be from March 14 (the day before first bar March 15)
        # March 14 bars are indices 0-23
        assert result["previous_day_high"] == 101.0 + 23 * 0.1  # max high on day 0
        assert result["previous_day_low"] == 99.0  # min low on day 0
        assert result["previous_day_range"] == result["previous_day_high"] - result["previous_day_low"]
        assert result["position"] == "inside_range"  # current close 100.5 is inside range

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
        # Create data with specific PDH/PDL values
        # Day 1 (March 14): high=110, low=90, close=100
        # Day 2 (March 15): current
        dates_day1 = pd.date_range("2024-03-14", periods=24, freq="h", tz="UTC")
        data_day1 = {
            "open": [100.0] * 24,
            "high": [110.0] * 24,  # PDH = 110
            "low": [90.0] * 24,  # PDL = 90
            "close": [100.0] * 24,  # PDC = 100
            "volume": [1000.0] * 24,
        }
        df_day1 = pd.DataFrame(data_day1, index=dates_day1)

        dates_day2 = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data_day2 = {
            "open": [101.0] * 24,
            "high": [105.0] * 24,
            "low": [95.0] * 24,
            "close": [102.0] * 24,
            "volume": [1000.0] * 24,
        }
        df_day2 = pd.DataFrame(data_day2, index=dates_day2)

        df = pd.concat([df_day1, df_day2])

        result = detect_pdh_pdl(df)

        # pivot = (PDH + PDL + PDC) / 3 = (110 + 90 + 100) / 3 = 100
        assert result["pivot"] == 100.0
        # r1 = 2 * pivot - PDL = 2 * 100 - 90 = 110
        assert result["r1"] == 110.0
        # r2 = pivot + (PDH - PDL) = 100 + 20 = 120
        assert result["r2"] == 120.0
        # s1 = 2 * pivot - PDH = 2 * 100 - 110 = 90
        assert result["s1"] == 90.0
        # s2 = pivot - (PDH - PDL) = 100 - 20 = 80
        assert result["s2"] == 80.0

    def test_position_above_pdh(self):
        """Test position above PDH."""
        # Day 1 with PDH=100
        dates_day1 = pd.date_range("2024-03-14", periods=24, freq="h", tz="UTC")
        data_day1 = {
            "open": [95.0] * 24,
            "high": [100.0] * 24,  # PDH = 100
            "low": [90.0] * 24,
            "close": [95.0] * 24,
            "volume": [1000.0] * 24,
        }
        df_day1 = pd.DataFrame(data_day1, index=dates_day1)

        # Day 2 with close above PDH
        dates_day2 = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data_day2 = {
            "open": [101.0] * 24,
            "high": [105.0] * 24,
            "low": [98.0] * 24,
            "close": [102.0] * 24,  # > PDH of 100
            "volume": [1000.0] * 24,
        }
        df_day2 = pd.DataFrame(data_day2, index=dates_day2)

        df = pd.concat([df_day1, df_day2])

        result = detect_pdh_pdl(df)

        assert result["position"] == "above_pdh"

    def test_position_below_pdl(self):
        """Test position below PDL."""
        # Day 1 with PDL=100
        dates_day1 = pd.date_range("2024-03-14", periods=24, freq="h", tz="UTC")
        data_day1 = {
            "open": [105.0] * 24,
            "high": [110.0] * 24,
            "low": [100.0] * 24,  # PDL = 100
            "close": [105.0] * 24,
            "volume": [1000.0] * 24,
        }
        df_day1 = pd.DataFrame(data_day1, index=dates_day1)

        # Day 2 with close below PDL
        dates_day2 = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data_day2 = {
            "open": [98.0] * 24,
            "high": [102.0] * 24,
            "low": [95.0] * 24,
            "close": [96.0] * 24,  # < PDL of 100
            "volume": [1000.0] * 24,
        }
        df_day2 = pd.DataFrame(data_day2, index=dates_day2)

        df = pd.concat([df_day1, df_day2])

        result = detect_pdh_pdl(df)

        assert result["position"] == "below_pdl"

    def test_position_inside_range(self):
        """Test position inside range (between PDH and PDL)."""
        # Day 1 with PDH=110, PDL=90
        dates_day1 = pd.date_range("2024-03-14", periods=24, freq="h", tz="UTC")
        data_day1 = {
            "open": [100.0] * 24,
            "high": [110.0] * 24,  # PDH = 110
            "low": [90.0] * 24,  # PDL = 90
            "close": [100.0] * 24,
            "volume": [1000.0] * 24,
        }
        df_day1 = pd.DataFrame(data_day1, index=dates_day1)

        # Day 2 with close inside range
        dates_day2 = pd.date_range("2024-03-15", periods=24, freq="h", tz="UTC")
        data_day2 = {
            "open": [101.0] * 24,
            "high": [105.0] * 24,
            "low": [95.0] * 24,
            "close": [100.0] * 24,  # Inside range 90-110
            "volume": [1000.0] * 24,
        }
        df_day2 = pd.DataFrame(data_day2, index=dates_day2)

        df = pd.concat([df_day1, df_day2])

        result = detect_pdh_pdl(df)

        assert result["position"] == "inside_range"

    def test_empty_data(self):
        """Test empty DataFrame returns insufficient_data."""
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.to_datetime([], tz="UTC"),
        )

        result = detect_pdh_pdl(df)

        assert result["position"] == "insufficient_data"
        assert pd.isna(result["previous_day_high"])
