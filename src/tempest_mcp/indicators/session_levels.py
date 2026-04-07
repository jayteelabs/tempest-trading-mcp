"""Session levels: Asia, London, New York PDH/PDL detection."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

import pandas as pd
import pytz

NY_TZ = pytz.timezone("America/New_York")


def detect_session_levels(
    ohlcv_df: pd.DataFrame, session_type: Literal["asia", "london", "ny"]
) -> dict:
    """Detect session high/low levels for Asia, London, or NY session.

    Args:
        ohlcv_df: DataFrame with UTC-aware pd.Timestamp index and columns
                  [open, high, low, close, volume]
        session_type: One of "asia", "london", "ny"

    Returns:
        dict with session_type, session_start_utc, session_end_utc, high, low,
        range, midpoint, bars. Returns bars=0 and NaN values if no bars in window.
        Raises ValueError if session_type is invalid.

    Session windows (UTC):
        - Asia: 00:00–09:00 UTC (fixed, no DST)
        - London: 08:00–17:00 UTC (fixed, no DST)
        - NY: 09:30–16:00 US Eastern (converts to UTC with DST using pytz)

    Note:
        When the DataFrame spans multiple days, only bars from the most recent
        session day are used (not aggregated across all days).
    """
    valid_sessions = {"asia", "london", "ny"}
    if session_type not in valid_sessions:
        raise ValueError(f"Invalid session_type: '{session_type}'. Must be one of {valid_sessions}")

    if ohlcv_df.empty:
        return _empty_session_result(session_type)

    last_bar_time = ohlcv_df.index[-1]

    # Asia: 00:00–09:00 UTC (fixed)
    if session_type == "asia":
        start_hour, end_hour = 0, 9
        session_day_candidate = last_bar_time.date()
        mask = (ohlcv_df.index.hour >= start_hour) & (ohlcv_df.index.hour < end_hour)

    # London: 08:00–17:00 UTC (fixed)
    elif session_type == "london":
        start_hour, end_hour = 8, 17
        session_day_candidate = last_bar_time.date()
        mask = (ohlcv_df.index.hour >= start_hour) & (ohlcv_df.index.hour < end_hour)

    # NY: 09:30–16:00 US Eastern (converts to UTC with DST)
    else:  # ny

        def ny_to_utc_mask(idx: pd.DatetimeIndex) -> pd.Series:
            """Create boolean mask for bars falling within NY session window."""
            ny_times = idx.tz_convert(NY_TZ)
            start_hour, start_min = 9, 30
            end_hour, end_min = 16, 0
            in_start = (ny_times.hour > start_hour) | (
                (ny_times.hour == start_hour) & (ny_times.minute >= start_min)
            )
            in_end = (ny_times.hour < end_hour) | (
                (ny_times.hour == end_hour) & (ny_times.minute < end_min)
            )
            return in_start & in_end

        mask = ny_to_utc_mask(ohlcv_df.index)
        # Session day in NY time
        ny_last_bar = last_bar_time.tz_convert(NY_TZ)
        session_day_candidate = ny_last_bar.date()

    # Filter to bars matching session window
    session_bars = ohlcv_df[mask]

    if len(session_bars) == 0:
        return _empty_session_result(session_type)

    # Find the most recent session day that has bars
    # Use the last bar's date as the primary session day; if no bars on that day,
    # use the previous calendar day (handles case where last bar is before session starts)
    most_recent_session_day = session_day_candidate

    def filter_to_session_day(df: pd.DataFrame, session_day: date, stype: str) -> pd.DataFrame:
        """Filter bars to only those on the specified session day."""
        if stype == "ny":
            # Filter by NY date
            df_ny = df.index.tz_convert(NY_TZ)
            day_mask = pd.DatetimeIndex(df_ny).date == session_day
            return df[day_mask]
        else:
            # Filter by UTC date
            return df[df.index.date == session_day]

    # Check if most recent day has bars; if not, try previous day
    bars_on_day = filter_to_session_day(session_bars, most_recent_session_day, session_type)
    if len(bars_on_day) == 0:
        most_recent_session_day = session_day_candidate - timedelta(days=1)
        bars_on_day = filter_to_session_day(session_bars, most_recent_session_day, session_type)

    if len(bars_on_day) == 0:
        return _empty_session_result(session_type)

    session_high = float(bars_on_day["high"].max())
    session_low = float(bars_on_day["low"].min())
    session_range = session_high - session_low
    midpoint = (session_high + session_low) / 2

    return {
        "session_type": session_type,
        "session_start_utc": bars_on_day.index[0],
        "session_end_utc": bars_on_day.index[-1],
        "high": session_high,
        "low": session_low,
        "range": session_range,
        "midpoint": midpoint,
        "bars": len(bars_on_day),
    }


def _empty_session_result(session_type: str) -> dict:
    """Return empty result dict for sessions with no bars."""
    return {
        "session_type": session_type,
        "session_start_utc": pd.NaT,
        "session_end_utc": pd.NaT,
        "high": float("nan"),
        "low": float("nan"),
        "range": float("nan"),
        "midpoint": float("nan"),
        "bars": 0,
    }


def detect_pdh_pdl(ohlcv_df: pd.DataFrame) -> dict:
    """Detect Previous Day High (PDH) and Previous Day Low (PDL).

    Args:
        ohlcv_df: DataFrame with UTC-aware pd.Timestamp index and columns
                  [open, high, low, close, volume]

    Returns:
        dict with previous_day_high, previous_day_low, previous_day_close,
        previous_day_range, pdh_timestamp_utc, pdl_timestamp_utc, pivot,
        r1, r2, s1, s2, position.
        Returns position="insufficient_data" if less than 2 full UTC calendar days.
    """
    if ohlcv_df.empty:
        return _empty_pdh_pdl_result(position="insufficient_data")

    # Need at least 2 full UTC calendar days of data
    # First bar time is the reference; we look for bars on the UTC calendar day before
    first_bar_time = ohlcv_df.index[0]
    first_bar_date = first_bar_time.date()

    # Find the UTC calendar day immediately before the first bar
    tz = pytz.UTC
    current_day_start = pd.Timestamp(first_bar_date, tz=tz)
    previous_day_start = current_day_start - pd.Timedelta(days=1)

    # Get all bars from the previous UTC calendar day
    previous_day_bars = ohlcv_df[
        (ohlcv_df.index >= previous_day_start) & (ohlcv_df.index < current_day_start)
    ]

    if len(previous_day_bars) == 0:
        return _empty_pdh_pdl_result(position="insufficient_data")

    pdh = float(previous_day_bars["high"].max())
    pdl = float(previous_day_bars["low"].min())
    pdc = float(previous_day_bars["close"].iloc[-1])  # Last close of the day
    pd_range = pdh - pdl

    # Find timestamps when PDH and PDL occurred (first occurrence if multiple bars share the value)
    pdh_mask = previous_day_bars["high"] == pdh
    pdl_mask = previous_day_bars["low"] == pdl
    pdh_timestamp = previous_day_bars[pdh_mask].index[0]
    pdl_timestamp = previous_day_bars[pdl_mask].index[0]

    # Pivot and levels
    pivot = (pdh + pdl + pdc) / 3
    r1 = 2 * pivot - pdl
    r2 = pivot + (pdh - pdl)
    s1 = 2 * pivot - pdh
    s2 = pivot - (pdh - pdl)

    # Determine current position
    current_close = float(ohlcv_df["close"].iloc[-1])

    if current_close > pdh:
        position = "above_pdh"
    elif current_close < pdl:
        position = "below_pdl"
    else:
        position = "inside_range"

    return {
        "previous_day_high": pdh,
        "previous_day_low": pdl,
        "previous_day_close": pdc,
        "previous_day_range": pd_range,
        "pdh_timestamp_utc": pdh_timestamp,
        "pdl_timestamp_utc": pdl_timestamp,
        "pivot": pivot,
        "r1": r1,
        "r2": r2,
        "s1": s1,
        "s2": s2,
        "position": position,
    }


def _empty_pdh_pdl_result(position: str = "insufficient_data") -> dict:
    """Return empty result dict for PDH/PDL when data is insufficient."""
    return {
        "previous_day_high": float("nan"),
        "previous_day_low": float("nan"),
        "previous_day_close": float("nan"),
        "previous_day_range": float("nan"),
        "pdh_timestamp_utc": pd.NaT,
        "pdl_timestamp_utc": pd.NaT,
        "pivot": float("nan"),
        "r1": float("nan"),
        "r2": float("nan"),
        "s1": float("nan"),
        "s2": float("nan"),
        "position": position,
    }
