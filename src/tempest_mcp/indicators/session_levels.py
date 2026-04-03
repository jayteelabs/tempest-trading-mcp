"""Session levels: Asia, London, New York PDH/PDL detection."""
from datetime import datetime, timedelta
import numpy as np
from tempest_mcp.config import ErrorCodes
from tempest_mcp.indicators.ta_wrapper import IndicatorError
from tempest_mcp.models.indicator import SessionLevels, SessionType

SESSION_TIMES = {
    SessionType.ASIA: {"start": 0, "end": 8},
    SessionType.LONDON: {"start": 7, "end": 16},
    SessionType.NEW_YORK: {"start": 13, "end": 22},
}

def calculate_session_levels(timestamps, high, low) -> SessionLevels:
    timestamps_arr = np.array(timestamps, dtype=np.float64)
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    if len(timestamps_arr) < 24:
        raise IndicatorError("Session levels require at least 24 candles", code=ErrorCodes.INSUFFICIENT_DATA)
    datetimes = [datetime.utcfromtimestamp(ts) for ts in timestamps_arr]
    current_dt = datetimes[-1]
    asia_high, asia_low = _find_session_levels(datetimes, high_arr, low_arr, SessionType.ASIA, current_dt)
    london_high, london_low = _find_session_levels(datetimes, high_arr, low_arr, SessionType.LONDON, current_dt)
    ny_high, ny_low = _find_session_levels(datetimes, high_arr, low_arr, SessionType.NEW_YORK, current_dt)
    return SessionLevels(symbol="", timeframe="", timestamp=timestamps_arr[-1], values={"asia_high": asia_high, "asia_low": asia_low, "london_high": london_high, "london_low": london_low, "ny_high": ny_high, "ny_low": ny_low})

def _find_session_levels(datetimes, high, low, session, current_dt):
    session_times = SESSION_TIMES[session]
    start_hour, end_hour = session_times["start"], session_times["end"]
    session_highs, session_lows = [], []
    for i, dt in enumerate(datetimes):
        if start_hour <= dt.hour < end_hour and dt.date() < current_dt.date():
            session_highs.append(high[i])
            session_lows.append(low[i])
    if session_highs:
        return float(max(session_highs)), float(min(session_lows))
    return float(np.max(high)), float(np.min(low))
