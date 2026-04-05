"""Trend indicator subpackage."""
import numpy as np
import talib

from tempest_mcp.indicators.trend.ema import (
    EMA_PERIODS,
    calculate_ema,
    calculate_ema_stack,
    death_cross,
    detect_ema_cross,
    golden_cross,
)
from tempest_mcp.models.indicator import SupertrendResult, VWAPResult


def calculate_vwap(high, low, close, volume) -> VWAPResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    volume_arr = np.array(volume, dtype=np.float64)
    typical_price = (high_arr + low_arr + close_arr) / 3
    cumulative_tp_volume = np.cumsum(typical_price * volume_arr)
    cumulative_volume = np.cumsum(volume_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        vwap = np.where(cumulative_volume > 0, cumulative_tp_volume / cumulative_volume, typical_price)
    return VWAPResult(symbol="", timeframe="", timestamp=0.0, values={"vwap": float(vwap[-1]) if len(vwap) > 0 else 0.0})


def calculate_supertrend(high, low, close, period: int = 10, multiplier: float = 3.0) -> SupertrendResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
    hl2 = (high_arr + low_arr) / 2
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    supertrend = np.zeros(len(close_arr))
    direction = np.zeros(len(close_arr))
    supertrend[0] = basic_upper[0] if not np.isnan(basic_upper[0]) else close_arr[0]
    direction[0] = 1
    for i in range(1, len(close_arr)):
        if np.isnan(basic_upper[i]) or np.isnan(basic_lower[i]):
            supertrend[i] = supertrend[i - 1]
            direction[i] = direction[i - 1]
            continue
        if direction[i - 1] == 1:
            if close_arr[i] < (basic_lower[i] if basic_lower[i] > supertrend[i - 1] or close_arr[i - 1] < supertrend[i - 1] else supertrend[i - 1]):
                direction[i] = -1
                supertrend[i] = basic_upper[i] if basic_upper[i] < supertrend[i - 1] or close_arr[i - 1] > supertrend[i - 1] else supertrend[i - 1]
            else:
                direction[i] = 1
                supertrend[i] = basic_lower[i] if basic_lower[i] > supertrend[i - 1] or close_arr[i - 1] < supertrend[i - 1] else supertrend[i - 1]
        else:
            if close_arr[i] > (basic_upper[i] if basic_upper[i] < supertrend[i - 1] or close_arr[i - 1] > supertrend[i - 1] else supertrend[i - 1]):
                direction[i] = 1
                supertrend[i] = basic_lower[i] if basic_lower[i] > supertrend[i - 1] or close_arr[i - 1] < supertrend[i - 1] else supertrend[i - 1]
            else:
                direction[i] = -1
                supertrend[i] = basic_upper[i] if basic_upper[i] < supertrend[i - 1] or close_arr[i - 1] > supertrend[i - 1] else supertrend[i - 1]
    supertrend = np.where(np.isnan(supertrend), close_arr, supertrend)
    trend_value = "bullish" if direction[-1] == 1 else "bearish"
    return SupertrendResult(symbol="", timeframe="", timestamp=0.0, values={"supertrend": float(supertrend[-1]), "trend": trend_value, "signal": int(direction[-1])})


__all__ = [
    "calculate_vwap",
    "calculate_supertrend",
    "EMA_PERIODS",
    "calculate_ema",
    "calculate_ema_stack",
    "death_cross",
    "detect_ema_cross",
    "golden_cross",
]
