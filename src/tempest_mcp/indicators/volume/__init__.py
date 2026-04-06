"""Volume indicator subpackage - combines OBV/MFI wrappers and VWAP engine."""

import numpy as np
import talib

from tempest_mcp.models.indicator import MFIResult, OBVResult

# Import VWAP engine functions
from tempest_mcp.indicators.volume.vwap import (
    SESSION_ANCHORS,
    calculate_vwap,
    calculate_vwap_bands,
    detect_vwap_cross,
)


def calculate_obv_result(close, volume, ema_period: int = 20) -> OBVResult:
    close_arr = np.array(close, dtype=np.float64)
    volume_arr = np.array(volume, dtype=np.float64)
    obv = talib.OBV(close_arr, volume_arr)
    obv_ema = talib.EMA(obv, timeperiod=ema_period)
    valid_obv = obv[~np.isnan(obv)]
    valid_ema = obv_ema[~np.isnan(obv_ema)]
    latest_obv = float(valid_obv[-1]) if len(valid_obv) > 0 else 0.0
    latest_ema = float(valid_ema[-1]) if len(valid_ema) > 0 else 0.0
    trend = "bullish" if latest_obv > latest_ema else "bearish"
    return OBVResult(symbol="", timeframe="", timestamp=0.0, values={"obv": latest_obv, "obv_ema": latest_ema, "trend": trend})


def calculate_mfi_result(high, low, close, volume, period: int = 14) -> MFIResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    volume_arr = np.array(volume, dtype=np.float64)
    mfi = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=period)
    valid_mfi = mfi[~np.isnan(mfi)]
    latest_mfi = float(valid_mfi[-1]) if len(valid_mfi) > 0 else 50.0
    return MFIResult(symbol="", timeframe="", timestamp=0.0, values={"mfi": latest_mfi, "overbought": latest_mfi > 80, "oversold": latest_mfi < 20})


__all__ = [
    # VWAP engine (pure pandas)
    "calculate_vwap",
    "calculate_vwap_bands",
    "detect_vwap_cross",
    "SESSION_ANCHORS",
    # Result wrappers (ta-lib)
    "calculate_obv_result",
    "calculate_mfi_result",
]
