"""Volume indicator subpackage - combines OBV/MFI wrappers and VWAP engine."""

import numpy as np
import pandas as pd

from tempest_mcp.indicators.volume.vwap import (
    SESSION_ANCHORS,
    calculate_vwap,
    calculate_vwap_bands,
    detect_vwap_cross,
)
from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Import ta-lib based result wrappers
try:
    import talib

    from tempest_mcp.models.indicator import MFIResult, OBVResult
    _HAS_TALIB = True
except ImportError:
    _HAS_TALIB = False


def calculate_obv(
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Calculate On-Balance Volume (OBV) using pure pandas.

    OBV is a cumulative indicator that adds volume on up closes
    and subtracts volume on down closes.

    Formula:
        If close[t] > close[t-1]: OBV[t] = OBV[t-1] + volume[t]
        If close[t] < close[t-1]: OBV[t] = OBV[t-1] - volume[t]
        If close[t] == close[t-1]: OBV[t] = OBV[t-1]
        First bar: OBV[0] = volume[0]

    Args:
        close: Series of close prices with UTC-aware DatetimeIndex.
        volume: Series of volume values with UTC-aware DatetimeIndex.

    Returns:
        pd.Series containing OBV values with UTC-aware index.
        Returns empty Series if input length is 0.

    Raises:
        ValueError: If close and volume have different lengths.
        ValueError: If close and volume have different DatetimeIndex values.

    Example:
        >>> close = pd.Series([100, 102, 101, 103], index=pd.date_range('2024-01-01', periods=4, tz='UTC'))
        >>> volume = pd.Series([1000, 1100, 1050, 1200], index=close.index)
        >>> obv = calculate_obv(close, volume)
    """
    if len(close) == 0:
        return pd.Series(dtype=float)

    if len(close) != len(volume):
        raise ValueError("close and volume must have the same length")

    # Ensure UTC-aware index
    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is None:
        close = close.copy()
        close.index = close.index.tz_localize("UTC")

    if isinstance(volume.index, pd.DatetimeIndex) and volume.index.tz is None:
        volume = volume.copy()
        volume.index = volume.index.tz_localize("UTC")

    # Validate indices are identical before alignment
    if not close.index.equals(volume.index):
        raise ValueError("close and volume must have the same DatetimeIndex")

    # Align indices
    aligned = pd.DataFrame({"close": close, "volume": volume}, index=close.index)
    aligned = aligned.dropna()

    if len(aligned) == 0:
        return pd.Series(dtype=float)

    # Calculate price changes
    close_vals = aligned["close"].values
    volume_vals = aligned["volume"].values

    # Initialize OBV array
    obv = np.zeros(len(close_vals))
    obv[0] = volume_vals[0]  # First bar: OBV[0] = volume[0]

    # Calculate OBV for subsequent bars
    for i in range(1, len(close_vals)):
        if close_vals[i] > close_vals[i - 1]:
            obv[i] = obv[i - 1] + volume_vals[i]
        elif close_vals[i] < close_vals[i - 1]:
            obv[i] = obv[i - 1] - volume_vals[i]
        else:
            obv[i] = obv[i - 1]  # No change on flat

    return pd.Series(obv, index=aligned.index)


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Money Flow Index (MFI) using pure pandas.

    MFI is a volume-weighted RSI that measures the flow of money
    into and out of an asset over a specified period.

    Formula:
        Typical Price = (High + Low + Close) / 3
        Raw Money Flow = Typical Price × Volume
        For each period:
            - Sum positive flows where TP > prior TP within the period
            - Sum negative flows where TP < prior TP within the period
        Money Ratio = Sum of Positive Raw Money Flow / Sum of Negative Raw Money Flow
        MFI = 100 - (100 / (1 + Money Ratio))

    Args:
        high: Series of high prices with UTC-aware DatetimeIndex.
        low: Series of low prices with UTC-aware DatetimeIndex.
        close: Series of close prices with UTC-aware DatetimeIndex.
        volume: Series of volume values with UTC-aware DatetimeIndex.
        period: Number of periods for MFI calculation (default 14).

    Returns:
        pd.Series containing MFI values with UTC-aware index.
        Output range is [0, 100].
        Returns empty Series if input length < period + 1.

    Raises:
        ValueError: If period <= 0.

    Example:
        >>> import numpy as np
        >>> n = 50
        >>> idx = pd.date_range('2024-01-01', periods=n, tz='UTC')
        >>> high = pd.Series(np.random.uniform(100, 110, n), index=idx)
        >>> low = pd.Series(np.random.uniform(90, 100, n), index=idx)
        >>> close = pd.Series(np.random.uniform(95, 105, n), index=idx)
        >>> volume = pd.Series(np.random.uniform(1000, 5000, n), index=idx)
        >>> mfi = calculate_mfi(high, low, close, volume, period=14)
    """
    if period <= 0:
        raise ValueError("Period must be a positive integer")

    if len(high) == 0 or len(low) == 0 or len(close) == 0 or len(volume) == 0:
        return pd.Series(dtype=float)

    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume):
        raise ValueError("All input Series must have the same length")

    # Need at least period + 1 bars for the first complete MFI calculation
    if len(high) < period + 1:
        logger.debug(
            "Insufficient data for MFI(%d): %d < %d + 1",
            period,
            len(high),
            period,
        )
        return pd.Series(dtype=float)

    # Ensure UTC-aware index
    if isinstance(high.index, pd.DatetimeIndex) and high.index.tz is None:
        high = high.copy()
        high.index = high.index.tz_localize("UTC")
    if isinstance(low.index, pd.DatetimeIndex) and low.index.tz is None:
        low = low.copy()
        low.index = low.index.tz_localize("UTC")
    if isinstance(close.index, pd.DatetimeIndex) and close.index.tz is None:
        close = close.copy()
        close.index = close.index.tz_localize("UTC")
    if isinstance(volume.index, pd.DatetimeIndex) and volume.index.tz is None:
        volume = volume.copy()
        volume.index = volume.index.tz_localize("UTC")

    # Align all series
    df = pd.DataFrame({
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }, index=high.index)
    df = df.dropna()

    if len(df) < period + 1:
        return pd.Series(dtype=float)

    # Calculate Typical Price
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0

    # Calculate Raw Money Flow
    raw_money_flow = typical_price * df["volume"]

    # Create result series
    mfi = pd.Series(index=df.index, dtype=float)

    # Calculate MFI for each complete period
    # MFI value at index i represents the MFI for period ending at i
    for i in range(period, len(typical_price)):
        # Sum flows for the period [i - period + 1, i]
        period_flows = raw_money_flow.iloc[i - period + 1:i + 1]
        period_tp = typical_price.iloc[i - period + 1:i + 1]

        # Compare each TP to the previous one WITHIN the period
        # tp_change[j] = tp[j] - tp[j-1] where j is within the period
        positive_flow = 0.0
        negative_flow = 0.0

        for j in range(1, len(period_flows)):
            # j=1 compares period_tp[1] to period_tp[0] (both within period)
            if period_tp.iloc[j] > period_tp.iloc[j - 1]:
                positive_flow += period_flows.iloc[j]
            elif period_tp.iloc[j] < period_tp.iloc[j - 1]:
                negative_flow += period_flows.iloc[j]
            # If equal, no flow added

        # Calculate Money Ratio
        if positive_flow == 0 and negative_flow == 0:
            # Neutral - no flow in either direction
            mfi.iloc[i] = 50.0
        elif negative_flow == 0:
            # No selling pressure - MFI = 100
            mfi.iloc[i] = 100.0
        elif positive_flow == 0:
            # No buying pressure - MFI = 0
            mfi.iloc[i] = 0.0
        else:
            money_ratio = positive_flow / negative_flow
            mfi_val = 100.0 - (100.0 / (1.0 + money_ratio))
            # Clamp to [0, 100] due to floating point
            mfi.iloc[i] = max(0.0, min(100.0, mfi_val))

    # Drop NaN values (first period values will be NaN since we start at index = period)
    mfi = mfi.dropna()

    return mfi


def calculate_obv_result(close, volume, ema_period: int = 20) -> "OBVResult":
    """Calculate OBV result wrapper using ta-lib.

    DEPRECATED: Use calculate_obv() for pure pandas implementation.
    """
    if not _HAS_TALIB:
        raise ImportError("ta-lib not available - install with: pip install ta-lib")

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


def calculate_mfi_result(high, low, close, volume, period: int = 14) -> "MFIResult":
    """Calculate MFI result wrapper using ta-lib.

    DEPRECATED: Use calculate_mfi() for pure pandas implementation.
    """
    if not _HAS_TALIB:
        raise ImportError("ta-lib not available - install with: pip install ta-lib")

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
    # Pure pandas volume indicators
    "calculate_obv",
    "calculate_mfi",
    # Result wrappers (ta-lib)
    "calculate_obv_result",
    "calculate_mfi_result",
]
