"""ATR Indicator Engine - Average True Range calculations.

Implements ATR calculation using Wilder's smoothing method.
ATR measures market volatility by decomposing the entire range of an asset
for each period.

True Range (TR) = max(H - L, |H - PC|, |L - PC|) where PC = previous close
ATR = Wilder's smoothing of TR (equivalent to EMA with alpha = 1/period)
"""

import pandas as pd

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Default ATR period
ATR_DEFAULT_PERIOD = 14


def calculate_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Average True Range (ATR) using Wilder's smoothing.

    ATR measures market volatility by calculating the average of True Range
    values over a specified period. Wilder's smoothing uses alpha = 1/period,
    which is different from standard EMA (alpha = 2/(period+1)).

    Args:
        high: Series of high prices with datetime index (UTC-aware).
        low: Series of low prices with datetime index (UTC-aware).
        close: Series of close prices with datetime index (UTC-aware).
        period: Number of periods for ATR calculation (default 14).
                Must be a positive integer.

    Returns:
        pd.Series containing ATR values, aligned with input index.
        Returns empty Series if:
            - input length is 0
            - input length < period

    Raises:
        ValueError: If period is not a positive integer.

    Example:
        >>> high = pd.Series([105, 110, 108], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> low = pd.Series([100, 102, 104], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> close = pd.Series([103, 108, 106], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> atr = calculate_atr(high, low, close, period=14)
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    # Handle empty input
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logger.debug("Empty input series for ATR calculation")
        return pd.Series(dtype=float)

    # Insufficient data check
    if len(close) < period:
        logger.debug(
            "Insufficient data for ATR(%d): %d < %d",
            period,
            len(close),
            period,
        )
        return pd.Series(dtype=float)

    # Align all series to same index
    aligned = pd.DataFrame({"high": high, "low": low, "close": close})
    aligned = aligned.dropna()

    if len(aligned) < period:
        logger.debug(
            "Insufficient aligned data for ATR(%d): %d < %d",
            period,
            len(aligned),
            period,
        )
        return pd.Series(dtype=float)

    # Calculate True Range (TR)
    # TR = max(H - L, |H - PC|, |L - PC|)
    high_vals = aligned["high"]
    low_vals = aligned["low"]
    close_vals = aligned["close"]

    # Previous close (shifted by 1)
    prev_close = close_vals.shift(1)

    # Calculate the three components of True Range
    tr1 = high_vals - low_vals  # H - L
    tr2 = (high_vals - prev_close).abs()  # |H - PC|
    tr3 = (low_vals - prev_close).abs()  # |L - PC|

    # True Range is the maximum of the three
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # First TR value will be NaN (no previous close), use H-L for first bar
    tr.iloc[0] = high_vals.iloc[0] - low_vals.iloc[0]

    # Calculate ATR using Wilder's smoothing with SMA seed
    # First ATR = SMA of first `period` TR values
    # Subsequent: ATR[t] = (ATR[t-1] * (period - 1) + TR[t]) / period
    # This is mathematically equivalent to ewm(alpha=1/period, adjust=False)
    # but seeded with an SMA of the first `period` values (Wilder's original method)
    atr = pd.Series(index=tr.index, dtype=float)
    atr.iloc[period - 1] = tr.iloc[:period].mean()  # SMA seed
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period

    return atr


__all__ = [
    "calculate_atr",
    "ATR_DEFAULT_PERIOD",
]
