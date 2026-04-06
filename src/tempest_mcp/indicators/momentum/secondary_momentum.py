"""Secondary Momentum Indicator Engine.

Implements momentum indicators using pure-pandas calculations:
- CCI (Commodity Channel Index)
- Williams %R (Williams Percent Range)
- ROC (Rate of Change)

All functions use standard SMA or rolling calculations as specified.
No EWM/Wilder smoothing in this module.
"""

import numpy as np
import pandas as pd

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Default periods
CCI_DEFAULT_PERIOD = 20
WILLIAMS_R_DEFAULT_PERIOD = 14
ROC_DEFAULT_PERIOD = 12


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate Commodity Channel Index (CCI).

    CCI measures the variation of a security's price from its statistical mean.
    It is unbounded and typically oscillates between ±100.

    Args:
        high: Series of high prices with datetime index (UTC-aware).
        low: Series of low prices with datetime index (UTC-aware).
        close: Series of close prices with datetime index (UTC-aware).
        period: Number of periods for calculation (default 20). Must be positive.

    Returns:
        pd.Series containing CCI values, aligned with input index.
        Returns empty Series if period <= 0 or input is empty.
        Returns Series of NaN if input length < period (insufficient data).

    Raises:
        ValueError: If period is not a positive integer.

    Example:
        >>> high = pd.Series([105, 110, 108], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> low = pd.Series([100, 102, 104], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> close = pd.Series([103, 108, 106], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> cci = calculate_cci(high, low, close, period=20)
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    # Handle empty input
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logger.debug("Empty input series for CCI calculation")
        return pd.Series(dtype=float)

    # Insufficient data check
    if len(close) < period:
        logger.debug(
            "Insufficient data for CCI(%d): %d < %d",
            period,
            len(close),
            period,
        )
        return pd.Series(dtype=float, index=close.index)

    # Align all series - NaN values propagate naturally through rolling calculations
    aligned = pd.DataFrame({"high": high, "low": low, "close": close})

    high_vals = aligned["high"]
    low_vals = aligned["low"]
    close_vals = aligned["close"]

    # Calculate Typical Price (TP)
    tp = (high_vals + low_vals + close_vals) / 3.0

    # Calculate SMA of TP
    tp_sma = tp.rolling(window=period).mean()

    # Calculate Mean Absolute Deviation of TP
    # Using rolling().apply() with raw=True for efficiency
    def mean_abs_dev(x):
        return np.mean(np.abs(x - np.mean(x)))

    mad = tp.rolling(window=period).apply(mean_abs_dev, raw=True)

    # Calculate CCI: (TP - SMA(TP)) / (0.015 × MAD)
    # 0.015 scales CCI so ~70-80% of values fall within ±100 in a ranging market
    cci = (tp - tp_sma) / (0.015 * mad)

    # Return aligned with original close index
    return pd.Series(cci.values, index=close.index)


def calculate_williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Calculate Williams %R (Williams Percent Range).

    Williams %R is a momentum indicator that measures overbought/oversold levels.
    It oscillates between 0 and -100, where:
    - Values close to 0 indicate overbought conditions
    - Values close to -100 indicate oversold conditions

    Args:
        high: Series of high prices with datetime index (UTC-aware).
        low: Series of low prices with datetime index (UTC-aware).
        close: Series of close prices with datetime index (UTC-aware).
        period: Number of periods for lookback (default 14). Must be positive.

    Returns:
        pd.Series containing Williams %R values (-100 to 0), aligned with input index.
        Returns empty Series if period <= 0 or input is empty.
        Returns Series of NaN if input length < period (insufficient data).
        Returns Series of -50.0 where highest high equals lowest low (flat market).

    Raises:
        ValueError: If period is not a positive integer.

    Example:
        >>> high = pd.Series([105, 110, 108], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> low = pd.Series([100, 102, 104], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> close = pd.Series([103, 108, 106], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> williams_r = calculate_williams_r(high, low, close, period=14)
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    # Handle empty input
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logger.debug("Empty input series for Williams %R calculation")
        return pd.Series(dtype=float)

    # Insufficient data check
    if len(close) < period:
        logger.debug(
            "Insufficient data for Williams R (%d): %d < %d",
            period,
            len(close),
            period,
        )
        return pd.Series(dtype=float, index=close.index)

    # Align all series - NaN values propagate naturally through rolling calculations
    aligned = pd.DataFrame({"high": high, "low": low, "close": close})

    high_vals = aligned["high"]
    low_vals = aligned["low"]
    close_vals = aligned["close"]

    # Calculate rolling highest high and lowest low
    highest_high = high_vals.rolling(window=period).max()
    lowest_low = low_vals.rolling(window=period).min()

    # Calculate the range
    range_val = highest_high - lowest_low

    # Calculate Williams %R: -100 × (HH - C) / (HH - LL)
    # Where HH = highest high, C = close, LL = lowest low
    # Handle division by zero: when HH == LL (flat market), return -50.0
    williams_r = -100.0 * (highest_high - close_vals) / range_val

    # For flat market bars (HH == LL), return -50.0 instead of NaN
    # Use np.where to handle the flat market case
    williams_r = np.where(
        range_val == 0,
        -50.0,
        williams_r,
    )

    # Return aligned with original close index
    return pd.Series(williams_r, index=close.index)


def calculate_roc(
    prices: pd.Series,
    period: int = 12,
) -> pd.Series:
    """Calculate Rate of Change (ROC).

    ROC measures the percentage change in price from period bars ago.
    Positive values indicate upward momentum, negative values indicate downward momentum.

    Args:
        prices: Series of price values with datetime index (UTC-aware).
        period: Number of periods for lookback (default 12). Must be positive.

    Returns:
        pd.Series containing ROC values (as percentage), aligned with input index.
        Returns empty Series if period <= 0 or input is empty.
        Returns Series with first 'period' values as NaN if input length < period + 1
        (insufficient data for lookback).

    Raises:
        ValueError: If period is not a positive integer.

    Example:
        >>> prices = pd.Series([100, 102, 101, 105], index=pd.date_range('2024-01-01', periods=4, tz='UTC'))
        >>> roc = calculate_roc(prices, period=2)
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    # Handle empty input
    if len(prices) == 0:
        logger.debug("Empty input series for ROC calculation")
        return pd.Series(dtype=float)

    # Insufficient data check: need at least period + 1 values for one valid ROC calculation
    if len(prices) < period + 1:
        logger.debug(
            "Insufficient data for ROC(%d): %d < %d",
            period,
            len(prices),
            period + 1,
        )
        return pd.Series(dtype=float, index=prices.index)

    # Calculate ROC directly on prices - NaN values propagate through shift operation
    # ROC = 100 × (current - price[period]) / price[period]
    price_shift = prices.shift(period)
    roc = 100.0 * (prices - price_shift) / price_shift

    return roc


__all__ = [
    # CCI
    "calculate_cci",
    "CCI_DEFAULT_PERIOD",
    # Williams %R
    "calculate_williams_r",
    "WILLIAMS_R_DEFAULT_PERIOD",
    # ROC
    "calculate_roc",
    "ROC_DEFAULT_PERIOD",
]
