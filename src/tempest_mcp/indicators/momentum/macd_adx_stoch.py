"""MACD, ADX, Stochastic Indicator Engine.

Implements momentum indicators using pure-pandas calculations:
- MACD histogram (Moving Average Convergence Divergence)
- ADX (Average Directional Index)
- Stochastic Oscillator

All functions use Wilder's smoothing where specified (ADX) and standard EMA
for MACD (per convention).
"""

import pandas as pd

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Default periods
MACD_DEFAULT_FAST = 12
MACD_DEFAULT_SLOW = 26
MACD_DEFAULT_SIGNAL = 9
ADX_DEFAULT_PERIOD = 14
STOCH_DEFAULT_K_PERIOD = 14
STOCH_DEFAULT_D_PERIOD = 3
STOCH_DEFAULT_SMOOTH_K = 3


def calculate_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Calculate MACD histogram and components.

    MACD measures momentum by comparing fast and slow EMAs.
    Uses standard EMA (not Wilder) per MACD convention.

    Args:
        prices: Series of price values (typically close) with datetime index (UTC-aware).
        fast: Fast EMA period (default 12). Must be positive.
        slow: Slow EMA period (default 26). Must be positive and > fast.
        signal: Signal line period (default 9). Must be positive.

    Returns:
        Dictionary with keys:
            - "macd": MACD line (Fast EMA - Slow EMA)
            - "signal": Signal line (EMA of MACD line)
            - "histogram": MACD line - Signal line

        All values are pd.Series aligned with input index.
        Returns empty Series for all keys if input length < slow.

    Raises:
        ValueError: If any period is not positive or if slow <= fast.

    Example:
        >>> prices = pd.Series(range(100, 150), index=pd.date_range('2024-01-01', periods=50, tz='UTC'))
        >>> macd_data = calculate_macd(prices, fast=12, slow=26, signal=9)
        >>> print(macd_data.keys())
        dict_keys(['macd', 'signal', 'histogram'])
    """
    # Validate periods
    if not isinstance(fast, int) or fast <= 0:
        raise ValueError("Fast period must be a positive integer")
    if not isinstance(slow, int) or slow <= 0:
        raise ValueError("Slow period must be a positive integer")
    if not isinstance(signal, int) or signal <= 0:
        raise ValueError("Signal period must be a positive integer")
    if slow <= fast:
        raise ValueError("Slow period must be greater than fast period")

    # Handle empty input
    if len(prices) == 0:
        logger.debug("Empty input series for MACD calculation")
        return {
            "macd": pd.Series(dtype=float),
            "signal": pd.Series(dtype=float),
            "histogram": pd.Series(dtype=float),
        }

    # Insufficient data check
    if len(prices) < slow:
        logger.debug(
            "Insufficient data for MACD(%d, %d, %d): %d < %d",
            fast,
            slow,
            signal,
            len(prices),
            slow,
        )
        return {
            "macd": pd.Series(dtype=float),
            "signal": pd.Series(dtype=float),
            "histogram": pd.Series(dtype=float),
        }

    # Calculate fast and slow EMAs using standard EMA (adjust=False)
    # EMA with span uses alpha = 2/(span+1)
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()

    # MACD line = Fast EMA - Slow EMA
    macd_line = fast_ema - slow_ema

    # Signal line = EMA of MACD line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram = MACD - Signal
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> dict[str, pd.Series]:
    """Calculate ADX (Average Directional Index) with +DI and -DI.

    ADX measures trend strength regardless of direction.
    Uses Wilder's smoothing (alpha = 1/period) for all smoothing operations.

    Args:
        high: Series of high prices with datetime index (UTC-aware).
        low: Series of low prices with datetime index (UTC-aware).
        close: Series of close prices with datetime index (UTC-aware).
        period: Smoothing period (default 14). Must be positive.

    Returns:
        Dictionary with keys:
            - "adx": ADX values
            - "plus_di": +DI values
            - "minus_di": -DI values

        All values are pd.Series aligned with the post-dropna index of the
        aligned input series. If NaN values are present in the input, they are
        dropped during alignment and outputs reflect only clean (non-NaN) bars.
        Returns empty Series for all keys if input length < period * 2.

    Raises:
        ValueError: If period is not a positive integer.

    Example:
        >>> high = pd.Series([105, 110, 108], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> low = pd.Series([100, 102, 104], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> close = pd.Series([103, 108, 106], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> adx_data = calculate_adx(high, low, close, period=14)
    """
    if not isinstance(period, int) or period <= 0:
        raise ValueError("Period must be a positive integer")

    # Handle empty input
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logger.debug("Empty input series for ADX calculation")
        return {
            "adx": pd.Series(dtype=float),
            "plus_di": pd.Series(dtype=float),
            "minus_di": pd.Series(dtype=float),
        }

    # Insufficient data check - need at least period * 2 bars for proper smoothing
    if len(close) < period * 2:
        logger.debug(
            "Insufficient data for ADX(%d): %d < %d",
            period,
            len(close),
            period * 2,
        )
        return {
            "adx": pd.Series(dtype=float),
            "plus_di": pd.Series(dtype=float),
            "minus_di": pd.Series(dtype=float),
        }

    # Align all series to same index
    aligned = pd.DataFrame({"high": high, "low": low, "close": close})
    aligned = aligned.dropna()

    if len(aligned) < period * 2:
        logger.debug(
            "Insufficient aligned data for ADX(%d): %d < %d",
            period,
            len(aligned),
            period * 2,
        )
        return {
            "adx": pd.Series(dtype=float),
            "plus_di": pd.Series(dtype=float),
            "minus_di": pd.Series(dtype=float),
        }

    high_vals = aligned["high"]
    low_vals = aligned["low"]
    close_vals = aligned["close"]

    # Calculate +DM and -DM
    # +DM = max(high - prev_high, 0)
    # -DM = max(prev_low - low, 0)
    up_move = high_vals.diff()
    down_move = -low_vals.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # Calculate True Range using same method as ATR
    prev_close = close_vals.shift(1)
    tr1 = high_vals - low_vals
    tr2 = (high_vals - prev_close).abs()
    tr3 = (low_vals - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # First TR value: use H-L (no previous close)
    tr.iloc[0] = high_vals.iloc[0] - low_vals.iloc[0]

    # Smooth DM and TR using Wilder's smoothing
    # Wilder's smoothing uses alpha = 1/period
    # ewm(alpha=1/period, adjust=False)
    alpha = 1.0 / period

    smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()
    smoothed_tr = tr.ewm(alpha=alpha, adjust=False).mean()

    # Calculate +DI and -DI
    # +DI = 100 * smoothed_plus_dm / smoothed_tr
    # -DI = 100 * smoothed_minus_dm / smoothed_tr
    # Handle division by zero
    plus_di = 100.0 * smoothed_plus_dm / smoothed_tr.where(smoothed_tr != 0, pd.NA)
    minus_di = 100.0 * smoothed_minus_dm / smoothed_tr.where(smoothed_tr != 0, pd.NA)

    # Fill any NaN from division by zero with 0
    plus_di = plus_di.fillna(0.0)
    minus_di = minus_di.fillna(0.0)

    # Calculate DX
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.where(di_sum != 0, pd.NA)
    dx = dx.fillna(0.0)

    # Calculate ADX (Wilder's smoothing of DX)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }


def calculate_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> dict[str, pd.Series]:
    """Calculate Stochastic Oscillator.

    The Stochastic Oscillator compares a security's closing price to its
    price range over a given time period.

    Args:
        high: Series of high prices with datetime index (UTC-aware).
        low: Series of low prices with datetime index (UTC-aware).
        close: Series of close prices with datetime index (UTC-aware).
        k_period: %K lookback period (default 14). Must be positive.
        d_period: %D smoothing period (default 3). Must be positive.
        smooth_k: Smoothing period for %K (default 3). Use 1 for no smoothing.

    Returns:
        Dictionary with keys:
            - "percent_k": %K values (smoothed if smooth_k > 1)
            - "percent_d": %D values (SMA of %K)

        All values are pd.Series aligned with the post-dropna index of the
        aligned input series, clamped to [0, 100]. If NaN values are present
        in the input, they are dropped during alignment and outputs reflect
        only clean (non-NaN) bars.
        Returns empty Series for all keys if input length < k_period.

    Raises:
        ValueError: If any period is not a positive integer.

    Example:
        >>> high = pd.Series([105, 110, 115], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> low = pd.Series([100, 102, 108], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> close = pd.Series([103, 108, 112], index=pd.date_range('2024-01-01', periods=3, tz='UTC'))
        >>> stoch = calculate_stochastic(high, low, close, k_period=14, d_period=3, smooth_k=3)
    """
    # Validate periods
    if not isinstance(k_period, int) or k_period <= 0:
        raise ValueError("K period must be a positive integer")
    if not isinstance(d_period, int) or d_period <= 0:
        raise ValueError("D period must be a positive integer")
    if not isinstance(smooth_k, int) or smooth_k <= 0:
        raise ValueError("Smooth K period must be a positive integer")

    # Handle empty input
    if len(high) == 0 or len(low) == 0 or len(close) == 0:
        logger.debug("Empty input series for Stochastic calculation")
        return {
            "percent_k": pd.Series(dtype=float),
            "percent_d": pd.Series(dtype=float),
        }

    # Insufficient data check
    if len(close) < k_period:
        logger.debug(
            "Insufficient data for Stochastic(%d): %d < %d",
            k_period,
            len(close),
            k_period,
        )
        return {
            "percent_k": pd.Series(dtype=float),
            "percent_d": pd.Series(dtype=float),
        }

    # Align all series to same index
    aligned = pd.DataFrame({"high": high, "low": low, "close": close})
    aligned = aligned.dropna()

    if len(aligned) < k_period:
        logger.debug(
            "Insufficient aligned data for Stochastic(%d): %d < %d",
            k_period,
            len(aligned),
            k_period,
        )
        return {
            "percent_k": pd.Series(dtype=float),
            "percent_d": pd.Series(dtype=float),
        }

    high_vals = aligned["high"]
    low_vals = aligned["low"]
    close_vals = aligned["close"]

    # Calculate rolling high and low
    highest_high = high_vals.rolling(window=k_period).max()
    lowest_low = low_vals.rolling(window=k_period).min()

    # Calculate raw %K
    # %K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
    range_val = highest_high - lowest_low
    # Avoid division by zero - only fill 50.0 where range is exactly 0
    # Do NOT fill warm-up NaNs (from incomplete rolling windows) - keep those as NaN
    percent_k_raw = 100.0 * (close_vals - lowest_low) / range_val.where(range_val != 0, pd.NA)
    # Only assign neutral 50 for zero-range bars, not for warm-up periods
    percent_k_raw = percent_k_raw.where(~((percent_k_raw.isna()) & (range_val == 0)), 50.0)

    # Apply smoothing to %K if smooth_k > 1
    if smooth_k > 1:
        percent_k = percent_k_raw.rolling(window=smooth_k).mean()
    else:
        percent_k = percent_k_raw

    # Calculate %D (SMA of %K)
    percent_d = percent_k.rolling(window=d_period).mean()

    # Clamp output to [0, 100] - Stochastic can exceed bounds during gaps
    percent_k = percent_k.clip(0, 100)
    percent_d = percent_d.clip(0, 100)

    return {
        "percent_k": percent_k,
        "percent_d": percent_d,
    }


__all__ = [
    # MACD
    "calculate_macd",
    "MACD_DEFAULT_FAST",
    "MACD_DEFAULT_SLOW",
    "MACD_DEFAULT_SIGNAL",
    # ADX
    "calculate_adx",
    "ADX_DEFAULT_PERIOD",
    # Stochastic
    "calculate_stochastic",
    "STOCH_DEFAULT_K_PERIOD",
    "STOCH_DEFAULT_D_PERIOD",
    "STOCH_DEFAULT_SMOOTH_K",
]
