"""Volatility indicators subpackage.

Provides ATR, Historical Volatility, Bollinger Width calculations.
Pure-pandas implementations plus ta-lib result wrappers.
"""

import pandas as pd

from tempest_mcp.indicators.volatility.atr import (
    ATR_DEFAULT_PERIOD,
    calculate_atr,
)
from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)


def calculate_bollinger_width(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """Calculate Bollinger Width using pure pandas.

    Bollinger Width = (Upper Band - Lower Band) / Middle Band
    This is a dimensionless, normalized measure of volatility.

    Formula:
        Middle Band = period SMA of prices
        Upper Band = Middle Band + (std_dev × population std dev of prices over period)
        Lower Band = Middle Band - (std_dev × population std dev of prices over period)
        Width = (Upper Band - Lower Band) / Middle Band

    Args:
        prices: Series of price values with UTC-aware DatetimeIndex.
        period: Number of periods for SMA and std dev calculation (default 20).
        std_dev: Number of standard deviations for band calculation (default 2.0).

    Returns:
        pd.Series containing Bollinger Width values with UTC-aware index.
        Returns empty Series if input length < period.

    Raises:
        ValueError: If period <= 0 or std_dev <= 0.

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105], index=pd.date_range('2024-01-01', periods=5, tz='UTC'))
        >>> bw = calculate_bollinger_width(prices, period=20)
    """
    if period <= 0:
        raise ValueError("Period must be a positive integer")

    if std_dev <= 0:
        raise ValueError("std_dev must be a positive number")

    if len(prices) == 0:
        return pd.Series(dtype=float)

    if len(prices) < period:
        logger.debug(
            "Insufficient data for Bollinger Width(%d): %d < %d",
            period,
            len(prices),
            period,
        )
        return pd.Series(dtype=float)

    # Ensure UTC-aware index
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize("UTC")

    # Drop NaN values
    prices = prices.dropna()

    if len(prices) < period:
        return pd.Series(dtype=float)

    # Calculate middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Calculate population standard deviation (ddof=0)
    rolling_std = prices.rolling(window=period).std(ddof=0)

    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * rolling_std)
    lower_band = middle_band - (std_dev * rolling_std)

    # Calculate Bollinger Width: (Upper - Lower) / Middle
    # Handle division by zero where middle_band is 0
    width = (upper_band - lower_band) / middle_band

    # Replace inf/-inf with NaN, then drop NaN at the start (where we don't have enough data)
    width = width.replace([float("inf"), float("-inf")], float("nan"))
    width = width.dropna()

    return width


def calculate_historical_volatility(
    prices: pd.Series,
    period: int = 252,
    annualize: bool = True,
) -> pd.Series:
    """Calculate Historical Volatility using pure pandas.

    Historical Volatility measures the degree of variation in asset prices
    over a given period, based on log returns.

    Formula:
        Log Return[t] = ln(price[t] / price[t-1])
        Std Dev of log returns over period (population std dev, ddof=0)
        If annualize=True: multiply by sqrt(252)

    Args:
        prices: Series of price values with UTC-aware DatetimeIndex.
        period: Number of periods for std dev calculation (default 252).
        annualize: If True, multiply by sqrt(252) for annualization (default True).

    Returns:
        pd.Series containing Historical Volatility values with UTC-aware index.
        Returns empty Series if period < 2 (intentional floor) or if input length < period + 1.

    Raises:
        ValueError: If period < 2.

    Example:
        >>> prices = pd.Series([100, 102, 101, 103, 105], index=pd.date_range('2024-01-01', periods=5, tz='UTC'))
        >>> hv = calculate_historical_volatility(prices, period=20)
    """
    import numpy as np

    if period < 2:
        raise ValueError("Period must be at least 2")

    if len(prices) == 0:
        return pd.Series(dtype=float)

    # Need period + 1 prices to calculate period log returns
    if len(prices) < period + 1:
        logger.debug(
            "Insufficient data for Historical Volatility(%d): %d < %d + 1",
            period,
            len(prices),
            period,
        )
        return pd.Series(dtype=float)

    # Ensure UTC-aware index
    if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is None:
        prices = prices.copy()
        prices.index = prices.index.tz_localize("UTC")

    # Drop NaN values
    prices = prices.dropna()

    if len(prices) < period + 1:
        return pd.Series(dtype=float)

    # Calculate log returns: ln(price[t] / price[t-1])
    log_returns = pd.Series(
        data=np.log(prices.values[1:] / prices.values[:-1]),
        index=prices.index[1:],
    )

    # Calculate rolling std dev of log returns (population std dev, ddof=0)
    hv = log_returns.rolling(window=period).std(ddof=0)

    # Annualize if requested
    if annualize:
        hv = hv * (252**0.5)

    # Drop NaN values (first period-1 values will be NaN)
    hv = hv.dropna()

    return hv


# Pure-pandas Historical Volatility (no ta-lib dependency)
def calculate_historical_volatility_pure(
    close, period: int = 20, trading_periods: int = 252
) -> dict[str, float]:
    """Calculate Historical Volatility using pure pandas/numpy."""
    import numpy as np
    import pandas as pd

    close_arr = np.array(close, dtype=np.float64)
    log_returns = np.log(close_arr[1:] / close_arr[:-1])
    log_series = pd.Series(log_returns)
    hv = log_series.rolling(period).std() * np.sqrt(trading_periods)
    valid_hv = hv.dropna()
    latest_hv = float(valid_hv.iloc[-1]) if len(valid_hv) > 0 else 0.0
    return {
        "hv": latest_hv,
        "hv_percent": latest_hv * 100,
        "percentile": 50.0,
    }


# Ta-lib based result wrappers (optional)
try:
    import numpy as np
    import talib

    from tempest_mcp.models.indicator import (
        ATRResult,
        BollingerWidthResult,
        HistoricalVolatilityResult,
    )

    def calculate_atr_result(high, low, close, period: int = 14) -> "ATRResult":
        """Calculate ATR result wrapper using ta-lib."""
        high_arr = np.array(high, dtype=np.float64)
        low_arr = np.array(low, dtype=np.float64)
        close_arr = np.array(close, dtype=np.float64)
        atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
        valid_atr = atr[~np.isnan(atr)]
        latest_atr = float(valid_atr[-1]) if len(valid_atr) > 0 else 0.0
        latest_close = float(close_arr[-1])
        atr_percent = (latest_atr / latest_close * 100) if latest_close > 0 else 0.0
        return ATRResult(
            symbol="",
            timeframe="",
            timestamp=0.0,
            values={"atr": latest_atr, "atr_percent": atr_percent},
        )

    def calculate_bollinger_width_result(
        close, period: int = 20, std_dev: float = 2.0
    ) -> "BollingerWidthResult":
        """Calculate Bollinger Width result wrapper using ta-lib."""
        close_arr = np.array(close, dtype=np.float64)
        upper, middle, lower = talib.BBANDS(
            close_arr, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
        )
        valid_idx = ~np.isnan(upper)
        latest_upper = float(upper[valid_idx][-1]) if np.any(valid_idx) else 0.0
        latest_middle = float(middle[valid_idx][-1]) if np.any(valid_idx) else 0.0
        latest_lower = float(lower[valid_idx][-1]) if np.any(valid_idx) else 0.0
        width = latest_upper - latest_lower
        width_percent = (width / latest_middle * 100) if latest_middle > 0 else 0.0
        return BollingerWidthResult(
            symbol="",
            timeframe="",
            timestamp=0.0,
            values={
                "upper": latest_upper,
                "middle": latest_middle,
                "lower": latest_lower,
                "width": width,
                "width_percent": width_percent,
            },
        )

    def calculate_historical_volatility_result(
        close, period: int = 20, trading_periods: int = 252
    ) -> "HistoricalVolatilityResult":
        """Calculate Historical Volatility result wrapper using ta-lib."""
        close_arr = np.array(close, dtype=np.float64)
        log_returns = np.log(close_arr[1:] / close_arr[:-1])
        # Use ta-lib STDDEV for consistency with other ta-lib wrappers
        stddev = talib.STDDEV(log_returns, timeperiod=period)
        hv = stddev * np.sqrt(trading_periods)
        valid_hv = hv[~np.isnan(hv)]
        latest_hv = float(valid_hv[-1]) if len(valid_hv) > 0 else 0.0
        return HistoricalVolatilityResult(
            symbol="",
            timeframe="",
            timestamp=0.0,
            values={"hv": latest_hv, "hv_percent": latest_hv * 100, "percentile": 50.0},
        )

    _HAS_TALIB = True

except ImportError:
    _HAS_TALIB = False

    # Stub functions that raise ImportError when called
    def calculate_atr_result(*args, **kwargs):
        raise ImportError("ta-lib not available - install with: pip install ta-lib")

    def calculate_bollinger_width_result(*args, **kwargs):
        raise ImportError("ta-lib not available - install with: pip install ta-lib")

    def calculate_historical_volatility_result(*args, **kwargs):
        raise ImportError("ta-lib not available - install with: pip install ta-lib")


__all__ = [
    # Pure-pandas ATR engine (always available)
    "calculate_atr",
    "ATR_DEFAULT_PERIOD",
    # Pure-pandas volatility indicators (always available)
    "calculate_bollinger_width",
    "calculate_historical_volatility",
    # Result wrappers (ta-lib based, optional)
    "calculate_atr_result",
    "calculate_bollinger_width_result",
    "calculate_historical_volatility_result",
]
