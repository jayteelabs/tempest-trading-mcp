"""Volatility indicators: ATR, Historical Volatility, Bollinger Width."""
import numpy as np
import talib
from tempest_mcp.config import ErrorCodes
from tempest_mcp.models.indicator import ATRResult, BollingerWidthResult, HistoricalVolatilityResult

def calculate_atr_result(high, low, close, period: int = 14) -> ATRResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)
    valid_atr = atr[~np.isnan(atr)]
    latest_atr = float(valid_atr[-1]) if len(valid_atr) > 0 else 0.0
    latest_close = float(close_arr[-1])
    atr_percent = (latest_atr / latest_close * 100) if latest_close > 0 else 0.0
    return ATRResult(symbol="", timeframe="", timestamp=0.0, values={"atr": latest_atr, "atr_percent": atr_percent})

def calculate_bollinger_width(close, period: int = 20, std_dev: float = 2.0) -> BollingerWidthResult:
    close_arr = np.array(close, dtype=np.float64)
    upper, middle, lower = talib.BBANDS(close_arr, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0)
    valid_idx = ~np.isnan(upper)
    latest_upper = float(upper[valid_idx][-1]) if np.any(valid_idx) else 0.0
    latest_middle = float(middle[valid_idx][-1]) if np.any(valid_idx) else 0.0
    latest_lower = float(lower[valid_idx][-1]) if np.any(valid_idx) else 0.0
    width = latest_upper - latest_lower
    width_percent = (width / latest_middle * 100) if latest_middle > 0 else 0.0
    return BollingerWidthResult(symbol="", timeframe="", timestamp=0.0, values={"upper": latest_upper, "middle": latest_middle, "lower": latest_lower, "width": width, "width_percent": width_percent})

def calculate_historical_volatility(close, period: int = 20, trading_periods: int = 252) -> HistoricalVolatilityResult:
    close_arr = np.array(close, dtype=np.float64)
    log_returns = np.log(close_arr[1:] / close_arr[:-1])
    hv_series = np.zeros(len(log_returns))
    hv_series[:] = np.nan
    for i in range(period - 1, len(log_returns)):
        window = log_returns[i - period + 1 : i + 1]
        hv_series[i] = np.std(window) * np.sqrt(trading_periods)
    valid_hv = hv_series[~np.isnan(hv_series)]
    latest_hv = float(valid_hv[-1]) if len(valid_hv) > 0 else 0.0
    return HistoricalVolatilityResult(symbol="", timeframe="", timestamp=0.0, values={"hv": latest_hv, "hv_percent": latest_hv * 100, "percentile": 50.0})
