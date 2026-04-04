"""Momentum indicators: RSI, MACD, Stochastic, CCI, Williams %R, ROC."""
import numpy as np
import talib

from tempest_mcp.models.indicator import (
    CCIResult,
    MACDResult,
    ROCResult,
    RSIResult,
    StochasticResult,
    WilliamsRResult,
)


def calculate_rsi_result(close, period: int = 14) -> RSIResult:
    close_arr = np.array(close, dtype=np.float64)
    rsi = talib.RSI(close_arr, timeperiod=period)
    valid_rsi = rsi[~np.isnan(rsi)]
    latest_rsi = float(valid_rsi[-1]) if len(valid_rsi) > 0 else 50.0
    return RSIResult(symbol="", timeframe="", timestamp=0.0, values={"rsi": latest_rsi, "overbought": latest_rsi > 70, "oversold": latest_rsi < 30})

def calculate_macd_result(close, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> MACDResult:
    close_arr = np.array(close, dtype=np.float64)
    macd, signal, hist = talib.MACD(close_arr, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    valid_idx = ~np.isnan(macd)
    latest_macd = float(macd[valid_idx][-1]) if np.any(valid_idx) else 0.0
    latest_signal = float(signal[valid_idx][-1]) if np.any(valid_idx) else 0.0
    latest_hist = float(hist[valid_idx][-1]) if np.any(valid_idx) else 0.0
    trend = "bullish" if latest_hist > 0 else "bearish"
    return MACDResult(symbol="", timeframe="", timestamp=0.0, values={"macd": latest_macd, "signal": latest_signal, "histogram": latest_hist, "trend": trend})

def calculate_stochastic_result(high, low, close, k_period: int = 14, d_period: int = 3) -> StochasticResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, fastk_period=k_period, slowk_period=d_period, slowk_matype=0, slowd_period=d_period, slowd_matype=0)
    valid_idx = ~np.isnan(slowk)
    latest_k = float(slowk[valid_idx][-1]) if np.any(valid_idx) else 50.0
    latest_d = float(slowd[valid_idx][-1]) if np.any(valid_idx) else 50.0
    return StochasticResult(symbol="", timeframe="", timestamp=0.0, values={"k": latest_k, "d": latest_d, "overbought": latest_k > 80, "oversold": latest_k < 20})

def calculate_cci_result(high, low, close, period: int = 20) -> CCIResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    cci = talib.CCI(high_arr, low_arr, close_arr, timeperiod=period)
    valid_cci = cci[~np.isnan(cci)]
    latest_cci = float(valid_cci[-1]) if len(valid_cci) > 0 else 0.0
    return CCIResult(symbol="", timeframe="", timestamp=0.0, values={"cci": latest_cci, "overbought": latest_cci > 100, "oversold": latest_cci < -100})

def calculate_williams_r_result(high, low, close, period: int = 14) -> WilliamsRResult:
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    close_arr = np.array(close, dtype=np.float64)
    wr = talib.WILLR(high_arr, low_arr, close_arr, timeperiod=period)
    valid_wr = wr[~np.isnan(wr)]
    latest_wr = float(valid_wr[-1]) if len(valid_wr) > 0 else -50.0
    return WilliamsRResult(symbol="", timeframe="", timestamp=0.0, values={"williams_r": latest_wr, "overbought": latest_wr > -20, "oversold": latest_wr < -80})

def calculate_roc_result(close, period: int = 12) -> ROCResult:
    close_arr = np.array(close, dtype=np.float64)
    roc = talib.ROC(close_arr, timeperiod=period)
    valid_roc = roc[~np.isnan(roc)]
    latest_roc = float(valid_roc[-1]) if len(valid_roc) > 0 else 0.0
    return ROCResult(symbol="", timeframe="", timestamp=0.0, values={"roc": latest_roc, "momentum": "bullish" if latest_roc > 0 else "bearish"})
