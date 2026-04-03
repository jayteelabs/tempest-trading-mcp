"""ta-lib wrapper for high-performance indicator calculations."""
import numpy as np
import talib
from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger
logger = get_logger(__name__)

class IndicatorError(Exception):
    def __init__(self, message: str, code: int = ErrorCodes.INDICATOR_ERROR):
        super().__init__(message)
        self.code = code

def _validate_input(data, name: str, min_length: int = 1) -> np.ndarray:
    if data is None:
        raise IndicatorError(f"{name} cannot be None", code=ErrorCodes.MISSING_PARAMETER)
    arr = np.array(data, dtype=np.float64)
    if len(arr) < min_length:
        raise IndicatorError(f"{name} requires at least {min_length} values, got {len(arr)}", code=ErrorCodes.INSUFFICIENT_DATA)
    return arr

def calculate_ema(close, period: int = 20) -> np.ndarray:
    close_arr = _validate_input(close, "close", min_length=period)
    return talib.EMA(close_arr, timeperiod=period)

def calculate_sma(close, period: int = 20) -> np.ndarray:
    close_arr = _validate_input(close, "close", min_length=period)
    return talib.SMA(close_arr, timeperiod=period)

def calculate_rsi(close, period: int = 14) -> np.ndarray:
    close_arr = _validate_input(close, "close", min_length=period + 1)
    return talib.RSI(close_arr, timeperiod=period)

def calculate_macd(close, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    close_arr = _validate_input(close, "close", min_length=slow_period + signal_period)
    return talib.MACD(close_arr, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)

def calculate_atr(high, low, close, period: int = 14) -> np.ndarray:
    high_arr = _validate_input(high, "high", min_length=period + 1)
    low_arr = _validate_input(low, "low", min_length=period + 1)
    close_arr = _validate_input(close, "close", min_length=period + 1)
    return talib.ATR(high_arr, low_arr, close_arr, timeperiod=period)

def calculate_adx(high, low, close, period: int = 14):
    high_arr = _validate_input(high, "high", min_length=period * 2)
    low_arr = _validate_input(low, "low", min_length=period * 2)
    close_arr = _validate_input(close, "close", min_length=period * 2)
    adx = talib.ADX(high_arr, low_arr, close_arr, timeperiod=period)
    plus_di = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=period)
    minus_di = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=period)
    return adx, plus_di, minus_di
