"""Structure indicators: Fibonacci, Pivot Points."""
import numpy as np
from tempest_mcp.config import ErrorCodes
from tempest_mcp.indicators.ta_wrapper import IndicatorError

def calculate_fibonacci_levels(high, low, trend: str = "up"):
    high_arr = np.array(high, dtype=np.float64)
    low_arr = np.array(low, dtype=np.float64)
    swing_high = float(np.max(high_arr))
    swing_low = float(np.min(low_arr))
    diff = swing_high - swing_low
    if trend.lower() == "up":
        return {"swing_high": swing_high, "swing_low": swing_low, "fib_382": swing_low + diff * 0.382, "fib_500": swing_low + diff * 0.500, "fib_618": swing_low + diff * 0.618}
    return {"swing_high": swing_high, "swing_low": swing_low, "fib_382": swing_high - diff * 0.382, "fib_500": swing_high - diff * 0.500, "fib_618": swing_high - diff * 0.618}

def calculate_pivot_points(high, low, close, method: str = "standard"):
    h, l, c = float(high[-1]), float(low[-1]), float(close[-1])
    pp = (h + l + c) / 3
    return {"pivot": pp, "r1": 2 * pp - l, "s1": 2 * pp - h, "r2": pp + (h - l), "s2": pp - (h - l)}
