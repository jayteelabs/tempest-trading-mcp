"""Momentum indicators subpackage.

Provides RSI, MACD, Stochastic, CCI, Williams %R, and ROC calculations.
"""

from tempest_mcp.indicators.momentum.rsi import (
    CENTERLINE,
    OVERBOUGHT_THRESHOLD,
    OVERSOLD_THRESHOLD,
    RSI_DEFAULT_PERIOD,
    calculate_rsi,
    detect_rsi_cross,
    detect_rsi_divergence,
    detect_rsi_extremes,
)

# Optional ta-lib based result wrappers
try:
    import numpy as np
    import talib

    from tempest_mcp.models.indicator import (
        MACDResult,
        RSIResult,
        StochasticResult,
    )

    def calculate_rsi_result(close, period: int = 14) -> "RSIResult":
        """Calculate RSI result wrapper using ta-lib."""
        close_arr = np.array(close, dtype=np.float64)
        rsi = talib.RSI(close_arr, timeperiod=period)
        valid_rsi = rsi[~np.isnan(rsi)]
        latest_rsi = float(valid_rsi[-1]) if len(valid_rsi) > 0 else 50.0
        return RSIResult(
            symbol="",
            timeframe="",
            timestamp=0.0,
            values={"rsi": latest_rsi, "overbought": latest_rsi > 70, "oversold": latest_rsi < 30},
        )

    def calculate_macd_result(
        close, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
    ) -> "MACDResult":
        """Calculate MACD result wrapper using ta-lib."""
        close_arr = np.array(close, dtype=np.float64)
        macd, signal, hist = talib.MACD(
            close_arr, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period
        )
        valid_idx = ~np.isnan(macd)
        latest_macd = float(macd[valid_idx][-1]) if np.any(valid_idx) else 0.0
        latest_signal = float(signal[valid_idx][-1]) if np.any(valid_idx) else 0.0
        latest_hist = float(hist[valid_idx][-1]) if np.any(valid_idx) else 0.0
        trend = "bullish" if latest_hist > 0 else "bearish"
        return MACDResult(
            symbol="",
            timeframe="",
            timestamp=0.0,
            values={
                "macd": latest_macd,
                "signal": latest_signal,
                "histogram": latest_hist,
                "trend": trend,
            },
        )

    def calculate_stochastic_result(
        high, low, close, k_period: int = 14, d_period: int = 3
    ) -> "StochasticResult":
        """Calculate Stochastic result wrapper using ta-lib."""
        high_arr = np.array(high, dtype=np.float64)
        low_arr = np.array(low, dtype=np.float64)
        close_arr = np.array(close, dtype=np.float64)
        slowk, slowd = talib.STOCH(
            high_arr,
            low_arr,
            close_arr,
            fastk_period=k_period,
            slowk_period=d_period,
            slowk_matype=0,
            slowd_period=d_period,
            slowd_matype=0,
        )
        valid_idx = ~np.isnan(slowk)
        latest_k = float(slowk[valid_idx][-1]) if np.any(valid_idx) else 50.0
        latest_d = float(slowd[valid_idx][-1]) if np.any(valid_idx) else 50.0
        return StochasticResult(
            symbol="",
            timeframe="",
            timestamp=0.0,
            values={
                "k": latest_k,
                "d": latest_d,
                "overbought": latest_k > 80,
                "oversold": latest_k < 20,
            },
        )

    _HAS_TALIB = True

except ImportError:
    _HAS_TALIB = False

    # Stub functions that raise ImportError when called
    def calculate_rsi_result(*args, **kwargs):
        raise ImportError("ta-lib not available - install with: pip install ta-lib")

    def calculate_macd_result(*args, **kwargs):
        raise ImportError("ta-lib not available - install with: pip install ta-lib")

    def calculate_stochastic_result(*args, **kwargs):
        raise ImportError("ta-lib not available - install with: pip install ta-lib")


__all__ = [
    # RSI engine functions (pure pandas, always available)
    "calculate_rsi",
    "detect_rsi_extremes",
    "detect_rsi_divergence",
    "detect_rsi_cross",
    # Result wrappers (ta-lib based, optional)
    "calculate_rsi_result",
    "calculate_macd_result",
    "calculate_stochastic_result",
    # Constants
    "RSI_DEFAULT_PERIOD",
    "OVERSOLD_THRESHOLD",
    "OVERBOUGHT_THRESHOLD",
    "CENTERLINE",
]
