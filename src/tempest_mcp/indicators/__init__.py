"""Technical indicator engine."""

from .momentum import (
    calculate_macd_result,
    calculate_rsi,
    calculate_rsi_result,
    calculate_stochastic_result,
    detect_rsi_cross,
    detect_rsi_divergence,
    detect_rsi_extremes,
)
from .session_levels import calculate_session_levels
from .ta_wrapper import (
    calculate_adx,
    calculate_atr,
    calculate_macd,
)
from .trend import (
    calculate_adx_result,
    calculate_ema,
    calculate_ema_result,
    calculate_ema_stack,
    calculate_supertrend,
    death_cross,
    detect_ema_cross,
    golden_cross,
)
from .volatility import (
    calculate_atr_result,
    calculate_bollinger_width,
    calculate_historical_volatility,
)
from .volume import (
    SESSION_ANCHORS,
    calculate_mfi_result,
    calculate_obv_result,
    calculate_vwap,
    calculate_vwap_bands,
    detect_vwap_cross,
)

__all__ = [
    # Trend indicators
    "calculate_ema",
    "calculate_ema_result",
    "calculate_ema_stack",
    "calculate_supertrend",
    "detect_ema_cross",
    "golden_cross",
    "death_cross",
    # Momentum indicators (RSI engine - pure pandas)
    "calculate_rsi",
    "detect_rsi_extremes",
    "detect_rsi_divergence",
    "detect_rsi_cross",
    # Momentum indicators (result wrappers - ta-lib)
    "calculate_rsi_result",
    "calculate_macd_result",
    "calculate_stochastic_result",
    # Volume indicators (VWAP engine - pure pandas)
    "calculate_vwap",
    "calculate_vwap_bands",
    "detect_vwap_cross",
    "SESSION_ANCHORS",
    # Volume indicators (result wrappers - ta-lib)
    "calculate_obv_result",
    "calculate_mfi_result",
    # Other indicators
    "calculate_macd",
    "calculate_atr",
    "calculate_adx",
    "calculate_adx_result",
    "calculate_atr_result",
    "calculate_bollinger_width",
    "calculate_historical_volatility",
    "calculate_session_levels",
]
