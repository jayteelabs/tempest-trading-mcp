"""Trend indicator subpackage."""
from tempest_mcp.indicators.trend.ema import (
    EMA_PERIODS,
    calculate_ema,
    calculate_ema_stack,
    death_cross,
    detect_ema_cross,
    golden_cross,
)

__all__ = [
    "EMA_PERIODS",
    "calculate_ema",
    "calculate_ema_stack",
    "death_cross",
    "detect_ema_cross",
    "golden_cross",
]
