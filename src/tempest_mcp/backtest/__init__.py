"""Backtesting engine."""

from tempest_mcp.backtest.commission import (
    CommissionModel,
    apply_slippage,
    calculate_commission,
    calculate_net_pnl,
)
from tempest_mcp.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    PositionDirection,
    SignalAction,
    Trade,
)
from tempest_mcp.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardWindowResult,
    run_walk_forward,
)

__all__ = [
    "BacktestEngine",
    "Trade",
    "BacktestResult",
    "CommissionModel",
    "calculate_commission",
    "apply_slippage",
    "calculate_net_pnl",
    "SignalAction",
    "PositionDirection",
    # Walk-forward engine
    "WalkForwardConfig",
    "WalkForwardResult",
    "WalkForwardSummary",
    "WalkForwardWindowResult",
    "run_walk_forward",
]
