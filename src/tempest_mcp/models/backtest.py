"""Backtest result and trade models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

Price: TypeAlias = float
Volume: TypeAlias = float
PnL: TypeAlias = float
Percentage: TypeAlias = float


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass(frozen=True)
class CommissionModel:
    commission_rate: Percentage = 0.001
    slippage_rate: Percentage = 0.0005
    min_commission: Price = 0.0

    def calculate_commission(self, price: Price, volume: Volume) -> Price:
        commission = price * volume * self.commission_rate
        return max(commission, self.min_commission)

    def apply_slippage(self, price: Price, side: OrderSide) -> Price:
        if side == OrderSide.BUY:
            return price * (1 + self.slippage_rate)
        else:
            return price * (1 - self.slippage_rate)


@dataclass(frozen=True)
class BacktestTrade:
    trade_id: str
    symbol: str
    side: OrderSide
    entry_price: Price
    exit_price: Price | None
    volume: Volume
    entry_time: float
    exit_time: float | None
    commission: Price = 0.0
    slippage: Price = 0.0
    pnl: PnL = 0.0
    pnl_percent: Percentage = 0.0


@dataclass(frozen=True)
class Position:
    symbol: str
    side: OrderSide
    entry_price: Price
    volume: Volume
    unrealized_pnl: PnL = 0.0
    margin_used: Price = 0.0


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    symbol: str
    timeframe: str
    start_time: float
    end_time: float
    initial_capital: Price
    final_capital: Price
    total_return: Percentage
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: Percentage
    avg_win: PnL
    avg_loss: PnL
    max_drawdown: Percentage
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    profit_factor: float | None = None
    trades: list[BacktestTrade] = field(default_factory=list)


@dataclass(frozen=True)
class StrategyResult:
    strategy_id: str
    backtest_result: BacktestResult
    metrics: dict[str, float] = field(default_factory=dict)


def calculate_performance_metrics(trades: list[BacktestTrade], returns: list[float], risk_free_rate: float = 0.0) -> dict[str, float]:
    import numpy as np
    if not returns:
        return {"sharpe_ratio": 0.0, "sortino_ratio": 0.0, "max_drawdown": 0.0, "volatility": 0.0}
    returns_array = np.array(returns)
    avg_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    sharpe = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0.0
    downside_returns = returns_array[returns_array < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
    sortino = (avg_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    volatility = std_return * np.sqrt(252) if std_return > 0 else 0.0
    return {"sharpe_ratio": float(sharpe), "sortino_ratio": float(sortino), "max_drawdown": float(abs(max_dd)), "volatility": float(volatility)}
