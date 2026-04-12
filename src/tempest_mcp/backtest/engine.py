"""Backtesting engine — ENG-16 spec implementation, extended for ENG-58 bidirectional support."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from tempest_mcp.backtest.commission import apply_slippage


# ---------------------------------------------------------------------------
# Enums for bidirectional signal and position state (ENG-58)
# ---------------------------------------------------------------------------


class SignalAction(Enum):
    """Bidirectional signal contract — replaces int sign semantics.

    Each variant is unambiguous about intent:
    - LONG_ENTRY  — open / extend a long position
    - LONG_EXIT   — close an existing long position
    - SHORT_ENTRY — open / extend a short position
    - SHORT_EXIT  — close an existing short position
    - HOLD        — no action
    """
    LONG_ENTRY = "long_entry"
    LONG_EXIT = "long_exit"
    SHORT_ENTRY = "short_entry"
    SHORT_EXIT = "short_exit"
    HOLD = "hold"

    @classmethod
    def from_int(cls, value: int) -> "SignalAction":
        """Convert legacy int signal to SignalAction (backwards compatibility)."""
        mapping = {
            1: cls.LONG_ENTRY,
            -1: cls.LONG_EXIT,
            0: cls.HOLD,
        }
        if value not in mapping:
            raise ValueError(f"Invalid signal int value: {value}. Use SignalAction enum directly.")
        return mapping[value]


class PositionDirection(Enum):
    """Internal position state — distinct from trade side.

    Enforces FLAT as mandatory intermediate state for directional flips.
    """
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


# ---------------------------------------------------------------------------
# Dataclasses per ENG-16 spec (do NOT import from models.backtest)
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Represents a single closed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    direction: PositionDirection  # long or short
    gross_pnl: float            # direction-aware PnL at exit
    net_pnl: float              # after commission + slippage
    commission: float           # total commission paid (both sides)
    slippage_cost: float
    bars_held: int              # number of bars between entry and exit


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    trades: list[Trade]           # closed trades only
    equity_curve: pd.Series        # index=timestamp, values=equity
    metrics: dict[str, float]      # all computed metrics
    open_position: bool            # True if last position is unclosed
    initial_capital: float
    final_equity: float


# ---------------------------------------------------------------------------
# BacktestEngine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Bidirectional backtest engine with commission and slippage (ENG-58)."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_bps: float = 5.0,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_bps = slippage_bps
        self._cash = initial_capital
        # Internal position state: {entry_price, size, entry_time, entry_idx, direction}
        self._position: dict[str, Any] | None = None
        self._position_direction: PositionDirection = PositionDirection.FLAT
        self._trades: list[Trade] = []
        self._equity_curve: list[float] = []
        self._has_open_position = False

    def run(self, ohlcv_df: pd.DataFrame, signals: pd.Series) -> BacktestResult:
        """Run backtest on OHLCV data with entry/exit signals.

        Signals fire at close, order executes at next bar open (no lookahead).
        Supports both long and short positions. FLAT is required as intermediate
        state for any directional flip (LONG->SHORT or SHORT->LONG must go through FLAT).
        """
        if len(ohlcv_df) < 2:
            raise ValueError(f"Backtest requires at least 2 rows, got {len(ohlcv_df)}")

        # Normalize signals to SignalAction enum
        normalized = self._normalize_signals(signals, ohlcv_df.index)

        self._cash = self.initial_capital
        self._position = None
        self._position_direction = PositionDirection.FLAT
        self._trades = []
        self._equity_curve = []
        self._has_open_position = False

        # Process bars from index 1 onwards (first bar signal is ignored — no prior bar for execution)
        for i in range(1, len(ohlcv_df)):
            prev_idx = i - 1
            curr_idx = i

            # Execute on next bar open if prior bar had signal
            prev_signal = normalized.iloc[prev_idx]
            self._process_signal(prev_signal, ohlcv_df, prev_idx)

            # Record equity after potential trade execution
            equity = self._calculate_equity(ohlcv_df, curr_idx)
            self._equity_curve.append(equity)

        # Check if position is still open at end
        if self._position is not None:
            self._has_open_position = True

        # Build equity curve series
        equity_series = pd.Series(self._equity_curve, index=ohlcv_df.index[1:])

        # Compute metrics
        metrics = self._compute_metrics()

        return BacktestResult(
            trades=self._trades,
            equity_curve=equity_series,
            metrics=metrics,
            open_position=self._has_open_position,
            initial_capital=self.initial_capital,
            final_equity=self._equity_curve[-1] if self._equity_curve else self.initial_capital,
        )

    def _normalize_signals(self, signals: pd.Series, index: pd.Index) -> pd.Series:
        """Normalize integer or SignalAction signals to SignalAction enum series."""
        normalized = signals.reindex(index).fillna(SignalAction.HOLD.value if signals.dtype == object else 0)
        if normalized.dtype == object:
            # Already SignalAction or string values
            return normalized.apply(
                lambda x: x if isinstance(x, SignalAction) else SignalAction(x)
            )
        else:
            # Legacy int signals
            return normalized.apply(
                lambda x: SignalAction.from_int(int(x))
            )

    def _process_signal(self, signal: SignalAction, df: pd.DataFrame, idx: int) -> None:
        """Process a single signal against current position state."""
        if signal == SignalAction.HOLD:
            return

        if signal == SignalAction.LONG_ENTRY:
            if self._position_direction == PositionDirection.LONG:
                # Already in long position — no-op
                return
            if self._position_direction != PositionDirection.FLAT:
                raise ValueError(
                    f"Invalid transition: cannot LONG_ENTRY when position is {self._position_direction.value}. "
                    f"Must close position first (transition through FLAT)."
                )
            self._open_position(df, idx, PositionDirection.LONG)

        elif signal == SignalAction.SHORT_ENTRY:
            if self._position_direction == PositionDirection.SHORT:
                # Already in short position — no-op
                return
            if self._position_direction != PositionDirection.FLAT:
                raise ValueError(
                    f"Invalid transition: cannot SHORT_ENTRY when position is {self._position_direction.value}. "
                    f"Must close position first (transition through FLAT)."
                )
            self._open_position(df, idx, PositionDirection.SHORT)

        elif signal == SignalAction.LONG_EXIT:
            if self._position_direction == PositionDirection.FLAT:
                # Already flat — no-op
                return
            if self._position_direction != PositionDirection.LONG:
                raise ValueError(
                    f"Invalid transition: cannot LONG_EXIT when position is {self._position_direction.value}. "
                    f"No long position to close."
                )
            self._close_position(df, idx)

        elif signal == SignalAction.SHORT_EXIT:
            if self._position_direction == PositionDirection.FLAT:
                # Already flat — no-op
                return
            if self._position_direction != PositionDirection.SHORT:
                raise ValueError(
                    f"Invalid transition: cannot SHORT_EXIT when position is {self._position_direction.value}. "
                    f"No short position to close."
                )
            self._close_position(df, idx)

    def _open_position(self, df: pd.DataFrame, idx: int, direction: PositionDirection) -> None:
        """Open a position at the next bar open after signal at idx."""
        if self._position is not None:
            return  # Already have a position (should not happen if state transitions are correct)

        entry_price = df["open"].iloc[idx + 1]  # execute at next bar open

        # Determine slippage direction: LONG_ENTRY pays slippage (buy), SHORT_ENTRY receives slippage (sell)
        slippage_direction = 1 if direction == PositionDirection.LONG else -1
        entry_price = apply_slippage(entry_price, 1.0, slippage_direction, self.slippage_bps)

        # Size = round(cash / entry_price, 8)
        size = round(self._cash / entry_price, 8)
        if size <= 0:
            return

        commission = entry_price * size * self.commission_pct
        self._cash -= commission

        self._position = {
            "entry_price": entry_price,
            "size": size,
            "entry_time": df.index[idx + 1],
            "entry_idx": idx + 1,
            "commission": commission,
            "direction": direction,
        }
        self._position_direction = direction

    def _close_position(self, df: pd.DataFrame, idx: int) -> None:
        """Close the current position at the next bar open after signal at idx."""
        if self._position is None:
            return

        direction = self._position["direction"]
        exit_price = df["open"].iloc[idx + 1]  # execute at next bar open

        # Slippage: for closing a long we sell (direction=-1), for closing a short we buy (direction=1)
        slippage_direction = -1 if direction == PositionDirection.LONG else 1
        exit_price = apply_slippage(exit_price, 1.0, slippage_direction, self.slippage_bps)

        size = self._position["size"]
        entry_price = self._position["entry_price"]

        # Direction-aware gross PnL:
        # Long:  (exit_price - entry_price) * size  — profit when price rises
        # Short: (entry_price - exit_price) * size  — profit when price falls
        if direction == PositionDirection.LONG:
            gross_pnl = (exit_price - entry_price) * size
        else:  # SHORT
            gross_pnl = (entry_price - exit_price) * size

        exit_commission = exit_price * size * self.commission_pct
        total_commission = self._position["commission"] + exit_commission

        # Slippage cost approximation (symmetric for both directions)
        mid_price = (entry_price + exit_price) / 2
        slippage_cost = size * (self.slippage_bps / 10000) * mid_price

        net_pnl = gross_pnl - total_commission - slippage_cost
        self._cash += net_pnl

        bars_held = (idx + 1) - self._position["entry_idx"]

        trade = Trade(
            entry_time=self._position["entry_time"],
            exit_time=df.index[idx + 1],
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            direction=direction,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            commission=total_commission,
            slippage_cost=slippage_cost,
            bars_held=bars_held,
        )
        self._trades.append(trade)
        self._position = None
        self._position_direction = PositionDirection.FLAT

    def _calculate_equity(self, df: pd.DataFrame, idx: int) -> float:
        """Calculate current equity (cash + unrealized PnL)."""
        equity = self._cash
        if self._position is not None:
            current_price = df["close"].iloc[idx]
            direction = self._position["direction"]
            entry_price = self._position["entry_price"]
            size = self._position["size"]
            # Direction-aware unrealized PnL:
            # Long:  (current - entry) * size
            # Short: (entry - current) * size
            if direction == PositionDirection.LONG:
                unrealized = (current_price - entry_price) * size
            else:  # SHORT
                unrealized = (entry_price - current_price) * size
            equity += unrealized
        return equity

    def _compute_metrics(self) -> dict[str, float]:
        """Compute backtest performance metrics."""
        metrics: dict[str, float] = {}

        # Basic counts
        total_trades = len(self._trades)
        metrics["total_trades"] = total_trades

        # Return metrics
        final_equity = self._equity_curve[-1] if self._equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        metrics["total_return"] = total_return

        # Win rate
        if total_trades > 0:
            wins = sum(1 for t in self._trades if t.net_pnl > 0)
            losses = sum(1 for t in self._trades if t.net_pnl <= 0)
            win_rate = wins / total_trades
            metrics["win_rate"] = win_rate
            metrics["wins"] = wins
            metrics["losses"] = losses

            # Average win/loss
            win_values = [t.net_pnl for t in self._trades if t.net_pnl > 0]
            loss_values = [t.net_pnl for t in self._trades if t.net_pnl <= 0]
            avg_win = np.mean(win_values) if win_values else 0.0
            avg_loss = abs(np.mean(loss_values)) if loss_values else 0.0
            metrics["avg_win"] = avg_win
            metrics["avg_loss"] = avg_loss

            # Profit factor
            gross_profit = sum(t.net_pnl for t in self._trades if t.net_pnl > 0)
            gross_loss = abs(sum(t.net_pnl for t in self._trades if t.net_pnl <= 0))
            if gross_loss == 0:
                profit_factor = float("inf") if gross_profit > 0 else 0.0
            else:
                profit_factor = gross_profit / gross_loss
            metrics["profit_factor"] = profit_factor

            # Expectancy
            loss_rate = 1 - win_rate
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            metrics["expectancy"] = expectancy
        else:
            metrics["win_rate"] = 0.0
            metrics["wins"] = 0
            metrics["losses"] = 0
            metrics["avg_win"] = 0.0
            metrics["avg_loss"] = 0.0
            metrics["profit_factor"] = 0.0
            metrics["expectancy"] = 0.0

        # Max drawdown
        equity_arr = np.array(self._equity_curve) if self._equity_curve else np.array([self.initial_capital])
        running_max = np.maximum.accumulate(equity_arr)
        drawdowns = (equity_arr - running_max) / running_max
        max_drawdown = float(abs(np.min(drawdowns))) if len(drawdowns) > 0 else 0.0
        metrics["max_drawdown"] = max_drawdown

        # Sharpe ratio (daily returns, annualized)
        if len(self._equity_curve) > 1:
            returns = np.diff(self._equity_curve) / self._equity_curve[:-1]
            mean_ret = np.mean(returns)
            std_ret = np.std(returns)
            if std_ret > 0:
                sharpe = np.sqrt(252) * mean_ret / std_ret
            else:
                sharpe = 0.0
            metrics["sharpe_ratio"] = float(sharpe)
        else:
            metrics["sharpe_ratio"] = 0.0

        return metrics
