"""Backtesting engine."""
from dataclasses import dataclass, field
import uuid
import numpy as np
from tempest_mcp.backtest.commission import CommissionModel, create_binance_model
from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.backtest import BacktestResult, BacktestTrade, OrderSide, Position, calculate_performance_metrics
from tempest_mcp.models.market import Kline

logger = get_logger(__name__)

class BacktestError(Exception):
    def __init__(self, message: str, code: int = ErrorCodes.INDICATOR_ERROR):
        super().__init__(message)
        self.code = code

@dataclass
class BacktestEngine:
    initial_capital: float = 10000.0
    commission_model: CommissionModel = field(default_factory=create_binance_model)
    leverage: float = 1.0
    _capital: float = field(default=0.0, init=False)
    _position: Position | None = field(default=None, init=False)
    _trades: list[BacktestTrade] = field(default_factory=list, init=False)
    _equity_curve: list[float] = field(default_factory=list, init=False)
    _returns: list[float] = field(default_factory=list, init=False)

    def __post_init__(self):
        self._capital = self.initial_capital

    def reset(self):
        self._capital = self.initial_capital
        self._position = None
        self._trades = []
        self._equity_curve = []
        self._returns = []

    def run(self, klines: list[Kline], strategy_func, strategy_id: str = "default", symbol: str = "", timeframe: str = "1h") -> BacktestResult:
        self.reset()
        if len(klines) < 50:
            raise BacktestError(f"Backtest requires at least 50 candles, got {len(klines)}", code=ErrorCodes.INSUFFICIENT_DATA)
        data = {"open": np.array([k.open for k in klines]), "high": np.array([k.high for k in klines]), "low": np.array([k.low for k in klines]), "close": np.array([k.close for k in klines]), "volume": np.array([k.volume for k in klines]), "timestamp": [k.timestamp for k in klines]}
        logger.info("Starting backtest", strategy=strategy_id, symbol=symbol, candles=len(klines))
        for i in range(50, len(klines)):
            context = {"index": i, "open": data["open"][: i + 1], "high": data["high"][: i + 1], "low": data["low"][: i + 1], "close": data["close"][: i + 1], "volume": data["volume"][: i + 1], "timestamp": data["timestamp"][: i + 1], "current_price": data["close"][i], "position": self._position}
            try:
                signal = strategy_func(context)
            except Exception as e:
                signal = 0
            self._process_signal(signal, data["close"][i], data["timestamp"][i], self._calculate_atr(data, i))
            equity = self._calculate_equity(data["close"][i])
            self._equity_curve.append(equity)
            if i > 50:
                ret = (equity - self._equity_curve[-2]) / self._equity_curve[-2]
                self._returns.append(ret)
        if self._position:
            self._close_position(data["close"][-1], data["timestamp"][-1], self._calculate_atr(data, len(data["close"]) - 1))
        result = self._calculate_result(strategy_id, symbol, timeframe, data["timestamp"][0], data["timestamp"][-1])
        logger.info("Backtest complete", strategy=strategy_id, total_trades=result.total_trades, win_rate=result.win_rate)
        return result

    def _process_signal(self, signal, price, timestamp, atr):
        if signal == 1 and self._position is None:
            self._open_position(OrderSide.BUY, price, timestamp, atr)
        elif signal == -1 and self._position:
            self._close_position(price, timestamp, atr)

    def _open_position(self, side, price, timestamp, atr):
        available = self._capital * 0.95
        executed_price, commission, slippage = self.commission_model.calculate_total_cost(price, available / price, side, atr)
        volume = available / executed_price
        self._position = Position(symbol="", side=side, entry_price=executed_price, volume=volume)
        self._capital -= commission + slippage

    def _close_position(self, price, timestamp, atr):
        if not self._position:
            return
        close_side = OrderSide.SELL if self._position.side == OrderSide.BUY else OrderSide.BUY
        executed_price, commission, slippage = self.commission_model.calculate_total_cost(price, self._position.volume, close_side, atr)
        pnl = (executed_price - self._position.entry_price) * self._position.volume if self._position.side == OrderSide.BUY else (self._position.entry_price - executed_price) * self._position.volume
        pnl -= commission + slippage
        pnl_percent = pnl / (self._position.entry_price * self._position.volume)
        trade = BacktestTrade(trade_id=str(uuid.uuid4())[:8], symbol="", side=self._position.side, entry_price=self._position.entry_price, exit_price=executed_price, volume=self._position.volume, entry_time=timestamp, exit_time=timestamp, commission=commission, slippage=slippage, pnl=pnl, pnl_percent=pnl_percent * 100)
        self._trades.append(trade)
        self._capital += pnl
        self._position = None

    def _calculate_equity(self, current_price):
        equity = self._capital
        if self._position:
            unrealized = (current_price - self._position.entry_price) * self._position.volume if self._position.side == OrderSide.BUY else 0
            equity += unrealized + self._position.entry_price * self._position.volume
        return equity

    def _calculate_atr(self, data, index, period=14):
        if index < period + 1:
            return None
        import talib
        atr = talib.ATR(data["high"][: index + 1], data["low"][: index + 1], data["close"][: index + 1], timeperiod=period)
        valid_atr = atr[~np.isnan(atr)]
        return float(valid_atr[-1]) if len(valid_atr) > 0 else None

    def _calculate_result(self, strategy_id, symbol, timeframe, start_time, end_time):
        final_capital = self._capital
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        total_trades = len(self._trades)
        winning_trades = sum(1 for t in self._trades if t.pnl > 0)
        losing_trades = sum(1 for t in self._trades if t.pnl <= 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        wins = [t.pnl for t in self._trades if t.pnl > 0]
        losses = [t.pnl for t in self._trades if t.pnl <= 0]
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        max_drawdown = self._calculate_max_drawdown()
        metrics = calculate_performance_metrics(self._trades, self._returns)
        return BacktestResult(strategy_id=strategy_id, symbol=symbol, timeframe=timeframe, start_time=start_time, end_time=end_time, initial_capital=self.initial_capital, final_capital=final_capital, total_return=total_return, total_trades=total_trades, winning_trades=winning_trades, losing_trades=losing_trades, win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss, max_drawdown=max_drawdown, sharpe_ratio=metrics.get("sharpe_ratio"), sortino_ratio=metrics.get("sortino_ratio"), profit_factor=self._calculate_profit_factor(), trades=self._trades)

    def _calculate_max_drawdown(self):
        if len(self._equity_curve) < 2:
            return 0.0
        equity = np.array(self._equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        return float(abs(np.min(drawdowns)))

    def _calculate_profit_factor(self):
        gross_profit = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trades if t.pnl < 0))
        if gross_loss == 0:
            return None if gross_profit == 0 else float("inf")
        return gross_profit / gross_loss
