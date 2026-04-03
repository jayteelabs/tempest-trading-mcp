"""Backtest MCP tools."""
from datetime import datetime, timedelta
from typing import Any, Callable
from tempest_mcp.backtest.commission import create_binance_model
from tempest_mcp.backtest.engine import BacktestEngine, BacktestError
from tempest_mcp.config import ErrorCodes
from tempest_mcp.data.yf_adapter import YFAdapter, YFinanceError
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.market import Kline

logger = get_logger(__name__)

def _success_response(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}

def _error_response(code: int, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}

STRATEGIES = {}

def _register_strategy(name: str):
    def decorator(func):
        STRATEGIES[name] = func
        return func
    return decorator

@_register_strategy("rsi_mean_reversion")
def _rsi_mean_reversion(ctx):
    import talib
    import numpy as np
    close = ctx["close"]
    rsi = talib.RSI(np.array(close), timeperiod=14)
    current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50
    position = ctx.get("position")
    if current_rsi < 30 and position is None:
        return 1
    elif current_rsi > 70 and position is not None:
        return -1
    return 0

@_register_strategy("ema_crossover")
def _ema_crossover(ctx):
    import talib
    import numpy as np
    close = ctx["close"]
    ema20 = talib.EMA(np.array(close), timeperiod=20)
    ema50 = talib.EMA(np.array(close), timeperiod=50)
    if np.isnan(ema20[-1]) or np.isnan(ema50[-1]):
        return 0
    position = ctx.get("position")
    if ema20[-1] > ema50[-1] and ema20[-2] <= ema50[-2] and position is None:
        return 1
    if ema20[-1] < ema50[-1] and ema20[-2] >= ema50[-2] and position is not None:
        return -1
    return 0

@_register_strategy("buy_hold")
def _buy_hold(ctx):
    if ctx.get("position") is None:
        return 1
    return 0

async def backtest_strategy(symbol: str, strategy_id: str = "rsi_mean_reversion", timeframe: str = "1h", period: str = "1y", initial_capital: float = 10000.0, exchange: str = "binance", source: str = "yf") -> dict[str, Any]:
    logger.info("Tool invoked: backtest_strategy", symbol=symbol, strategy=strategy_id)
    try:
        if strategy_id not in STRATEGIES:
            return _error_response(ErrorCodes.INVALID_PARAMETER, f"Unknown strategy: {strategy_id}. Available: {list(STRATEGIES.keys())}")
        strategy_func = STRATEGIES[strategy_id]
        if source == "yf":
            adapter = YFAdapter()
            df = adapter.get_historical_prices(symbol, period)
            klines = [Kline(timestamp=idx.timestamp(), open=row["open"], high=row["high"], low=row["low"], close=row["close"], volume=row.get("volume", 0)) for idx, row in df.iterrows()]
        else:
            return _error_response(ErrorCodes.INVALID_PARAMETER, "Only yf source supported for backtest")
        engine = BacktestEngine(initial_capital=initial_capital, commission_model=create_binance_model())
        result = engine.run(klines=klines, strategy_func=strategy_func, strategy_id=strategy_id, symbol=symbol, timeframe=timeframe)
        return _success_response({"strategy_id": result.strategy_id, "symbol": result.symbol, "initial_capital": result.initial_capital, "final_capital": result.final_capital, "total_return": result.total_return, "total_trades": result.total_trades, "win_rate": result.win_rate, "max_drawdown": result.max_drawdown, "sharpe_ratio": result.sharpe_ratio})
    except (YFinanceError, BacktestError) as e:
        return _error_response(e.code, e.message)
    except Exception as e:
        logger.error("backtest_strategy failed", symbol=symbol, error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))

async def compare_strategies(symbol: str, strategy_ids: list[str] | None = None, timeframe: str = "1h", period: str = "1y", initial_capital: float = 10000.0, exchange: str = "binance", source: str = "yf") -> dict[str, Any]:
    logger.info("Tool invoked: compare_strategies", symbol=symbol)
    if strategy_ids is None:
        strategy_ids = ["rsi_mean_reversion", "ema_crossover", "buy_hold"]
    invalid = [s for s in strategy_ids if s not in STRATEGIES]
    if invalid:
        return _error_response(ErrorCodes.INVALID_PARAMETER, f"Unknown strategies: {invalid}")
    try:
        adapter = YFAdapter()
        df = adapter.get_historical_prices(symbol, period)
        klines = [Kline(timestamp=idx.timestamp(), open=row["open"], high=row["high"], low=row["low"], close=row["close"], volume=row.get("volume", 0)) for idx, row in df.iterrows()]
        results = []
        for strategy_id in strategy_ids:
            strategy_func = STRATEGIES[strategy_id]
            engine = BacktestEngine(initial_capital=initial_capital, commission_model=create_binance_model())
            result = engine.run(klines=klines, strategy_func=strategy_func, strategy_id=strategy_id, symbol=symbol, timeframe=timeframe)
            results.append({"strategy_id": result.strategy_id, "total_return": result.total_return, "win_rate": result.win_rate, "max_drawdown": result.max_drawdown, "total_trades": result.total_trades})
        results.sort(key=lambda x: x["total_return"], reverse=True)
        return _success_response({"symbol": symbol, "strategies": results, "best_strategy": results[0]["strategy_id"] if results else None})
    except Exception as e:
        logger.error("compare_strategies failed", symbol=symbol, error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))
