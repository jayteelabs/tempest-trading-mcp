"""Backtest MCP tool stubs — ENG-5 skeleton."""
from typing import Any

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)


async def backtest_strategy(
    symbol: str,
    strategy_id: str = "rsi_mean_reversion",
    timeframe: str = "1h",
    period: str = "1y",
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    source: str = "yf"
) -> dict[str, Any]:
    """Run a backtest for a single strategy on a symbol."""
    logger.info("Tool invoked: backtest_strategy", symbol=symbol, strategy=strategy_id)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "backtest_strategy",
            "symbol": symbol,
            "strategy_id": strategy_id,
            "timeframe": timeframe,
            "period": period,
            "note": "Implementation pending — backtest engine (ENG-4 successor tickets)"
        }
    }
