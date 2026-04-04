"""Technical indicator MCP tool stubs — ENG-5 skeleton."""
from typing import Any

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)


async def indicator_rsi(
    symbol: str,
    period: int = 14,
    timeframe: str = "1h",
    limit: int = 100,
    exchange: str = "binance"
) -> dict[str, Any]:
    """Calculate Relative Strength Index (RSI) for a symbol."""
    logger.info("Tool invoked: indicator_rsi", symbol=symbol, period=period)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "indicator_rsi",
            "symbol": symbol,
            "period": period,
            "timeframe": timeframe,
            "values": [],
            "note": "Implementation pending — indicator engine (ENG-4 successor tickets)"
        }
    }
