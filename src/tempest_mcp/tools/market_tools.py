"""Market data MCP tool stubs — ENG-5 skeleton."""
from typing import Any

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)


async def fetch_ticker(symbol: str, exchange: str = "binance") -> dict[str, Any]:
    """Fetch real-time ticker price + volume for a symbol."""
    logger.info("Tool invoked: fetch_ticker", symbol=symbol, exchange=exchange)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "fetch_ticker",
            "symbol": symbol,
            "exchange": exchange,
            "note": "Implementation pending — data layer (ENG-4 successor tickets)"
        }
    }


async def fetch_klines(
    symbol: str,
    timeframe: str = "1h",
    since: str | None = None,
    limit: int = 100,
    exchange: str = "binance",
    source: str = "ccxt"
) -> dict[str, Any]:
    """Fetch OHLCV klines (candlestick data) for a symbol."""
    logger.info("Tool invoked: fetch_klines", symbol=symbol, timeframe=timeframe, limit=limit)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "fetch_klines",
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "note": "Implementation pending — data layer (ENG-4 successor tickets)"
        }
    }


async def fetch_orderbook(symbol: str, limit: int = 20, exchange: str = "binance") -> dict[str, Any]:
    """Fetch order book (bid/ask depth) for a symbol."""
    logger.info("Tool invoked: fetch_orderbook", symbol=symbol, limit=limit)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "fetch_orderbook",
            "symbol": symbol,
            "exchange": exchange,
            "limit": limit,
            "note": "Implementation pending — data layer (ENG-4 successor tickets)"
        }
    }
