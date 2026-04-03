"""Market data MCP tools."""
from datetime import datetime
from typing import Any
from tempest_mcp.config import ErrorCodes
from tempest_mcp.data.ccxt_adapter import CCXTAdapter, CCXTError
from tempest_mcp.data.yf_adapter import YFAdapter, YFinanceError
from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

def _success_response(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}

def _error_response(code: int, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}

async def fetch_ticker(symbol: str, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: fetch_ticker", symbol=symbol, exchange=exchange)
    try:
        adapter = CCXTAdapter(exchange=exchange)
        ticker = adapter.fetch_ticker(symbol)
        return _success_response({"symbol": ticker.symbol, "exchange": ticker.exchange, "price": ticker.price, "volume_24h": ticker.volume_24h, "high_24h": ticker.high_24h, "low_24h": ticker.low_24h, "change_percent_24h": ticker.change_percent_24h, "bid": ticker.bid, "ask": ticker.ask, "timestamp": ticker.timestamp})
    except CCXTError as e:
        return _error_response(e.code, e.message)
    except Exception as e:
        logger.error("fetch_ticker failed", symbol=symbol, error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))

async def fetch_klines(symbol: str, timeframe: str = "1h", since: str | None = None, limit: int = 100, exchange: str = "binance", source: str = "ccxt") -> dict[str, Any]:
    logger.info("Tool invoked: fetch_klines", symbol=symbol, timeframe=timeframe, limit=limit)
    try:
        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
            except ValueError:
                try:
                    since_dt = datetime.fromtimestamp(float(since))
                except ValueError:
                    return _error_response(ErrorCodes.INVALID_PARAMETER, f"Invalid since format: {since}")
        if source == "yf":
            adapter = YFAdapter()
            kline_data = adapter.fetch_klines(symbol, timeframe, since_dt, limit)
        else:
            adapter = CCXTAdapter(exchange=exchange)
            kline_data = adapter.fetch_klines(symbol, timeframe, since_dt, limit)
        klines_list = [{"timestamp": k.timestamp, "open": k.open, "high": k.high, "low": k.low, "close": k.close, "volume": k.volume} for k in kline_data.klines]
        return _success_response({"symbol": kline_data.symbol, "timeframe": kline_data.timeframe, "exchange": kline_data.exchange, "klines": klines_list, "count": len(klines_list)})
    except (CCXTError, YFinanceError) as e:
        return _error_response(e.code, e.message)
    except Exception as e:
        logger.error("fetch_klines failed", symbol=symbol, error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))

async def fetch_orderbook(symbol: str, limit: int = 20, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: fetch_orderbook", symbol=symbol, limit=limit)
    try:
        adapter = CCXTAdapter(exchange=exchange)
        orderbook = adapter.fetch_orderbook(symbol, limit)
        return _success_response({"symbol": orderbook.symbol, "exchange": orderbook.exchange, "bids": [{"price": b.price, "volume": b.volume} for b in orderbook.bids], "asks": [{"price": a.price, "volume": a.volume} for a in orderbook.asks], "timestamp": orderbook.timestamp})
    except CCXTError as e:
        return _error_response(e.code, e.message)
    except Exception as e:
        logger.error("fetch_orderbook failed", symbol=symbol, error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))
