"""Screener MCP tools."""
from typing import Any
from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.indicator import SessionType
from tempest_mcp.screener.scanner import ScanFilter, Screener

logger = get_logger(__name__)

def _success_response(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}

def _error_response(code: int, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}

def _parse_filters(filter_names: list[str]) -> list[ScanFilter]:
    filter_map = {"rsi_oversold": ScanFilter.RSI_OVERSOLD, "rsi_overbought": ScanFilter.RSI_OVERBOUGHT, "trend_bullish": ScanFilter.TREND_BULLISH, "trend_bearish": ScanFilter.TREND_BEARISH, "high_volatility": ScanFilter.HIGH_VOLATILITY, "low_volatility": ScanFilter.LOW_VOLATILITY, "volume_spike": ScanFilter.VOLUME_SPIKE}
    return [filter_map.get(f) for f in filter_names if f in filter_map]

async def screener_scan(symbols: list[str] | None = None, filters: list[str] | None = None, min_score: float = 0.0, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: screener_scan", symbols=len(symbols) if symbols else "default")
    try:
        scan_filters = _parse_filters(filters) if filters else []
        screener = Screener(symbols=tuple(symbols) if symbols else None, exchange=exchange, filters=scan_filters, min_score=min_score)
        results = screener.scan()
        formatted = [{"symbol": r.symbol, "exchange": r.exchange, "price": r.price, "filters_matched": r.filters_matched, "indicator_values": r.indicator_values, "score": r.score} for r in results]
        return _success_response({"total_scanned": len(symbols) if symbols else len(screener.symbols), "results_count": len(formatted), "results": formatted})
    except Exception as e:
        logger.error("screener_scan failed", error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))

async def session_breakout_scan(session: str, symbols: list[str] | None = None, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: session_breakout_scan", session=session)
    try:
        session_map = {"asia": SessionType.ASIA, "london": SessionType.LONDON, "new_york": SessionType.NEW_YORK, "ny": SessionType.NEW_YORK}
        session_type = session_map.get(session.lower())
        if session_type is None:
            return _error_response(ErrorCodes.INVALID_PARAMETER, f"Invalid session: {session}")
        screener = Screener(symbols=tuple(symbols) if symbols else None, exchange=exchange)
        results = screener.session_breakout_scan(session_type)
        formatted = [{"symbol": r.symbol, "exchange": r.exchange, "price": r.price, "filters_matched": r.filters_matched, "indicator_values": r.indicator_values, "score": r.score} for r in results]
        return _success_response({"session": session, "results_count": len(formatted), "results": formatted})
    except Exception as e:
        logger.error("session_breakout_scan failed", error=str(e))
        return _error_response(ErrorCodes.INTERNAL_ERROR, str(e))
