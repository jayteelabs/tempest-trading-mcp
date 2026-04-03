"""Technical indicator MCP tools."""
from typing import Any
import numpy as np
from tempest_mcp.config import ErrorCodes
from tempest_mcp.data.ccxt_adapter import CCXTAdapter, CCXTError
from tempest_mcp.indicators.momentum import calculate_rsi_result, calculate_macd_result, calculate_stochastic_result, calculate_cci_result, calculate_williams_r_result, calculate_roc_result
from tempest_mcp.indicators.trend import calculate_vwap, calculate_supertrend, calculate_ema_result, calculate_adx_result
from tempest_mcp.indicators.volatility import calculate_atr_result, calculate_bollinger_width, calculate_historical_volatility
from tempest_mcp.indicators.volume import calculate_obv_result, calculate_mfi_result
from tempest_mcp.indicators.session_levels import calculate_session_levels
from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

def _success_response(data: dict[str, Any]) -> dict[str, Any]:
    return {"success": True, "data": data}

def _error_response(code: int, message: str) -> dict[str, Any]:
    return {"success": False, "error": {"code": code, "message": message}}

async def _fetch_price_data(symbol: str, timeframe: str = "1h", limit: int = 200, exchange: str = "binance"):
    adapter = CCXTAdapter(exchange=exchange)
    kline_data = adapter.fetch_klines(symbol, timeframe, limit=limit)
    klines = kline_data.klines
    return [k.timestamp for k in klines], [k.open for k in klines], [k.high for k in klines], [k.low for k in klines], [k.close for k in klines], [k.volume for k in klines]

async def indicator_ema(symbol: str, periods: list[int] | None = None, timeframe: str = "1h", limit: int = 200, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_ema", symbol=symbol)
    if periods is None:
        periods = [7, 25, 50, 200]
    try:
        _, _, _, _, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_ema_result(close, periods)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_vwap(symbol: str, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_vwap", symbol=symbol)
    try:
        _, _, high, low, close, volume = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_vwap(high, low, close, volume)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_rsi(symbol: str, period: int = 14, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_rsi", symbol=symbol)
    try:
        _, _, _, _, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_rsi_result(close, period)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_macd(symbol: str, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_macd", symbol=symbol)
    try:
        _, _, _, _, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_macd_result(close)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_atr(symbol: str, period: int = 14, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_atr", symbol=symbol)
    try:
        _, _, high, low, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_atr_result(high, low, close, period)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_supertrend(symbol: str, period: int = 10, multiplier: float = 3.0, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_supertrend", symbol=symbol)
    try:
        _, _, high, low, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_supertrend(high, low, close, period, multiplier)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_session_levels(symbol: str, timeframe: str = "1h", limit: int = 48, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_session_levels", symbol=symbol)
    try:
        timestamps, _, high, low, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_session_levels(timestamps, high, low)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_adx(symbol: str, period: int = 14, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_adx", symbol=symbol)
    try:
        _, _, high, low, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_adx_result(high, low, close, period)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_stochastic(symbol: str, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_stochastic", symbol=symbol)
    try:
        _, _, high, low, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_stochastic_result(high, low, close)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_cci(symbol: str, period: int = 20, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_cci", symbol=symbol)
    try:
        _, _, high, low, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_cci_result(high, low, close, period)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_williams_r(symbol: str, period: int = 14, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_williams_r", symbol=symbol)
    try:
        _, _, high, low, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_williams_r_result(high, low, close, period)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_roc(symbol: str, period: int = 12, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_roc", symbol=symbol)
    try:
        _, _, _, _, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_roc_result(close, period)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_bollinger_width(symbol: str, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_bollinger_width", symbol=symbol)
    try:
        _, _, _, _, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_bollinger_width(close)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_obv(symbol: str, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_obv", symbol=symbol)
    try:
        _, _, _, _, close, volume = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_obv_result(close, volume)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_mfi(symbol: str, timeframe: str = "1h", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_mfi", symbol=symbol)
    try:
        _, _, high, low, close, volume = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_mfi_result(high, low, close, volume)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))

async def indicator_historical_volatility(symbol: str, timeframe: str = "1d", limit: int = 100, exchange: str = "binance") -> dict[str, Any]:
    logger.info("Tool invoked: indicator_historical_volatility", symbol=symbol)
    try:
        _, _, _, _, close, _ = await _fetch_price_data(symbol, timeframe, limit, exchange)
        result = calculate_historical_volatility(close)
        return _success_response({"symbol": symbol, "timeframe": timeframe, "values": result.values})
    except Exception as e:
        return _error_response(ErrorCodes.INDICATOR_ERROR, str(e))
