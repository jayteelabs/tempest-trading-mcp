"""Technical indicator MCP tools — live data backed implementations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tempest_mcp.data._contracts import SUPPORTED_EXCHANGES, SUPPORTED_TIMEFRAMES
from tempest_mcp.data._factory import get_ohlcv_intake
from tempest_mcp.data._intake import OhlcvRequest
from tempest_mcp.data._symbols import normalize_to_ccxt_market
from tempest_mcp.indicators.momentum.rsi import (
    OVERBOUGHT_THRESHOLD,
    OVERSOLD_THRESHOLD,
    calculate_rsi,
)
from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# Zone thresholds for RSI interpretation
_ZONE_OVERSOLD = "oversold"
_ZONE_NEUTRAL = "neutral"
_ZONE_OVERBOUGHT = "overbought"


def _is_valid_exchange(exchange: str) -> bool:
    """Check if exchange is supported."""
    return exchange.lower() in SUPPORTED_EXCHANGES


def _failure_envelope(code: int, message: str) -> dict[str, Any]:
    """Build a deterministic failure envelope.

    Args:
        code: Error code (1xxx = validation, 2xxx = data/fetch)
        message: Human-readable error message

    Returns:
        Failure envelope dict matching repo conventions
    """
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _detect_zone(rsi_value: float) -> str:
    """Detect RSI zone based on thresholds.

    Args:
        rsi_value: RSI value (0-100)

    Returns:
        Zone string: 'oversold', 'neutral', or 'overbought'
    """
    if rsi_value <= OVERSOLD_THRESHOLD:
        return _ZONE_OVERSOLD
    elif rsi_value >= OVERBOUGHT_THRESHOLD:
        return _ZONE_OVERBOUGHT
    return _ZONE_NEUTRAL


async def indicator_rsi(
    symbol: str,
    period: int = 14,
    timeframe: str = "1h",
    limit: int = 100,
    exchange: str = "binance",
) -> dict[str, Any]:
    """Calculate Relative Strength Index (RSI) for a symbol.

    Fetches historical OHLCV data via the repo's historical adapter path
    and computes RSI using the existing calculate_rsi() engine.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "BTC/USDT", "BTC-USDT")
        period: RSI period (must be >= 2)
        timeframe: Data timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo)
        limit: Maximum number of RSI rows to return
        exchange: Exchange name (binance, bybit, coinbase, kraken)

    Returns:
        Success envelope:
            {
                "success": True,
                "data": {
                    "tool": "indicator_rsi",
                    "symbol": "BTC/USDT",  # canonical format
                    "exchange": "binance",
                    "timeframe": "1h",
                    "period": 14,
                    "limit": 100,
                    "source_used": "ccxt",  # "ccxt" | "yfinance"
                    "values": [{"timestamp": "...", "rsi": 45.5}, ...],
                    "latest": {"timestamp": "...", "rsi": 52.3, "zone": "neutral"}
                }
            }
        Failure envelope (validation):
            {"success": False, "error": {"code": 1001-1005, "message": "..."}}
        Failure envelope (data):
            {"success": False, "error": {"code": 2001-2002, "message": "..."}}
    """
    logger.info(
        "indicator_rsi invoked",
        symbol=symbol,
        period=period,
        timeframe=timeframe,
        limit=limit,
        exchange=exchange,
    )

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    # Validate symbol format
    try:
        canonical_symbol = normalize_to_ccxt_market(symbol)
    except ValueError as exc:
        logger.warning("indicator_rsi symbol validation failed", symbol=symbol, error=str(exc))
        return _failure_envelope(1001, f"Invalid symbol format: {symbol!r}")

    # Validate exchange
    exchange_lower = exchange.lower()
    if not _is_valid_exchange(exchange_lower):
        supported = ", ".join(SUPPORTED_EXCHANGES)
        return _failure_envelope(
            1002,
            f"Unsupported exchange: {exchange}. Supported: {supported}",
        )

    # Validate timeframe
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        return _failure_envelope(
            1003,
            f"Invalid timeframe: {timeframe}. Supported: {supported}",
        )

    # Validate period (must be integer >= 2)
    if not isinstance(period, int) or period < 2:
        return _failure_envelope(1004, "period must be a positive integer >= 2")

    # Validate limit (must be integer between 1 and 1000)
    if not isinstance(limit, int) or limit < 1 or limit > 1000:
        return _failure_envelope(1005, "limit must be an integer between 1 and 1000")

    # -------------------------------------------------------------------------
    # Data Fetch
    # -------------------------------------------------------------------------

    try:
        intake = get_ohlcv_intake(exchange_lower)
        # Fetch enough candles for RSI computation plus a small buffer while
        # keeping `limit` as the public maximum RSI row count.
        ohlcv_result = intake.fetch(
            OhlcvRequest(
                symbol=canonical_symbol,
                timeframe=timeframe,
                exchange=exchange_lower,
                start=None,
                end=None,
                limit=limit,
                warmup_bars=period + 5,
            )
        )
        ohlcv_df = ohlcv_result.frame
    except Exception as exc:
        logger.error(
            "indicator_rsi adapter fetch failed",
            symbol=canonical_symbol,
            exchange=exchange_lower,
            error=str(exc),
        )
        return _failure_envelope(2001, f"No historical data available for {symbol} on {exchange}")

    # -------------------------------------------------------------------------
    # RSI Computation
    # -------------------------------------------------------------------------

    if ohlcv_df.empty:
        logger.warning(
            "indicator_rsi no data returned",
            symbol=canonical_symbol,
            exchange=exchange_lower,
        )
        return _failure_envelope(
            2001,
            f"No historical data available for {symbol} on {exchange}",
        )

    # Extract close prices - ensure UTC-aware datetime index
    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        logger.warning("indicator_rsi non-datetime index, converting", index_type=type(ohlcv_df.index))
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df.index = pd.to_datetime(ohlcv_df.index, utc=True)

    close_prices = ohlcv_df["close"]

    # Compute RSI
    try:
        rsi_series = calculate_rsi(close_prices, period=period)
    except Exception as exc:
        logger.error(
            "indicator_rsi RSI computation failed",
            symbol=canonical_symbol,
            period=period,
            error=str(exc),
        )
        return _failure_envelope(2002, f"RSI computation failed: {exc}")

    # Filter out NaN values and align with original index
    rsi_valid = rsi_series.dropna()

    if rsi_valid.empty:
        logger.warning(
            "indicator_rsi insufficient data for RSI",
            symbol=canonical_symbol,
            period=period,
            candle_count=len(ohlcv_df),
        )
        return _failure_envelope(
            2001,
            f"No historical data available for {symbol} on {exchange}",
        )

    # -------------------------------------------------------------------------
    # Build Response
    # -------------------------------------------------------------------------

    # Get up to `limit` rows (oldest → newest)
    rsi_result = rsi_valid.tail(limit)

    # Build values list
    values = []
    for ts, rsi_val in rsi_result.items():
        # Format timestamp as ISO string with UTC timezone
        if isinstance(ts, pd.Timestamp):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)

        values.append({
            "timestamp": ts_str,
            "rsi": round(float(rsi_val), 4),
        })

    # Build latest entry from most recent RSI value
    latest_rsi = rsi_valid.iloc[-1]
    latest_ts = rsi_valid.index[-1]
    if isinstance(latest_ts, pd.Timestamp):
        latest_ts_str = latest_ts.isoformat()
    else:
        latest_ts_str = str(latest_ts)

    latest_zone = _detect_zone(float(latest_rsi))

    latest = {
        "timestamp": latest_ts_str,
        "rsi": round(float(latest_rsi), 4),
        "zone": latest_zone,
    }

    # Normalize symbol to canonical format for response
    response_symbol = canonical_symbol  # Already in BTC/USDT format

    result = {
        "success": True,
        "data": {
            "tool": "indicator_rsi",
            "symbol": response_symbol,
            "exchange": exchange_lower,
            "timeframe": timeframe,
            "period": period,
            "limit": limit,
            "source_used": ohlcv_result.source_used,
            "values": values,
            "latest": latest,
        },
    }

    logger.info(
        "indicator_rsi success",
        symbol=response_symbol,
        exchange=exchange_lower,
        values_count=len(values),
        source_used=ohlcv_result.source_used,
    )

    return result
