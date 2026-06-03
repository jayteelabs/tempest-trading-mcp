"""Market data MCP tool handlers — ENG-122.

Implements real behavior for fetch_ticker, fetch_klines, and fetch_orderbook
wired to the repo's live CCXT adapter and historical routing layer.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import structlog

from tempest_mcp.data import get_live_adapter
from tempest_mcp.data._contracts import SUPPORTED_EXCHANGES, SUPPORTED_TIMEFRAMES
from tempest_mcp.data._factory import get_ohlcv_intake
from tempest_mcp.data._intake import OhlcvRequest
from tempest_mcp.data._symbols import normalize_to_ccxt_market
from tempest_mcp.time_utils import BUSINESS_TZ_NAME, coerce_window_datetime_to_utc

logger = structlog.get_logger()

def _failure_envelope(code: int, message: str) -> dict:
    """Create a deterministic MCP failure envelope."""
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _is_valid_exchange(exchange: str) -> bool:
    """Check if exchange is supported."""
    return exchange.lower() in SUPPORTED_EXCHANGES


def _normalize_symbol(symbol: str) -> str | None:
    """Normalize symbol to canonical CCXT format (BTC/USDT).

    Returns None if symbol is invalid.
    """
    try:
        return normalize_to_ccxt_market(symbol)
    except ValueError:
        return None


def _parse_since(since_str: str | None) -> datetime | None:
    """Parse ISO-8601 since string to UTC datetime.

    Naive datetimes are interpreted in America/New_York, then converted to UTC.
    Returns None if since_str is None or invalid.
    """
    if since_str is None:
        return None

    try:
        # Parse ISO-8601 string
        dt = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        # If timezone naive, interpret in business timezone then convert to UTC
        if dt.tzinfo is None:
            logger.warning(
                "naive_datetime_interpreted_as_business_tz",
                param="since",
                dt=str(dt),
                timezone=BUSINESS_TZ_NAME,
            )
            dt = coerce_window_datetime_to_utc(dt)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except ValueError:
        return None


async def fetch_ticker(
    symbol: str,
    exchange: str = "binance",
) -> dict:
    """Fetch real-time ticker price + metadata for a crypto symbol.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSD", "BTC/USDT")
        exchange: Exchange name (default: "binance")

    Returns:
        MCP envelope:
        - success=True with data: {tool, symbol, exchange, price, bid, ask,
          change_pct_24h, volume_24h, timestamp}
        - success=False with error: {code, message} for validation or data errors

    Notes:
        - price is always present on success (float)
        - bid/ask/change_pct_24h/volume_24h may be None if upstream doesn't provide them
        - Uses exchange-aware CCXT adapter for the specified exchange
    """
    logger.info("fetch_ticker", symbol=symbol, exchange=exchange)

    # Validate exchange
    exchange_lower = exchange.lower()
    if not _is_valid_exchange(exchange_lower):
        return _failure_envelope(
            1004,
            f"Invalid exchange: {exchange!r}. Must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}",
        )

    # Validate and normalize symbol
    if not symbol or not isinstance(symbol, str):
        return _failure_envelope(1004, "symbol must be a non-empty string")

    ccxt_symbol = _normalize_symbol(symbol)
    if ccxt_symbol is None:
        return _failure_envelope(
            1004,
            f"Invalid symbol format: {symbol!r}. Expected formats: BTCUSDT, ETHUSD, BTC/USDT",
        )

    # Get exchange-aware live adapter
    adapter = get_live_adapter(exchange_name=exchange_lower)

    # Fetch ticker snapshot
    snapshot = adapter.fetch_ticker_snapshot(symbol)

    # Check for failure (price is NaN)
    if math.isnan(snapshot["price"]):
        return _failure_envelope(
            3000,
            f"Unable to fetch ticker data for {symbol} on {exchange}",
        )

    # Format timestamp
    ts = snapshot["timestamp"]
    ts_iso = ts.isoformat() if ts is not None else None

    return {
        "success": True,
        "data": {
            "tool": "fetch_ticker",
            "symbol": ccxt_symbol,
            "exchange": exchange_lower,
            "price": snapshot["price"],
            "bid": snapshot["bid"],
            "ask": snapshot["ask"],
            "change_pct_24h": snapshot["change_pct_24h"],
            "volume_24h": snapshot["volume_24h"],
            "timestamp": ts_iso,
        },
    }


async def fetch_klines(
    symbol: str,
    timeframe: str = "1h",
    since: str | None = None,
    limit: int = 100,
    exchange: str = "binance",
    source: str = "ccxt",
) -> dict:
    """Fetch OHLCV klines (candlestick data) for a symbol.

    Routes through the historical abstraction layer (CCXT primary, yfinance fallback).
    The source parameter is accepted for compatibility but is validated to only "ccxt",
    which means "use the exchange-backed historical route".

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSD", "BTC/USDT")
        timeframe: OHLCV interval (default: "1h")
        since: ISO-8601 start time (naive interpreted as America/New_York)
        limit: Max candles to return (1-1000)
        exchange: Exchange name (default: "binance")
        source: Must be "ccxt" (compatibility-only; actual route is CCXT+yfinance fallback)

    Returns:
        MCP envelope:
        - success=True with data: {tool, symbol, exchange, timeframe, limit, since,
          source, source_used, rows: [{timestamp, open, high, low, close, volume}, ...]}
        - success=False with error: {code, message}

    Notes:
        - source="ccxt" is required; other values return validation error
        - source_used in response indicates which adapter actually fulfilled ("ccxt" or "yfinance")
        - 4h timeframe via yfinance fallback returns empty data (4h not supported by yfinance)
    """
    logger.info("fetch_klines", symbol=symbol, timeframe=timeframe, since=since, limit=limit, exchange=exchange)

    # Validate exchange
    exchange_lower = exchange.lower()
    if not _is_valid_exchange(exchange_lower):
        return _failure_envelope(
            1004,
            f"Invalid exchange: {exchange!r}. Must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}",
        )

    # Validate timeframe
    if timeframe not in SUPPORTED_TIMEFRAMES:
        return _failure_envelope(
            1004,
            f"Invalid timeframe: {timeframe!r}. Must be one of: {', '.join(sorted(SUPPORTED_TIMEFRAMES))}",
        )

    # Validate limit
    if not isinstance(limit, int) or limit < 1 or limit > 1000:
        return _failure_envelope(
            1004,
            "limit must be an integer between 1 and 1000",
        )

    # Validate source (compatibility-only, must be "ccxt")
    if source != "ccxt":
        return _failure_envelope(
            1004,
            'source must be "ccxt" (historical routing is CCXT primary with yfinance fallback)',
        )

    # Validate and normalize symbol
    if not symbol or not isinstance(symbol, str):
        return _failure_envelope(1004, "symbol must be a non-empty string")

    ccxt_symbol = _normalize_symbol(symbol)
    if ccxt_symbol is None:
        return _failure_envelope(
            1004,
            f"Invalid symbol format: {symbol!r}. Expected formats: BTCUSDT, ETHUSD, BTC/USDT",
        )

    # Parse since
    since_dt = _parse_since(since)
    if since is not None and since_dt is None:
        return _failure_envelope(1004, "since must be a valid ISO-8601 datetime string")
    since_iso = since_dt.isoformat() if since_dt is not None else None

    # Fetch OHLCV through the historical intake seam.
    intake = get_ohlcv_intake(exchange_name=exchange_lower)
    ohlcv_result = intake.fetch(
        OhlcvRequest(
            symbol=ccxt_symbol,
            timeframe=timeframe,
            exchange=exchange_lower,
            start=since_dt,
            end=None,
            limit=limit,
            auto_adjust=True,
        )
    )
    df = ohlcv_result.frame

    # Check for empty result
    if df.empty:
        return _failure_envelope(
            3000,
            f"Unable to fetch klines data for {symbol} on {exchange} ({timeframe})",
        )

    # Convert DataFrame to rows
    rows = []
    for ts, row in df.iterrows():
        rows.append({
            "timestamp": ts.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })

    # Limit rows if needed
    if limit and len(rows) > limit:
        rows = rows[-limit:]

    return {
        "success": True,
        "data": {
            "tool": "fetch_klines",
            "symbol": ohlcv_result.canonical_symbol,
            "exchange": exchange_lower,
            "timeframe": timeframe,
            "limit": limit,
            "since": since_iso,
            "source": source,
            "source_used": ohlcv_result.source_used,
            "rows": rows,
        },
    }


async def fetch_orderbook(
    symbol: str,
    limit: int = 20,
    exchange: str = "binance",
) -> dict:
    """Fetch order book (bid/ask depth) for a symbol.

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT", "ETHUSD", "BTC/USDT")
        limit: Depth of orderbook (1-100)
        exchange: Exchange name (default: "binance")

    Returns:
        MCP envelope:
        - success=True with data: {tool, symbol, exchange, limit, timestamp,
          bids: [[price, amount], ...], asks: [[price, amount], ...]}
        - success=False with error: {code, message}

    Notes:
        - bids/asks are sorted: bids descending by price, asks ascending by price
        - one-sided snapshots are returned as success; only both sides empty are treated as a data-source error
        - Uses exchange-aware CCXT adapter for the specified exchange
    """
    logger.info("fetch_orderbook", symbol=symbol, limit=limit, exchange=exchange)

    # Validate exchange
    exchange_lower = exchange.lower()
    if not _is_valid_exchange(exchange_lower):
        return _failure_envelope(
            1004,
            f"Invalid exchange: {exchange!r}. Must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}",
        )

    # Validate limit
    if not isinstance(limit, int) or limit < 1 or limit > 100:
        return _failure_envelope(
            1004,
            "limit must be an integer between 1 and 100",
        )

    # Validate and normalize symbol
    if not symbol or not isinstance(symbol, str):
        return _failure_envelope(1004, "symbol must be a non-empty string")

    ccxt_symbol = _normalize_symbol(symbol)
    if ccxt_symbol is None:
        return _failure_envelope(
            1004,
            f"Invalid symbol format: {symbol!r}. Expected formats: BTCUSDT, ETHUSD, BTC/USDT",
        )

    # Get exchange-aware live adapter
    adapter = get_live_adapter(exchange_name=exchange_lower)

    # Fetch orderbook snapshot
    snapshot = adapter.fetch_orderbook_snapshot(symbol, limit=limit)

    # Check for failure (timestamp None or empty bids/asks)
    if snapshot["timestamp"] is None or (len(snapshot["bids"]) == 0 and len(snapshot["asks"]) == 0):
        return _failure_envelope(
            3000,
            f"Unable to fetch orderbook data for {symbol} on {exchange}",
        )

    # Format timestamp
    ts = snapshot["timestamp"]
    ts_iso = ts.isoformat() if ts is not None else None

    # Format bids/asks - CCXT returns [[price, amount], ...]
    bids = [[float(p), float(a)] for p, a in snapshot["bids"]]
    asks = [[float(p), float(a)] for p, a in snapshot["asks"]]

    return {
        "success": True,
        "data": {
            "tool": "fetch_orderbook",
            "symbol": ccxt_symbol,
            "exchange": exchange_lower,
            "limit": limit,
            "timestamp": ts_iso,
            "bids": bids,
            "asks": asks,
        },
    }
