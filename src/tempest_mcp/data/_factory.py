"""Data adapter factory with caching."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from tempest_mcp.data._hist import HistoricalDataSource

# Supported exchange names
SUPPORTED_EXCHANGES: tuple[str, ...] = ("binance", "bybit", "coinbase", "kraken")


@lru_cache(maxsize=4)
def get_live_adapter(exchange_name: Literal["binance", "bybit", "coinbase", "kraken"] = "binance"):
    """Get a live-market-data adapter for the specified exchange.

    Returns a CCXTAdapter configured for the requested exchange.
    Results are cached per exchange (maxsize=4) for efficiency.

    Args:
        exchange_name: Target exchange (default: "binance")

    Returns:
        CCXTAdapter instance for exchange-backed live market data.

    Example:
        >>> adapter = get_live_adapter("binance")
        >>> adapter = get_live_adapter("bybit")
    """
    from tempest_mcp.data.ccxt_adapter import CCXTAdapter

    return CCXTAdapter(exchange_name=exchange_name)


@lru_cache(maxsize=4)
def get_historical_adapter(exchange_name: Literal["binance", "bybit", "coinbase", "kraken"] = "binance") -> HistoricalDataSource:
    """Get the historical data adapter for the specified exchange (singleton per exchange).

    Returns a HistoricalDataSource configured with a CCXT adapter for the requested
    exchange as primary and yfinance as fallback.
    Uses lru_cache(maxsize=4) for per-exchange singleton semantics.

    Data Source Priority (D3):
    - Primary: CCXT via exchange public REST (no API keys required)
    - Fallback: yfinance (for stocks and data gaps CCXT doesn't cover)

    Args:
        exchange_name: Target exchange for CCXT primary (default: "binance")

    Returns:
        HistoricalDataSource instance configured for the exchange

    Example:
        >>> adapter = get_historical_adapter("binance")
        >>> df = adapter.fetch_ohlcv("BTC/USDT", interval="1d", start=None, end=None)
    """
    return HistoricalDataSource(exchange_name=exchange_name)
