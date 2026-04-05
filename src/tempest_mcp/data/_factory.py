"""Data adapter factory with caching."""

from __future__ import annotations

from functools import lru_cache

from tempest_mcp.data._hist import HistoricalDataSource


@lru_cache(maxsize=1)
def get_historical_adapter() -> HistoricalDataSource:
    """Get the historical data adapter (singleton).

    Returns a HistoricalDataSource with CCXT as primary and yfinance as fallback.
    Uses lru_cache(maxsize=1) for singleton semantics.

    Data Source Priority (D3):
    - Primary: CCXT via Binance/Bybit public REST (no API keys required)
    - Fallback: yfinance (for stocks and data gaps CCXT doesn't cover)

    Returns:
        HistoricalDataSource instance

    Example:
        >>> adapter = get_historical_adapter()
        >>> df = adapter.fetch_ohlcv("BTC/USDT", interval="1d", start=None, end=None)
    """
    return HistoricalDataSource()
