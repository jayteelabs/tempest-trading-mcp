"""Data adapter factory with caching."""

from __future__ import annotations

from functools import lru_cache

from tempest_mcp.data._hist import HistoricalDataSource


@lru_cache(maxsize=1)
def get_historical_adapter() -> HistoricalDataSource:
    """Get the historical data adapter (singleton).

    Returns a HistoricalDataSource wrapping YFAdapter.
    Uses lru_cache(maxsize=1) for singleton semantics.

    Returns:
        HistoricalDataSource instance

    Example:
        >>> adapter = get_historical_adapter()
        >>> df = adapter.fetch_ohlcv("BTC-USD", interval="1d", start=None, end=None)
    """
    return HistoricalDataSource()
