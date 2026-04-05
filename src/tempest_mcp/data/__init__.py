"""
Data layer for Tempest MCP.

This module provides adapters for fetching market data from various sources:
- CCXT: Real-time + historical data for crypto via Binance/Bybit public REST APIs (primary)
- Yahoo Finance: Historical data for stocks and data gaps CCXT doesn't cover (fallback)

Data Source Priority (D3):
- Primary: CCXT via Binance/Bybit public REST (all crypto + stocks, no API keys)
- Fallback: yfinance (for stocks and data gaps CCXT doesn't cover)

IMPORTANT (2026-04-05): TradingView is NOT used for data. There is no official
TradingView data API that accepts a key and returns OHLCV. tv_adapter.py is a
deprecated stub. TRADINGVIEW_API_KEY is not used.

Design Decisions:
- D3: CCXT primary + yfinance fallback (TradingView deprecated)
- D11: Symbol conversion via _symbols.py
- D19: Historical data abstraction layer (HistoricalDataSource)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

# Backward-compatible exports from YFAdapter and CCXTAdapter (ENG-4/ENG-6)
# Must be at module level (not inside TYPE_CHECKING) for runtime exports
from tempest_mcp.data.ccxt_adapter import CCXTAdapter
from tempest_mcp.data.yf_adapter import YFAdapter

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger()

__all__ = [
    # Historical data adapters
    "YFAdapter",
    "CCXTAdapter",
    # Live data protocol and factory
    "LiveDataAdapter",
    "get_live_adapter",
    # Historical data protocol and factory (D19)
    "HistoricalDataSource",
    "HistoricalDataAdapter",
    "get_historical_adapter",
    "DataSourceRouter",
    # Symbol utilities
    "normalize_to_ccxt",
    "normalize_to_tradingview",
    "normalize_to_yf",
    "get_base_currency",
    "validate_symbol",
]


@runtime_checkable
class LiveDataAdapter(Protocol):
    """Protocol defining the interface for live data adapters.

    All adapters must implement these three methods:
    - fetch_live_price: Get latest trade price
    - fetch_ohlcv_live: Get OHLCV candlestick data
    - fetch_orderbook_snapshot: Get order book depth
    """

    def fetch_live_price(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> float:
        """Fetch latest trade price for a symbol.

        Args:
            symbol: Symbol in any supported format
            exchange: Target exchange (default: "binance")

        Returns:
            Latest trade price as float, or float('nan') on error
        """
        ...

    def fetch_ohlcv_live(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data.

        Args:
            symbol: Symbol in any supported format
            timeframe: Timeframe string (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV columns and UTC-aware index,
            or empty DataFrame on error
        """
        ...

    def fetch_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict:
        """Fetch order book snapshot.

        Args:
            symbol: Symbol in any supported format
            limit: Depth of orderbook to fetch

        Returns:
            Dict with bids, asks, and timestamp,
            or empty structure on error
        """
        ...


def get_live_adapter() -> LiveDataAdapter:
    """Get the live data adapter.

    Always returns CCXTAdapter — TradingView has no OHLCV data API.
    tv_adapter.py is a deprecated stub, retained for backward compatibility.

    Returns:
        CCXTAdapter instance (real-time crypto data via Binance/Bybit public REST)

    Example:
        >>> adapter = get_live_adapter()
        >>> price = adapter.fetch_live_price("BTCUSDT")
    """
    from tempest_mcp.data.ccxt_adapter import CCXTAdapter

    logger.info(
        "adapter_selected",
        adapter="CCXTAdapter",
        reason="TradingView has no OHLCV data API - CCXT is primary",
    )
    return CCXTAdapter()


# Lazy re-exports via __getattr__ to avoid E402 for _symbols imports
def __getattr__(name: str):
    if name in (
        "get_base_currency",
        "normalize_to_ccxt",
        "normalize_to_tradingview",
        "normalize_to_yf",
        "validate_symbol",
        # Historical data exports (D19)
        "HistoricalDataSource",
        "HistoricalDataAdapter",
        "get_historical_adapter",
        "DataSourceRouter",
    ):
        if name in ("HistoricalDataSource", "HistoricalDataAdapter"):
            import tempest_mcp.data._hist as _hist
            return getattr(_hist, name)
        if name == "get_historical_adapter":
            import tempest_mcp.data._factory as _factory
            return getattr(_factory, name)
        if name == "DataSourceRouter":
            import tempest_mcp.data._router as _router
            return getattr(_router, name)
        import tempest_mcp.data._symbols as _symbols
        return getattr(_symbols, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
