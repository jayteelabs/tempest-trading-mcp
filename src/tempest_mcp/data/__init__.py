"""Data layer for Tempest MCP.

Architecture summary:
- CCXT is the active primary market-data source for live crypto data and raw
  exchange-backed historical retrieval.
- yfinance is historical fallback only, used when CCXT returns no usable
  historical data or when symbol coverage differs.
- TradingView is not an active data source. ``tv_adapter.py`` is retained only
  as a deprecated compatibility shim and must not be used for new work.

Data source priority (D3):
- Primary: CCXT via Binance/Bybit public REST (no API keys required)
- Historical fallback: yfinance (stocks and CCXT historical gaps)

Design decisions:
- D3: CCXT primary + yfinance fallback (TradingView deprecated)
- D11: Symbol conversion via _symbols.py
- D19: Historical data abstraction layer (HistoricalDataSource)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

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
    "get_ohlcv_intake",
    "OhlcvIntake",
    "OhlcvRequest",
    "OhlcvResult",
    "DataSourceRouter",
    # Symbol utilities
    "normalize_to_ccxt",
    "normalize_to_tradingview",
    "normalize_to_yf",
    "get_base_currency",
    "validate_symbol",
    # Supported exchanges
    "SUPPORTED_EXCHANGES",
]

# Supported exchange names for market tools
SUPPORTED_EXCHANGES: tuple[str, ...] = ("binance", "bybit", "coinbase", "kraken")


@runtime_checkable
class LiveDataAdapter(Protocol):
    """Protocol defining the interface for live data adapters.

    All adapters must implement these methods:
    - fetch_live_price: Get latest trade price
    - fetch_ohlcv_live: Get OHLCV candlestick data
    - fetch_orderbook_snapshot: Get order book depth
    - fetch_ticker_snapshot: Get structured ticker snapshot (ENG-122)
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

    def fetch_ticker_snapshot(
        self,
        symbol: str,
    ) -> dict:
        """Fetch a structured ticker snapshot.

        Args:
            symbol: Symbol in any supported format

        Returns:
            Dict with price, bid, ask, change_pct_24h, volume_24h, timestamp,
            or dict with price=float('nan') and None fields on error
        """
        ...


def get_live_adapter(exchange_name: Literal["binance", "bybit", "coinbase", "kraken"] = "binance") -> LiveDataAdapter:
    """Get the active live-market-data adapter for the specified exchange.

    The live path uses CCXT for live price, OHLCV, and order book retrieval.
    Results are cached per exchange for efficiency.

    Args:
        exchange_name: Target exchange (default: "binance")

    Returns:
        CCXTAdapter instance for exchange-backed live market data.

    Example:
        >>> adapter = get_live_adapter("binance")
        >>> price = adapter.fetch_live_price("BTCUSDT")
    """
    from tempest_mcp.data._factory import get_live_adapter as _get_live_adapter

    return _get_live_adapter(exchange_name=exchange_name)


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
        "get_ohlcv_intake",
        "OhlcvIntake",
        "OhlcvRequest",
        "OhlcvResult",
        "DataSourceRouter",
    ):
        if name in ("HistoricalDataSource", "HistoricalDataAdapter"):
            import tempest_mcp.data._hist as _hist

            return getattr(_hist, name)
        if name in ("OhlcvIntake", "OhlcvRequest", "OhlcvResult"):
            import tempest_mcp.data._intake as _intake

            return getattr(_intake, name)
        if name == "get_ohlcv_intake":
            import tempest_mcp.data._factory as _factory

            return getattr(_factory, name)
        if name == "get_historical_adapter":
            import tempest_mcp.data._factory as _factory

            return getattr(_factory, name)
        if name == "DataSourceRouter":
            import tempest_mcp.data._router as _router

            return getattr(_router, name)
        import tempest_mcp.data._symbols as _symbols

        return getattr(_symbols, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
