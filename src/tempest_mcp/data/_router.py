"""Data source router for historical/live data routing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tempest_mcp.data import HistoricalDataSource, LiveDataAdapter


class DataSourceRouter:
    """Routes data requests to appropriate adapters based on data type.

    Usage:
        router = DataSourceRouter()
        hist_adapter = router.route_historical("binance")   # HistoricalDataSource (CCXT primary, yfinance fallback)
        live_adapter = router.route_live("bybit")          # LiveDataAdapter (CCXT only)
    """

    def route_historical(self, exchange: Literal["binance", "bybit", "coinbase", "kraken"] = "binance") -> HistoricalDataSource:
        """Route to historical data adapter for the specified exchange.

        Args:
            exchange: Target exchange (default: "binance")

        Returns:
            HistoricalDataSource wrapping CCXTAdapter (primary) with yfinance fallback.
        """
        from tempest_mcp.data._factory import get_historical_adapter

        return get_historical_adapter(exchange_name=exchange)

    def route_live(self, exchange: Literal["binance", "bybit", "coinbase", "kraken"] = "binance") -> LiveDataAdapter:
        """Route to live data adapter for the specified exchange.

        Args:
            exchange: Target exchange (default: "binance")

        Returns:
            LiveDataAdapter (CCXTAdapter — TradingView has no OHLCV data API)
        """
        from tempest_mcp.data import LiveDataAdapter, get_live_adapter

        adapter = get_live_adapter(exchange_name=exchange)
        if not isinstance(adapter, LiveDataAdapter):
            raise TypeError(
                f"get_live_adapter() returned {type(adapter).__name__}, expected LiveDataAdapter"
            )
        return adapter
