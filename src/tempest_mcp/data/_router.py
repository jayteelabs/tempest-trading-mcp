"""Data source router for historical/live data routing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tempest_mcp.data._factory import get_historical_adapter

if TYPE_CHECKING:
    from tempest_mcp.data import HistoricalDataSource, LiveDataAdapter


class DataSourceRouter:
    """Routes data requests to appropriate adapters based on data type.

    Usage:
        router = DataSourceRouter()
        hist_adapter = router.route_historical()   # HistoricalDataSource (CCXT primary, yfinance fallback)
        live_adapter = router.route_live()          # LiveDataAdapter (CCXT only)
    """

    def route_historical(self) -> HistoricalDataSource:
        """Route to historical data adapter.

        Returns:
            HistoricalDataSource wrapping CCXTAdapter (primary) with yfinance fallback.
        """
        return get_historical_adapter()

    def route_live(self) -> LiveDataAdapter:
        """Route to live data adapter.

        Returns:
            LiveDataAdapter (CCXTAdapter — TradingView has no OHLCV data API)
        """
        from tempest_mcp.data import LiveDataAdapter, get_live_adapter

        adapter = get_live_adapter()
        if not isinstance(adapter, LiveDataAdapter):
            raise TypeError(
                f"get_live_adapter() returned {type(adapter).__name__}, expected LiveDataAdapter"
            )
        return adapter
