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
        hist_adapter = router.route_historical()   # HistoricalDataSource
        live_adapter = router.route_live()          # LiveDataAdapter
    """

    def route_historical(self) -> HistoricalDataSource:
        """Route to historical data adapter.

        Returns:
            HistoricalDataSource wrapping YFAdapter
        """
        return get_historical_adapter()

    def route_live(self) -> LiveDataAdapter:
        """Route to live data adapter.

        Returns:
            LiveDataAdapter (TradingViewAdapter if API key set, else CCXTAdapter)
        """
        from tempest_mcp.data import get_live_adapter

        return get_live_adapter()
