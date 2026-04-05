"""Historical data source abstraction layer."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import pandas as pd


@runtime_checkable
class HistoricalDataAdapter(Protocol):
    """Protocol defining the interface for historical data adapters.

    All adapters must implement:
    - fetch_ohlcv: Get historical OHLCV candlestick data
    """

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbol: Asset symbol in yfinance format (e.g., BTC-USD, ETH-USD)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            auto_adjust: Whether to adjust for splits/dividends

        Returns:
            DataFrame with [open, high, low, close, volume] and UTC-aware index,
            or empty DataFrame on error
        """
        ...


class HistoricalDataSource:
    """Wraps YFAdapter to conform to HistoricalDataAdapter protocol.

    This class wraps the existing YFAdapter and exposes fetch_ohlcv()
    to conform to the HistoricalDataAdapter protocol.
    """

    def __init__(self) -> None:
        from tempest_mcp.data.yf_adapter import YFAdapter
        from tempest_mcp.data.yf_adapter import fetch_ohlcv as _fetch_ohlcv

        self._yf_adapter = YFAdapter()
        self._fetch_ohlcv = _fetch_ohlcv

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data via YFAdapter.

        Args:
            symbol: Asset symbol in yfinance format (e.g., BTC-USD, ETH-USD)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            auto_adjust: Whether to adjust for splits/dividends

        Returns:
            DataFrame with [open, high, low, close, volume] and UTC-aware index,
            or empty DataFrame on error
        """
        return self._fetch_ohlcv(
            symbol=symbol,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )
