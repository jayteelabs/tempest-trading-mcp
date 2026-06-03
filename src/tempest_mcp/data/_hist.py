"""Backward-compatible historical data source wrapper.

New historical OHLCV callers should use :mod:`tempest_mcp.data._intake`.
This module remains for legacy imports that expect ``fetch_ohlcv`` to return a
``(DataFrame, source_used)`` tuple.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

import pandas as pd

from tempest_mcp.data._contracts import empty_ohlcv_frame
from tempest_mcp.data._intake import OhlcvIntake, OhlcvRequest


def _empty_ohlcv() -> pd.DataFrame:
    """Return empty OHLCV DataFrame with canonical columns."""
    return empty_ohlcv_frame()


@runtime_checkable
class HistoricalDataAdapter(Protocol):
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        pass


class HistoricalDataSource:
    """Compatibility wrapper around the historical OHLCV intake seam."""

    _CCXT_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"})
    _YF_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"})

    def __init__(
        self,
        exchange_name: Literal["binance", "bybit", "coinbase", "kraken"] = "binance",
        *,
        intake: OhlcvIntake | None = None,
    ) -> None:
        self.exchange_name = exchange_name
        self._ccxt = None
        self._intake = intake or OhlcvIntake(exchange_name=exchange_name)

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> tuple[pd.DataFrame, str]:
        """Fetch OHLCV data and return the legacy tuple shape.

        ``source_used='empty'`` is internal to the new seam; legacy callers only
        know ``ccxt`` / ``yfinance``, so empty seam results map back to ``ccxt``.
        """
        intake = (
            OhlcvIntake(exchange_name=self.exchange_name, ccxt_adapter=self._ccxt)
            if self._ccxt is not None
            else self._intake
        )
        result = intake.fetch(
            OhlcvRequest(
                symbol=symbol,
                timeframe=interval,
                exchange=self.exchange_name,
                start=start,
                end=end,
                auto_adjust=auto_adjust,
            )
        )
        source_used = "ccxt" if result.source_used == "empty" else result.source_used
        return result.frame, source_used
