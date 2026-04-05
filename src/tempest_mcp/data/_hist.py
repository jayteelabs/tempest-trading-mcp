"""Historical data source abstraction layer.

Data Source Priority (D3):
- Primary: CCXT via Binance/Bybit public REST (all crypto + stocks)
- Fallback: yfinance (for stocks and data gaps CCXT doesn't cover)

CCXT is tried first. If it returns empty DataFrame, yfinance is used as fallback.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
            symbol: Asset symbol in any supported format (CCXT or yfinance native)
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
    """Primary CCXT + fallback yfinance historical data source.

    This class wraps CCXT as the primary historical data adapter with
    automatic fallback to yfinance when CCXT fails or returns empty data.

    Data Source Priority (D3):
    - Primary: CCXT via Binance/Bybit public REST (no API keys required)
    - Fallback: yfinance (for stocks and data gaps CCXT doesn't cover)

    Usage:
        >>> source = HistoricalDataSource()
        >>> df = source.fetch_ohlcv("BTC/USDT", interval="1d", start=start, end=end)
    """

    def __init__(self) -> None:
        from tempest_mcp.data.ccxt_adapter import CCXTAdapter

        self._ccxt = CCXTAdapter()

    def _ccxt_timeframe(self, interval: str) -> str:
        """Map generic interval to CCXT timeframe string."""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "4h": "4h",
            "1d": "1d",
            "1wk": "1w",
            "1mo": "1M",
        }
        return mapping.get(interval, "1d")

    def _symbol_to_ccxt(self, symbol: str) -> str:
        """Normalize symbol to CCXT format (e.g. BTC-USD -> BTC/USDT)."""
        if "/" in symbol:
            return symbol.upper()
        if "-" in symbol and symbol.endswith("-USD"):
            base = symbol.replace("-USD", "")
            return f"{base.upper()}/USDT"
        if "-" in symbol:
            parts = symbol.split("-")
            return f"{parts[0].upper()}/{parts[1].upper()}"
        return symbol.upper()

    def _symbol_to_yf(self, symbol: str) -> str:
        """Convert CCXT symbol format to yfinance format for fallback."""
        if "/" in symbol:
            base, quote = symbol.split("/")
            if quote.upper() in ("USDT", "USD", "USDC", "US"):
                return f"{base.upper()}-USD"
            return f"{base.upper()}-{quote.upper()}"
        return symbol

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data via CCXT primary, yfinance fallback.

        Tries CCXT first (supports Binance/Bybit public REST). On failure
        or empty result, falls back to yfinance for stocks and data gaps.

        Args:
            symbol: Asset symbol (CCXT format preferred, e.g. BTC/USDT)
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo)
            start: Start datetime (UTC)
            end: End datetime (UTC)
            auto_adjust: Whether to adjust for splits/dividends (yfinance only)

        Returns:
            DataFrame with [open, high, low, close, volume] and UTC-aware index,
            or empty DataFrame on error
        """
        from tempest_mcp.data.yf_adapter import fetch_ohlcv as _yf_fetch_ohlcv

        # Normalize to CCXT symbol format
        ccxt_symbol = self._symbol_to_ccxt(symbol)
        timeframe = self._ccxt_timeframe(interval)
        limit = 1000  # CCXT max

        # Compute since timestamp for CCXT
        since_ms: int | None = None
        if start is not None:
            aware_start = start
            if aware_start.tzinfo is None:
                aware_start = aware_start.replace(tzinfo=timezone.utc)
            since_ms = int(aware_start.timestamp() * 1000)

        # Try CCXT primary
        ccxt_result = self._ccxt.fetch_ohlcv_historical(
            symbol=ccxt_symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit,
        )

        if not ccxt_result.empty:
            return ccxt_result

        # CCXT failed or empty — fallback to yfinance
        yf_symbol = self._symbol_to_yf(ccxt_symbol)
        return _yf_fetch_ohlcv(
            symbol=yf_symbol,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )
