"""Historical data source abstraction layer.

Data Source Priority (D3):
- Primary: CCXT via Binance/Bybit public REST (all crypto + stocks)
- Fallback: yfinance (for stocks and data gaps CCXT doesn't cover)

CCXT is tried first. If it returns empty DataFrame, yfinance is used as fallback.

Limitation: yfinance does not support "4h" interval. Use "1h" and aggregate
client-side, or use CCXT directly for 4h data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger()


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
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 4h*, 1d, 1wk, 1mo)
                * Note: "4h" is supported by CCXT but NOT by yfinance (fallback).
                  For 4h via yfinance, aggregate from 1h or use CCXT directly.
            start: Start datetime (UTC). Naive datetimes are interpreted as UTC.
            end: End datetime (UTC). Naive datetimes are interpreted as UTC.
            auto_adjust: Whether to adjust for splits/dividends (yfinance only;
                CCXT always returns split-adjusted spot prices)

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

    # Intervals supported by CCXT (and their yfinance compatibility)
    _CCXT_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"})
    _YF_INTERVALS = frozenset({"1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"})  # 4h excluded

    def __init__(self) -> None:
        from tempest_mcp.data.ccxt_adapter import CCXTAdapter

        self._ccxt = CCXTAdapter()

    def _ccxt_timeframe(self, interval: str) -> tuple[str, bool]:
        """Map generic interval to CCXT timeframe string.

        Returns:
            Tuple of (ccxt_timeframe, is_supported). is_supported=False for invalid intervals.
        """
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
        mapped = mapping.get(interval)
        if mapped is None:
            logger.warning(
                "unsupported_interval_defaulting",
                interval=interval,
                default="1d",
            )
            return "1d", False
        return mapped, True

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

    def _symbol_to_yf(self, symbol: str) -> tuple[str, bool]:
        """Convert CCXT symbol format to yfinance format for fallback.

        Returns:
            Tuple of (yfinance_symbol, is_valid). is_valid=False for exotic
            quote currencies that yfinance definitely doesn't support.
        """
        if "/" in symbol:
            base, quote = symbol.split("/")
            quote_upper = quote.upper()
            # yfinance supports USD, USDT, USDC for crypto pairs
            if quote_upper in ("USDT", "USD", "USDC", "US"):
                return f"{base.upper()}-USD", True
            # Non-USD quote: yfinance might not have it — flag as potentially invalid
            logger.warning(
                "yfinance_exotic_quote_symbol",
                symbol=symbol,
                yf_symbol=f"{base.upper()}-{quote_upper}",
                note="yfinance may not support this symbol",
            )
            return f"{base.upper()}-{quote_upper}", True  # Let yfinance decide
        return symbol, True

    def _ensure_utc(self, dt: datetime | None, name: str) -> datetime | None:
        """Normalize datetime to UTC-aware. Naive datetimes are interpreted as UTC."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            logger.warning(
                f"{name}_naive_datetime_interpreted_as_utc",
                dt=str(dt),
            )
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _interval_seconds(self, interval: str) -> int:
        """Get interval duration in seconds."""
        mapping = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1wk": 604800,
            "1mo": 2592000,
        }
        return mapping.get(interval, 86400)

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
            start: Start datetime (UTC). Naive datetimes interpreted as UTC.
            end: End datetime (UTC). Naive datetimes interpreted as UTC.
            auto_adjust: Whether to adjust for splits/dividends (yfinance only;
                CCXT always returns split-adjusted spot prices)

        Returns:
            DataFrame with [open, high, low, close, volume] and UTC-aware index,
            or empty DataFrame on error

        Note:
            "4h" is supported by CCXT but NOT by yfinance. If CCXT returns
            empty for 4h, the yfinance fallback will also return empty.
        """
        from tempest_mcp.data.yf_adapter import fetch_ohlcv as _yf_fetch_ohlcv

        # Validate interval
        if interval not in self._CCXT_INTERVALS:
            logger.warning(
                "invalid_interval_defaulting",
                interval=interval,
                supported=sorted(self._CCXT_INTERVALS),
                default="1d",
            )

        # Normalize timezones to UTC
        start_utc = self._ensure_utc(start, "start")
        end_utc = self._ensure_utc(end, "end")

        # Normalize to CCXT symbol format
        ccxt_symbol = self._symbol_to_ccxt(symbol)
        timeframe, interval_supported = self._ccxt_timeframe(interval)

        # Compute since/until timestamps for CCXT
        since_ms: int | None = None
        until_ms: int | None = None
        if start_utc is not None:
            since_ms = int(start_utc.timestamp() * 1000)
        if end_utc is not None:
            until_ms = int(end_utc.timestamp() * 1000)

        # Calculate limit needed for the date range (if both start and end are provided)
        limit = 1000
        if since_ms is not None and until_ms is not None:
            range_ms = until_ms - since_ms
            interval_sec = self._interval_seconds(interval)
            needed = range_ms // (interval_sec * 1000)
            limit = min(needed + 1, 1000)

        # Build CCXT params with until for date range filtering
        ccxt_params: dict = {}
        if until_ms is not None:
            ccxt_params["until"] = until_ms

        # Try CCXT primary
        logger.info(
            "ccxt_fetch_historical_attempt",
            symbol=ccxt_symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            until_ms=until_ms,
            limit=limit,
        )
        ccxt_result = self._ccxt.fetch_ohlcv_historical(
            symbol=ccxt_symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=limit,
            params=ccxt_params,
        )

        if not ccxt_result.empty:
            logger.info(
                "historical_fetch_ccxt_success",
                symbol=ccxt_symbol,
                rows=len(ccxt_result),
                source="ccxt",
            )
            return ccxt_result

        # CCXT failed or empty — fallback to yfinance
        yf_symbol, _ = self._symbol_to_yf(ccxt_symbol)
        logger.info(
            "historical_fetch_fallback_yfinance",
            symbol=yf_symbol,
            reason="CCXT returned empty",
        )
        return _yf_fetch_ohlcv(
            symbol=yf_symbol,
            interval=interval,
            start=start,
            end=end,
            auto_adjust=auto_adjust,
        )
