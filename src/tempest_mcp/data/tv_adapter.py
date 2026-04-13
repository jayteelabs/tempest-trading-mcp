"""Deprecated TradingView compatibility shim.

This module is intentionally not part of the active data architecture.

Important:
- There is no official TradingView OHLCV data API in use by this project
- ``TRADINGVIEW_API_KEY`` does not activate a real TradingView market-data path
- New code must use ``CCXTAdapter`` for live/exchange data and ``YFAdapter`` or
  ``HistoricalDataSource`` for historical fallback behavior

Why this file still exists:
- Backward compatibility for older imports and legacy call sites
- Explicit migration surface while the project transitions away from
  TradingView-first assumptions

Active source priority (D3):
- Primary market/live data: CCXT via Binance/Bybit public REST
- Historical fallback: yfinance
- Orderbook: CCXT only
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from tempest_mcp.data import LiveDataAdapter

import pandas as pd
import structlog

from tempest_mcp.data._symbols import (
    _validate_limit,
    _validate_timeframe,
    normalize_to_ccxt,
    normalize_to_tradingview,
    validate_symbol,
)
from tempest_mcp.errors import TradingViewError

logger = structlog.get_logger()

# OHLCV column names
OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]

# Allowed exchange values for validation
ALLOWED_EXCHANGES: frozenset[str] = frozenset({"binance", "bybit", "coinbase", "kraken"})

# Rate limit tracking with thread-safe lock
_rate_limit_tracker: dict[str, float] = {}
_rate_limit_lock = threading.Lock()


class TradingViewAdapter:
    """Deprecated compatibility adapter that forwards behavior to CCXT.

    This class remains import-compatible for legacy callers but should not be
    treated as a real TradingView integration. In practice, successful paths in
    this shim degrade to CCXT-backed behavior.

    Attributes:
        api_key: Legacy TradingView API key field retained for compatibility
        rate_limit: Legacy per-minute limit retained for compatibility behavior

    Example:
        >>> adapter = TradingViewAdapter(api_key="unused-legacy-key")
        >>> price = adapter.fetch_live_price("BTCUSD")
        >>> df = adapter.fetch_ohlcv_live("BTCUSD", "1m", 100)
        >>> orderbook = adapter.fetch_orderbook_snapshot("BTCUSD", 20)
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: int = 60,
    ) -> None:
        """Initialize the deprecated compatibility adapter.

        Args:
            api_key: Legacy field retained for compatibility; not required for
                any active market-data source
            rate_limit: Compatibility-only per-minute throttle value
        """
        self.api_key = api_key or os.environ.get("TRADINGVIEW_API_KEY")
        self.rate_limit = max(1, rate_limit)

        # Lazy-import CCXT adapter for orderbook delegation (D16)
        self._ccxt_adapter: LiveDataAdapter | None = None

        logger.info(
            "tradingview_adapter_initialized",
            has_api_key=bool(self.api_key),
            rate_limit=self.rate_limit,
            deprecated=True,
            behavior="compatibility_shim",
        )

    @property
    def ccxt_adapter(self) -> LiveDataAdapter:
        """Get or create the underlying CCXT adapter used by this shim."""
        if self._ccxt_adapter is None:
            # Lazy import to avoid circular dependency
            from tempest_mcp.data.ccxt_adapter import CCXTAdapter

            self._ccxt_adapter = CCXTAdapter()
        return self._ccxt_adapter

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting (thread-safe)."""
        current_time = time.time()
        min_interval = 60.0 / self.rate_limit

        with _rate_limit_lock:
            last_request = _rate_limit_tracker.get("tradingview", 0)
            if current_time - last_request < min_interval:
                time.sleep(min_interval - (current_time - last_request))
            _rate_limit_tracker["tradingview"] = time.time()

    def _make_api_request(
        self,
        endpoint: str,
        params: dict,
    ) -> dict | None:
        """Placeholder for legacy TradingView request behavior.

        No real TradingView API integration is implemented. Returning ``None``
        keeps this shim on the documented fallback path to CCXT.

        Args:
            endpoint: API endpoint path
            params: Request parameters

        Returns:
            Always ``None`` so callers fall back to CCXT-backed behavior.
        """
        # Intentionally unimplemented: this project does not use TradingView as
        # an active data provider.
        logger.warning(
            "tradingview_api_not_implemented",
            endpoint=endpoint,
            message="TradingView integration is deprecated; using CCXT fallback path",
        )
        return None

    def _sanitize_for_log(self, value: str) -> str:
        """Strip control characters to prevent log injection."""
        return "".join(c for c in value if c.isprintable() or c in "\t\n")

    def fetch_live_price(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> float:
        """Fetch the latest trade price for a symbol.

        Args:
            symbol: Symbol in any compatible format (e.g., "BTCUSD", "BTCUSDT")
            exchange: Target exchange (validated against allowed values)

        Returns:
            Latest trade price as float via compatibility fallback.
            Returns ``float('nan')`` on error.

        Note:
            New code should call ``CCXTAdapter.fetch_live_price()`` directly.
        """
        # Validate exchange param
        safe_exchange = self._sanitize_for_log(exchange)
        if safe_exchange not in ALLOWED_EXCHANGES:
            logger.error(
                "invalid_exchange_param",
                exchange=safe_exchange,
                allowed=list(ALLOWED_EXCHANGES),
            )
            return float("nan")

        try:
            # Normalize symbol into the active CCXT canonical format first.
            ccxt_symbol = normalize_to_ccxt(symbol)

            if not validate_symbol(symbol):
                logger.error(
                    "invalid_symbol",
                    symbol=self._sanitize_for_log(symbol),
                    ccxt_symbol=ccxt_symbol,
                )
                return float("nan")

            self._check_rate_limit()

            if not self.api_key:
                logger.warning(
                    "tradingview_no_api_key",
                    message="Deprecated TradingView shim using CCXT fallback",
                )
                return self.ccxt_adapter.fetch_live_price(ccxt_symbol, exchange)

            # Legacy alias normalization retained only for compatibility logs.
            tv_symbol = normalize_to_tradingview(symbol)

            # TradingView API: v1-data Multiple symbols real-time bars
            response = self._make_api_request(
                "v1-data/multiple-symbols-real-time-bars",
                {"symbols": tv_symbol, "exchange": exchange},
            )

            if response is None:
                logger.info(
                    "tradingview_fallback_to_ccxt",
                    symbol=tv_symbol,
                    reason="API returned None",
                )
                return self.ccxt_adapter.fetch_live_price(ccxt_symbol, exchange)

            # Extract price from response
            price = float("nan")

            logger.info(
                "fetch_live_price_success",
                source="tradingview",
                symbol=tv_symbol,
                price=price,
            )

            return price

        except TradingViewError as e:
            logger.error(
                "fetch_live_price_tradingview_error",
                source="tradingview",
                symbol=self._sanitize_for_log(symbol),
                error=str(e),
                code=e.code,
            )
            ccxt_symbol = normalize_to_ccxt(symbol)
            try:
                return self.ccxt_adapter.fetch_live_price(ccxt_symbol, exchange)
            except Exception:
                return float("nan")

        except Exception as e:
            logger.error(
                "fetch_live_price_unexpected_error",
                source="tradingview",
                symbol=self._sanitize_for_log(symbol),
                error=str(e),
            )
            ccxt_symbol = normalize_to_ccxt(symbol)
            try:
                return self.ccxt_adapter.fetch_live_price(ccxt_symbol, exchange)
            except Exception:
                return float("nan")

    def fetch_ohlcv_live(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch OHLCV candlestick data for a symbol.

        Args:
            symbol: Symbol in any compatible format (e.g., "BTCUSD", "BTCUSDT")
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles to fetch (default: 100, max: 1000)

        Returns:
            DataFrame with columns [open, high, low, close, volume] and
            UTC-aware DatetimeIndex. Returns empty DataFrame on error.

        Note:
            New code should call ``CCXTAdapter.fetch_ohlcv_live()`` directly.
        """
        # Validate timeframe
        if not _validate_timeframe(timeframe):
            logger.error(
                "invalid_timeframe",
                timeframe=self._sanitize_for_log(timeframe),
                supported=sorted(
                    _validate_timeframe.__doc__.split() if _validate_timeframe.__doc__ else []
                ),
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Validate and clamp limit
        limit = _validate_limit(limit)

        try:
            ccxt_symbol = normalize_to_ccxt(symbol)

            if not validate_symbol(symbol):
                logger.error(
                    "invalid_symbol",
                    symbol=self._sanitize_for_log(symbol),
                    ccxt_symbol=ccxt_symbol,
                )
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            self._check_rate_limit()

            if not self.api_key:
                logger.warning(
                    "tradingview_no_api_key",
                    message="Deprecated TradingView shim using CCXT fallback",
                )
                return self.ccxt_adapter.fetch_ohlcv_live(ccxt_symbol, timeframe, limit)

            tv_symbol = normalize_to_tradingview(symbol)

            response = self._make_api_request(
                "v1-data/symbol-real-time-bars",
                {"symbol": tv_symbol, "timeframe": timeframe, "limit": limit},
            )

            if response is None:
                logger.info(
                    "tradingview_fallback_to_ccxt",
                    symbol=tv_symbol,
                    timeframe=timeframe,
                    reason="API returned None",
                )
                return self.ccxt_adapter.fetch_ohlcv_live(ccxt_symbol, timeframe, limit)

            df = pd.DataFrame(columns=OHLCV_COLUMNS)

            logger.info(
                "fetch_ohlcv_success",
                source="tradingview",
                symbol=tv_symbol,
                timeframe=timeframe,
                rows=len(df),
            )

            return df

        except TradingViewError as e:
            logger.error(
                "fetch_ohlcv_tradingview_error",
                source="tradingview",
                symbol=self._sanitize_for_log(symbol),
                timeframe=timeframe,
                error=str(e),
                code=e.code,
            )
            ccxt_symbol = normalize_to_ccxt(symbol)
            try:
                return self.ccxt_adapter.fetch_ohlcv_live(ccxt_symbol, timeframe, limit)
            except Exception:
                return pd.DataFrame(columns=OHLCV_COLUMNS)

        except Exception as e:
            logger.error(
                "fetch_ohlcv_unexpected_error",
                source="tradingview",
                symbol=self._sanitize_for_log(symbol),
                timeframe=timeframe,
                error=str(e),
            )
            ccxt_symbol = normalize_to_ccxt(symbol)
            try:
                return self.ccxt_adapter.fetch_ohlcv_live(ccxt_symbol, timeframe, limit)
            except Exception:
                return pd.DataFrame(columns=OHLCV_COLUMNS)

    def fetch_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict:
        """Fetch order book snapshot for a symbol.

        TradingView is not an active order book source. This compatibility shim
        delegates directly to CCXT.

        Args:
            symbol: Symbol in any format (e.g., "BTCUSD", "BTCUSDT")
            limit: Depth of orderbook to fetch (default: 20, max: 1000)

        Returns:
            Dict with keys:
            - bids: List of [price, amount] pairs, sorted by price desc
            - asks: List of [price, amount] pairs, sorted by price asc
            - timestamp: UTC-aware pandas Timestamp

        Note:
            Per D14, this function NEVER raises exceptions to callers.
            On failure, logs ERROR and returns empty bids/asks with None timestamp.
        """
        limit = _validate_limit(limit)

        logger.info(
            "tradingview_delegating_orderbook_to_ccxt",
            symbol=self._sanitize_for_log(symbol),
            limit=limit,
            design_decision="D16",
        )

        # D16: TV→CCXT hybrid - delegate to CCXT for orderbook
        # Wrap in try/except to enforce D14 (no exception propagation)
        try:
            ccxt_symbol = normalize_to_ccxt(symbol)
            return self.ccxt_adapter.fetch_orderbook_snapshot(ccxt_symbol, limit)
        except Exception as e:
            logger.error(
                "fetch_orderbook_snapshot_error",
                source="ccxt_delegation",
                symbol=self._sanitize_for_log(symbol),
                error=str(e),
            )
            return {
                "bids": [],
                "asks": [],
                "timestamp": None,
            }
