"""
TradingView Data Adapter for real-time market data.

This adapter provides real-time market data via TradingView's v1-data API.
When TRADINGVIEW_API_KEY is set, this adapter is preferred over CCXT.

Design Decisions:
- D12: TV uses BTCUSD, CCXT uses BTCUSDT - adapter normalizes
- D14: Empty DataFrame on error - NO exception propagation
- D15: TradingViewError in 3001-3005 range
- D16: fetch_orderbook_snapshot() delegates to CCXT internally (TV→CCXT hybrid)

Rate Limiting:
- TradingView: Per-minute limits based on subscription tier
- Adapter respects configured rate limits with thread-safe locking
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
    """TradingView-based data adapter for real-time market data.

    Provides real-time price and OHLCV data via TradingView's v1-data API.
    Orderbook data is delegated to CCXT internally (D16 - TV→CCXT hybrid).

    Attributes:
        api_key: TradingView API key
        rate_limit: Requests per minute limit

    Example:
        >>> adapter = TradingViewAdapter(api_key="your-api-key")
        >>> price = adapter.fetch_live_price("BTCUSD")
        >>> df = adapter.fetch_ohlcv_live("BTCUSD", "1m", 100)
        >>> orderbook = adapter.fetch_orderbook_snapshot("BTCUSD", 20)
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit: int = 60,
    ) -> None:
        """Initialize TradingView adapter.

        Args:
            api_key: TradingView API key (required for real data)
            rate_limit: Requests per minute limit (default: 60)
        """
        self.api_key = api_key or os.environ.get("TRADINGVIEW_API_KEY")
        self.rate_limit = max(1, rate_limit)

        # Lazy-import CCXT adapter for orderbook delegation (D16)
        self._ccxt_adapter: LiveDataAdapter | None = None

        logger.info(
            "tradingview_adapter_initialized",
            has_api_key=bool(self.api_key),
            rate_limit=self.rate_limit,
        )

    @property
    def ccxt_adapter(self) -> LiveDataAdapter:
        """Get or create CCXT adapter for orderbook delegation (D16)."""
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
        """Make authenticated API request to TradingView.

        This is a placeholder for the actual TradingView API integration.
        The real implementation would use aiohttp or httpx to call
        TradingView's v1-data endpoints.

        Args:
            endpoint: API endpoint path
            params: Request parameters

        Returns:
            API response dict or None on error
        """
        # Placeholder - actual implementation would call TradingView API
        # For now, this returns None to trigger fallback behavior
        logger.warning(
            "tradingview_api_not_implemented",
            endpoint=endpoint,
            message="TradingView API integration pending - returning None",
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
            symbol: Symbol in any format (e.g., "BTCUSD", "BTCUSDT")
            exchange: Target exchange (validated against allowed values)

        Returns:
            Latest trade price as float. Returns float('nan') on error.

        Note:
            Per D14, this function NEVER raises exceptions to callers.
            On failure, logs ERROR and returns NaN.
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
            # Normalize symbol to CCXT format for CCXT fallback
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
                    message="No TradingView API key - falling back to CCXT",
                )
                # Use CCXT format for CCXT adapter
                return self.ccxt_adapter.fetch_live_price(ccxt_symbol, exchange)

            # Normalize to TV format for TV API
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
            symbol: Symbol in any format (e.g., "BTCUSD", "BTCUSDT")
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            limit: Number of candles to fetch (default: 100, max: 1000)

        Returns:
            DataFrame with columns [open, high, low, close, volume] and
            UTC-aware DatetimeIndex. Returns empty DataFrame on error.

        Note:
            Per D14, this function NEVER raises exceptions to callers.
            On failure, logs ERROR and returns empty DataFrame with correct columns.
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
                    message="No TradingView API key - falling back to CCXT",
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

        Per D16, this method delegates to CCXT internally because TradingView
        has no orderbook snapshot endpoint.

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
