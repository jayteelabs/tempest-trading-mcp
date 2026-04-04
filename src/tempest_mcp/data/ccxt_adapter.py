"""
CCXT Data Adapter for cryptocurrency market data.

This adapter provides real-time market data via CCXT library, supporting
Binance and other exchanges through their public REST APIs.

Design Decisions:
- D13: Orderbook data via CCXT adapter (TV has no orderbook endpoint)
- D14: Empty DataFrame on error - NO exception propagation
- D15: CCXTError in 3101-3105 range

Rate Limiting (D14):
- Binance: 1200 requests/minute weighted
- Adapter respects exchange-specific rate limits
"""

import threading
import time
import warnings
from typing import Final

import ccxt
import pandas as pd
import structlog

from tempest_mcp.data._symbols import (
    _validate_limit,
    _validate_timeframe,
    normalize_to_ccxt,
    validate_symbol,
)

logger = structlog.get_logger()

# OHLCV column names
OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]

# Rate limit tracking with thread-safe lock
_rate_limit_tracker: dict[str, float] = {}
_rate_limit_lock = threading.Lock()


class CCXTAdapter:
    """CCXT-based data adapter for cryptocurrency market data.

    Provides real-time price, OHLCV, and orderbook data via CCXT library.
    Uses public REST APIs - no API keys required for read operations.

    Attributes:
        exchange: CCXT exchange instance
        exchange_name: Name of the exchange (default: "binance")

    Example:
        >>> adapter = CCXTAdapter()
        >>> price = adapter.fetch_live_price("BTCUSDT")
        >>> df = adapter.fetch_ohlcv_live("BTCUSDT", "1m", 100)
        >>> orderbook = adapter.fetch_orderbook_snapshot("BTCUSDT", 20)
    """

    def __init__(self, exchange_name: str = "binance") -> None:
        """Initialize CCXT adapter.

        Args:
            exchange_name: Exchange to use (default: "binance")
        """
        self.exchange_name = exchange_name.lower()

        # Get exchange class
        exchange_classes = {
            "binance": ccxt.binance,
            "bybit": ccxt.bybit,
            "kraken": ccxt.kraken,
            "coinbase": ccxt.coinbase,
        }

        exchange_class = exchange_classes.get(self.exchange_name, ccxt.binance)
        self.exchange = exchange_class(
            {
                "enableRateLimit": True,
                "options": {
                    "defaultType": "spot",
                },
            }
        )

        logger.info(
            "ccxt_adapter_initialized",
            exchange=self.exchange_name,
        )

    def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting (thread-safe)."""
        current_time = time.time()
        min_interval = 0.05

        with _rate_limit_lock:
            last_request = _rate_limit_tracker.get(self.exchange_name, 0)
            if current_time - last_request < min_interval:
                time.sleep(min_interval - (current_time - last_request))
            _rate_limit_tracker[self.exchange_name] = time.time()

    def fetch_live_price(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> float:
        """Fetch the latest trade price for a symbol.

        Args:
            symbol: Symbol in any format (e.g., "BTCUSD", "BTCUSDT")
            exchange: Target exchange (default: "binance").
                Note: Multi-exchange support requires creating separate
                CCXTAdapter instances. The exchange param is deprecated
                and will be removed in a future version.

        Returns:
            Latest trade price as float. Returns float('nan') on error.

        Note:
            Per D14, this function NEVER raises exceptions to callers.
            On failure, logs ERROR and returns NaN.

        Example:
            >>> adapter = CCXTAdapter()
            >>> price = adapter.fetch_live_price("BTCUSDT")
            >>> isinstance(price, float)
            True
        """
        # Warn if exchange param differs from adapter's exchange (future removal)
        if exchange != self.exchange_name:
            warnings.warn(
                f"The 'exchange' parameter is deprecated and will be ignored. "
                f"This CCXTAdapter instance is configured for '{self.exchange_name}'. "
                f"For a different exchange, create a separate CCXTAdapter instance: "
                f"CCXTAdapter(exchange_name='{exchange}')",
                DeprecationWarning,
                stacklevel=2,
            )

        try:
            # Normalize symbol to CCXT format
            ccxt_symbol = normalize_to_ccxt(symbol)

            if not validate_symbol(ccxt_symbol):
                logger.error(
                    "invalid_symbol",
                    symbol=symbol,
                    ccxt_symbol=ccxt_symbol,
                )
                return float("nan")

            self._check_rate_limit()

            # Fetch ticker from exchange
            ticker = self.exchange.fetch_ticker(ccxt_symbol)
            price = float(ticker["last"])

            logger.info(
                "fetch_live_price_success",
                source="ccxt",
                exchange=self.exchange_name,
                symbol=ccxt_symbol,
                price=price,
            )

            return price

        except ccxt.NetworkError as e:
            logger.error(
                "fetch_live_price_network_error",
                source="ccxt",
                symbol=symbol,
                error=str(e),
            )
            return float("nan")

        except ccxt.ExchangeError as e:
            logger.error(
                "fetch_live_price_exchange_error",
                source="ccxt",
                symbol=symbol,
                error=str(e),
            )
            return float("nan")

        except Exception as e:
            logger.error(
                "fetch_live_price_unexpected_error",
                source="ccxt",
                symbol=symbol,
                error=str(e),
            )
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
            limit: Number of candles to fetch (default: 100)

        Returns:
            DataFrame with columns [open, high, low, close, volume] and
            UTC-aware DatetimeIndex. Returns empty DataFrame on error.

        Note:
            Per D14, this function NEVER raises exceptions to callers.
            On failure, logs ERROR and returns empty DataFrame with correct columns.

        Example:
            >>> adapter = CCXTAdapter()
            >>> df = adapter.fetch_ohlcv_live("BTCUSDT", "1m", 100)
            >>> list(df.columns)
            ['open', 'high', 'low', 'close', 'volume']
        """
        # Validate timeframe
        if not _validate_timeframe(timeframe):
            logger.error(
                "invalid_timeframe",
                timeframe=timeframe,
                supported=sorted({"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"}),
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Validate and clamp limit
        limit = _validate_limit(limit)

        try:
            # Normalize symbol to CCXT format
            ccxt_symbol = normalize_to_ccxt(symbol)

            if not validate_symbol(ccxt_symbol):
                logger.error(
                    "invalid_symbol",
                    symbol=symbol,
                    ccxt_symbol=ccxt_symbol,
                )
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            self._check_rate_limit()

            # Fetch OHLCV from exchange
            # CCXT returns: [[timestamp, open, high, low, close, volume], ...]
            ohlcv_data = self.exchange.fetch_ohlcv(
                ccxt_symbol,
                timeframe=timeframe,
                limit=limit,
            )

            if not ohlcv_data:
                logger.error(
                    "fetch_ohlcv_empty_response",
                    source="ccxt",
                    symbol=ccxt_symbol,
                    timeframe=timeframe,
                )
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )

            # Set UTC-aware index
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

            # Select only OHLCV columns
            df = df[OHLCV_COLUMNS]

            logger.info(
                "fetch_ohlcv_success",
                source="ccxt",
                exchange=self.exchange_name,
                symbol=ccxt_symbol,
                timeframe=timeframe,
                rows=len(df),
            )

            return df

        except ccxt.NetworkError as e:
            logger.error(
                "fetch_ohlcv_network_error",
                source="ccxt",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        except ccxt.ExchangeError as e:
            logger.error(
                "fetch_ohlcv_exchange_error",
                source="ccxt",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        except Exception as e:
            logger.error(
                "fetch_ohlcv_unexpected_error",
                source="ccxt",
                symbol=symbol,
                timeframe=timeframe,
                error=str(e),
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

    def fetch_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict:
        """Fetch order book snapshot for a symbol.

        Args:
            symbol: Symbol in any format (e.g., "BTCUSD", "BTCUSDT")
            limit: Depth of orderbook to fetch (default: 20)

        Returns:
            Dict with keys:
            - bids: List of [price, amount] pairs, sorted by price desc
            - asks: List of [price, amount] pairs, sorted by price asc
            - timestamp: UTC-aware pandas Timestamp

        Note:
            Per D14, this function NEVER raises exceptions to callers.
            On failure, logs ERROR and returns empty bids/asks with None timestamp.

        Example:
            >>> adapter = CCXTAdapter()
            >>> ob = adapter.fetch_orderbook_snapshot("BTCUSDT", 20)
            >>> "bids" in ob and "asks" in ob
            True
        """
        # Validate and clamp limit
        limit = _validate_limit(limit)

        try:
            # Normalize symbol to CCXT format
            ccxt_symbol = normalize_to_ccxt(symbol)

            if not validate_symbol(ccxt_symbol):
                logger.error(
                    "invalid_symbol",
                    symbol=symbol,
                    ccxt_symbol=ccxt_symbol,
                )
                return {
                    "bids": [],
                    "asks": [],
                    "timestamp": None,
                }

            self._check_rate_limit()

            # Fetch orderbook from exchange
            orderbook = self.exchange.fetch_order_book(ccxt_symbol, limit=limit)

            result = {
                "bids": orderbook.get("bids", [])[:limit],
                "asks": orderbook.get("asks", [])[:limit],
                "timestamp": pd.Timestamp.now(tz="UTC"),
            }

            logger.info(
                "fetch_orderbook_success",
                source="ccxt",
                exchange=self.exchange_name,
                symbol=ccxt_symbol,
                bid_depth=len(result["bids"]),
                ask_depth=len(result["asks"]),
            )

            return result

        except ccxt.NetworkError as e:
            logger.error(
                "fetch_orderbook_network_error",
                source="ccxt",
                symbol=symbol,
                error=str(e),
            )
            return {
                "bids": [],
                "asks": [],
                "timestamp": None,
            }

        except ccxt.ExchangeError as e:
            logger.error(
                "fetch_orderbook_exchange_error",
                source="ccxt",
                symbol=symbol,
                error=str(e),
            )
            return {
                "bids": [],
                "asks": [],
                "timestamp": None,
            }

        except Exception as e:
            logger.error(
                "fetch_orderbook_unexpected_error",
                source="ccxt",
                symbol=symbol,
                error=str(e),
            )
            return {
                "bids": [],
                "asks": [],
                "timestamp": None,
            }
