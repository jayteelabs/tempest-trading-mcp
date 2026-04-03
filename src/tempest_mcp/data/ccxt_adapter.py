"""CCXT data adapter for real-time market data. NO API KEYS REQUIRED."""

from dataclasses import dataclass
from datetime import datetime
import ccxt
import numpy as np
from tempest_mcp.config import ErrorCodes, get_config
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.market import Kline, KlineData, OrderBook, OrderBookLevel, Ticker

logger = get_logger(__name__)


class CCXTError(Exception):
    def __init__(self, message: str, code: int = ErrorCodes.CCXT_ERROR):
        super().__init__(message)
        self.code = code
        self.message = message


SUPPORTED_EXCHANGES = ("binance", "bybit")


@dataclass
class CCXTAdapter:
    exchange: str = "binance"
    timeout: int = 30
    _exchange_instance: ccxt.Exchange | None = None

    def __post_init__(self):
        config = get_config()
        if self.exchange == "binance":
            self.exchange = config.default_exchange
        if self.timeout == 30:
            self.timeout = config.ccxt_timeout
        if self.exchange.lower() not in SUPPORTED_EXCHANGES:
            raise CCXTError(f"Unsupported exchange: {self.exchange}. Supported: {SUPPORTED_EXCHANGES}", code=ErrorCodes.INVALID_EXCHANGE)
        self._create_exchange()

    def _create_exchange(self) -> None:
        exchange_class = getattr(ccxt, self.exchange.lower())
        self._exchange_instance = exchange_class({"enableRateLimit": True, "timeout": self.timeout * 1000, "options": {"defaultType": "spot"}})

    @property
    def client(self) -> ccxt.Exchange:
        if self._exchange_instance is None:
            self._create_exchange()
        return self._exchange_instance

    def fetch_ticker(self, symbol: str) -> Ticker:
        try:
            logger.debug("Fetching ticker", symbol=symbol, exchange=self.exchange)
            ticker_data = self.client.fetch_ticker(symbol)
            return Ticker(
                symbol=symbol, exchange=self.exchange, price=float(ticker_data.get("last", 0)),
                timestamp=float(ticker_data.get("timestamp", 0) / 1000),
                volume_24h=float(ticker_data.get("baseVolume", 0)), high_24h=float(ticker_data.get("high", 0)),
                low_24h=float(ticker_data.get("low", 0)), change_percent_24h=float(ticker_data.get("percentage", 0)),
                bid=float(ticker_data.get("bid", 0)), ask=float(ticker_data.get("ask", 0)),
            )
        except ccxt.BadSymbol as e:
            raise CCXTError(f"Invalid symbol: {symbol}", code=ErrorCodes.INVALID_SYMBOL)
        except ccxt.NetworkError as e:
            logger.error("CCXT network error", symbol=symbol, error=str(e))
            raise CCXTError(f"Network error fetching {symbol}: {e}", code=ErrorCodes.NETWORK_ERROR)
        except Exception as e:
            logger.error("CCXT ticker fetch failed", symbol=symbol, error=str(e))
            raise CCXTError(f"Failed to fetch ticker for {symbol}: {e}", code=ErrorCodes.CCXT_ERROR)

    def fetch_klines(self, symbol: str, timeframe: str = "1h", since: datetime | None = None, limit: int = 100) -> KlineData:
        try:
            since_ms = int(since.timestamp() * 1000) if since else None
            logger.debug("Fetching klines", symbol=symbol, timeframe=timeframe, limit=limit, exchange=self.exchange)
            ohlcv = self.client.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=min(limit, 1000))
            if not ohlcv:
                raise CCXTError(f"No kline data found for {symbol}", code=ErrorCodes.DATA_NOT_FOUND)
            klines = [Kline(timestamp=c[0] / 1000, open=float(c[1]), high=float(c[2]), low=float(c[3]), close=float(c[4]), volume=float(c[5])) for c in ohlcv]
            logger.info("Fetched klines", symbol=symbol, count=len(klines), timeframe=timeframe, exchange=self.exchange)
            return KlineData(symbol=symbol, timeframe=timeframe, exchange=self.exchange, klines=klines)
        except ccxt.BadSymbol as e:
            raise CCXTError(f"Invalid symbol: {symbol}", code=ErrorCodes.INVALID_SYMBOL)
        except CCXTError:
            raise
        except Exception as e:
            logger.error("CCXT klines fetch failed", symbol=symbol, error=str(e))
            raise CCXTError(f"Failed to fetch klines for {symbol}: {e}", code=ErrorCodes.CCXT_ERROR)

    def fetch_orderbook(self, symbol: str, limit: int = 20) -> OrderBook:
        try:
            logger.debug("Fetching orderbook", symbol=symbol, limit=limit, exchange=self.exchange)
            ob = self.client.fetch_order_book(symbol, limit=limit)
            bids = [OrderBookLevel(price=float(b[0]), volume=float(b[1])) for b in ob.get("bids", [])]
            asks = [OrderBookLevel(price=float(a[0]), volume=float(a[1])) for a in ob.get("asks", [])]
            return OrderBook(
                symbol=symbol, exchange=self.exchange, bids=bids, asks=asks,
                timestamp=ob.get("timestamp", 0) / 1000 if ob.get("timestamp") else 0,
            )
        except ccxt.BadSymbol as e:
            raise CCXTError(f"Invalid symbol: {symbol}", code=ErrorCodes.INVALID_SYMBOL)
        except Exception as e:
            logger.error("CCXT orderbook fetch failed", symbol=symbol, error=str(e))
            raise CCXTError(f"Failed to fetch orderbook for {symbol}: {e}", code=ErrorCodes.CCXT_ERROR)

    def get_symbols(self) -> list[str]:
        try:
            markets = self.client.load_markets()
            return list(markets.keys())
        except Exception as e:
            logger.error("Failed to fetch symbols", error=str(e))
            raise CCXTError(f"Failed to fetch symbols: {e}", code=ErrorCodes.CCXT_ERROR)
