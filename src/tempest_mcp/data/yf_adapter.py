"""Yahoo Finance data adapter for historical market data. NO API KEYS REQUIRED."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from tempest_mcp.config import ErrorCodes, get_config
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.market import Kline, KlineData, Ticker

logger = get_logger(__name__)


class YFinanceError(Exception):
    def __init__(self, message: str, code: int = ErrorCodes.YFINANCE_ERROR):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class YFAdapter:
    cache_ttl: int = 300

    def __post_init__(self):
        config = get_config()
        if self.cache_ttl == 300:
            self.cache_ttl = config.yf_cache_ttl

    def _convert_symbol(self, symbol: str) -> str:
        if "/" in symbol:
            base, quote = symbol.split("/")
            if quote.upper() in ("USDT", "USD", "USDC"):
                return f"{base.upper()}-USD"
            return f"{base.upper()}-{quote.upper()}"
        return symbol.upper()

    def _convert_timeframe(self, timeframe: str) -> str:
        mapping = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "1h", "1d": "1d", "1w": "1wk", "1M": "1mo"}
        return mapping.get(timeframe, "1d")

    def fetch_ticker(self, symbol: str) -> Ticker:
        yf_symbol = self._convert_symbol(symbol)
        try:
            logger.debug("Fetching ticker", symbol=symbol, yf_symbol=yf_symbol)
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            if not info:
                raise YFinanceError(f"No data found for symbol: {symbol}", code=ErrorCodes.DATA_NOT_FOUND)
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            volume = info.get("volume") or info.get("regularMarketVolume", 0)
            high = info.get("dayHigh") or info.get("regularMarketDayHigh", 0)
            low = info.get("dayLow") or info.get("regularMarketDayLow", 0)
            change_percent = info.get("regularMarketChangePercent", 0)
            return Ticker(
                symbol=symbol, exchange="yahoo", price=float(price),
                timestamp=datetime.now().timestamp(), volume_24h=float(volume) if volume else None,
                high_24h=float(high) if high else None, low_24h=float(low) if low else None,
                change_percent_24h=float(change_percent) if change_percent else None,
            )
        except YFinanceError:
            raise
        except Exception as e:
            logger.error("YFinance ticker fetch failed", symbol=symbol, error=str(e))
            raise YFinanceError(f"Failed to fetch ticker for {symbol}: {e}", code=ErrorCodes.YFINANCE_ERROR)

    def fetch_klines(self, symbol: str, timeframe: str = "1d", since: datetime | None = None, limit: int = 100) -> KlineData:
        yf_symbol = self._convert_symbol(symbol)
        interval = self._convert_timeframe(timeframe)
        if since is None:
            period_days = {"1m": limit / 1440, "5m": limit / 288, "15m": limit / 96, "30m": limit / 48, "1h": limit / 24, "1d": limit, "1wk": limit * 7, "1mo": limit * 30}
            days = period_days.get(interval, limit)
            since = datetime.now() - timedelta(days=days)
        try:
            logger.debug("Fetching klines", symbol=symbol, yf_symbol=yf_symbol, timeframe=timeframe, limit=limit)
            end = datetime.now()
            df = yf.download(yf_symbol, start=since, end=end, interval=interval, progress=False, auto_adjust=False)
            if df.empty:
                raise YFinanceError(f"No kline data found for {symbol}", code=ErrorCodes.DATA_NOT_FOUND)
            klines = []
            for idx, row in df.iterrows():
                ts = idx.timestamp() if isinstance(idx, pd.Timestamp) else float(idx)
                klines.append(Kline(
                    timestamp=ts, open=float(row["Open"]), high=float(row["High"]),
                    low=float(row["Low"]), close=float(row["Close"]), volume=float(row["Volume"]) if "Volume" in row else 0.0,
                ))
            klines = klines[-limit:] if len(klines) > limit else klines
            logger.info("Fetched klines", symbol=symbol, count=len(klines), timeframe=timeframe)
            return KlineData(symbol=symbol, timeframe=timeframe, exchange="yahoo", klines=klines)
        except YFinanceError:
            raise
        except Exception as e:
            logger.error("YFinance klines fetch failed", symbol=symbol, error=str(e))
            raise YFinanceError(f"Failed to fetch klines for {symbol}: {e}", code=ErrorCodes.YFINANCE_ERROR)

    def get_historical_prices(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        yf_symbol = self._convert_symbol(symbol)
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, auto_adjust=False)
            if df.empty:
                raise YFinanceError(f"No historical data for {symbol}", code=ErrorCodes.DATA_NOT_FOUND)
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            return df
        except YFinanceError:
            raise
        except Exception as e:
            logger.error("YFinance historical fetch failed", symbol=symbol, error=str(e))
            raise YFinanceError(f"Failed to fetch historical data for {symbol}: {e}", code=ErrorCodes.YFINANCE_ERROR)
