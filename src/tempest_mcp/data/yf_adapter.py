"""Yahoo Finance data adapter for historical market data. NO API KEYS REQUIRED."""

import time
from datetime import datetime, timedelta, timezone
from typing import Final

import pandas as pd
import yfinance as yf

from tempest_mcp.config import ErrorCodes, get_config
from tempest_mcp.data._symbols import normalize_to_yf
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.market import Kline, KlineData, Ticker

logger = get_logger(__name__)

# Error code constants (3xxx range per D8)
_YF_RATE_LIMIT_CODE: Final[int] = 3004
_YF_NETWORK_CODE: Final[int] = 3005
_YF_GENERAL_CODE: Final[int] = 3001

# Check if yfinance has YFRateLimitError at module load time
_YF_HAS_RATE_LIMIT_ERROR = hasattr(yf.exceptions, "YFRateLimitError")


class TempestError(Exception):
    """Base exception for all Tempest errors (D8 error hierarchy)."""

    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code
        self.message = message


class DataSourceError(TempestError):
    """Base exception for data source errors (3xxx range)."""

    def __init__(self, message: str, code: int = ErrorCodes.DATA_SOURCE_ERROR):
        super().__init__(message, code)


class YFinanceError(DataSourceError):
    """Yahoo Finance specific errors (code 3xxx)."""

    def __init__(self, message: str, code: int = ErrorCodes.YFINANCE_ERROR):
        super().__init__(message, code)


def _get_empty_ohlcv() -> pd.DataFrame:
    """Return empty DataFrame with correct OHLCV columns."""
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def _is_rate_limit_error(e: Exception) -> bool:
    """Check if exception is a yfinance rate limit error, safely."""
    if _YF_HAS_RATE_LIMIT_ERROR:
        return isinstance(e, yf.exceptions.YFRateLimitError)
    # Fallback: check message content
    error_str = str(e).lower()
    return "rate limit" in error_str or "429" in error_str


def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data from Yahoo Finance.

    Args:
        symbol: Asset symbol in yfinance native format (e.g., BTC-USD, ETH-USD).
                Option B: no symbol conversion needed — BTC-USD and ETH-USD work as-is.
        interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo). Default: 1d.
        start: Start datetime (UTC). Defaults to 30 days ago for 1d interval.
        end: End datetime (UTC). Defaults to now.
        auto_adjust: Whether to adjust prices for splits/dividends. Default: True.

    Returns:
        pd.DataFrame with columns [open, high, low, close, volume] and UTC-aware index.
        Returns empty DataFrame with correct columns on error (no exception propagated).

    Raises:
        This function does NOT raise exceptions. All errors are logged and return empty DataFrame.
    """
    config = get_config()
    # Validate timeout and max_retries - ensure positive values
    timeout: int = max(1, config.yf_timeout)
    max_retries: int = max(1, config.yf_retries)

    # Normalize dates to UTC
    now_utc = datetime.now(timezone.utc)
    if start is None:
        # Default: 30 days for daily, shorter for intraday
        start = now_utc - timedelta(days=30)

    # Ensure timezone-aware - naive datetimes are interpreted as UTC
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
        logger.warning("fetch_ohlcv: naive start datetime interpreted as UTC", start=start)
    if end is not None and end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
        logger.warning("fetch_ohlcv: naive end datetime interpreted as UTC", end=end)

    # Map interval to yfinance format
    interval_map: dict[str, str] = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "1d": "1d",
        "1wk": "1wk",
        "1mo": "1mo",
    }
    yf_interval = interval_map.get(interval)
    if yf_interval is None:
        logger.error(
            "fetch_ohlcv: unsupported interval",
            interval=interval,
            supported_intervals=list(interval_map.keys()),
        )
        return _get_empty_ohlcv()

    # 4h is not supported by yfinance - reject with clear error
    if interval == "4h":
        logger.error(
            "fetch_ohlcv: 4h interval not supported by yfinance. Use 1h and aggregate client-side.",
            interval=interval,
        )
        return _get_empty_ohlcv()

    # Determine end for yfinance query
    # If end is None (defaulting to now), don't pass end to yfinance to avoid
    # returning open/NaN data near midnight - use yfinance default behavior
    query_end = end if end is not None else None

    last_exception: Exception | None = None
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers=symbol,
                start=start,
                end=query_end,
                interval=yf_interval,
                auto_adjust=auto_adjust,
                progress=False,
                timeout=timeout,
            )

            if df.empty:
                # Empty DataFrame could be transient network issue — retry before giving up
                logger.warning(
                    "fetch_ohlcv: empty DataFrame, retrying",
                    symbol=symbol,
                    interval=interval,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
                # Don't return here — continue to retry loop

            # Flatten multi-index columns if present (yfinance returns multi-index when single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Ensure lowercase columns
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Ensure volume is numeric
            if "volume" in df.columns:
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

            # Keep only required columns
            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            for col in ohlcv_cols:
                if col not in df.columns:
                    df[col] = 0.0

            result = df[ohlcv_cols].copy()

            # Handle all-NaN DataFrame (yfinance returns this for some symbols/data gaps)
            if result.isnull().all().all():
                logger.warning(
                    "fetch_ohlcv: all-NaN data for symbol",
                    symbol=symbol,
                    interval=interval,
                    attempt=attempt + 1,
                )
                return _get_empty_ohlcv()

            # Ensure index is UTC-aware
            if isinstance(result.index, pd.DatetimeIndex):
                if result.index.tz is None:
                    result.index = result.index.tz_localize(timezone.utc)
                else:
                    result.index = result.index.tz_convert(timezone.utc)

            row_count = len(result)
            logger.info(
                "fetch_ohlcv: success",
                symbol=symbol,
                interval=interval,
                row_count=row_count,
            )
            return result

        except Exception as e:
            if _is_rate_limit_error(e):
                last_exception = YFinanceError(
                    f"Rate limit exceeded for {symbol}",
                    code=_YF_RATE_LIMIT_CODE,
                )
                logger.warning(
                    "fetch_ohlcv: rate limit, retrying",
                    symbol=symbol,
                    interval=interval,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                )
            elif isinstance(e, OSError):
                # Network errors — retry
                last_exception = YFinanceError(
                    f"Network error fetching {symbol}: {e}",
                    code=_YF_NETWORK_CODE,
                )
                logger.warning(
                    "fetch_ohlcv: network error, retrying",
                    symbol=symbol,
                    interval=interval,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )
            else:
                last_exception = YFinanceError(
                    f"Failed to fetch OHLCV for {symbol}: {e}",
                    code=_YF_GENERAL_CODE,
                )
                logger.warning(
                    "fetch_ohlcv: error, retrying",
                    symbol=symbol,
                    interval=interval,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )

        # Exponential backoff before retry (skip on last attempt)
        if attempt < max_retries - 1:
            sleep_time = 2**attempt
            logger.debug(
                "fetch_ohlcv: sleeping before retry",
                symbol=symbol,
                interval=interval,
                sleep_seconds=sleep_time,
            )
            time.sleep(sleep_time)

    # All retries exhausted — log error and return empty DataFrame
    logger.error(
        "fetch_ohlcv: all retries exhausted",
        symbol=symbol,
        interval=interval,
        max_retries=max_retries,
        error=str(last_exception) if last_exception else None,
    )
    return _get_empty_ohlcv()


class YFAdapter:
    """Yahoo Finance adapter class (existing implementation)."""

    cache_ttl: int = 300

    def __post_init__(self) -> None:
        config = get_config()
        if self.cache_ttl == 300:
            self.cache_ttl = config.yf_cache_ttl

    def _convert_symbol(self, symbol: str) -> str:
        return normalize_to_yf(symbol)

    def _convert_timeframe(self, timeframe: str) -> str:
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "1d": "1d",
            "1w": "1wk",
            "1M": "1mo",
        }
        return mapping.get(timeframe, "1d")

    def fetch_ticker(self, symbol: str) -> Ticker:
        """Fetch ticker data. DEPRECATED: use fetch_ohlcv for OHLCV data."""

        yf_symbol = self._convert_symbol(symbol)
        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info
            if not info:
                raise YFinanceError(
                    f"No data found for symbol: {symbol}", code=ErrorCodes.DATA_NOT_FOUND
                )
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
            volume = info.get("volume") or info.get("regularMarketVolume", 0)
            high = info.get("dayHigh") or info.get("regularMarketDayHigh", 0)
            low = info.get("dayLow") or info.get("regularMarketDayLow", 0)
            change_percent = info.get("regularMarketChangePercent", 0)
            return Ticker(
                symbol=symbol,
                exchange="yahoo",
                price=float(price),
                timestamp=datetime.now().timestamp(),
                volume_24h=float(volume) if volume else None,
                high_24h=float(high) if high else None,
                low_24h=float(low) if low else None,
                change_percent_24h=float(change_percent) if change_percent else None,
            )
        except YFinanceError:
            raise
        except Exception as e:
            logger.error("YFinance ticker fetch failed", symbol=symbol, error=str(e))
            raise YFinanceError(
                f"Failed to fetch ticker for {symbol}: {e}", code=ErrorCodes.YFINANCE_ERROR
            ) from e

    def fetch_klines(
        self, symbol: str, timeframe: str = "1d", since: datetime | None = None, limit: int = 100
    ) -> KlineData:
        """Fetch klines. DEPRECATED: use fetch_ohlcv for OHLCV data."""

        yf_symbol = self._convert_symbol(symbol)
        interval = self._convert_timeframe(timeframe)
        if since is None:
            period_days = {
                "1m": limit / 1440,
                "5m": limit / 288,
                "15m": limit / 96,
                "30m": limit / 48,
                "1h": limit / 24,
                "1d": limit,
                "1wk": limit * 7,
                "1mo": limit * 30,
            }
            days = period_days.get(interval, limit)
            since = datetime.now() - timedelta(days=days)
        try:
            end = datetime.now()
            df = yf.download(
                yf_symbol,
                start=since,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
            )
            if df.empty:
                raise YFinanceError(
                    f"No kline data found for {symbol}", code=ErrorCodes.DATA_NOT_FOUND
                )
            klines = []
            for idx, row in df.iterrows():
                ts = idx.timestamp() if isinstance(idx, pd.Timestamp) else float(idx)
                klines.append(
                    Kline(
                        timestamp=ts,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=float(row["Volume"]) if "Volume" in row else 0.0,
                    )
                )
            klines = klines[-limit:] if len(klines) > limit else klines
            logger.info("Fetched klines", symbol=symbol, count=len(klines), timeframe=timeframe)
            return KlineData(symbol=symbol, timeframe=timeframe, exchange="yahoo", klines=klines)
        except YFinanceError:
            raise
        except Exception as e:
            logger.error("YFinance klines fetch failed", symbol=symbol, error=str(e))
            raise YFinanceError(
                f"Failed to fetch klines for {symbol}: {e}", code=ErrorCodes.YFINANCE_ERROR
            ) from e

    def get_historical_prices(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        yf_symbol = self._convert_symbol(symbol)
        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, auto_adjust=False)
            if df.empty:
                raise YFinanceError(
                    f"No historical data for {symbol}", code=ErrorCodes.DATA_NOT_FOUND
                )
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            return df
        except YFinanceError:
            raise
        except Exception as e:
            logger.error("YFinance historical fetch failed", symbol=symbol, error=str(e))
            raise YFinanceError(
                f"Failed to fetch historical data for {symbol}: {e}", code=ErrorCodes.YFINANCE_ERROR
            ) from e
