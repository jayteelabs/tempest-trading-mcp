"""Historical OHLCV intake seam.

This module owns historical request normalization, provider routing, fallback,
canonical frame shaping, row-cap semantics, and source attribution. Live OHLCV,
ticker, orderbook, and screener live paths intentionally remain outside this seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal, Protocol

import pandas as pd
import structlog

from tempest_mcp.data._contracts import (
    CCXT_TIMEFRAME_MAP,
    MAX_LIMIT,
    MIN_LIMIT,
    SUPPORTED_EXCHANGES,
    SUPPORTED_TIMEFRAMES,
    TIMEFRAME_SECONDS,
    YF_INTERVAL_MAP,
    canonicalize_ohlcv_frame,
    empty_ohlcv_frame,
)
from tempest_mcp.data._symbols import normalize_to_ccxt_market, normalize_to_yf
from tempest_mcp.time_utils import BUSINESS_TZ_NAME, coerce_window_datetime_to_utc

logger = structlog.get_logger()

ExchangeName = Literal["binance", "bybit", "coinbase", "kraken"]
OhlcvSource = Literal["ccxt", "yfinance", "empty"]
OhlcvFailureReason = Literal[
    "invalid_symbol",
    "invalid_exchange",
    "invalid_timeframe",
    "invalid_window",
    "empty_ohlcv",
    "fetch_error",
    "unsupported_yfinance_interval",
]


class CcxtHistoricalAdapter(Protocol):
    def fetch_ohlcv_historical(
        self,
        symbol: str,
        timeframe: str = "1d",
        since: int | None = None,
        limit: int = MAX_LIMIT,
        params: dict | None = None,
    ) -> pd.DataFrame: ...


class YFinanceHistoricalAdapter(Protocol):
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame: ...


@dataclass(frozen=True, slots=True)
class OhlcvRequest:
    symbol: str
    timeframe: str = "1d"
    exchange: str = "binance"
    start: datetime | None = None
    end: datetime | None = None
    limit: int | None = None
    warmup_bars: int = 0
    auto_adjust: bool = True


@dataclass(frozen=True, slots=True)
class OhlcvResult:
    frame: pd.DataFrame
    canonical_symbol: str
    exchange: str
    timeframe: str
    source_used: OhlcvSource
    provider_symbol: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    failure_reason: OhlcvFailureReason | None = None


class _YFinanceAdapter:
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str = "1d",
        start: datetime | None = None,
        end: datetime | None = None,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        from tempest_mcp.data.yf_adapter import fetch_ohlcv

        return fetch_ohlcv(symbol, interval=interval, start=start, end=end, auto_adjust=auto_adjust)


class OhlcvIntake:
    def __init__(
        self,
        exchange_name: ExchangeName = "binance",
        *,
        ccxt_adapter: CcxtHistoricalAdapter | None = None,
        yf_adapter: YFinanceHistoricalAdapter | None = None,
    ) -> None:
        self.exchange_name = exchange_name.lower()
        if ccxt_adapter is None:
            from tempest_mcp.data.ccxt_adapter import CCXTAdapter

            ccxt_adapter = CCXTAdapter(exchange_name=self.exchange_name)
        self._ccxt = ccxt_adapter
        self._yf = yf_adapter or _YFinanceAdapter()

    def fetch(self, request: OhlcvRequest) -> OhlcvResult:
        exchange = request.exchange.lower() if request.exchange else self.exchange_name
        warnings: list[str] = []

        invalid_result = self._validate_request(request, exchange)
        if invalid_result is not None:
            return invalid_result

        start_utc = self._ensure_utc(request.start, "start", warnings)
        end_utc = self._ensure_utc(request.end, "end", warnings)
        if start_utc is not None and end_utc is not None and start_utc >= end_utc:
            return self._empty_result(
                request,
                exchange,
                warnings=tuple(warnings),
                failure_reason="invalid_window",
            )

        fetch_start_utc = self._derive_fetch_start(request, start_utc, end_utc)
        fetch_limit = self._derive_fetch_limit(request, fetch_start_utc, end_utc)

        if self._is_direct_yfinance_symbol(request.symbol):
            return self._fetch_yfinance_direct(
                request, exchange, fetch_start_utc, end_utc, fetch_limit, warnings
            )

        try:
            ccxt_symbol = normalize_to_ccxt_market(
                request.symbol, exchange=exchange
            )  # public canonical/display symbol
        except ValueError as exc:
            logger.warning("ohlcv_intake_invalid_symbol", symbol=request.symbol, error=str(exc))
            return self._empty_result(
                request,
                exchange,
                warnings=tuple(warnings),
                failure_reason="invalid_symbol",
            )

        ccxt_frame = self._fetch_ccxt(
            request, ccxt_symbol, fetch_start_utc, end_utc, fetch_limit, warnings
        )
        if not ccxt_frame.empty:
            return OhlcvResult(
                frame=self._trim_frame(ccxt_frame, request),
                canonical_symbol=ccxt_symbol,
                exchange=exchange,
                timeframe=request.timeframe,
                source_used="ccxt",
                provider_symbol=ccxt_symbol,
                warnings=tuple(warnings),
            )

        if request.timeframe not in YF_INTERVAL_MAP:
            return self._empty_result(
                request,
                exchange,
                canonical_symbol=ccxt_symbol,
                provider_symbol=ccxt_symbol,
                warnings=tuple(warnings),
                failure_reason="unsupported_yfinance_interval",
            )

        try:
            yf_symbol = normalize_to_yf(ccxt_symbol)
        except ValueError as exc:
            logger.warning(
                "ohlcv_intake_invalid_yfinance_symbol", symbol=ccxt_symbol, error=str(exc)
            )
            return self._empty_result(
                request,
                exchange,
                canonical_symbol=ccxt_symbol,
                provider_symbol=ccxt_symbol,
                warnings=tuple(warnings),
                failure_reason="invalid_symbol",
            )

        yf_frame = self._fetch_yfinance(request, yf_symbol, fetch_start_utc, end_utc, warnings)
        if not yf_frame.empty:
            return OhlcvResult(
                frame=self._trim_frame(yf_frame, request),
                canonical_symbol=ccxt_symbol,
                exchange=exchange,
                timeframe=request.timeframe,
                source_used="yfinance",
                provider_symbol=yf_symbol,
                warnings=tuple(warnings),
            )

        return self._empty_result(
            request,
            exchange,
            canonical_symbol=ccxt_symbol,
            provider_symbol=yf_symbol,
            warnings=tuple(warnings),
            failure_reason="empty_ohlcv",
        )

    def _validate_request(self, request: OhlcvRequest, exchange: str) -> OhlcvResult | None:
        if exchange not in SUPPORTED_EXCHANGES:
            return self._empty_result(request, exchange, failure_reason="invalid_exchange")
        if request.timeframe not in SUPPORTED_TIMEFRAMES:
            return self._empty_result(request, exchange, failure_reason="invalid_timeframe")
        if request.limit is not None and (
            isinstance(request.limit, bool)
            or not isinstance(request.limit, int)
            or request.limit < MIN_LIMIT
            or request.limit > MAX_LIMIT
        ):
            return self._empty_result(request, exchange, failure_reason="invalid_window")
        if (
            isinstance(request.warmup_bars, bool)
            or not isinstance(request.warmup_bars, int)
            or request.warmup_bars < 0
        ):
            return self._empty_result(request, exchange, failure_reason="invalid_window")
        if not request.symbol or not isinstance(request.symbol, str):
            return self._empty_result(request, exchange, failure_reason="invalid_symbol")
        return None

    def _ensure_utc(self, dt: datetime | None, name: str, warnings: list[str]) -> datetime | None:
        if dt is None:
            return None
        if dt.tzinfo is None:
            warnings.append(f"{name}_naive_interpreted_as_{BUSINESS_TZ_NAME}")
        return coerce_window_datetime_to_utc(dt)

    def _requested_fetch_cap(self, request: OhlcvRequest) -> int | None:
        if request.limit is None:
            return None
        requested = request.limit + request.warmup_bars
        return max(MIN_LIMIT, min(MAX_LIMIT, requested))

    def _derive_fetch_start(
        self,
        request: OhlcvRequest,
        start_utc: datetime | None,
        end_utc: datetime | None,
    ) -> datetime | None:
        cap = self._requested_fetch_cap(request)
        if cap is None or end_utc is None:
            return start_utc

        interval_seconds = TIMEFRAME_SECONDS.get(request.timeframe, TIMEFRAME_SECONDS["1d"])
        latest_window_start = end_utc - timedelta(seconds=interval_seconds * cap)
        if start_utc is not None and start_utc > latest_window_start:
            return start_utc
        return latest_window_start

    def _derive_fetch_limit(
        self,
        request: OhlcvRequest,
        fetch_start_utc: datetime | None,
        end_utc: datetime | None,
    ) -> int:
        requested = self._requested_fetch_cap(request)
        if fetch_start_utc is not None and end_utc is not None:
            seconds = max(0, (end_utc - fetch_start_utc).total_seconds())
            interval_seconds = TIMEFRAME_SECONDS.get(request.timeframe, TIMEFRAME_SECONDS["1d"])
            needed = max(1, int((seconds + interval_seconds - 1) // interval_seconds))
            return min(MAX_LIMIT, needed)
        if fetch_start_utc is not None:
            return MAX_LIMIT
        if requested is not None:
            return requested
        return MAX_LIMIT

    def _fetch_ccxt(
        self,
        request: OhlcvRequest,
        ccxt_symbol: str,
        start_utc: datetime | None,
        end_utc: datetime | None,
        fetch_limit: int,
        warnings: list[str],
    ) -> pd.DataFrame:
        since_ms = int(start_utc.timestamp() * 1000) if start_utc is not None else None
        until_ms = int(end_utc.timestamp() * 1000) if end_utc is not None else None
        params: dict[str, int | bool] = {"until": until_ms} if until_ms is not None else {}
        if since_ms is not None and request.limit is not None:
            params["paginate"] = True
        if until_ms is not None:
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            if until_ms > now_ms + 60000:
                warnings.append("end_in_future")
        try:
            frame = self._ccxt.fetch_ohlcv_historical(
                symbol=ccxt_symbol,
                timeframe=CCXT_TIMEFRAME_MAP.get(request.timeframe, request.timeframe),
                since=since_ms,
                limit=fetch_limit,
                params=params,
            )
        except Exception as exc:
            logger.warning("ohlcv_intake_ccxt_exception", symbol=ccxt_symbol, error=str(exc))
            warnings.append("ccxt_fetch_error")
            return empty_ohlcv_frame()
        return canonicalize_ohlcv_frame(frame)

    def _fetch_yfinance_direct(
        self,
        request: OhlcvRequest,
        exchange: str,
        start_utc: datetime | None,
        end_utc: datetime | None,
        fetch_limit: int,
        warnings: list[str],
    ) -> OhlcvResult:
        yf_symbol = request.symbol.strip().upper()
        canonical_symbol = self._canonical_from_direct_yfinance_symbol(yf_symbol, exchange)
        if request.timeframe not in YF_INTERVAL_MAP:
            return self._empty_result(
                request,
                exchange,
                canonical_symbol=canonical_symbol,
                provider_symbol=yf_symbol,
                warnings=tuple(warnings),
                failure_reason="unsupported_yfinance_interval",
            )
        try:
            provider_symbol = normalize_to_yf(yf_symbol)
        except ValueError as exc:
            logger.warning(
                "ohlcv_intake_invalid_direct_yfinance_symbol", symbol=yf_symbol, error=str(exc)
            )
            return self._empty_result(
                request,
                exchange,
                canonical_symbol=canonical_symbol,
                provider_symbol=yf_symbol,
                warnings=tuple(warnings),
                failure_reason="invalid_symbol",
            )
        frame = self._fetch_yfinance(request, provider_symbol, start_utc, end_utc, warnings)
        if frame.empty:
            return self._empty_result(
                request,
                exchange,
                canonical_symbol=canonical_symbol,
                provider_symbol=provider_symbol,
                warnings=tuple(warnings),
                failure_reason="empty_ohlcv",
            )
        # Direct yfinance is seam/provider-level only; keep provider-native symbols
        # separate from the Tempest canonical/display symbol.
        return OhlcvResult(
            frame=self._trim_frame(frame, request, fallback_limit=fetch_limit),
            canonical_symbol=canonical_symbol,
            exchange=exchange,
            timeframe=request.timeframe,
            source_used="yfinance",
            provider_symbol=provider_symbol,
            warnings=tuple(warnings),
        )

    def _fetch_yfinance(
        self,
        request: OhlcvRequest,
        yf_symbol: str,
        start_utc: datetime | None,
        end_utc: datetime | None,
        warnings: list[str],
    ) -> pd.DataFrame:
        try:
            frame = self._yf.fetch_ohlcv(
                symbol=yf_symbol,
                interval=YF_INTERVAL_MAP[request.timeframe],
                start=start_utc,
                end=end_utc,
                auto_adjust=request.auto_adjust,
            )
        except Exception as exc:
            logger.warning("ohlcv_intake_yfinance_exception", symbol=yf_symbol, error=str(exc))
            warnings.append("yfinance_fetch_error")
            return empty_ohlcv_frame()
        return canonicalize_ohlcv_frame(frame)

    def _trim_frame(
        self,
        frame: pd.DataFrame,
        request: OhlcvRequest,
        fallback_limit: int | None = None,
    ) -> pd.DataFrame:
        result = canonicalize_ohlcv_frame(frame)
        cap = None
        if request.limit is not None:
            cap = request.limit + request.warmup_bars
        elif fallback_limit is not None:
            cap = fallback_limit
        if cap is not None and cap > 0 and len(result) > cap:
            return result.tail(cap)
        return result

    def _empty_result(
        self,
        request: OhlcvRequest,
        exchange: str,
        *,
        canonical_symbol: str | None = None,
        provider_symbol: str | None = None,
        warnings: tuple[str, ...] = (),
        failure_reason: OhlcvFailureReason,
    ) -> OhlcvResult:
        return OhlcvResult(
            frame=empty_ohlcv_frame(),
            canonical_symbol=canonical_symbol or request.symbol,
            exchange=exchange,
            timeframe=request.timeframe,
            source_used="empty",
            provider_symbol=provider_symbol,
            warnings=warnings,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _is_direct_yfinance_symbol(symbol: str) -> bool:
        return isinstance(symbol, str) and symbol.strip().upper().endswith("-USD")

    @staticmethod
    def _canonical_from_direct_yfinance_symbol(symbol: str, exchange: str) -> str:
        symbol_upper = symbol.strip().upper()
        base, separator, quote = symbol_upper.partition("-")
        if separator and base.isalnum() and quote == "USD":
            return normalize_to_ccxt_market(f"{base}USDT", exchange=exchange)
        return symbol_upper
