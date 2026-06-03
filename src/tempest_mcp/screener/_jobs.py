"""Internal screening job policy helpers."""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import pandas as pd


@dataclass(frozen=True, slots=True)
class ScreeningJobKey:
    symbol: str
    exchange: str
    timeframe: str | None = None
    window_days: int | None = None


@dataclass(frozen=True, slots=True)
class ScreeningJobFailure:
    symbol: str
    exchange: str
    reason: str
    timeframe: str | None = None
    window_days: int | None = None


TItem = TypeVar("TItem")
TFailure = TypeVar("TFailure")


@dataclass(frozen=True, slots=True)
class ScreeningOutcome(Generic[TItem, TFailure]):
    items: tuple[TItem, ...]
    failures: tuple[TFailure, ...]
    effective_exchange: str
    resolved_symbols: tuple[str, ...] | None

    @property
    def is_success(self) -> bool:
        return bool(self.items) or not self.failures


class LiveOhlcvFetcher(Protocol):
    def fetch_ohlcv_live(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame: ...


def resolve_screening_symbols(requested_symbols: Sequence[str] | None, default_symbols: Sequence[str]) -> tuple[str, ...]:
    """Resolve explicit/default symbols with order-preserving deduplication."""
    return tuple(dict.fromkeys(default_symbols if requested_symbols is None else requested_symbols))


def run_symbol_jobs(*, symbols: Sequence[str], exchange: str, fetcher: LiveOhlcvFetcher, timeframe: str, limit: int, empty_failure: Callable[[ScreeningJobKey], TFailure], fetch_failure: Callable[[ScreeningJobKey, Exception], TFailure], evaluate: Callable[[ScreeningJobKey, pd.DataFrame], tuple[TItem | None, TFailure | None]], sort_items: Callable[[Iterable[TItem]], list[TItem]], sort_failures: Callable[[Iterable[TFailure]], list[TFailure]] | None = None) -> tuple[list[TItem], list[TFailure]]:
    """Run per-symbol live OHLCV jobs without owning scan-specific math."""
    items: list[TItem] = []
    failures: list[TFailure] = []
    for symbol in symbols:
        key = ScreeningJobKey(symbol=symbol, exchange=exchange, timeframe=timeframe)
        try:
            df = fetcher.fetch_ohlcv_live(symbol, timeframe=timeframe, limit=limit)
            if df.empty:
                failures.append(empty_failure(key))
                continue
        except Exception as exc:
            failures.append(fetch_failure(key, exc))
            continue
        item, failure = evaluate(key, df)
        if failure is not None:
            failures.append(failure)
        elif item is not None:
            items.append(item)
    sorted_failures = sort_failures(failures) if sort_failures else failures
    return sort_items(items), sorted_failures


def run_symbol_horizon_jobs(*, symbols: Sequence[str], exchange: str, fetcher: LiveOhlcvFetcher, horizons: Sequence[tuple[str, int]], limit_for_horizon: Callable[[str, int], int], empty_failure: Callable[[ScreeningJobKey], TFailure], fetch_failure: Callable[[ScreeningJobKey, Exception], TFailure], evaluate: Callable[[ScreeningJobKey, pd.DataFrame, int], tuple[TItem | None, TFailure | None]], sort_items: Callable[[Iterable[TItem]], list[TItem]], sort_failures: Callable[[Iterable[TFailure]], list[TFailure]]) -> tuple[list[TItem], list[TFailure]]:
    """Run per-symbol/per-horizon live OHLCV jobs."""
    items: list[TItem] = []
    failures: list[TFailure] = []
    for symbol in symbols:
        for timeframe, window_days in horizons:
            key = ScreeningJobKey(symbol, exchange, timeframe, window_days)
            limit = limit_for_horizon(timeframe, window_days)
            try:
                df = fetcher.fetch_ohlcv_live(symbol, timeframe=timeframe, limit=limit)
                if df.empty:
                    failures.append(empty_failure(key))
                    continue
            except Exception as exc:
                failures.append(fetch_failure(key, exc))
                continue
            item, failure = evaluate(key, df, limit)
            if failure is not None:
                failures.append(failure)
            elif item is not None:
                items.append(item)
    return sort_items(items), sort_failures(failures)


def sort_scan_results(results: Iterable[Any]) -> list[Any]:
    return sorted(results, key=lambda r: (-r.score, -len(r.filters_matched), r.symbol, r.exchange))


def sort_scan_failures_for_session(failures: Iterable[Any]) -> list[Any]:
    return sorted(failures, key=lambda f: (f.symbol, f.exchange))


def sort_order_block_candidates(candidates: Iterable[Any]) -> list[Any]:
    horizon_priority = {("4h", 7): 0, ("1h", 1): 1}
    return sorted(candidates, key=lambda c: (-c.score, horizon_priority.get((c.timeframe, c.window_days), 99), c.symbol, c.exchange))


def sort_order_block_failures(failures: Iterable[Any]) -> list[Any]:
    return sorted(failures, key=lambda f: (f.symbol, f.exchange, f.timeframe, f.window_days, f.reason))


def serialize_scan_result(result: Any) -> dict[str, Any]:
    return {"symbol": result.symbol, "exchange": result.exchange, "timestamp": result.timestamp, "price": result.price, "filters_matched": result.filters_matched, "indicator_values": result.indicator_values, "score": result.score}


def serialize_scan_failure(failure: Any) -> dict[str, Any]:
    return {"symbol": failure.symbol, "exchange": failure.exchange, "reason": failure.reason}


def serialize_order_block_candidate(candidate: Any) -> dict[str, Any]:
    return {"symbol": candidate.symbol, "exchange": candidate.exchange, "timeframe": candidate.timeframe, "window_days": candidate.window_days, "timestamp": candidate.timestamp, "price": candidate.price, "zone_type": candidate.zone_type, "zone_high": candidate.zone_high, "zone_low": candidate.zone_low, "freshness_candles": candidate.freshness_candles, "score": candidate.score}


def serialize_order_block_failure(failure: Any) -> dict[str, Any]:
    return {"symbol": failure.symbol, "exchange": failure.exchange, "timeframe": failure.timeframe, "window_days": failure.window_days, "reason": failure.reason}


def screening_success(items: Sequence[Any], failures: Sequence[Any]) -> bool:
    return bool(items) or not failures
