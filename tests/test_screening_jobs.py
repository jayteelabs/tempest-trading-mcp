"""Focused tests for the internal C4 screening job policy seam."""

import pandas as pd

from tempest_mcp.screener._jobs import (
    ScreeningJobKey,
    resolve_screening_symbols,
    run_symbol_horizon_jobs,
    run_symbol_jobs,
    screening_success,
    serialize_order_block_failure,
    serialize_scan_failure,
    sort_order_block_failures,
    sort_scan_failures_for_session,
    sort_scan_results,
)
from tempest_mcp.screener.scanner import OrderBlockFailure, ScanFailure, ScanResult


class RecordingFetcher:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def fetch_ohlcv_live(self, symbol: str, timeframe: str, limit: int):
        self.calls.append((symbol, timeframe, limit))
        response = self.responses[(symbol, timeframe)]
        if isinstance(response, Exception):
            raise response
        return response


def frame(rows=3):
    return pd.DataFrame(
        {"open": [1.0] * rows, "high": [2.0] * rows, "low": [0.5] * rows, "close": [1.5] * rows, "volume": [10.0] * rows},
        index=pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
    )


def test_resolve_screening_symbols_dedupes_explicit_and_defaults():
    assert resolve_screening_symbols(["B", "A", "B"], ("D",)) == ("B", "A")
    assert resolve_screening_symbols(None, ("D", "E", "D")) == ("D", "E")


def test_run_symbol_jobs_maps_success_empty_and_fetch_failure_in_request_order():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    fetcher = RecordingFetcher({("A", "1h"): frame(), ("B", "1h"): empty, ("C", "1h"): RuntimeError("boom")})

    def evaluate(key: ScreeningJobKey, df: pd.DataFrame):
        return ScanResult(key.symbol, key.exchange, 1.0, 10.0, [], {}, 0.0), None

    results, failures = run_symbol_jobs(
        symbols=("A", "B", "C"),
        exchange="binance",
        fetcher=fetcher,
        timeframe="1h",
        limit=100,
        empty_failure=lambda key: ScanFailure(key.symbol, key.exchange, "empty_ohlcv"),
        fetch_failure=lambda key, _exc: ScanFailure(key.symbol, key.exchange, "fetch_error"),
        evaluate=evaluate,
        sort_items=sort_scan_results,
    )

    assert [r.symbol for r in results] == ["A"]
    assert [(f.symbol, f.reason) for f in failures] == [("B", "empty_ohlcv"), ("C", "fetch_error")]
    assert fetcher.calls == [("A", "1h", 100), ("B", "1h", 100), ("C", "1h", 100)]


def test_session_failure_sort_and_success_policy_are_shared():
    failures = [ScanFailure("Z", "binance", "fetch_error"), ScanFailure("A", "binance", "empty_ohlcv")]
    assert [f.symbol for f in sort_scan_failures_for_session(failures)] == ["A", "Z"]
    assert screening_success([], failures) is False
    assert screening_success([], []) is True


def test_run_symbol_horizon_jobs_uses_fixed_horizon_order_and_sorts_failures():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    fetcher = RecordingFetcher({("B", "1h"): empty, ("B", "4h"): RuntimeError("down")})

    def evaluate(key: ScreeningJobKey, df: pd.DataFrame, limit: int):
        return None, None

    candidates, failures = run_symbol_horizon_jobs(
        symbols=("B",),
        exchange="binance",
        fetcher=fetcher,
        horizons=(("1h", 1), ("4h", 7)),
        limit_for_horizon=lambda timeframe, window_days: 24 if timeframe == "1h" else 42,
        empty_failure=lambda key: OrderBlockFailure(key.symbol, key.exchange, key.timeframe or "", key.window_days or 0, "data_unavailable"),
        fetch_failure=lambda key, _exc: OrderBlockFailure(key.symbol, key.exchange, key.timeframe or "", key.window_days or 0, "data_unavailable"),
        evaluate=evaluate,
        sort_items=list,
        sort_failures=sort_order_block_failures,
    )

    assert candidates == []
    assert fetcher.calls == [("B", "1h", 24), ("B", "4h", 42)]
    assert [serialize_order_block_failure(f) for f in failures] == [
        {"symbol": "B", "exchange": "binance", "timeframe": "1h", "window_days": 1, "reason": "data_unavailable"},
        {"symbol": "B", "exchange": "binance", "timeframe": "4h", "window_days": 7, "reason": "data_unavailable"},
    ]


def test_serializers_preserve_public_failure_shapes_without_source_used():
    assert serialize_scan_failure(ScanFailure("BTC/USDT", "binance", "fetch_error")) == {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "reason": "fetch_error",
    }
