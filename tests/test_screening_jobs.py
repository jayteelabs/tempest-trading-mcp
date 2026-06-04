"""Focused tests for the internal C4 screening job policy seam."""

import pandas as pd
import pytest

from tempest_mcp.config import ErrorCodes
from tempest_mcp.models.indicator import SessionType
from tempest_mcp.screener import scanner as scanner_module
from tempest_mcp.screener._jobs import (
    ScreeningJobKey,
    resolve_screening_symbols,
    run_symbol_horizon_jobs,
    run_symbol_jobs,
    screening_success,
    serialize_order_block_candidate,
    serialize_order_block_failure,
    serialize_scan_failure,
    sort_order_block_candidates,
    sort_order_block_failures,
    sort_scan_failures_for_session,
    sort_scan_results,
)
from tempest_mcp.screener.scanner import (
    ORDER_BLOCK_HORIZONS,
    OrderBlockCandidate,
    OrderBlockFailure,
    ScanFailure,
    ScanResult,
    Screener,
)
from tempest_mcp.tools import screener_tools


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
        {
            "open": [1.0] * rows,
            "high": [2.0] * rows,
            "low": [0.5] * rows,
            "close": [1.5] * rows,
            "volume": [10.0] * rows,
        },
        index=pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
    )


def test_resolve_screening_symbols_dedupes_explicit_and_defaults():
    assert resolve_screening_symbols(["B", "A", "B"], ("D",)) == ("B", "A")
    assert resolve_screening_symbols(None, ("D", "E", "D")) == ("D", "E")


def test_run_symbol_jobs_maps_success_empty_and_fetch_failure_in_request_order():
    empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    fetcher = RecordingFetcher(
        {("A", "1h"): frame(), ("B", "1h"): empty, ("C", "1h"): RuntimeError("boom")}
    )

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
    failures = [
        ScanFailure("Z", "binance", "fetch_error"),
        ScanFailure("A", "binance", "empty_ohlcv"),
    ]
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
        empty_failure=lambda key: OrderBlockFailure(
            key.symbol, key.exchange, key.timeframe or "", key.window_days or 0, "data_unavailable"
        ),
        fetch_failure=lambda key, _exc: OrderBlockFailure(
            key.symbol, key.exchange, key.timeframe or "", key.window_days or 0, "data_unavailable"
        ),
        evaluate=evaluate,
        sort_items=list,
        sort_failures=sort_order_block_failures,
    )

    assert candidates == []
    assert fetcher.calls == [("B", "1h", 24), ("B", "4h", 42)]
    assert [serialize_order_block_failure(f) for f in failures] == [
        {
            "symbol": "B",
            "exchange": "binance",
            "timeframe": "1h",
            "window_days": 1,
            "reason": "data_unavailable",
        },
        {
            "symbol": "B",
            "exchange": "binance",
            "timeframe": "4h",
            "window_days": 7,
            "reason": "data_unavailable",
        },
    ]


def test_serializers_preserve_public_failure_shapes_without_source_used():
    assert serialize_scan_failure(ScanFailure("BTC/USDT", "binance", "fetch_error")) == {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "reason": "fetch_error",
    }


def test_sort_order_block_candidates_prefers_swing_horizon_on_score_tie():
    day = OrderBlockCandidate(
        "BTC/USDT", "binance", "1h", 1, 1.0, 10.0, "bullish", 11.0, 9.0, 2, 0.9
    )
    swing = OrderBlockCandidate(
        "ETH/USDT", "binance", "4h", 7, 1.0, 10.0, "bearish", 11.0, 9.0, 2, 0.9
    )

    assert sort_order_block_candidates([day, swing]) == [swing, day]


def test_session_breakout_scan_uses_symbol_job_policy(monkeypatch):
    observed = {}

    def fake_run_symbol_jobs(**kwargs):
        observed.update(kwargs)
        key = ScreeningJobKey("BTC/USDT", "binance", "1h")
        result, failure = kwargs["evaluate"](key, frame(rows=48))
        assert failure is None
        return [result], []

    def fake_evaluate(self, symbol, session, df, *, proximity_pct, volume_multiplier):
        assert symbol == "BTC/USDT"
        assert session is SessionType.NEW_YORK
        assert len(df) == 48
        assert proximity_pct == 1.5
        assert volume_multiplier == 3.0
        return ScanResult(symbol, self.exchange, 1.0, 100.0, ["pdh_breakout"], {}, 20.0), None

    monkeypatch.setattr(scanner_module, "run_symbol_jobs", fake_run_symbol_jobs)
    monkeypatch.setattr(Screener, "_evaluate_session_breakout_frame", fake_evaluate)

    screener = Screener(symbols=("BTC/USDT",), exchange="binance")
    results, failures = screener.session_breakout_scan(
        SessionType.NEW_YORK,
        proximity_pct=1.5,
        volume_multiplier=3.0,
    )

    assert [result.symbol for result in results] == ["BTC/USDT"]
    assert failures == []
    assert observed["timeframe"] == "1h"
    assert observed["limit"] == 48
    assert observed["sort_failures"] is scanner_module.sort_scan_failures_for_session


def test_session_breakout_scan_maps_evaluation_exception_to_indicator_error(monkeypatch):
    def fail_evaluate(self, symbol, session, df, *, proximity_pct, volume_multiplier):
        raise RuntimeError("indicator exploded")

    monkeypatch.setattr(Screener, "_evaluate_session_breakout_frame", fail_evaluate)

    screener = Screener(symbols=("BTC/USDT",), exchange="binance")
    screener._adapter = RecordingFetcher({("BTC/USDT", "1h"): frame(rows=48)})

    results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

    assert results == []
    assert failures == [ScanFailure("BTC/USDT", "binance", "indicator_error")]


def test_session_evaluation_returns_acceptance_critical_failure_reasons(monkeypatch):
    screener = Screener(symbols=("BTC/USDT",), exchange="binance")

    import tempest_mcp.indicators.session_levels as session_levels

    monkeypatch.setattr(session_levels, "detect_session_levels", lambda df, session: {"bars": 0})
    _, failure = screener._evaluate_session_breakout_frame(
        "BTC/USDT", SessionType.ASIA, frame(rows=48), proximity_pct=1.0, volume_multiplier=2.0
    )
    assert failure == ScanFailure("BTC/USDT", "binance", "insufficient_session_data")

    monkeypatch.setattr(
        session_levels,
        "detect_session_levels",
        lambda df, session: {"high": 2, "low": 1, "bars": 8},
    )
    monkeypatch.setattr(
        session_levels, "detect_pdh_pdl", lambda df: {"position": "insufficient_data"}
    )
    _, failure = screener._evaluate_session_breakout_frame(
        "BTC/USDT", SessionType.ASIA, frame(rows=48), proximity_pct=1.0, volume_multiplier=2.0
    )
    assert failure == ScanFailure("BTC/USDT", "binance", "insufficient_pdh_pdl_data")


def test_order_block_scan_uses_horizon_job_policy(monkeypatch):
    observed = {}

    def fake_run_symbol_horizon_jobs(**kwargs):
        observed.update(kwargs)
        key = ScreeningJobKey("BTC/USDT", "binance", "4h", 7)
        result, failure = kwargs["evaluate"](key, frame(rows=42), 42)
        assert failure is None
        return [result], []

    def fake_evaluate(self, key, df, *, atr_period, impulse_atr_mult, max_zone_age_bars):
        assert (key.symbol, key.timeframe, key.window_days) == ("BTC/USDT", "4h", 7)
        assert atr_period == 14
        assert impulse_atr_mult == 1.25
        assert max_zone_age_bars == 20
        return OrderBlockCandidate(
            key.symbol,
            key.exchange,
            key.timeframe,
            key.window_days,
            1.0,
            10.0,
            "bullish",
            11.0,
            9.0,
            1,
            0.95,
        ), None

    monkeypatch.setattr(scanner_module, "run_symbol_horizon_jobs", fake_run_symbol_horizon_jobs)
    monkeypatch.setattr(Screener, "_evaluate_order_block_frame", fake_evaluate)

    screener = Screener(symbols=("BTC/USDT",), exchange="binance")
    candidates, failures = screener.order_block_scan(impulse_atr_mult=1.25)

    assert [candidate.timeframe for candidate in candidates] == ["4h"]
    assert failures == []
    assert observed["horizons"] == ORDER_BLOCK_HORIZONS
    assert observed["sort_items"] is scanner_module.sort_order_block_candidates
    assert observed["sort_failures"] is scanner_module.sort_order_block_failures


def test_order_block_evaluation_maps_validation_internal_and_no_zone_failures(monkeypatch):
    screener = Screener(symbols=("BTC/USDT",), exchange="binance")
    key = ScreeningJobKey("BTC/USDT", "binance", "1h", 1)
    import tempest_mcp.strategies.backtest_order_blocks as order_blocks

    monkeypatch.setattr(
        order_blocks,
        "detect_active_order_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad")),
    )
    _, failure = screener._evaluate_order_block_frame(
        key, frame(rows=24), atr_period=14, impulse_atr_mult=1.0, max_zone_age_bars=20
    )
    assert failure.reason == "order_block_validation_failed"

    monkeypatch.setattr(
        order_blocks,
        "detect_active_order_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    _, failure = screener._evaluate_order_block_frame(
        key, frame(rows=24), atr_period=14, impulse_atr_mult=1.0, max_zone_age_bars=20
    )
    assert failure.reason == "internal_error"

    monkeypatch.setattr(order_blocks, "detect_active_order_blocks", lambda *args, **kwargs: [])
    _, failure = screener._evaluate_order_block_frame(
        key, frame(rows=24), atr_period=14, impulse_atr_mult=1.0, max_zone_age_bars=20
    )
    assert failure.reason == "no_active_order_blocks"


def test_run_symbol_horizon_jobs_maps_insufficient_bars_through_order_block_callback():
    fetcher = RecordingFetcher({("BTC/USDT", "1h"): frame(rows=3)})

    def evaluate(key: ScreeningJobKey, df: pd.DataFrame, limit: int):
        if len(df) < limit:
            return None, OrderBlockFailure(
                key.symbol,
                key.exchange,
                key.timeframe or "",
                key.window_days or 0,
                "insufficient_bars",
            )
        return None, None

    _, failures = run_symbol_horizon_jobs(
        symbols=("BTC/USDT",),
        exchange="binance",
        fetcher=fetcher,
        horizons=(("1h", 1),),
        limit_for_horizon=lambda _timeframe, _window_days: 24,
        empty_failure=lambda key: OrderBlockFailure(
            key.symbol, key.exchange, key.timeframe or "", key.window_days or 0, "data_unavailable"
        ),
        fetch_failure=lambda key, _exc: OrderBlockFailure(
            key.symbol, key.exchange, key.timeframe or "", key.window_days or 0, "data_unavailable"
        ),
        evaluate=evaluate,
        sort_items=sort_order_block_candidates,
        sort_failures=sort_order_block_failures,
    )

    assert [failure.reason for failure in failures] == ["insufficient_bars"]


@pytest.mark.asyncio
async def test_public_full_failure_envelopes_remain_distinct(monkeypatch):
    monkeypatch.setattr(
        Screener,
        "scan",
        lambda self: ([], [ScanFailure("BTC/USDT", "binance", "empty_ohlcv")]),
    )
    screener_response = await screener_tools.screener_scan(symbols=["BTC/USDT"])
    assert screener_response["success"] is False
    assert screener_response["error"]["code"] == ErrorCodes.DATA_SOURCE_ERROR
    assert "data" not in screener_response

    monkeypatch.setattr(
        Screener,
        "session_breakout_scan",
        lambda self, **kwargs: ([], [ScanFailure("BTC/USDT", "binance", "empty_ohlcv")]),
    )
    session_response = await screener_tools.session_breakout_scan(
        session="ny", symbols=["BTC/USDT"]
    )
    assert session_response["success"] is False
    assert session_response["data"]["results"] == []
    assert session_response["data"]["failures"] == [
        {"symbol": "BTC/USDT", "exchange": "binance", "reason": "empty_ohlcv"}
    ]

    monkeypatch.setattr(
        Screener,
        "order_block_scan",
        lambda self, **kwargs: (
            [],
            [OrderBlockFailure("BTC/USDT", "binance", "1h", 1, "data_unavailable")],
        ),
    )
    order_response = await screener_tools.order_block_screener_scan(symbols=["BTC/USDT"])
    assert order_response["success"] is False
    assert order_response["error"]["message"] == "All symbol/horizon jobs failed"
    assert order_response["data"]["candidates"] == []


def test_order_block_serialization_is_json_shape_and_omits_source_used():
    candidate = OrderBlockCandidate(
        "BTC/USDT", "binance", "4h", 7, 1.0, 10.0, "bullish", 11.0, 9.0, 1, 0.95
    )
    serialized = serialize_order_block_candidate(candidate)
    assert serialized == {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "timeframe": "4h",
        "window_days": 7,
        "timestamp": 1.0,
        "price": 10.0,
        "zone_type": "bullish",
        "zone_high": 11.0,
        "zone_low": 9.0,
        "freshness_candles": 1,
        "score": 0.95,
    }
    assert "source_used" not in serialized
