"""Focused tests for the internal OHLCV lifecycle helper."""

from datetime import datetime

import pandas as pd
import pytest

from tempest_mcp.config import ErrorCodes
from tempest_mcp.tools._ohlcv_lifecycle import (
    OhlcvLifecycleRequest,
    backtest_window_payload,
    build_backtest_window_request,
    internal_error,
    min_bars_check,
    parse_iso_datetime,
    run_ohlcv_lifecycle,
    validation_error,
)
from tempest_mcp.tools.backtest_window import ResolvedBacktestWindow


def _window():
    return ResolvedBacktestWindow(
        symbol="BTC/USDT",
        trade_style="custom",
        timeframe="1h",
        start_at_utc=pd.Timestamp("2024-01-01T00:00:00Z"),
        end_at_utc=pd.Timestamp("2024-01-01T02:00:00Z"),
        estimated_bars=2,
        exchange="binance",
    )


def _request():
    return OhlcvLifecycleRequest(
        tool_name="unit_tool",
        symbol="BTC/USDT",
        trade_style="custom",
        timeframe="1h",
        start_at=parse_iso_datetime("start_at", "2024-01-01T00:00:00Z"),
        end_at=parse_iso_datetime("end_at", "2024-01-01T02:00:00Z"),
        exchange="binance",
        max_bars=10,
    )


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, *args, **kwargs):
        self.errors.append((args, kwargs))


def test_parse_iso_datetime_preserves_existing_semantics():
    parsed = parse_iso_datetime("start_at", "2024-01-01T00:00:00Z")
    assert parsed == datetime.fromisoformat("2024-01-01T00:00:00+00:00")
    assert parse_iso_datetime("start_at", parsed) is parsed
    with pytest.raises(ValueError, match="start_at must be a valid ISO 8601 datetime"):
        parse_iso_datetime("start_at", object())


def test_error_envelopes_use_existing_codes():
    assert validation_error("bad") == {
        "success": False,
        "error": {"code": ErrorCodes.INVALID_PARAMETER, "message": "bad"},
    }
    assert internal_error("boom") == {
        "success": False,
        "error": {"code": ErrorCodes.INTERNAL_ERROR, "message": "boom"},
    }


def test_request_construction_and_window_payload():
    built = build_backtest_window_request(_request())
    assert built.symbol == "BTC/USDT"
    assert built.trade_style == "custom"
    assert built.timeframe == "1h"
    assert built.max_bars == 10
    payload = backtest_window_payload(_window())
    assert list(payload) == [
        "trade_style",
        "timeframe",
        "start_at_utc",
        "end_at_utc",
        "estimated_bars",
        "exchange",
    ]
    assert "source_used" not in payload


def test_lifecycle_success_callback_receives_df_and_window(monkeypatch):
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    window = _window()

    def fake_fetch(request):
        assert request.symbol == "BTC/USDT"
        return df, window

    monkeypatch.setattr("tempest_mcp.tools._ohlcv_lifecycle.resolve_and_fetch_backtest_ohlcv", fake_fetch)
    seen = {}

    def callback(ohlcv_df, resolved_window):
        seen["same_df"] = ohlcv_df is df
        seen["same_window"] = resolved_window is window
        return {"tool": "unit_tool", "window": backtest_window_payload(resolved_window)}

    result = run_ohlcv_lifecycle(
        _request(),
        logger=_Logger(),
        callback=callback,
        sufficiency_check=min_bars_check(2),
        calculation_error_message="Unit calculation failed",
        fetch_ohlcv=fake_fetch,
    )
    assert result["success"] is True
    assert seen == {"same_df": True, "same_window": True}
    assert "source_used" not in result["data"]["window"]


def test_lifecycle_error_mapping(monkeypatch):
    logger = _Logger()

    def raises_value_error(_request):
        raise ValueError("bad window")

    monkeypatch.setattr("tempest_mcp.tools._ohlcv_lifecycle.resolve_and_fetch_backtest_ohlcv", raises_value_error)
    result = run_ohlcv_lifecycle(
        _request(), logger=logger, callback=lambda *_: {}, calculation_error_message="Calc failed", fetch_ohlcv=raises_value_error
    )
    assert result["error"] == {"code": ErrorCodes.INVALID_PARAMETER, "message": "bad window"}

    def raises_runtime(_request):
        raise RuntimeError("network")

    monkeypatch.setattr("tempest_mcp.tools._ohlcv_lifecycle.resolve_and_fetch_backtest_ohlcv", raises_runtime)
    result = run_ohlcv_lifecycle(
        _request(), logger=logger, callback=lambda *_: {}, calculation_error_message="Calc failed", fetch_ohlcv=raises_runtime
    )
    assert result["error"] == {"code": ErrorCodes.INTERNAL_ERROR, "message": "Data fetch failed"}


def test_lifecycle_insufficient_and_callback_exception_mapping(monkeypatch):
    df = pd.DataFrame({"close": [1.0]}, index=pd.date_range("2024-01-01", periods=1, tz="UTC"))
    monkeypatch.setattr(
        "tempest_mcp.tools._ohlcv_lifecycle.resolve_and_fetch_backtest_ohlcv",
        lambda _request: (df, _window()),
    )
    result = run_ohlcv_lifecycle(
        _request(),
        logger=_Logger(),
        callback=lambda *_: {"unreachable": True},
        sufficiency_check=min_bars_check(2),
        calculation_error_message="Calc failed",
        fetch_ohlcv=lambda _request: (df, _window()),
    )
    assert result["error"]["code"] == ErrorCodes.INVALID_PARAMETER
    assert result["error"]["message"] == "Insufficient data: only 1 bars returned (minimum 2 required)"

    df2 = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    monkeypatch.setattr(
        "tempest_mcp.tools._ohlcv_lifecycle.resolve_and_fetch_backtest_ohlcv",
        lambda _request: (df2, _window()),
    )
    result = run_ohlcv_lifecycle(
        _request(),
        logger=_Logger(),
        callback=lambda *_: (_ for _ in ()).throw(ValueError("bad params")),
        calculation_error_message="Calc failed",
        fetch_ohlcv=lambda _request: (df2, _window()),
    )
    assert result["error"] == {"code": ErrorCodes.INVALID_PARAMETER, "message": "bad params"}

    result = run_ohlcv_lifecycle(
        _request(),
        logger=_Logger(),
        callback=lambda *_: (_ for _ in ()).throw(RuntimeError("boom")),
        calculation_error_message="Calc failed",
        fetch_ohlcv=lambda _request: (df2, _window()),
    )
    assert result["error"] == {"code": ErrorCodes.INTERNAL_ERROR, "message": "Calc failed"}
