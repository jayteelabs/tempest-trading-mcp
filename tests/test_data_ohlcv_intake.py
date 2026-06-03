"""Tests for the historical OHLCV intake seam — ENG-177."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from tempest_mcp.data._intake import OhlcvIntake, OhlcvRequest


def _frame(rows: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": range(rows),
            "high": range(1, rows + 1),
            "low": range(rows),
            "close": range(1, rows + 1),
            "volume": [100.0] * rows,
        },
        index=idx,
    )


class FakeCcxt:
    def __init__(self, frame: pd.DataFrame | None = None, exc: Exception | None = None):
        self.frame = frame if frame is not None else pd.DataFrame()
        self.exc = exc
        self.calls = []

    def fetch_ohlcv_historical(self, symbol, timeframe="1d", since=None, limit=1000, params=None):
        self.calls.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "since": since,
                "limit": limit,
                "params": params or {},
            }
        )
        if self.exc:
            raise self.exc
        frame = self.frame
        if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
            return frame
        if since is not None:
            since_ts = pd.to_datetime(since, unit="ms", utc=True)
            frame = frame[frame.index >= since_ts]
        until = (params or {}).get("until")
        if until is not None:
            until_ts = pd.to_datetime(until, unit="ms", utc=True)
            frame = frame[frame.index < until_ts]
        return frame.head(limit)


class FakeYf:
    def __init__(self, frame: pd.DataFrame | None = None):
        self.frame = frame if frame is not None else pd.DataFrame()
        self.calls = []

    def fetch_ohlcv(self, symbol, interval="1d", start=None, end=None, auto_adjust=True):
        self.calls.append(
            {
                "symbol": symbol,
                "interval": interval,
                "start": start,
                "end": end,
                "auto_adjust": auto_adjust,
            }
        )
        return self.frame


def test_ccxt_success_skips_yfinance_and_canonicalizes():
    ccxt = FakeCcxt(_frame())
    yf = FakeYf(_frame())
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=yf)

    result = intake.fetch(OhlcvRequest(symbol="BTCUSDT", timeframe="1h", limit=2))

    assert result.source_used == "ccxt"
    assert result.canonical_symbol == "BTC/USDT"
    assert result.provider_symbol == "BTC/USDT"
    assert list(result.frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(result.frame) == 2
    assert yf.calls == []
    assert ccxt.calls[0]["limit"] == 2


def test_ccxt_empty_falls_back_to_yfinance():
    ccxt = FakeCcxt(pd.DataFrame())
    yf = FakeYf(_frame())
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=yf)

    result = intake.fetch(OhlcvRequest(symbol="BTCUSDT", timeframe="1d", limit=3))

    assert result.source_used == "yfinance"
    assert result.canonical_symbol == "BTC/USDT"
    assert result.provider_symbol == "BTC-USD"
    assert yf.calls[0]["symbol"] == "BTC-USD"


def test_ccxt_exception_falls_back_with_warning():
    ccxt = FakeCcxt(exc=RuntimeError("boom"))
    yf = FakeYf(_frame())
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=yf)

    result = intake.fetch(OhlcvRequest(symbol="BTCUSDT", timeframe="1d"))

    assert result.source_used == "yfinance"
    assert "ccxt_fetch_error" in result.warnings


def test_direct_yfinance_usd_path_does_not_call_ccxt():
    ccxt = FakeCcxt(_frame())
    yf = FakeYf(_frame())
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=yf)

    result = intake.fetch(OhlcvRequest(symbol="BTC-USD", timeframe="1d", limit=2))

    assert result.source_used == "yfinance"
    assert result.canonical_symbol == "BTC/USDT"
    assert result.provider_symbol == "BTC-USD"
    assert ccxt.calls == []


def test_invalid_symbol_returns_internal_empty_result():
    intake = OhlcvIntake(ccxt_adapter=FakeCcxt(), yf_adapter=FakeYf())

    result = intake.fetch(OhlcvRequest(symbol="INVALID@#$", timeframe="1d"))

    assert result.source_used == "empty"
    assert result.failure_reason == "invalid_symbol"
    assert result.frame.empty


def test_invalid_exchange_timeframe_and_window_return_failures():
    intake = OhlcvIntake(ccxt_adapter=FakeCcxt(), yf_adapter=FakeYf())

    assert (
        intake.fetch(OhlcvRequest(symbol="BTCUSDT", exchange="bad")).failure_reason
        == "invalid_exchange"
    )
    assert (
        intake.fetch(OhlcvRequest(symbol="BTCUSDT", timeframe="2h")).failure_reason
        == "invalid_timeframe"
    )
    assert (
        intake.fetch(
            OhlcvRequest(
                symbol="BTCUSDT",
                start=datetime(2024, 1, 2, tzinfo=timezone.utc),
                end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            )
        ).failure_reason
        == "invalid_window"
    )


def test_direct_yfinance_4h_is_unsupported():
    intake = OhlcvIntake(ccxt_adapter=FakeCcxt(_frame()), yf_adapter=FakeYf(_frame()))

    result = intake.fetch(OhlcvRequest(symbol="BTC-USD", timeframe="4h"))

    assert result.source_used == "empty"
    assert result.failure_reason == "unsupported_yfinance_interval"


def test_window_limit_and_rsi_warmup_fetch_size():
    ccxt = FakeCcxt(_frame(50))
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=FakeYf())

    result = intake.fetch(OhlcvRequest(symbol="BTCUSDT", timeframe="1h", limit=10, warmup_bars=19))

    assert ccxt.calls[0]["limit"] == 29
    assert len(result.frame) == 29


def test_start_only_limit_returns_latest_rows_after_provider_limiting():
    ccxt = FakeCcxt(_frame(20))
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=FakeYf())

    result = intake.fetch(
        OhlcvRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            limit=5,
        )
    )

    assert list(result.frame.index.hour) == [15, 16, 17, 18, 19]


def test_start_end_limit_returns_latest_rows_inside_window_after_provider_limiting():
    ccxt = FakeCcxt(_frame(20))
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=FakeYf())

    result = intake.fetch(
        OhlcvRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 1, 20, tzinfo=timezone.utc),
            limit=5,
        )
    )

    assert list(result.frame.index.hour) == [15, 16, 17, 18, 19]


def test_end_limit_returns_latest_rows_ending_at_end_after_provider_limiting():
    ccxt = FakeCcxt(_frame(20))
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=FakeYf())

    result = intake.fetch(
        OhlcvRequest(
            symbol="BTCUSDT",
            timeframe="1h",
            end=datetime(2024, 1, 1, 20, tzinfo=timezone.utc),
            limit=5,
        )
    )

    assert list(result.frame.index.hour) == [15, 16, 17, 18, 19]


def test_start_end_window_sets_since_until_and_naive_warning():
    ccxt = FakeCcxt(_frame())
    intake = OhlcvIntake(ccxt_adapter=ccxt, yf_adapter=FakeYf())

    result = intake.fetch(
        OhlcvRequest(
            symbol="BTCUSDT", timeframe="1h", start=datetime(2024, 1, 1), end=datetime(2024, 1, 2)
        )
    )

    assert ccxt.calls[0]["since"] is not None
    assert ccxt.calls[0]["params"]["until"] is not None
    assert ccxt.calls[0]["limit"] == 24
    assert any("naive_interpreted" in warning for warning in result.warnings)


def test_all_providers_empty_returns_canonical_empty():
    intake = OhlcvIntake(ccxt_adapter=FakeCcxt(pd.DataFrame()), yf_adapter=FakeYf(pd.DataFrame()))

    result = intake.fetch(OhlcvRequest(symbol="BTCUSDT", timeframe="1d"))

    assert result.source_used == "empty"
    assert result.failure_reason == "empty_ohlcv"
    assert list(result.frame.columns) == ["open", "high", "low", "close", "volume"]
