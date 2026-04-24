"""Tests for backtest_window.py — ENG-17 Phase 2 window resolution and validation."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tempest_mcp.tools.backtest_window import (
    SUPPORTED_TIMEFRAMES,
    BacktestWindowRequest,
    ResolvedBacktestWindow,
    _estimate_candle_count,
    resolve_backtest_window,
)


class TestResolveBacktestWindow:
    """Tests for resolve_backtest_window() — preset resolution and validation."""

    def test_day_trade_preset_uses_1h_24h(self):
        """day_trade preset defaults to 1h timeframe over 24h."""
        request = BacktestWindowRequest(symbol="BTC/USDT", trade_style="day_trade")
        result = resolve_backtest_window(request)

        assert result.trade_style == "day_trade"
        assert result.timeframe == "1h"
        assert result.symbol == "BTC/USDT"
        # Duration should be ~1 day
        duration = result.end_at_utc - result.start_at_utc
        assert timedelta(hours=23) <= duration <= timedelta(hours=25)

    def test_swing_trade_preset_uses_4h_7d(self):
        """swing_trade preset defaults to 4h timeframe over 7 days."""
        request = BacktestWindowRequest(symbol="ETH/USDT", trade_style="swing_trade")
        result = resolve_backtest_window(request)

        assert result.trade_style == "swing_trade"
        assert result.timeframe == "4h"
        duration = result.end_at_utc - result.start_at_utc
        assert timedelta(days=6) <= duration <= timedelta(days=8)

    def test_custom_requires_start_and_end(self):
        """custom trade_style requires both start_at and end_at."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1),
            end_at=datetime(2024, 1, 2),
        )
        result = resolve_backtest_window(request)

        assert result.trade_style == "custom"
        # Naive datetime is interpreted in America/New_York, then converted to UTC
        # So the UTC timestamp is 5 hours later than the naive input
        assert result.start_at_utc == pd.Timestamp("2024-01-01", tz="America/New_York").tz_convert(
            "UTC"
        )
        assert result.end_at_utc == pd.Timestamp("2024-01-02", tz="America/New_York").tz_convert(
            "UTC"
        )

    def test_custom_rejects_missing_start_at(self):
        """custom without start_at raises ValueError."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            end_at=datetime(2024, 1, 2),
        )
        with pytest.raises(ValueError, match="start_at and end_at"):
            resolve_backtest_window(request)

    def test_custom_rejects_missing_end_at(self):
        """custom without end_at raises ValueError."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1),
        )
        with pytest.raises(ValueError, match="start_at and end_at"):
            resolve_backtest_window(request)

    def test_preset_rejects_start_at(self):
        """Non-custom trade_style rejects explicit start_at (strict reject)."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="day_trade",
            start_at=datetime(2024, 1, 1),
        )
        with pytest.raises(ValueError, match="does not support start_at"):
            resolve_backtest_window(request)

    def test_preset_rejects_end_at(self):
        """Non-custom trade_style rejects explicit end_at (strict reject)."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="swing_trade",
            end_at=datetime(2024, 1, 7),
        )
        with pytest.raises(ValueError, match="does not support start_at or end_at"):
            resolve_backtest_window(request)

    def test_start_at_after_end_at_raises(self):
        """start_at >= end_at raises ValueError."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 10),
            end_at=datetime(2024, 1, 1),
        )
        with pytest.raises(ValueError, match="start_at.*must be before end_at"):
            resolve_backtest_window(request)

    def test_start_at_equals_end_at_raises(self):
        """start_at == end_at raises ValueError."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1),
            end_at=datetime(2024, 1, 1),
        )
        with pytest.raises(ValueError, match="start_at.*must be before end_at"):
            resolve_backtest_window(request)

    def test_custom_allows_explicit_timeframe_override(self):
        """custom trade_style accepts explicit timeframe override."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1),
            end_at=datetime(2024, 1, 2),
            timeframe="4h",
        )
        result = resolve_backtest_window(request)
        assert result.timeframe == "4h"

    def test_invalid_trade_style_raises(self):
        """Unsupported trade_style is rejected explicitly."""
        request = BacktestWindowRequest(symbol="BTC/USDT", trade_style="invalid")
        with pytest.raises(ValueError, match="trade_style must be one of"):
            resolve_backtest_window(request)

    def test_invalid_timeframe_raises(self):
        """Unsupported timeframe is rejected instead of silently defaulting."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="day_trade",
            timeframe="2h",
        )
        with pytest.raises(ValueError, match="timeframe must be one of"):
            resolve_backtest_window(request)

    def test_invalid_max_bars_raises(self):
        """max_bars must be a positive integer."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="day_trade",
            max_bars=0,
        )
        with pytest.raises(ValueError, match="max_bars must be an integer greater than 0"):
            resolve_backtest_window(request)

    def test_window_too_large_exceeds_hard_cap(self):
        """Window exceeding MAX_BARS_HARD_CAP raises ValueError."""
        # 2 years at 1h = ~17,520 bars, well over 1000 cap
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2022, 1, 1),
            end_at=datetime(2024, 1, 1),
            timeframe="1h",
        )
        with pytest.raises(ValueError, match="exceeds hard safety cap"):
            resolve_backtest_window(request)

    def test_window_exceeds_caller_max_bars(self):
        """Window exceeding caller-supplied max_bars raises ValueError."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1),
            end_at=datetime(2024, 1, 10),
            timeframe="1h",
            max_bars=10,  # 10 bars at 1h over 9 days = ~216 bars
        )
        with pytest.raises(ValueError, match="exceeds caller-supplied max_bars"):
            resolve_backtest_window(request)

    def test_naive_datetime_interpreted_in_business_timezone(self):
        """Naive datetime is interpreted in America/New_York before UTC conversion."""
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1, 12, 0, 0),  # naive
            end_at=datetime(2024, 1, 2, 12, 0, 0),  # naive
        )
        result = resolve_backtest_window(request)
        assert result.start_at_utc == pd.Timestamp(
            "2024-01-01 12:00:00", tz="America/New_York"
        ).tz_convert("UTC")
        assert result.end_at_utc == pd.Timestamp(
            "2024-01-02 12:00:00", tz="America/New_York"
        ).tz_convert("UTC")

    def test_estimated_bars_calculation(self):
        """Estimated bars is computed from duration and timeframe."""
        # 24 hours at 1h = 24 bars
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
            timeframe="1h",
        )
        result = resolve_backtest_window(request)
        # 24 hours / 1 hour = 24 bars (approximately)
        assert 20 <= result.estimated_bars <= 30


class TestEstimateCandleCount:
    """Tests for _estimate_candle_count()."""

    def test_1h_over_24h(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        count = _estimate_candle_count(pd.Timestamp(start), pd.Timestamp(end), "1h")
        assert count == 24

    def test_4h_over_7d(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 8, tzinfo=timezone.utc)
        count = _estimate_candle_count(pd.Timestamp(start), pd.Timestamp(end), "4h")
        assert count == 42  # 7 days * 6 (4h intervals per day)

    def test_1d_over_30d(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 31, tzinfo=timezone.utc)
        count = _estimate_candle_count(pd.Timestamp(start), pd.Timestamp(end), "1d")
        assert count == 30

    def test_unknown_timeframe_raises(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="timeframe must be one of"):
            _estimate_candle_count(pd.Timestamp(start), pd.Timestamp(end), "unknown")

    def test_supported_timeframes_are_covered(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        for timeframe in SUPPORTED_TIMEFRAMES:
            count = _estimate_candle_count(pd.Timestamp(start), pd.Timestamp(end), timeframe)
            assert count >= 1

    def test_minimum_one_bar(self):
        """Even very short windows return at least 1 bar."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 0, 1, tzinfo=timezone.utc)
        count = _estimate_candle_count(pd.Timestamp(start), pd.Timestamp(end), "1h")
        assert count == 1


class TestBacktestWindowRequest:
    """Tests for BacktestWindowRequest dataclass."""

    def test_defaults(self):
        """Default values are applied correctly."""
        request = BacktestWindowRequest(symbol="BTC/USDT")
        assert request.trade_style == "day_trade"
        assert request.timeframe is None
        assert request.start_at is None
        assert request.end_at is None
        assert request.exchange == "binance"
        assert request.max_bars is None

    def test_all_params_set(self):
        """All parameters can be set."""
        request = BacktestWindowRequest(
            symbol="ETH/USDT",
            trade_style="custom",
            timeframe="4h",
            start_at=datetime(2024, 1, 1),
            end_at=datetime(2024, 1, 7),
            exchange="bybit",
            max_bars=500,
        )
        assert request.symbol == "ETH/USDT"
        assert request.trade_style == "custom"
        assert request.timeframe == "4h"
        assert request.exchange == "bybit"
        assert request.max_bars == 500


class TestResolvedBacktestWindow:
    """Tests for ResolvedBacktestWindow dataclass."""

    def test_frozen_immutable(self):
        """ResolvedBacktestWindow is frozen and immutable."""
        window = ResolvedBacktestWindow(
            symbol="BTC/USDT",
            trade_style="day_trade",
            timeframe="1h",
            start_at_utc=pd.Timestamp("2024-01-01", tz="UTC"),
            end_at_utc=pd.Timestamp("2024-01-02", tz="UTC"),
            estimated_bars=24,
            exchange="binance",
        )
        with pytest.raises(AttributeError):
            window.symbol = "ETH/USDT"

class TestFetchResolvedOhlcv:
    """Regression tests for fetch_resolved_ohlcv() — ENG-127 fix.

    Verifies that:
    1. fetch_resolved_ohlcv unpacks the tuple returned by HistoricalDataSource.fetch_ohlcv
    2. window.exchange is passed to HistoricalDataSource
    3. Only the DataFrame is returned (not the tuple)
    """

    def test_unpacks_tuple_and_returns_dataframe(self, monkeypatch):
        """fetch_resolved_ohlcv must unpack (DataFrame, source_used) tuple, not return it.

        Prior to the ENG-127 fix, fetch_resolved_ohlcv stored the tuple directly
        as df, causing every downstream caller to receive a tuple instead of a DataFrame.
        """
        import pandas as pd

        from tempest_mcp.tools.backtest_window import (
            BacktestWindowRequest,
            fetch_resolved_ohlcv,
            resolve_backtest_window,
        )

        # Build a minimal resolved window
        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
            end_at=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
            timeframe="1h",
            exchange="binance",
        )
        resolved = resolve_backtest_window(request)

        # Mock HistoricalDataSource to return the tuple that _hist.py actually returns
        mock_df = pd.DataFrame(
            {"open": [100], "high": [105], "low": [99], "close": [103], "volume": [1000]},
            index=pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")]),
        )

        class MockHistoricalDataSource:
            def __init__(self, exchange_name: str = "binance"):
                self.exchange_name = exchange_name

            def fetch_ohlcv(self, symbol, interval, start, end):
                # Must return tuple[DataFrame, str] as _hist.py now does
                return mock_df, "ccxt"

        # More precise: replace the class in the module
        import tempest_mcp.tools.backtest_window as bw_module
        bw_module.HistoricalDataSource = MockHistoricalDataSource  # type: ignore

        result = fetch_resolved_ohlcv(resolved)

        # The fix ensures result is a DataFrame, not a tuple
        assert isinstance(result, pd.DataFrame), (
            f"fetch_resolved_ohlcv must return pd.DataFrame, got {type(result).__name__}. "
            "ENG-127: tuple unpacking regression."
        )
        assert len(result) == 1

    def test_exchange_passed_to_historical_data_source(self, monkeypatch):
        """window.exchange must be forwarded to HistoricalDataSource constructor.

        Prior to the ENG-127 fix, HistoricalDataSource() was called with no arguments,
        always defaulting to 'binance' regardless of the resolved window's exchange.
        """
        import pandas as pd

        from tempest_mcp.tools.backtest_window import (
            BacktestWindowRequest,
            fetch_resolved_ohlcv,
            resolve_backtest_window,
        )

        request = BacktestWindowRequest(
            symbol="ETH/USDT",
            trade_style="custom",
            start_at=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
            end_at=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
            timeframe="1h",
            exchange="bybit",
        )
        resolved = resolve_backtest_window(request)

        captured_exchange_name: list[str] = []

        class MockHistoricalDataSource:
            def __init__(self, exchange_name: str = "binance"):
                captured_exchange_name.append(exchange_name)

            def fetch_ohlcv(self, symbol, interval, start, end):
                return pd.DataFrame(), "ccxt"

        import tempest_mcp.tools.backtest_window as bw_module
        bw_module.HistoricalDataSource = MockHistoricalDataSource  # type: ignore

        fetch_resolved_ohlcv(resolved)

        assert captured_exchange_name == ["bybit"], (
            f"HistoricalDataSource must be instantiated with window.exchange='bybit', "
            f"got {captured_exchange_name!r}. ENG-127: exchange plumbing regression."
        )


class TestResolveAndFetchBacktestOhlcv:
    """Regression tests for resolve_and_fetch_backtest_ohlcv() — ENG-127 fix."""

    def test_returns_dataframe_not_tuple(self, monkeypatch):
        """resolve_and_fetch_backtest_ohlcv must return (DataFrame, window) not (tuple, window).

        This is the canonical entry point used by all backtest and analytical tools.
        The bug caused all callers to receive a (tuple[DataFrame, str], window) instead
        of (DataFrame, window), collapsing every affected tool into a code-9000 failure.
        """
        import pandas as pd

        from tempest_mcp.tools.backtest_window import (
            BacktestWindowRequest,
            resolve_and_fetch_backtest_ohlcv,
        )

        mock_df = pd.DataFrame(
            {"open": [100, 101], "high": [105, 106], "low": [99, 100], "close": [103, 104], "volume": [1000, 1100]},
            index=pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC"),
        )

        class MockHistoricalDataSource:
            def __init__(self, exchange_name: str = "binance"):
                pass

            def fetch_ohlcv(self, symbol, interval, start, end):
                return mock_df, "ccxt"

        import tempest_mcp.tools.backtest_window as bw_module
        bw_module.HistoricalDataSource = MockHistoricalDataSource  # type: ignore

        request = BacktestWindowRequest(
            symbol="BTC/USDT",
            trade_style="custom",
            start_at=pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(),
            end_at=pd.Timestamp("2024-01-03", tz="UTC").to_pydatetime(),
            timeframe="1h",
        )

        ohlcv_df, resolved_window = resolve_and_fetch_backtest_ohlcv(request)

        assert isinstance(ohlcv_df, pd.DataFrame), (
            f"resolve_and_fetch_backtest_ohlcv must return DataFrame as first tuple element, "
            f"got {type(ohlcv_df).__name__}. ENG-127 regression."
        )
        assert len(ohlcv_df) == 2
        assert resolved_window.symbol == "BTC/USDT"
