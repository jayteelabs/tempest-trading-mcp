"""Tests for the PDH/PDL + Session Levels backtest strategy (ENG-19)."""

from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.strategies.backtest_pdh_session import run_pdh_session_backtest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(
    year: int,
    month: int,
    day: int,
    hour: int = 0,
    minute: int = 0,
) -> pd.Timestamp:
    return pd.Timestamp(datetime(year, month, day, hour, minute), tz=timezone.utc)


def _ohlcv(
    timestamps: list[pd.Timestamp],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Build OHLCV DataFrame from explicit flat lists."""
    if volumes is None:
        volumes = [1000.0] * len(timestamps)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=pd.DatetimeIndex(timestamps, tz=timezone.utc),
    )


def _hourly_range(
    start: pd.Timestamp,
    n: int,
    *,
    start_price: float = 100.0,
    step: float = 0.0,
    high_bias: float = 0.005,
    low_bias: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Build n hourly bars with near-deterministic O/H/L/C.

    ``step`` adds a fixed amount to close each bar (useful for trends).
    ``high_bias`` and ``low_bias`` are fractions of price for intrabar spread.
    """
    rng = np.random.RandomState(seed)
    dates = [start + pd.Timedelta(hours=i) for i in range(n)]
    opens, highs, lows, closes, vols = [], [], [], [], []
    price = start_price
    for _ in range(n):
        opens.append(price)
        rng_val = rng.uniform(-low_bias, high_bias)
        close = price * (1 + rng_val)
        highs.append(max(price, close) * (1 + rng.uniform(0, high_bias / 2)))
        lows.append(min(price, close) * (1 - rng.uniform(0, low_bias / 2)))
        closes.append(close)
        vols.append(float(rng.randint(1000, 5000)))
        price = close + step
    return _ohlcv(dates, opens, highs, lows, closes, vols)


def _intraday_hours_range(
    start: pd.Timestamp,
    day_count: int,
    *,
    hours: range,
    price: float = 100.0,
) -> pd.DataFrame:
    """Build flat OHLCV bars for selected hours on consecutive UTC dates."""
    timestamps = [
        start + pd.Timedelta(days=day_offset, hours=hour)
        for day_offset in range(day_count)
        for hour in hours
    ]
    bar_count = len(timestamps)
    return _ohlcv(
        timestamps,
        [price] * bar_count,
        [price + 1.0] * bar_count,
        [price - 1.0] * bar_count,
        [price] * bar_count,
    )


def _patch_at(module_qualified_name: str, attr: str, mock: MagicMock):
    """Patch ``attr`` in ``module_qualified_name`` using patch.object on the imported name."""
    module = __import__(module_qualified_name, fromlist=[attr])
    return patch.object(module, attr, mock)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([], tz=timezone.utc)
        with pytest.raises(ValueError, match="empty"):
            run_pdh_session_backtest(df)

    def test_invalid_atr_period_raises(self):
        df = _hourly_range(_ts(2024, 1, 1), 20)
        with pytest.raises(ValueError, match="atr_period"):
            run_pdh_session_backtest(df, atr_period=0)
        with pytest.raises(ValueError, match="atr_period"):
            run_pdh_session_backtest(df, atr_period=-1)

    def test_invalid_session_type_raises(self):
        df = _hourly_range(_ts(2024, 1, 1), 20)
        with pytest.raises(ValueError, match="Invalid session_type"):
            run_pdh_session_backtest(df, session_types=["london", "junk"])

    def test_missing_columns_raises(self):
        df = _hourly_range(_ts(2024, 1, 1), 20).drop(columns=["high"])
        with pytest.raises(ValueError, match="missing required columns"):
            run_pdh_session_backtest(df)


# ---------------------------------------------------------------------------
# Session gating tests
# ---------------------------------------------------------------------------


class TestSessionGating:
    def test_no_entry_during_asia_only_hours(self):
        """Data containing only Asia-session hours must not produce entry signals."""
        df = _intraday_hours_range(_ts(2023, 12, 31), 4, hours=range(9))
        signals, _ = run_pdh_session_backtest(df)
        non_hold = signals[signals != SignalAction.HOLD]
        assert len(non_hold) == 0, f"Expected no signals, got indices {list(non_hold.index)}"

    def test_asia_only_data_produces_no_signals(self):
        """Asia-only data (all bars 00:00-08:00 UTC) produces no signals."""
        df = _intraday_hours_range(_ts(2023, 12, 31), 4, hours=range(9), price=101.0)
        signals, _ = run_pdh_session_backtest(df)
        non_hold = signals[signals != SignalAction.HOLD]
        assert len(non_hold) == 0


# ---------------------------------------------------------------------------
# Breakout entry tests  (mock-based)
# ---------------------------------------------------------------------------


class TestBreakoutEntries:
    def _mock_pdh(self, pdh=105.0, pdl=97.0, position="above_pdh"):
        m = MagicMock()
        m.return_value = {
            "previous_day_high": pdh,
            "previous_day_low": pdl,
            "position": position,
        }
        return m

    def _mock_session(self, bars=8):
        m = MagicMock()
        m.return_value = {
            "bars": bars,
            "high": 106.0,
            "low": 100.0,
            "session_start_utc": _ts(2024, 1, 2, 8),
            "session_end_utc": _ts(2024, 1, 2, 16),
        }
        return m

    def _df_london_bar(self, bar_idx: int, close: float, high: float, low: float):
        df = _hourly_range(_ts(2023, 12, 31), 120, start_price=100.0, seed=42)
        df.iloc[bar_idx, df.columns.get_loc("high")] = high
        df.iloc[bar_idx, df.columns.get_loc("low")] = low
        df.iloc[bar_idx, df.columns.get_loc("close")] = close
        return df

    def test_long_entry_close_above_pdh(self):
        df = self._df_london_bar(57, close=106.0, high=106.5, low=100.0)
        mock_p = self._mock_pdh()
        mock_s = self._mock_session()
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[57] == SignalAction.LONG_ENTRY

    def test_short_entry_close_below_pdl(self):
        df = self._df_london_bar(57, close=95.0, high=101.0, low=94.5)
        mock_p = self._mock_pdh(pdh=103.0, pdl=97.0, position="inside_range")
        mock_s = self._mock_session()
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[57] == SignalAction.SHORT_ENTRY

    def test_no_entry_close_inside_range(self):
        df = self._df_london_bar(57, close=100.0, high=101.0, low=99.0)
        mock_p = self._mock_pdh(pdh=103.0, pdl=97.0, position="inside_range")
        mock_s = self._mock_session()
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[57] == SignalAction.HOLD

    def test_no_entry_asia_even_if_above_pdh(self):
        """Asia-session bar (bars=0) suppresses entry even when close > PDH."""
        df = self._df_london_bar(3, close=106.0, high=106.5, low=100.0)  # Asia bar
        mock_p = self._mock_pdh()
        mock_s = MagicMock(return_value={"bars": 0, "high": float("nan"), "low": float("nan")})
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[3] == SignalAction.HOLD

    def test_no_entry_when_current_bar_is_outside_session_window(self):
        """Prior London bars in the window must not qualify a non-session breakout bar."""
        df = self._df_london_bar(20, close=106.0, high=106.5, low=100.0)
        bar_time = df.index[20]
        mock_p = self._mock_pdh()
        mock_s = MagicMock(
            return_value={
                "bars": 9,
                "high": 106.0,
                "low": 100.0,
                "session_start_utc": bar_time.normalize() + pd.Timedelta(hours=8),
                "session_end_utc": bar_time.normalize() + pd.Timedelta(hours=16),
            }
        )
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[20] == SignalAction.HOLD


# ---------------------------------------------------------------------------
# SL / TP tests
# ---------------------------------------------------------------------------


class TestStopLossTakeProfit:
    def _mock_pdh_session(self, pdh=105.0, pdl=97.0):
        mock_p = MagicMock()
        mock_p.return_value = {
            "previous_day_high": pdh,
            "previous_day_low": pdl,
            "position": "above_pdh",
        }
        mock_s = MagicMock()
        mock_s.return_value = {
            "bars": 8,
            "high": 107.0,
            "low": 100.0,
            "session_start_utc": _ts(2024, 1, 2, 8),
            "session_end_utc": _ts(2024, 1, 2, 16),
        }
        return mock_p, mock_s

    def _df_with_atr_seed(self, bar_idx: int, close: float, next_bar: tuple) -> pd.DataFrame:
        """Create df with 14-bar ATR seed (range ≈ 4 → ATR ≈ 4.0)."""
        df = _hourly_range(_ts(2023, 12, 31), 120, start_price=100.0, seed=42)
        # Seed 14 bars with H=104, L=100 → TR=4 each → ATR=4.0
        for i in range(14):
            df.iloc[i, df.columns.get_loc("high")] = 104.0
            df.iloc[i, df.columns.get_loc("low")] = 100.0
            df.iloc[i, df.columns.get_loc("close")] = 102.0
        # Entry bar
        df.iloc[bar_idx, df.columns.get_loc("high")] = 106.5
        df.iloc[bar_idx, df.columns.get_loc("low")] = 100.0
        df.iloc[bar_idx, df.columns.get_loc("close")] = close
        # Next bar (SL/TP trigger bar)
        nb_open, nb_high, nb_low, nb_close = next_bar
        df.iloc[bar_idx + 1, df.columns.get_loc("open")] = nb_open
        df.iloc[bar_idx + 1, df.columns.get_loc("high")] = nb_high
        df.iloc[bar_idx + 1, df.columns.get_loc("low")] = nb_low
        df.iloc[bar_idx + 1, df.columns.get_loc("close")] = nb_close
        return df

    def test_entry_signal_stored_with_stop_distance(self):
        """LONG_ENTRY emitted at entry bar with ATR-based stop."""
        df = self._df_with_atr_seed(57, close=106.0, next_bar=(107.0, 108.0, 100.0, 105.0))
        mock_p, mock_s = self._mock_pdh_session()
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df, atr_multiplier=1.5)
        assert signals.iloc[57] == SignalAction.LONG_ENTRY

    def test_tp_hits_before_sl_emits_long_exit(self):
        """Intraday high >= TP → LONG_EXIT emitted on that bar."""
        # ATR≈4, sd=6, TP=106+12=118. Bar 58: open=107, high=120 (>= TP) → LONG_EXIT
        df = self._df_with_atr_seed(57, close=106.0, next_bar=(107.0, 120.0, 103.0, 115.0))
        mock_p, mock_s = self._mock_pdh_session()
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[57] == SignalAction.LONG_ENTRY
        assert signals.iloc[58] == SignalAction.LONG_EXIT

    def test_sl_hits_before_tp_emits_long_exit(self):
        """Intraday low <= SL → LONG_EXIT emitted on that bar."""
        # ATR≈4, sd=6, SL=106-6=100. Bar 58: open=103 (between), low=99 (<= SL) → LONG_EXIT
        df = self._df_with_atr_seed(57, close=106.0, next_bar=(103.0, 110.0, 99.0, 100.5))
        mock_p, mock_s = self._mock_pdh_session()
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[57] == SignalAction.LONG_ENTRY
        assert signals.iloc[58] == SignalAction.LONG_EXIT


# ---------------------------------------------------------------------------
# Engine integration tests
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_engine_run_called_once(self):
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        with patch.object(BacktestEngine, "run", return_value=None) as m:
            try:
                run_pdh_session_backtest(df)
            except Exception:
                pass
            m.assert_called_once()

    def test_signals_are_signal_action_values(self):
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        signals, _ = run_pdh_session_backtest(df)
        assert signals.dtype == object
        assert all(isinstance(s, SignalAction) for s in signals)

    def test_engine_state_after_run(self):
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        _, engine = run_pdh_session_backtest(df)
        assert isinstance(engine._trades, list)
        assert isinstance(engine._equity_curve, list)
        assert isinstance(engine._has_open_position, bool)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_insufficient_data_no_signals(self):
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        mock_p = MagicMock(return_value={"position": "insufficient_data"})
        mock_s = MagicMock(return_value={"bars": 8, "high": 106.0, "low": 100.0})
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        non_hold = signals[signals != SignalAction.HOLD]
        assert len(non_hold) == 0

    def test_zero_bars_session_no_signal(self):
        df = _hourly_range(_ts(2023, 12, 31), 120, start_price=100.0, seed=42)
        df.iloc[57, df.columns.get_loc("high")] = 106.0
        df.iloc[57, df.columns.get_loc("low")] = 100.0
        df.iloc[57, df.columns.get_loc("close")] = 106.0
        mock_p = MagicMock(
            return_value={
                "previous_day_high": 105.0,
                "previous_day_low": 97.0,
                "position": "above_pdh",
            }
        )
        mock_s = MagicMock(return_value={"bars": 0, "high": float("nan"), "low": float("nan")})
        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)
        assert signals.iloc[57] == SignalAction.HOLD

    def test_pdh_pdl_cache_recomputes_when_evaluated_bar_date_changes(self):
        timestamps = [_ts(2024, 1, 1) + pd.Timedelta(hours=i) for i in range(48)]
        opens = [95.0] * 48
        highs = [96.0] * 48
        lows = [94.0] * 48
        closes = [95.0] * 48

        breakout_idx = 34
        highs[breakout_idx] = 107.0
        closes[breakout_idx] = 106.0
        opens[breakout_idx + 1] = 106.0
        highs[breakout_idx + 1] = 110.0
        lows[breakout_idx + 1] = 104.0
        closes[breakout_idx + 1] = 108.0

        df = _ohlcv(timestamps, opens, highs, lows, closes)

        evaluated_dates: list[date] = []

        def mock_detect_pdh_pdl(window: pd.DataFrame) -> dict:
            bar_date = window.index[-1].date()
            evaluated_dates.append(bar_date)
            if bar_date == timestamps[0].date():
                return {"position": "insufficient_data"}
            return {
                "previous_day_high": 100.0,
                "previous_day_low": 90.0,
                "position": "above_pdh",
            }

        mock_session = MagicMock(
            return_value={
                "bars": 8,
                "high": 107.0,
                "low": 94.0,
                "session_start_utc": _ts(2024, 1, 2, 8),
                "session_end_utc": _ts(2024, 1, 2, 16),
            }
        )

        with _patch_at(
            "tempest_mcp.strategies.backtest_pdh_session",
            "detect_pdh_pdl",
            MagicMock(side_effect=mock_detect_pdh_pdl),
        ):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_session
            ):
                signals, _ = run_pdh_session_backtest(df)

        assert evaluated_dates == [timestamps[0].date(), timestamps[24].date()]
        assert signals.iloc[breakout_idx] == SignalAction.LONG_ENTRY

    def test_same_direction_entry_requires_breakout_reset_before_reentry(self):
        df = _hourly_range(_ts(2023, 12, 31), 120, start_price=100.0, seed=42)

        for i in range(14):
            df.iloc[i, df.columns.get_loc("high")] = 104.0
            df.iloc[i, df.columns.get_loc("low")] = 100.0
            df.iloc[i, df.columns.get_loc("close")] = 102.0

        df.iloc[57, df.columns.get_loc("high")] = 106.5
        df.iloc[57, df.columns.get_loc("low")] = 100.5
        df.iloc[57, df.columns.get_loc("close")] = 106.0

        df.iloc[58, df.columns.get_loc("open")] = 106.0
        df.iloc[58, df.columns.get_loc("high")] = 107.5
        df.iloc[58, df.columns.get_loc("low")] = 105.2
        df.iloc[58, df.columns.get_loc("close")] = 107.0

        df.iloc[59, df.columns.get_loc("open")] = 103.0
        df.iloc[59, df.columns.get_loc("high")] = 104.0
        df.iloc[59, df.columns.get_loc("low")] = 99.0
        df.iloc[59, df.columns.get_loc("close")] = 100.0

        df.iloc[60, df.columns.get_loc("open")] = 100.0
        df.iloc[60, df.columns.get_loc("high")] = 104.5
        df.iloc[60, df.columns.get_loc("low")] = 99.5
        df.iloc[60, df.columns.get_loc("close")] = 104.0

        df.iloc[61, df.columns.get_loc("open")] = 104.0
        df.iloc[61, df.columns.get_loc("high")] = 107.0
        df.iloc[61, df.columns.get_loc("low")] = 103.5
        df.iloc[61, df.columns.get_loc("close")] = 106.0

        mock_p = MagicMock(
            return_value={
                "previous_day_high": 105.0,
                "previous_day_low": 97.0,
                "position": "above_pdh",
            }
        )
        mock_s = MagicMock(
            return_value={
                "bars": 8,
                "high": 107.0,
                "low": 94.0,
                "session_start_utc": _ts(2024, 1, 2, 8),
                "session_end_utc": _ts(2024, 1, 2, 16),
            }
        )

        with _patch_at("tempest_mcp.strategies.backtest_pdh_session", "detect_pdh_pdl", mock_p):
            with _patch_at(
                "tempest_mcp.strategies.backtest_pdh_session", "detect_session_levels", mock_s
            ):
                signals, _ = run_pdh_session_backtest(df)

        assert signals.iloc[57] == SignalAction.LONG_ENTRY
        assert signals.iloc[58] == SignalAction.HOLD
        assert signals.iloc[59] == SignalAction.LONG_EXIT
        assert signals.iloc[60] == SignalAction.HOLD
        assert signals.iloc[61] == SignalAction.LONG_ENTRY


# ---------------------------------------------------------------------------
# Determinism test
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_deterministic_output(self):
        df1 = _hourly_range(_ts(2023, 12, 31), 120, start_price=100.0, seed=42)
        df2 = _hourly_range(_ts(2023, 12, 31), 120, start_price=100.0, seed=42)
        s1, _ = run_pdh_session_backtest(df1)
        s2, _ = run_pdh_session_backtest(df2)
        pd.testing.assert_series_equal(s1, s2)


# ---------------------------------------------------------------------------
# Additional tests for full-coverage real-data scenarios
# ---------------------------------------------------------------------------


class TestRealDataScenarios:
    """Tests using real detect_pdh_pdl / detect_session_levels on constructed data.

    These require the DataFrame to start early enough that detect_pdh_pdl's
    lookback (UTC calendar day before the first bar's date) is satisfied.
    For bars on Jan 3+, data must start on Dec 31 or earlier so that the
    lookback day (Dec 30 or earlier) is present in the DataFrame.
    """

    def test_no_false_signals_before_pdh_pdl_is_valid(self):
        """Before PDH/PDL is valid (insufficient_data), no breakout signals fire."""
        df = _intraday_hours_range(_ts(2023, 12, 31), 4, hours=range(9), price=102.0)
        signals, _ = run_pdh_session_backtest(df)
        # Asia-only data → no eligible session → always HOLD
        non_hold = signals[signals != SignalAction.HOLD]
        assert len(non_hold) == 0


# ---------------------------------------------------------------------------
# Phase 2 contract tests
# ---------------------------------------------------------------------------


class TestPhase2Contract:
    """Phase 2 shared contract: strategy consumes resolved OHLCV from caller.

    It does not own date-range planning or data fetching. Preset/plan
    parameters are informational — they do not change signal logic.
    """

    def test_initial_capital_passed_to_engine(self):
        """initial_capital is forwarded to BacktestEngine."""
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        _, engine = run_pdh_session_backtest(df, initial_capital=50_000.0)
        assert engine.initial_capital == 50_000.0

    def test_trade_style_parameter_accepted(self):
        """trade_style is accepted without error (informational only)."""
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        for style in ("day_trade", "swing_trade", "custom"):
            _, engine = run_pdh_session_backtest(df, trade_style=style)
            assert engine is not None  # No error raised

    def test_timeframe_parameter_accepted(self):
        """timeframe hint is accepted without error (informational only)."""
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        _, engine = run_pdh_session_backtest(df, timeframe="1h")
        assert engine is not None

    def test_start_at_end_at_parameters_accepted(self):
        """start_at and end_at are accepted without error (informational only)."""
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        _, engine = run_pdh_session_backtest(
            df,
            start_at=_ts(2023, 12, 31),
            end_at=_ts(2024, 1, 4),
        )
        assert engine is not None

    def test_exchange_parameter_accepted(self):
        """exchange name is accepted without error (informational only)."""
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        _, engine = run_pdh_session_backtest(df, exchange="binance")
        assert engine is not None

    def test_all_phase2_parameters_combined(self):
        """All Phase 2 preset/plan parameters work together."""
        df = _hourly_range(_ts(2023, 12, 31), 50, start_price=100.0)
        _, engine = run_pdh_session_backtest(
            df,
            trade_style="day_trade",
            timeframe="1h",
            start_at=_ts(2023, 12, 31),
            end_at=_ts(2024, 1, 4),
            exchange="binance",
            initial_capital=75_000.0,
        )
        assert engine.initial_capital == 75_000.0
