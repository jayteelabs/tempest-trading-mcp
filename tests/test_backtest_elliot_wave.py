"""Tests for the Elliot Wave Simplified backtest strategy (ENG-24)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import tempest_mcp.strategies.backtest_elliot_wave as elliot_wave
from tempest_mcp.backtest.engine import SignalAction


def _make_ohlcv(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """Create deterministic OHLCV DataFrame from price data."""
    idx = pd.date_range("2024-01-01 00:00", periods=len(closes), freq="h", tz="UTC")
    if volumes is None:
        volumes = [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "open": opens if opens else closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=idx,
    )


def _make_simple_ohlcv(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Create OHLCV from close prices with surrounding highs/lows."""
    idx = pd.date_range("2024-01-01 00:00", periods=len(closes), freq="h", tz="UTC")
    close_series = pd.Series(closes, index=idx)
    open_series = close_series.shift(1).fillna(close_series.iloc[0])
    high_series = pd.Series(np.maximum(open_series, close_series) + 0.5, index=idx)
    low_series = pd.Series(np.minimum(open_series, close_series) - 0.5, index=idx)
    if volumes is None:
        volumes = [1000.0] * len(closes)
    return pd.DataFrame(
        {
            "open": open_series,
            "high": high_series,
            "low": low_series,
            "close": close_series,
            "volume": volumes,
        },
        index=idx,
    )


class TestElliotWaveStrategyId:
    def test_strategy_id_is_elliot_wave(self):
        assert elliot_wave.STRATEGY_ID == "elliot_wave"


class TestElliotWaveValidation:
    def test_empty_dataframe_raises(self):
        df = _make_ohlcv([], [], [], [])
        with pytest.raises(ValueError, match="must not be empty"):
            elliot_wave.generate_elliot_wave_signals(df)

    def test_missing_required_columns_raises(self):
        df = _make_simple_ohlcv([100, 101, 102])
        df = df.drop(columns=["volume"])
        with pytest.raises(ValueError, match="missing required columns"):
            elliot_wave.generate_elliot_wave_signals(df)

    def test_invalid_swing_window_raises(self):
        df = _make_simple_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108])
        with pytest.raises(ValueError, match="swing_window must be >= 1"):
            elliot_wave.generate_elliot_wave_signals(df, swing_window=0)

    def test_invalid_wave3_retrace_band_raises(self):
        df = _make_simple_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108])
        with pytest.raises(ValueError, match="wave3_retrace_min"):
            elliot_wave.generate_elliot_wave_signals(df, wave3_retrace_min=0.8, wave3_retrace_max=0.3)

    def test_invalid_wavec_retrace_band_raises(self):
        df = _make_simple_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108])
        with pytest.raises(ValueError, match="wavec_retrace_min"):
            elliot_wave.generate_elliot_wave_signals(df, wavec_retrace_min=0.9, wavec_retrace_max=0.3)

    def test_invalid_risk_reward_ratio_raises(self):
        df = _make_simple_ohlcv([100, 101, 102, 103, 104, 105, 106, 107, 108])
        with pytest.raises(ValueError, match="risk_reward_ratio must be > 0"):
            elliot_wave.generate_elliot_wave_signals(df, risk_reward_ratio=0)


class TestElliotWaveShortWindow:
    """Short/insufficient windows must return deterministic HOLD series."""

    def test_very_short_window_returns_hold(self):
        # Window too short for swing detection
        df = _make_simple_ohlcv([100.0, 101.0, 102.0])
        signals = elliot_wave.generate_elliot_wave_signals(df)
        assert len(signals) == len(df)
        assert (signals == SignalAction.HOLD).all()

    def test_insufficient_for_wave3_returns_hold(self):
        # Not enough bars for Wave 3 pattern (need at least swing_window*2 + 6)
        df = _make_simple_ohlcv([100.0, 101.0, 102.0, 103.0, 104.0])
        signals = elliot_wave.generate_elliot_wave_signals(df, swing_window=2)
        assert len(signals) == len(df)
        assert (signals == SignalAction.HOLD).all()


class TestElliotWaveHelpers:
    """Test helper functions used internally."""

    def test_bps_to_price_conversion(self):
        assert abs(elliot_wave._bps_to_price(100.0, 10) - 0.1) < 1e-9
        assert abs(elliot_wave._bps_to_price(100.0, 5) - 0.05) < 1e-9
        assert abs(elliot_wave._bps_to_price(200.0, 50) - 1.0) < 1e-9

    def test_retrace_in_band_bullish(self):
        # L0=100, H1=110, L2=105 -> retrace = (110-105)/(110-100) = 0.5
        assert elliot_wave._retrace_in_band(100.0, 110.0, 105.0, 0.382, 0.786) is True
        # Above max: L2=108 -> retrace = (110-108)/10 = 0.2 < 0.382 (below min)
        assert elliot_wave._retrace_in_band(100.0, 110.0, 108.0, 0.382, 0.786) is False
        # Below min: L2=107 -> retrace = (110-107)/10 = 0.3 < 0.382
        assert elliot_wave._retrace_in_band(100.0, 110.0, 107.0, 0.382, 0.786) is False
        # Within band: L2=103 -> retrace = (110-103)/10 = 0.7 (within [0.382, 0.786])
        assert elliot_wave._retrace_in_band(100.0, 110.0, 103.0, 0.382, 0.786) is True

    def test_retrace_in_band_bearish(self):
        # H0=110, L1=100, H2=105 -> retrace = (105-100)/(110-100) = 0.5
        assert elliot_wave._retrace_in_band(110.0, 100.0, 105.0, 0.382, 0.786) is True

    def test_retrace_rejects_extension_beyond_origin(self):
        # C beyond A (extension, not retrace)
        assert elliot_wave._retrace_in_band(100.0, 110.0, 99.0, 0.382, 0.786) is False

    def test_detect_swing_highs_lows(self):
        # Clear swing high at index 2 and clear swing low at index 3.
        highs = pd.Series([5.0, 3.0, 7.0, 2.0, 6.0])
        lows = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0])
        is_high, is_low = elliot_wave._detect_swing_highs_lows(highs, lows, swing_window=1)
        assert is_high.iloc[2]
        assert is_low.iloc[3]
        assert not is_high.iloc[3]
        assert not is_low.iloc[2]

    def test_detect_swing_highs_lows_ignores_same_bar_tie(self):
        highs = pd.Series([5.0, 7.0, 5.0])
        lows = pd.Series([2.0, 1.0, 3.0])
        is_high, is_low = elliot_wave._detect_swing_highs_lows(highs, lows, swing_window=1)

        assert not is_high.iloc[1]
        assert not is_low.iloc[1]
        assert elliot_wave._find_swing_points(is_high, is_low, highs, lows) == []

    def test_find_swing_points(self):
        highs = pd.Series([5.0, 3.0, 7.0, 2.0, 6.0])
        lows = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0])
        is_high, is_low = elliot_wave._detect_swing_highs_lows(highs, lows, swing_window=1)
        points = elliot_wave._find_swing_points(is_high, is_low, highs, lows)
        assert points == [
            {"type": "high", "idx": 2, "price": 7.0},
            {"type": "low", "idx": 3, "price": 1.0},
        ]


class TestElliotWaveSignals:
    """Test signal generation for various setups."""

    def test_no_signals_on_trending_without_retrace(self):
        """Strong trend without proper retrace should not trigger entries."""
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0]
        df = _make_simple_ohlcv(closes)
        signals = elliot_wave.generate_elliot_wave_signals(df, swing_window=2)
        assert (signals == SignalAction.HOLD).all()

    def test_engine_compatible_output(self):
        """Output must be compatible with BacktestEngine.run."""
        closes = [100.0 + i * 0.5 for i in range(20)]
        df = _make_simple_ohlcv(closes)

        signals, engine = elliot_wave.run_elliot_wave_backtest(df, initial_capital=50_000.0)

        assert len(signals) == len(df)
        assert isinstance(signals.index, pd.DatetimeIndex)
        assert engine.initial_capital == 50_000.0
        # Engine should run without error
        assert hasattr(engine, "_trades")

    def test_signals_have_correct_dtype(self):
        """Signals must be object dtype with SignalAction values."""
        closes = [100.0 + i * 0.5 for i in range(15)]
        df = _make_simple_ohlcv(closes)
        signals = elliot_wave.generate_elliot_wave_signals(df)
        assert signals.dtype == object
        assert all(isinstance(s, SignalAction) for s in signals)

    def test_plan_parameters_preserved(self):
        """Informational plan parameters must not affect signal generation."""
        closes = [100.0 + i * 0.5 for i in range(15)]
        df = _make_simple_ohlcv(closes)

        signals1, _ = elliot_wave.run_elliot_wave_backtest(df, trade_style="day_trade")
        signals2, _ = elliot_wave.run_elliot_wave_backtest(df, trade_style="swing_trade")

        assert signals1.equals(signals2)

    def test_ohlcv_utc_index_handled(self):
        """UTC-aware indexes should be preserved."""
        closes = [100.0 + i * 0.5 for i in range(15)]
        df = _make_simple_ohlcv(closes)
        assert df.index.tz is not None

        signals = elliot_wave.generate_elliot_wave_signals(df)
        assert signals.index.tz is not None


class TestElliotWaveWaveC:
    """Test Wave C corrective pattern detection."""

    def test_wave_c_bounded_retrace(self, monkeypatch):
        """Wave C should produce an entry when the bounded ABC pattern breaks out."""
        df = _make_ohlcv(
            opens=[100.0, 109.0, 105.0, 108.0, 111.0, 103.0, 102.0, 102.0],
            highs=[101.0, 110.0, 106.0, 109.0, 112.0, 104.0, 103.0, 103.0],
            lows=[99.0, 103.0, 104.0, 107.0, 110.0, 102.0, 101.0, 101.0],
            closes=[100.0, 110.0, 104.0, 109.0, 111.0, 103.0, 102.0, 102.0],
        )
        monkeypatch.setattr(
            elliot_wave,
            "_find_swing_points",
            lambda *args, **kwargs: [
                {"type": "high", "idx": 1, "price": 110.0},
                {"type": "low", "idx": 2, "price": 104.0},
                {"type": "high", "idx": 3, "price": 109.0},
            ],
        )

        signals = elliot_wave.generate_elliot_wave_signals(
            df,
            swing_window=1,
            confirmation_bars=0,
            breakout_buffer_bps=0,
            invalidation_buffer_bps=0,
            atr_period=1,
            atr_stop_multiplier=0,
            risk_reward_ratio=1,
        )

        non_hold = signals[signals != SignalAction.HOLD]
        assert list(non_hold.values) == [SignalAction.LONG_ENTRY, SignalAction.LONG_EXIT]
        assert non_hold.index[0] == df.index[4]
        assert non_hold.index[1] == df.index[5]

    def test_wave_c_short_zero_confirmation_enters_on_breakout_bar(self, monkeypatch):
        df = _make_ohlcv(
            opens=[110.0, 106.0, 107.0, 105.0, 103.0, 103.0, 103.0, 103.0],
            highs=[111.0, 110.0, 108.0, 105.5, 104.0, 104.0, 104.0, 104.0],
            lows=[109.0, 104.0, 105.0, 103.5, 102.0, 102.0, 102.0, 102.0],
            closes=[110.0, 105.0, 106.0, 104.5, 103.0, 103.0, 103.0, 103.0],
        )
        monkeypatch.setattr(
            elliot_wave,
            "_find_swing_points",
            lambda *args, **kwargs: [
                {"type": "low", "idx": 1, "price": 104.0},
                {"type": "high", "idx": 2, "price": 109.0},
                {"type": "low", "idx": 3, "price": 105.0},
            ],
        )

        signals, engine = elliot_wave.run_elliot_wave_backtest(
            df,
            swing_window=1,
            confirmation_bars=0,
            breakout_buffer_bps=0,
            invalidation_buffer_bps=0,
            atr_period=1,
            atr_stop_multiplier=0,
            risk_reward_ratio=1,
        )

        non_hold = signals[signals != SignalAction.HOLD]
        assert list(non_hold.index) == [df.index[4]]
        assert non_hold.iloc[0] == SignalAction.SHORT_ENTRY
        assert engine._has_open_position is True


class TestElliotWaveExitLogic:
    """Test stop/invalidation exit logic at strategy layer."""

    def test_stop_uses_invalidation_level(self, monkeypatch):
        """Wave-origin breach should trigger a stored stop exit for Wave C long."""
        df = _make_ohlcv(
            opens=[100.0, 109.0, 105.0, 108.0, 111.0, 103.0, 102.0, 102.0],
            highs=[101.0, 110.0, 106.0, 109.0, 112.0, 104.0, 103.0, 103.0],
            lows=[99.0, 103.0, 104.0, 107.0, 112.0, 103.0, 101.0, 101.0],
            closes=[100.0, 110.0, 104.0, 109.0, 111.0, 103.0, 102.0, 102.0],
        )
        monkeypatch.setattr(
            elliot_wave,
            "_find_swing_points",
            lambda *args, **kwargs: [
                {"type": "high", "idx": 1, "price": 110.0},
                {"type": "low", "idx": 2, "price": 104.0},
                {"type": "high", "idx": 3, "price": 109.0},
            ],
        )

        signals = elliot_wave.generate_elliot_wave_signals(
            df,
            swing_window=1,
            confirmation_bars=0,
            breakout_buffer_bps=0,
            invalidation_buffer_bps=0,
            atr_period=1,
            atr_stop_multiplier=0,
            risk_reward_ratio=1,
        )

        assert signals.iloc[4] == SignalAction.LONG_ENTRY
        assert signals.iloc[5] == SignalAction.LONG_EXIT

    def test_atr_stop_multiplier_applied(self, monkeypatch):
        """ATR-based stop buffer should widen the static stop and change exit behavior."""
        df = _make_ohlcv(
            opens=[100.0, 109.0, 105.0, 108.0, 111.0, 107.0, 102.0, 102.0],
            highs=[101.0, 110.0, 106.0, 109.0, 112.0, 108.0, 103.0, 103.0],
            lows=[99.0, 103.0, 104.0, 107.0, 111.0, 107.0, 101.0, 101.0],
            closes=[100.0, 110.0, 104.0, 109.0, 111.0, 107.0, 102.0, 102.0],
        )
        monkeypatch.setattr(
            elliot_wave,
            "_find_swing_points",
            lambda *args, **kwargs: [
                {"type": "high", "idx": 1, "price": 110.0},
                {"type": "low", "idx": 2, "price": 104.0},
                {"type": "high", "idx": 3, "price": 109.0},
            ],
        )
        monkeypatch.setattr(
            elliot_wave,
            "calculate_atr",
            lambda high, low, close, period=14: pd.Series([4.0] * len(close), index=close.index),
        )

        signals_no_atr = elliot_wave.generate_elliot_wave_signals(
            df,
            swing_window=1,
            confirmation_bars=0,
            breakout_buffer_bps=0,
            invalidation_buffer_bps=0,
            atr_stop_multiplier=0,
            risk_reward_ratio=1,
        )
        signals_with_atr = elliot_wave.generate_elliot_wave_signals(
            df,
            swing_window=1,
            confirmation_bars=0,
            breakout_buffer_bps=0,
            invalidation_buffer_bps=0,
            atr_stop_multiplier=1,
            risk_reward_ratio=1,
        )

        assert signals_no_atr.iloc[4] == SignalAction.LONG_ENTRY
        assert signals_no_atr.iloc[5] == SignalAction.HOLD
        assert signals_with_atr.iloc[4] == SignalAction.LONG_ENTRY
        assert signals_with_atr.iloc[5] == SignalAction.LONG_EXIT


class TestElliotWaveEngineResult:
    """Test that strategy integrates with backtest engine properly."""

    def test_engine_runs_without_error(self):
        """Engine.run should complete without exceptions."""
        closes = [100.0 + i * 0.3 + np.sin(i / 3) * 2 for i in range(30)]
        df = _make_simple_ohlcv(closes)

        signals, engine = elliot_wave.run_elliot_wave_backtest(df)

        # Engine should complete
        assert engine is not None
        assert hasattr(engine, "_trades")
        assert hasattr(engine, "_equity_curve")

    def test_result_package_matches_contract(self):
        """Result package should have expected structure."""
        closes = [100.0 + i * 0.3 for i in range(25)]
        df = _make_simple_ohlcv(closes)

        signals, engine = elliot_wave.run_elliot_wave_backtest(df)

        # Verify engine attributes
        assert hasattr(engine, "initial_capital")
        # Engine.run returns BacktestResult via run() - verify the result
        result = engine.run(df, signals)
        assert hasattr(result, "final_equity")
        assert hasattr(result, "trades")
        assert hasattr(result, "equity_curve")
