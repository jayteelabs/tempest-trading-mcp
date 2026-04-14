"""Tests for the VWAP Anchored backtest strategy (ENG-21)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.backtest.engine import SignalAction
from tempest_mcp.strategies.backtest_vwap import run_vwap_anchored_backtest


def _make_ohlcv(closes: list[float], volumes: list[float] | None = None) -> pd.DataFrame:
    """Create deterministic OHLCV data from close prices."""
    idx = pd.date_range("2024-01-01 00:00", periods=len(closes), freq="h", tz="UTC")
    close = pd.Series(closes, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.Series(np.maximum(open_, close) + 0.4, index=idx)
    low = pd.Series(np.minimum(open_, close) - 0.4, index=idx)
    if volumes is None:
        volumes = [500.0] * len(closes)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volumes},
        index=idx,
    )


class TestVwapAnchorContract:
    def test_requested_anchor_is_forwarded_to_indicator(self):
        df = _make_ohlcv([100.0, 101.0, 102.0, 103.0])
        vwap = pd.Series([100.0, 100.5, 101.0, 101.5], index=df.index)
        ema_fast = pd.Series([99.0, 99.5, 100.0, 100.5], index=df.index)
        ema_slow = pd.Series([100.5, 100.5, 100.5, 100.5], index=df.index)

        with patch("tempest_mcp.strategies.backtest_vwap.calculate_vwap", return_value=vwap) as mock_vwap:
            with patch(
                "tempest_mcp.strategies.backtest_vwap.calculate_ema",
                side_effect=[ema_fast, ema_slow],
            ):
                signals, engine = run_vwap_anchored_backtest(
                    df,
                    vwap_anchor="daily",
                    trend_fast_period=2,
                    trend_slow_period=3,
                    volume_lookback=2,
                    volume_multiplier=1.0,
                )

        assert mock_vwap.call_count == 1
        assert mock_vwap.call_args.kwargs["anchor"] == "daily"
        assert len(signals) == len(df)
        assert len(engine._trades) == 0


class TestVwapStrategySignals:
    def test_long_entry_then_reward_exit_on_real_data(self):
        closes = [100, 99, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108]
        volumes = [500, 500, 500, 500, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
        df = _make_ohlcv(closes, volumes)

        signals, engine = run_vwap_anchored_backtest(
            df,
            vwap_anchor="asia",
            trend_fast_period=3,
            trend_slow_period=5,
            volume_lookback=3,
            volume_multiplier=1.2,
            rr_multiple=2.0,
        )

        non_hold = signals[signals != SignalAction.HOLD]
        assert list(non_hold.index) == [df.index[4], df.index[9]]
        assert non_hold.iloc[0] == SignalAction.LONG_ENTRY
        assert non_hold.iloc[1] == SignalAction.LONG_EXIT
        assert len(engine._trades) == 1

    def test_short_entry_then_vwap_reversion_exit_on_real_data(self):
        closes = [108, 107, 106, 105, 103, 101, 99, 98, 99, 101, 103, 105]
        volumes = [500, 500, 500, 500, 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000]
        df = _make_ohlcv(closes, volumes)

        signals, engine = run_vwap_anchored_backtest(
            df,
            vwap_anchor="asia",
            trend_fast_period=3,
            trend_slow_period=5,
            volume_lookback=3,
            volume_multiplier=1.2,
            rr_multiple=2.0,
        )

        non_hold = signals[signals != SignalAction.HOLD]
        assert list(non_hold.index) == [df.index[4], df.index[10]]
        assert non_hold.iloc[0] == SignalAction.SHORT_ENTRY
        assert non_hold.iloc[1] == SignalAction.SHORT_EXIT
        assert len(engine._trades) == 1

    def test_insufficient_history_returns_hold_only(self):
        df = _make_ohlcv([100.0, 100.5])

        signals, engine = run_vwap_anchored_backtest(
            df,
            vwap_anchor="asia",
            trend_fast_period=3,
            trend_slow_period=5,
            volume_lookback=3,
            volume_multiplier=1.2,
            rr_multiple=2.0,
        )

        assert (signals == SignalAction.HOLD).all()
        assert len(engine._trades) == 0

    def test_missing_required_columns_raises(self):
        df = _make_ohlcv([100.0, 101.0, 102.0])
        df = df.drop(columns=["volume"])

        with pytest.raises(ValueError, match="missing required columns"):
            run_vwap_anchored_backtest(df)

    def test_nan_vwap_bar_is_skipped(self):
        df = _make_ohlcv([100.0, 101.0, 102.0, 103.0], [1000.0, 1000.0, 1000.0, 1000.0])
        vwap = pd.Series([100.0, np.nan, 100.0, 100.0], index=df.index)
        ema_fast = pd.Series([99.0, 100.0, 101.0, 102.0], index=df.index)
        ema_slow = pd.Series([98.0, 99.0, 100.0, 101.0], index=df.index)

        with patch("tempest_mcp.strategies.backtest_vwap.calculate_vwap", return_value=vwap):
            with patch(
                "tempest_mcp.strategies.backtest_vwap.calculate_ema",
                side_effect=[ema_fast, ema_slow],
            ):
                signals, _ = run_vwap_anchored_backtest(
                    df,
                    trend_fast_period=2,
                    trend_slow_period=3,
                    volume_lookback=2,
                    volume_multiplier=1.0,
                )

        assert signals.loc[df.index[1]] == SignalAction.HOLD

    def test_zero_volume_bar_does_not_confirm_entry(self):
        df = _make_ohlcv([100.0, 99.0, 101.0, 102.0], [1000.0, 1000.0, 0.0, 1000.0])
        vwap = pd.Series([100.0, 100.0, 100.0, 100.0], index=df.index)
        ema_fast = pd.Series([99.0, 99.0, 100.0, 101.0], index=df.index)
        ema_slow = pd.Series([98.0, 98.0, 99.0, 100.0], index=df.index)

        with patch("tempest_mcp.strategies.backtest_vwap.calculate_vwap", return_value=vwap):
            with patch(
                "tempest_mcp.strategies.backtest_vwap.calculate_ema",
                side_effect=[ema_fast, ema_slow],
            ):
                signals, _ = run_vwap_anchored_backtest(
                    df,
                    trend_fast_period=2,
                    trend_slow_period=3,
                    volume_lookback=2,
                    volume_multiplier=1.0,
                )

        assert (signals == SignalAction.HOLD).all()
