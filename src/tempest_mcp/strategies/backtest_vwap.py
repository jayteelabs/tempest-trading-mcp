"""VWAP Anchored backtest strategy (ENG-21).

This strategy consumes a resolved OHLCV DataFrame and delegates price planning
and date-range resolution to the shared backtest contract. It uses anchored VWAP
for the session-aware mean line, a simple EMA trend filter, and volume
confirmation before emitting entry signals.

Signal model:
    LONG_ENTRY  — price rejects upward from VWAP with trend + volume confirmation
    SHORT_ENTRY — price rejects downward from VWAP with trend + volume confirmation
    LONG_EXIT   — exit on VWAP reversion, stop hit, or 2:1 reward target
    SHORT_EXIT  — exit on VWAP reversion, stop hit, or 2:1 reward target

The strategy is deterministic and returns a signal series plus a configured
BacktestEngine instance.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.indicators.trend.ema import calculate_ema
from tempest_mcp.indicators.volume.vwap import calculate_vwap

# Shared Phase 2 defaults
TRADE_STYLE_PRESETS = {
    "day_trade": {"timeframe": "1h", "duration_days": 1},
    "swing_trade": {"timeframe": "4h", "duration_days": 7},
}

_VALID_VWAP_ANCHORS = {"asia", "london", "ny", "daily"}


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a UTC-aware DatetimeIndex."""
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


def _reindex_or_empty(series: pd.Series, index: pd.Index) -> pd.Series:
    """Reindex a possibly-empty indicator series to the working index."""
    if series.empty:
        return pd.Series(index=index, dtype=float)
    return series.reindex(index)


def run_vwap_anchored_backtest(
    ohlcv_df: pd.DataFrame,
    vwap_anchor: Literal["asia", "london", "ny", "daily"] = "ny",
    trend_fast_period: int = 7,
    trend_slow_period: int = 25,
    volume_lookback: int = 20,
    volume_multiplier: float = 1.2,
    rr_multiple: float = 2.0,
    # --- Phase 2 preset/plan parameters (caller-facing, optional) -------------
    trade_style: Literal["day_trade", "swing_trade", "custom"] | None = None,
    timeframe: str | None = None,
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
    exchange: str | None = None,
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, BacktestEngine]:
    """Run the VWAP Anchored backtest strategy.

    Parameters
    ----------
    ohlcv_df:
        Resolved OHLCV DataFrame with UTC-aware index and columns
        [open, high, low, close, volume].
    vwap_anchor:
        Session anchor passed directly to ``calculate_vwap``.
    trend_fast_period / trend_slow_period:
        EMA periods used for trend confirmation. Fast must be smaller than slow.
    volume_lookback:
        Rolling lookback used for volume confirmation.
    volume_multiplier:
        Current volume must be greater than the rolling average multiplied by
        this factor.
    rr_multiple:
        Reward-to-risk target multiplier. ``2.0`` implements 2:1 R:R.
    trade_style / timeframe / start_at / end_at / exchange:
        Informational plan parameters kept for contract alignment; the strategy
        does not own date-range resolution.
    initial_capital:
        Starting capital for the backtest engine.

    Returns
    -------
    signals, engine:
        Signal series and configured engine after the run.

    Raises
    ------
    ValueError
        For malformed inputs, invalid periods, or missing OHLCV columns.
    """
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df must not be empty")
    if vwap_anchor not in _VALID_VWAP_ANCHORS:
        raise ValueError(f"Invalid vwap_anchor: '{vwap_anchor}'. Must be one of {_VALID_VWAP_ANCHORS}")
    if trend_fast_period <= 0 or trend_slow_period <= 0:
        raise ValueError("trend periods must be positive integers")
    if trend_fast_period >= trend_slow_period:
        raise ValueError("trend_fast_period must be smaller than trend_slow_period")
    if volume_lookback <= 0:
        raise ValueError("volume_lookback must be a positive integer")
    if volume_multiplier <= 0:
        raise ValueError("volume_multiplier must be positive")
    if rr_multiple <= 0:
        raise ValueError("rr_multiple must be positive")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv_df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {', '.join(sorted(missing))}")

    ohlcv_df = _ensure_utc_index(ohlcv_df)
    close = ohlcv_df["close"]
    high = ohlcv_df["high"]
    low = ohlcv_df["low"]
    volume = ohlcv_df["volume"]

    vwap_series = calculate_vwap(high, low, close, volume, anchor=vwap_anchor)
    fast_ema = _reindex_or_empty(calculate_ema(close, trend_fast_period), ohlcv_df.index)
    slow_ema = _reindex_or_empty(calculate_ema(close, trend_slow_period), ohlcv_df.index)
    volume_sma = volume.rolling(volume_lookback, min_periods=volume_lookback).mean()

    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)
    position: dict[str, float | str] | None = None

    for i in range(1, len(ohlcv_df)):
        bar_open = float(ohlcv_df["open"].iloc[i])
        bar_high = float(high.iloc[i])
        bar_low = float(low.iloc[i])
        bar_close = float(close.iloc[i])
        bar_volume = float(volume.iloc[i])
        vwap_val = vwap_series.iloc[i]
        fast_val = fast_ema.iloc[i]
        slow_val = slow_ema.iloc[i]
        vol_avg = volume_sma.iloc[i]

        if position is not None:
            direction = position["direction"]
            stop_price = float(position["stop_price"])
            tp_price = float(position["tp_price"])
            exit_signal = None

            if direction == "long":
                if bar_open <= stop_price or bar_low <= stop_price:
                    exit_signal = SignalAction.LONG_EXIT
                elif bar_high >= tp_price:
                    exit_signal = SignalAction.LONG_EXIT
                elif pd.notna(vwap_val) and bar_close <= float(vwap_val):
                    exit_signal = SignalAction.LONG_EXIT
            else:
                if bar_open >= stop_price or bar_high >= stop_price:
                    exit_signal = SignalAction.SHORT_EXIT
                elif bar_low <= tp_price:
                    exit_signal = SignalAction.SHORT_EXIT
                elif pd.notna(vwap_val) and bar_close >= float(vwap_val):
                    exit_signal = SignalAction.SHORT_EXIT

            if exit_signal is not None:
                signals.iloc[i] = exit_signal
                position = None
                continue

        if pd.isna(vwap_val) or pd.isna(fast_val) or pd.isna(slow_val) or pd.isna(vol_avg):
            continue

        volume_confirmed = bar_volume >= float(vol_avg) * volume_multiplier
        long_trend = float(fast_val) > float(slow_val) and bar_close > float(fast_val)
        short_trend = float(fast_val) < float(slow_val) and bar_close < float(fast_val)

        long_rejection = bar_low <= float(vwap_val) <= bar_close and bar_close > bar_open
        short_rejection = bar_high >= float(vwap_val) >= bar_close and bar_close < bar_open

        if long_rejection and long_trend and volume_confirmed:
            rejection_gap = bar_close - float(vwap_val)
            if rejection_gap <= 0:
                continue
            stop_price = float(vwap_val) - rejection_gap
            risk_distance = bar_close - stop_price
            tp_price = bar_close + rr_multiple * risk_distance
            signals.iloc[i] = SignalAction.LONG_ENTRY
            position = {
                "direction": "long",
                "entry_price": bar_close,
                "stop_price": stop_price,
                "tp_price": tp_price,
            }
        elif short_rejection and short_trend and volume_confirmed:
            rejection_gap = float(vwap_val) - bar_close
            if rejection_gap <= 0:
                continue
            stop_price = float(vwap_val) + rejection_gap
            risk_distance = stop_price - bar_close
            tp_price = bar_close - rr_multiple * risk_distance
            signals.iloc[i] = SignalAction.SHORT_ENTRY
            position = {
                "direction": "short",
                "entry_price": bar_close,
                "stop_price": stop_price,
                "tp_price": tp_price,
            }

    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(ohlcv_df, signals)

    return signals, engine
