"""PDH/PDL + Session Levels backtest strategy (ENG-19).

Phase 2 contract — strategy is a pure signal generator that consumes resolved
OHLCV from a shared window resolver. It does not own date-range planning or
rate-limit handling.

Entry signals:
    LONG_ENTRY  — close > PDH  and bar is in eligible session
    SHORT_ENTRY — close < PDL  and bar is in eligible session

Exit signals (strategy-generated, evaluated on bars after entry):
    LONG_EXIT / SHORT_EXIT — emitted when intrabar price hits SL or TP

SL/TP computation (set at entry, evaluated on subsequent bars):
    Long:  SL = entry_price - atr_multiplier * ATR
           TP = entry_price + 2 * atr_multiplier * ATR
    Short: SL = entry_price + atr_multiplier * ATR
           TP = entry_price - 2 * atr_multiplier * ATR

ATR is computed on the full DataFrame using Wilder's smoothing (period=atr_period).
SL/TP are evaluated per bar: open hit first → immediate exit; otherwise H/L check.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.indicators.session_levels import detect_pdh_pdl, detect_session_levels
from tempest_mcp.indicators.volatility.atr import calculate_atr

# Shared Phase 2 defaults
TRADE_STYLE_PRESETS = {
    "day_trade": {"timeframe": "1h", "duration_days": 1},
    "swing_trade": {"timeframe": "4h", "duration_days": 7},
}


def _bar_is_within_session(
    bar_time: pd.Timestamp,
    session_result: dict,
) -> bool:
    """Return True only when the current bar itself sits inside the session window."""
    session_start = session_result.get("session_start_utc")
    session_end = session_result.get("session_end_utc")

    return (
        session_result.get("bars", 0) > 0
        and pd.notna(session_start)
        and pd.notna(session_end)
        and session_start <= bar_time <= session_end
    )


def run_pdh_session_backtest(
    ohlcv_df: pd.DataFrame,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    session_types: list[Literal["asia", "london", "ny"]] | None = None,
    # --- Phase 2 preset/plan parameters (caller-facing, optional) -------------
    trade_style: Literal["day_trade", "swing_trade", "custom"] | None = None,
    timeframe: str | None = None,
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
    exchange: str | None = None,
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, BacktestEngine]:
    """Run PDH/PDL + Session Levels backtest.

    The strategy is a pure signal generator. It consumes a **resolved OHLCV
    DataFrame** from a shared window resolver and does not own date-range
    planning or data fetching.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        Resolved OHLCV DataFrame with UTC-aware DatetimeIndex and columns
        [open, high, low, close, volume]. The caller is responsible for
        ensuring sufficient warmup data (at least 1 prior UTC calendar day
        for PDH/PDL lookback).
    atr_period : int, default 14
        ATR period (Wilder's smoothing).
    atr_multiplier : float, default 1.5
        Stop distance = atr_multiplier × ATR value.
    session_types : list of str, optional
        Eligible sessions. Defaults to ["london", "ny"].
        Asia is always excluded.

    Phase 2 preset/plan parameters (caller-facing, optional):
    trade_style : {"day_trade", "swing_trade", "custom"}, optional
        Trading style preset. Defaults:
        - "day_trade"  → 1h timeframe over 24h
        - "swing_trade" → 4h timeframe over 7d
        - "custom" → explicit start_at + end_at
    timeframe : str, optional
        OHLCV timeframe hint (e.g. "1h", "4h"). Informational — the
        strategy operates on whatever timeframe the resolved DataFrame uses.
    start_at : pd.Timestamp, optional
        Backtest start (informational / plan parameter).
    end_at : pd.Timestamp, optional
        Backtest end (informational / plan parameter).
    exchange : str, optional
        Exchange name (informational / plan parameter).
    initial_capital : float, default 100_000.0
        Starting capital for the backtest engine.

    Returns
    -------
    signals : pd.Series
        SignalAction values indexed by ohlcv_df timestamp.
    engine : BacktestEngine
        Engine after running. Access trades via engine._trades,
        equity via engine._equity_curve, and metrics via engine._compute_metrics().

    Raises
    ------
    ValueError
        Empty DataFrame, atr_period <= 0, invalid session_types, missing columns.

    Shared Phase 2 defaults:
        day_trade  → 1h / 24h
        swing_trade → 4h / 7d
        custom → explicit start_at + end_at
    """
    if session_types is None:
        session_types = ["london", "ny"]

    # ---- Validation --------------------------------------------------------
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df must not be empty")
    if atr_period <= 0:
        raise ValueError(f"atr_period must be a positive integer, got {atr_period}")
    valid = {"asia", "london", "ny"}
    for s in session_types:
        if s not in valid:
            raise ValueError(f"Invalid session_type: '{s}'. Must be one of {valid}")
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv_df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {', '.join(sorted(missing))}")

    # Ensure UTC-aware index
    if ohlcv_df.index.tz is None:
        ohlcv_df = ohlcv_df.copy()
        ohlcv_df.index = ohlcv_df.index.tz_localize("UTC")

    # ---- Pre-compute ATR --------------------------------------------------
    atr_series = calculate_atr(
        ohlcv_df["high"],
        ohlcv_df["low"],
        ohlcv_df["close"],
        period=atr_period,
    )

    # ---- Signal generation -------------------------------------------------
    # signals[i] corresponds to bar i (decision on bar i's close; engine executes
    # the order on bar i+1's open — the BacktestEngine handles this offset).
    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # pdh_pdl_cache: evaluated_bar_date -> (pdh, pdl) to avoid O(n²)
    # detect_pdh_pdl calls while still recalculating when the evaluation day changes.
    pdh_pdl_cache: dict = {}

    # Track pending entry: cleared when engine processes the corresponding bar
    # pending_entry keys: direction, entry_bar, stop_price, tp_price
    pending_entry: dict | None = None
    prior_long_breakout = False
    prior_short_breakout = False

    for i in range(1, len(ohlcv_df)):
        bar_time = ohlcv_df.index[i]
        bar_open = float(ohlcv_df["open"].iloc[i])
        bar_high = float(ohlcv_df["high"].iloc[i])
        bar_low = float(ohlcv_df["low"].iloc[i])
        bar_close = float(ohlcv_df["close"].iloc[i])
        exited_this_bar = False

        # ---- SL/TP check for pending entry from prior bar --------------------
        if pending_entry is not None:
            direction = pending_entry["direction"]
            stop = pending_entry["stop_price"]
            tp = pending_entry["tp_price"]

            should_exit = False
            if direction == "long":
                if bar_open <= stop or bar_low <= stop:
                    should_exit = True
                elif bar_high >= tp:
                    should_exit = True
            elif direction == "short":
                if bar_open >= stop or bar_high >= stop:
                    should_exit = True
                elif bar_low <= tp:
                    should_exit = True

            if should_exit:
                signals.iloc[i] = (
                    SignalAction.LONG_EXIT if direction == "long" else SignalAction.SHORT_EXIT
                )
                pending_entry = None
                exited_this_bar = True

        # ---- PDH/PDL (cached per calendar day) -----------------------------
        window = ohlcv_df.iloc[: i + 1]
        bar_date = bar_time.date()
        if bar_date not in pdh_pdl_cache:
            result = detect_pdh_pdl(window)
            if result["position"] == "insufficient_data":
                pdh_pdl_cache[bar_date] = (float("nan"), float("nan"))
            else:
                pdh_pdl_cache[bar_date] = (
                    result["previous_day_high"],
                    result["previous_day_low"],
                )
        pdh, pdl = pdh_pdl_cache[bar_date]
        if pdh != pdh:  # NaN — insufficient_data
            prior_long_breakout = False
            prior_short_breakout = False
            continue

        current_long_breakout = bar_close > pdh
        current_short_breakout = bar_close < pdl

        # ---- Session eligibility check ---------------------------------------
        session_ok = False
        for st in session_types:
            res = detect_session_levels(window, st)
            if _bar_is_within_session(bar_time, res):
                session_ok = True
                break

        if pending_entry is not None or exited_this_bar or not session_ok:
            prior_long_breakout = current_long_breakout
            prior_short_breakout = current_short_breakout
            continue

        # ---- ATR value for stop construction --------------------------------
        atr_val = float(atr_series.iloc[i - 1]) if i - 1 < len(atr_series) else float("nan")
        if atr_val != atr_val:  # NaN
            prior_long_breakout = current_long_breakout
            prior_short_breakout = current_short_breakout
            continue
        stop_distance = atr_multiplier * atr_val

        # ---- Entry signals --------------------------------------------------
        if current_long_breakout and not prior_long_breakout:
            signals.iloc[i] = SignalAction.LONG_ENTRY
            pending_entry = {
                "direction": "long",
                "entry_bar": i,
                "stop_price": bar_close - stop_distance,
                "tp_price": bar_close + 2.0 * stop_distance,
            }
        elif current_short_breakout and not prior_short_breakout:
            signals.iloc[i] = SignalAction.SHORT_ENTRY
            pending_entry = {
                "direction": "short",
                "entry_bar": i,
                "stop_price": bar_close + stop_distance,
                "tp_price": bar_close - 2.0 * stop_distance,
            }

        prior_long_breakout = current_long_breakout
        prior_short_breakout = current_short_breakout

    # ---- Run engine -------------------------------------------------------
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(ohlcv_df, signals)

    return signals, engine
