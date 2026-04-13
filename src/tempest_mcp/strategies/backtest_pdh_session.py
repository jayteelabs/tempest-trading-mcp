"""PDH/PDL + Session Levels backtest strategy (ENG-19).

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


def run_pdh_session_backtest(
    ohlcv_df: pd.DataFrame,
    atr_period: int = 14,
    atr_multiplier: float = 1.5,
    session_types: list[Literal["asia", "london", "ny"]] | None = None,
) -> tuple[pd.Series, BacktestEngine]:
    """Run PDH/PDL + Session Levels backtest.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        DataFrame with UTC-aware DatetimeIndex and columns
        [open, high, low, close, volume].
    atr_period : int, default 14
        ATR period (Wilder's smoothing).
    atr_multiplier : float, default 1.5
        Stop distance = atr_multiplier × ATR value.
    session_types : list of str, optional
        Eligible sessions. Defaults to ["london", "ny"].
        Asia is always excluded.

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
    # signals[i] corresponds to bar i (决策 on bar i's close, engine executes
    # the order on bar i+1's open — the BacktestEngine handles this offset).
    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # pdh_pdl_cache: first_bar_date -> (pdh, pdl) to avoid O(n²) detect_pdh_pdl calls
    pdh_pdl_cache: dict = {}

    # Track pending entry (set when entry signal is emitted; cleared after engine processes it)
    # pending_entry["direction"]: PositionDirection
    # pending_entry["entry_bar"]: bar index i where LONG/SHORT_ENTRY was emitted
    # pending_entry["entry_price"]: price at which engine will open position
    # pending_entry["stop_price"]: SL price
    # pending_entry["tp_price"]: TP price
    pending_entry: dict | None = None

    for i in range(1, len(ohlcv_df)):
        bar_open = float(ohlcv_df["open"].iloc[i])
        bar_high = float(ohlcv_df["high"].iloc[i])
        bar_low = float(ohlcv_df["low"].iloc[i])
        bar_close = float(ohlcv_df["close"].iloc[i])

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
                signals.iloc[i] = SignalAction.LONG_EXIT if direction == "long" else SignalAction.SHORT_EXIT
                pending_entry = None
                continue  # Don't also check for new entry on same bar

        # ---- Session eligibility check ---------------------------------------
        session_ok = False
        for st in session_types:
            res = detect_session_levels(ohlcv_df.iloc[: i + 1], st)
            if res.get("bars", 0) > 0:
                session_ok = True
                break
        if not session_ok:
            continue

        # ---- PDH/PDL (cached) -----------------------------------------------
        window = ohlcv_df.iloc[: i + 1]
        first_date = window.index[0].date()
        if first_date not in pdh_pdl_cache:
            result = detect_pdh_pdl(window)
            if result["position"] == "insufficient_data":
                pdh_pdl_cache[first_date] = (float("nan"), float("nan"))
            else:
                pdh_pdl_cache[first_date] = (
                    result["previous_day_high"],
                    result["previous_day_low"],
                )
        pdh, pdl = pdh_pdl_cache[first_date]
        if pdh != pdh:  # NaN
            continue

        # ---- Entry signals --------------------------------------------------
        if bar_close > pdh:
            signals.iloc[i] = SignalAction.LONG_ENTRY
            atr_val = float(atr_series.iloc[i - 1]) if i - 1 < len(atr_series) else float("nan")
            if atr_val != atr_val:
                pending_entry = None
                continue
            sd = atr_multiplier * atr_val
            pending_entry = {
                "direction": "long",
                "entry_bar": i,
                "stop_price": bar_close - sd,
                "tp_price": bar_close + 2.0 * sd,
            }
        elif bar_close < pdl:
            signals.iloc[i] = SignalAction.SHORT_ENTRY
            atr_val = float(atr_series.iloc[i - 1]) if i - 1 < len(atr_series) else float("nan")
            if atr_val != atr_val:
                pending_entry = None
                continue
            sd = atr_multiplier * atr_val
            pending_entry = {
                "direction": "short",
                "entry_bar": i,
                "stop_price": bar_close + sd,
                "tp_price": bar_close - 2.0 * sd,
            }

    # ---- Run engine -------------------------------------------------------
    engine = BacktestEngine()
    engine.run(ohlcv_df, signals)

    return signals, engine
