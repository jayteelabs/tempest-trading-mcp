"""Elliot Wave Simplified backtest strategy (ENG-24).

This strategy consumes a resolved OHLCV DataFrame and detects simplified
Elliott Wave structures for trading Wave 3 and Wave C setups.

Signal model:
    LONG_ENTRY  — Wave 3 or Wave C long setup confirmed and breakout triggered
    SHORT_ENTRY — Wave 3 or Wave C short setup confirmed and breakout triggered
    LONG_EXIT   — stop or target hit for long position
    SHORT_EXIT  — stop or target hit for short position
    HOLD        — no action / insufficient window

The strategy is deterministic and returns a signal series plus a configured
BacktestEngine instance.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.indicators.volatility.atr import calculate_atr

# Strategy identifier
STRATEGY_ID = "elliot_wave"

# Default retracement bands (approved in design review)
_WAVE3_RETRACE_MIN = 0.382
_WAVE3_RETRACE_MAX = 0.786
_WAVEC_RETRACE_MIN = 0.382
_WAVEC_RETRACE_MAX = 0.886

# Minimum bars needed for swing detection (swing_window on each side + confirmation)
# _MIN_BARS_FOR_SWING = 3  # swing_window * 2 + 1 minimum for a single swing point
_MIN_BARS_WAVE3 = 6  # Need L0, H1, L2 for wave 3 long (minimum)
# _MIN_BARS_WAVEC = 5  # Need A, B, C for wave C


def _bps_to_price(price: float, bps: float) -> float:
    """Convert basis points to price units."""
    return price * bps / 10_000


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a UTC-aware DatetimeIndex."""
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


def _detect_swing_highs_lows(
    high: pd.Series,
    low: pd.Series,
    swing_window: int,
) -> tuple[pd.Series, pd.Series]:
    """Detect local swing highs and lows using rolling window.

    A swing high is a bar whose high is the maximum over [i-swing_window, i+swing_window].
    A swing low is a bar whose low is the minimum over the same window.

    Returns two boolean series aligned with the input index:
        is_swing_high[i] = True if high[i] is local max in window
        is_swing_low[i]  = True if low[i] is local min in window
    """
    is_swing_high = pd.Series(False, index=high.index)
    is_swing_low = pd.Series(False, index=low.index)

    for i in range(swing_window, len(high) - swing_window):
        window_high = high.iloc[i - swing_window : i + swing_window + 1]
        window_low = low.iloc[i - swing_window : i + swing_window + 1]

        is_high = high.iloc[i] == window_high.max()
        is_low = low.iloc[i] == window_low.min()

        # Resolve ties deterministically: a bar cannot be both swing high and swing low.
        if is_high and not is_low:
            is_swing_high.iloc[i] = True
        elif is_low and not is_high:
            is_swing_low.iloc[i] = True

    return is_swing_high, is_swing_low


def _find_swing_points(
    is_swing_high: pd.Series,
    is_swing_low: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> list[dict]:
    """Build ordered list of swing points with type, index, and price.

    Returns list of dicts: [{"type": "high"|"low", "idx": int, "price": float}]
    ordered by index.
    """
    points = []
    for i in range(len(is_swing_high)):
        if is_swing_high.iloc[i]:
            points.append({"type": "high", "idx": i, "price": float(high.iloc[i])})
        if is_swing_low.iloc[i]:
            points.append({"type": "low", "idx": i, "price": float(low.iloc[i])})
    # Sort by index
    points.sort(key=lambda p: p["idx"])
    return points


def _retrace_in_band(
    a: float,
    b: float,
    c: float,
    band_min: float,
    band_max: float,
) -> bool:
    """Check if point C's retrace of AB is within [band_min, band_max].

    For bullish: AB is the move from A to B (A low, B high)
    Retrace from B to C = (B - C) / (B - A)
    For bearish: AB is the move from A to B (A high, B low)
    Retrace from B to C = (C - B) / (A - B)
    """
    ab_range = abs(b - a)
    if ab_range == 0:
        return False

    if (b > a and c >= a) or (b < a and c <= a):
        # Correct direction for a retrace (C beyond A would be extension)
        retrace = abs(b - c) / ab_range
        return band_min <= retrace <= band_max
    return False


def _validate_inputs(
    ohlcv_df: pd.DataFrame,
    swing_window: int,
    confirmation_bars: int,
    wave3_retrace_min: float,
    wave3_retrace_max: float,
    wavec_retrace_min: float,
    wavec_retrace_max: float,
    breakout_buffer_bps: float,
    invalidation_buffer_bps: float,
    atr_period: int,
    atr_stop_multiplier: float,
    risk_reward_ratio: float,
) -> None:
    """Validate all strategy parameters and raise ValueError on bad inputs."""
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df must not be empty")
    if swing_window < 1:
        raise ValueError("swing_window must be >= 1")
    if confirmation_bars < 0:
        raise ValueError("confirmation_bars must be >= 0")
    if not (0 < wave3_retrace_min < wave3_retrace_max < 1):
        raise ValueError(
            "wave3_retrace_min and wave3_retrace_max must satisfy 0 < wave3_retrace_min < wave3_retrace_max < 1"
        )
    if not (0 < wavec_retrace_min < wavec_retrace_max < 1):
        raise ValueError(
            "wavec_retrace_min and wavec_retrace_max must satisfy 0 < wavec_retrace_min < wavec_retrace_max < 1"
        )
    if breakout_buffer_bps < 0:
        raise ValueError("breakout_buffer_bps must be >= 0")
    if invalidation_buffer_bps < 0:
        raise ValueError("invalidation_buffer_bps must be >= 0")
    if atr_period < 1:
        raise ValueError("atr_period must be >= 1")
    if atr_stop_multiplier < 0:
        raise ValueError("atr_stop_multiplier must be >= 0")
    if risk_reward_ratio <= 0:
        raise ValueError("risk_reward_ratio must be > 0")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv_df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {', '.join(sorted(missing))}")


def generate_elliot_wave_signals(
    ohlcv_df: pd.DataFrame,
    *,
    swing_window: int = 2,
    confirmation_bars: int = 1,
    wave3_retrace_min: float = _WAVE3_RETRACE_MIN,
    wave3_retrace_max: float = _WAVE3_RETRACE_MAX,
    wavec_retrace_min: float = _WAVEC_RETRACE_MIN,
    wavec_retrace_max: float = _WAVEC_RETRACE_MAX,
    breakout_buffer_bps: float = 5.0,
    invalidation_buffer_bps: float = 5.0,
    atr_period: int = 14,
    atr_stop_multiplier: float = 1.0,
    risk_reward_ratio: float = 2.0,
) -> pd.Series:
    """Generate Elliot Wave simplified signals for the given OHLCV window.

    Parameters
    ----------
    ohlcv_df:
        Resolved OHLCV DataFrame with UTC-aware index and columns
        [open, high, low, close, volume].
    swing_window:
        Window size for detecting swing highs/lows (default 2).
    confirmation_bars:
        Number of bars to confirm a breakout before entry (default 1).
    wave3_retrace_min / wave3_retrace_max:
        Retracement band for Wave 3 setups (default 0.382 / 0.786).
    wavec_retrace_min / wavec_retrace_max:
        Retracement band for Wave C setups (default 0.382 / 0.886).
    breakout_buffer_bps:
        Buffer in basis points above/below swing level for breakout (default 5).
    invalidation_buffer_bps:
        Buffer in basis points for stop/invalidation level (default 5).
    atr_period:
        Period for ATR calculation (default 14).
    atr_stop_multiplier:
        Multiplier for ATR-based stop buffer (default 1.0).
    risk_reward_ratio:
        Reward-to-risk ratio for target calculation (default 2.0).

    Returns
    -------
    pd.Series
        Signal series with values {LONG_ENTRY, LONG_EXIT, SHORT_ENTRY, SHORT_EXIT, HOLD}
        aligned with the input DataFrame index.

    Raises
    ------
    ValueError
        For invalid strategy parameters or missing OHLCV columns.
    """
    _validate_inputs(
        ohlcv_df,
        swing_window,
        confirmation_bars,
        wave3_retrace_min,
        wave3_retrace_max,
        wavec_retrace_min,
        wavec_retrace_max,
        breakout_buffer_bps,
        invalidation_buffer_bps,
        atr_period,
        atr_stop_multiplier,
        risk_reward_ratio,
    )

    # Short/insufficient window: return deterministic HOLD series
    min_required = swing_window * 2 + max(confirmation_bars + 1, _MIN_BARS_WAVE3)
    if len(ohlcv_df) < min_required:
        return pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    ohlcv_df = _ensure_utc_index(ohlcv_df)
    high = ohlcv_df["high"]
    low = ohlcv_df["low"]
    close = ohlcv_df["close"]

    # Compute ATR
    atr = calculate_atr(high, low, close, period=atr_period)
    if atr.empty:
        return pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # Detect swing points
    is_swing_high, is_swing_low = _detect_swing_highs_lows(high, low, swing_window)
    swing_points = _find_swing_points(is_swing_high, is_swing_low, high, low)

    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # Position state: None (flat) or dict with direction, entry_price, stop_price, tp_price
    position: dict | None = None

    # Track confirmed breakout state
    # For each potential setup, we need confirmation bars after breakout
    pending_long_setup: dict | None = None
    pending_short_setup: dict | None = None

    for i in range(swing_window, len(ohlcv_df) - 1):
        bar_high = float(high.iloc[i])
        bar_low = float(low.iloc[i])
        bar_close = float(close.iloc[i])
        atr_val = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
        entered_now = False

        # Check for stop/target exit first
        if position is not None:
            direction = position["direction"]
            stop_price = float(position["stop_price"])
            tp_price = float(position["tp_price"])
            exit_signal: SignalAction | None = None

            if direction == "long":
                # Exit uses the stored stop set at entry time.
                if bar_low <= stop_price:
                    exit_signal = SignalAction.LONG_EXIT
                elif bar_high >= tp_price:
                    exit_signal = SignalAction.LONG_EXIT
            elif direction == "short":
                if bar_high >= stop_price:
                    exit_signal = SignalAction.SHORT_EXIT
                elif bar_low <= tp_price:
                    exit_signal = SignalAction.SHORT_EXIT

            if exit_signal is not None:
                signals.iloc[i] = exit_signal
                position = None
                pending_long_setup = None
                pending_short_setup = None
                continue

        # Process pending setups (breakout confirmation)
        if pending_long_setup is not None:
            confirm_count = pending_long_setup["confirm_count"]
            breakout_idx = pending_long_setup["breakout_idx"]
            wave_origin = pending_long_setup["wave_origin"]
            entry_level = pending_long_setup["entry_level"]

            if i - breakout_idx >= confirm_count:
                # Confirmation bars elapsed - enter long
                invalidation_stop = wave_origin - _bps_to_price(
                    wave_origin, invalidation_buffer_bps
                )
                atr_stop = (
                    invalidation_stop
                    if atr_stop_multiplier <= 0
                    else bar_close - atr_stop_multiplier * atr_val
                )
                stop_price = max(invalidation_stop, atr_stop)
                risk_distance = bar_close - stop_price
                if risk_distance <= 0:
                    pending_long_setup = None
                else:
                    tp_price = bar_close + risk_reward_ratio * risk_distance

                    signals.iloc[i] = SignalAction.LONG_ENTRY
                    position = {
                        "direction": "long",
                        "entry_price": bar_close,
                        "stop_price": stop_price,
                        "tp_price": tp_price,
                        "wave_origin": wave_origin,
                    }
                    pending_long_setup = None
            else:
                # Check if breakout still valid (price still above entry level)
                if bar_low <= entry_level:
                    pending_long_setup = None

        if pending_short_setup is not None and position is None:
            confirm_count = pending_short_setup["confirm_count"]
            breakout_idx = pending_short_setup["breakout_idx"]
            wave_origin = pending_short_setup["wave_origin"]
            entry_level = pending_short_setup["entry_level"]

            if i - breakout_idx >= confirm_count:
                invalidation_stop = wave_origin + _bps_to_price(
                    wave_origin, invalidation_buffer_bps
                )
                atr_stop = (
                    invalidation_stop
                    if atr_stop_multiplier <= 0
                    else bar_close + atr_stop_multiplier * atr_val
                )
                stop_price = min(invalidation_stop, atr_stop)
                risk_distance = stop_price - bar_close
                if risk_distance <= 0:
                    pending_short_setup = None
                else:
                    tp_price = bar_close - risk_reward_ratio * risk_distance

                    signals.iloc[i] = SignalAction.SHORT_ENTRY
                    position = {
                        "direction": "short",
                        "entry_price": bar_close,
                        "stop_price": stop_price,
                        "tp_price": tp_price,
                        "wave_origin": wave_origin,
                    }
                    pending_short_setup = None
            else:
                if bar_high >= entry_level:
                    pending_short_setup = None

        # Look for new setups if we're flat
        if position is None and pending_long_setup is None and pending_short_setup is None:
            # Find recent swing points
            recent_swing = [
                p for p in swing_points if p["idx"] <= i and p["idx"] >= i - swing_window * 4
            ]
            if len(recent_swing) < 3:
                continue

            # Wave 3 Long detection: L0 -> H1 -> L2 with L2 > L0
            # Find pattern: low, high, low (swing low, swing high, swing low)
            for j in range(len(recent_swing) - 2):
                p0 = recent_swing[j]
                p1 = recent_swing[j + 1]
                p2 = recent_swing[j + 2]

                if p0["type"] != "low" or p1["type"] != "high" or p2["type"] != "low":
                    continue

                l0_price = p0["price"]
                h1_price = p1["price"]
                l2_price = p2["price"]

                # Conditions for Wave 3 long:
                # 1. L2 > L0 (higher low, still bullish)
                # 2. H1 > L0 (impulse move)
                # 3. Retrace of H1->L2 is within [wave3_retrace_min, wave3_retrace_max]
                if l2_price <= l0_price:
                    continue
                if h1_price <= l0_price:
                    continue

                if not _retrace_in_band(
                    l0_price, h1_price, l2_price, wave3_retrace_min, wave3_retrace_max
                ):
                    continue

                # Breakout level: above H1 + buffer
                breakout_level = h1_price + _bps_to_price(h1_price, breakout_buffer_bps)

                if entered_now is not None:
                    continue

                # Check if price is breaking out
                if bar_close > breakout_level:
                    if confirmation_bars == 0:
                        invalidation_stop = l0_price - _bps_to_price(
                            l0_price, invalidation_buffer_bps
                        )
                        atr_stop = (
                            invalidation_stop
                            if atr_stop_multiplier <= 0
                            else bar_close - atr_stop_multiplier * atr_val
                        )
                        stop_price = max(invalidation_stop, atr_stop)
                        risk_distance = bar_close - stop_price
                        if risk_distance > 0:
                            tp_price = bar_close + risk_reward_ratio * risk_distance
                            signals.iloc[i] = SignalAction.LONG_ENTRY
                            position = {
                                "direction": "long",
                                "entry_price": bar_close,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "wave_origin": l0_price,
                            }
                            entered_now = True
                    else:
                        pending_long_setup = {
                            "confirm_count": confirmation_bars,
                            "breakout_idx": i,
                            "wave_origin": l0_price,
                            "entry_level": breakout_level,
                        }

            # Wave 3 Short detection: H0 -> L1 -> H2 with H2 < H0
            for j in range(len(recent_swing) - 2):
                p0 = recent_swing[j]
                p1 = recent_swing[j + 1]
                p2 = recent_swing[j + 2]

                if p0["type"] != "high" or p1["type"] != "low" or p2["type"] != "high":
                    continue

                h0_price = p0["price"]
                l1_price = p1["price"]
                h2_price = p2["price"]

                if h2_price >= h0_price:
                    continue
                if l1_price >= h0_price:
                    continue

                if not _retrace_in_band(
                    h0_price, l1_price, h2_price, wave3_retrace_min, wave3_retrace_max
                ):
                    continue

                breakout_level = l1_price - _bps_to_price(l1_price, breakout_buffer_bps)

                if bar_close < breakout_level:
                    if confirmation_bars == 0:
                        invalidation_stop = h0_price + _bps_to_price(
                            h0_price, invalidation_buffer_bps
                        )
                        atr_stop = (
                            invalidation_stop
                            if atr_stop_multiplier <= 0
                            else bar_close + atr_stop_multiplier * atr_val
                        )
                        stop_price = min(invalidation_stop, atr_stop)
                        risk_distance = stop_price - bar_close
                        if risk_distance > 0:
                            tp_price = bar_close - risk_reward_ratio * risk_distance
                            signals.iloc[i] = SignalAction.SHORT_ENTRY
                            position = {
                                "direction": "short",
                                "entry_price": bar_close,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "wave_origin": h0_price,
                            }
                            entered_now = True
                    else:
                        pending_short_setup = {
                            "confirm_count": confirmation_bars,
                            "breakout_idx": i,
                            "wave_origin": h0_price,
                            "entry_level": breakout_level,
                        }

            # Wave C Long detection: A->B->C bounded retrace
            # Pattern: high, low, high (start high, pull back low, rally but not above start)
            for j in range(len(recent_swing) - 2):
                p0 = recent_swing[j]
                p1 = recent_swing[j + 1]
                p2 = recent_swing[j + 2]

                if p0["type"] != "high" or p1["type"] != "low" or p2["type"] != "high":
                    continue

                a_price = p0["price"]
                b_price = p1["price"]
                c_price = p2["price"]

                # Wave C long: C should be near A (bounded correction) but lower
                # Retrace of AB should be in [wavec_retrace_min, wavec_retrace_max]
                if c_price >= a_price:
                    continue
                if b_price >= a_price:
                    continue

                if not _retrace_in_band(
                    a_price, b_price, c_price, wavec_retrace_min, wavec_retrace_max
                ):
                    continue

                # Breakout above A (start of wave)
                breakout_level = a_price + _bps_to_price(a_price, breakout_buffer_bps)

                if bar_close > breakout_level:
                    if confirmation_bars == 0:
                        invalidation_stop = b_price - _bps_to_price(
                            b_price, invalidation_buffer_bps
                        )
                        atr_stop = (
                            invalidation_stop
                            if atr_stop_multiplier <= 0
                            else bar_close - atr_stop_multiplier * atr_val
                        )
                        stop_price = max(invalidation_stop, atr_stop)
                        risk_distance = bar_close - stop_price
                        if risk_distance > 0:
                            tp_price = bar_close + risk_reward_ratio * risk_distance
                            signals.iloc[i] = SignalAction.LONG_ENTRY
                            position = {
                                "direction": "long",
                                "entry_price": bar_close,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "wave_origin": b_price,  # Invalidation at B for Wave C
                            }
                            entered_now = True
                    else:
                        pending_long_setup = {
                            "confirm_count": confirmation_bars,
                            "breakout_idx": i,
                            "wave_origin": b_price,  # Invalidation at B for Wave C
                            "entry_level": breakout_level,
                        }

            # Wave C Short detection: A->B->C bounded retrace (inverse)
            for j in range(len(recent_swing) - 2):
                p0 = recent_swing[j]
                p1 = recent_swing[j + 1]
                p2 = recent_swing[j + 2]

                if p0["type"] != "low" or p1["type"] != "high" or p2["type"] != "low":
                    continue

                a_price = p0["price"]
                b_price = p1["price"]
                c_price = p2["price"]

                if c_price <= a_price:
                    continue
                if b_price <= a_price:
                    continue

                if not _retrace_in_band(
                    a_price, b_price, c_price, wavec_retrace_min, wavec_retrace_max
                ):
                    continue

                breakout_level = a_price - _bps_to_price(a_price, breakout_buffer_bps)

                if bar_close < breakout_level:
                    if confirmation_bars == 0:
                        invalidation_stop = b_price + _bps_to_price(
                            b_price, invalidation_buffer_bps
                        )
                        atr_stop = (
                            invalidation_stop
                            if atr_stop_multiplier <= 0
                            else bar_close + atr_stop_multiplier * atr_val
                        )
                        stop_price = min(invalidation_stop, atr_stop)
                        risk_distance = stop_price - bar_close
                        if risk_distance > 0:
                            tp_price = bar_close - risk_reward_ratio * risk_distance
                            signals.iloc[i] = SignalAction.SHORT_ENTRY
                            position = {
                                "direction": "short",
                                "entry_price": bar_close,
                                "stop_price": stop_price,
                                "tp_price": tp_price,
                                "wave_origin": b_price,
                            }
                            entered_now = True
                    else:
                        pending_short_setup = {
                            "confirm_count": confirmation_bars,
                            "breakout_idx": i,
                            "wave_origin": b_price,
                            "entry_level": breakout_level,
                        }

    return signals


def run_elliot_wave_backtest(
    ohlcv_df: pd.DataFrame,
    *,
    swing_window: int = 2,
    confirmation_bars: int = 1,
    wave3_retrace_min: float = _WAVE3_RETRACE_MIN,
    wave3_retrace_max: float = _WAVE3_RETRACE_MAX,
    wavec_retrace_min: float = _WAVEC_RETRACE_MIN,
    wavec_retrace_max: float = _WAVEC_RETRACE_MAX,
    breakout_buffer_bps: float = 5.0,
    invalidation_buffer_bps: float = 5.0,
    atr_period: int = 14,
    atr_stop_multiplier: float = 1.0,
    risk_reward_ratio: float = 2.0,
    # --- Phase 2 preset/plan parameters (caller-facing, informational) --------
    trade_style: Literal["day_trade", "swing_trade", "custom"] | None = None,
    timeframe: str | None = None,
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
    exchange: str | None = None,
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, BacktestEngine]:
    """Run the Elliot Wave Simplified backtest strategy.

    Parameters
    ----------
    ohlcv_df:
        Resolved OHLCV DataFrame with UTC-aware index and columns
        [open, high, low, close, volume].
    swing_window:
        Window size for detecting swing highs/lows (default 2).
    confirmation_bars:
        Number of bars to confirm a breakout before entry (default 1).
    wave3_retrace_min / wave3_retrace_max:
        Retracement band for Wave 3 setups (default 0.382 / 0.786).
    wavec_retrace_min / wavec_retrace_max:
        Retracement band for Wave C setups (default 0.382 / 0.886).
    breakout_buffer_bps:
        Buffer in basis points for breakout confirmation (default 5).
    invalidation_buffer_bps:
        Buffer in basis points for stop/invalidation level (default 5).
    atr_period:
        Period for ATR calculation (default 14).
    atr_stop_multiplier:
        Multiplier for ATR-based stop buffer (default 1.0).
    risk_reward_ratio:
        Reward-to-risk ratio for target calculation (default 2.0).
    trade_style / timeframe / start_at / end_at / exchange:
        Informational plan parameters kept for contract alignment.
    initial_capital:
        Starting capital for the backtest engine.

    Returns
    -------
    signals, engine:
        Signal series and configured engine after the run.

    Raises
    ------
    ValueError
        For malformed inputs, invalid parameters, or missing OHLCV columns.
    """
    signals = generate_elliot_wave_signals(
        ohlcv_df,
        swing_window=swing_window,
        confirmation_bars=confirmation_bars,
        wave3_retrace_min=wave3_retrace_min,
        wave3_retrace_max=wave3_retrace_max,
        wavec_retrace_min=wavec_retrace_min,
        wavec_retrace_max=wavec_retrace_max,
        breakout_buffer_bps=breakout_buffer_bps,
        invalidation_buffer_bps=invalidation_buffer_bps,
        atr_period=atr_period,
        atr_stop_multiplier=atr_stop_multiplier,
        risk_reward_ratio=risk_reward_ratio,
    )

    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(ohlcv_df, signals)

    return signals, engine


__all__ = ["generate_elliot_wave_signals", "run_elliot_wave_backtest", "STRATEGY_ID"]
