"""EMA Stack Trend Following backtest strategy (ENG-22).

This strategy consumes a resolved OHLCV DataFrame and delegates date-range
resolution to the shared backtest contract. It enters when the EMA stack
is aligned in the trend direction and exits on trend failure, stop loss,
or 2:1 reward-to-risk target.

Signal model:
    LONG_ENTRY  — EMA stack aligned bullish (golden cross confirmed)
    SHORT_ENTRY — EMA stack aligned bearish (death cross confirmed)
    LONG_EXIT   — trend failure, stop hit, or 2:1 reward target
    SHORT_EXIT  — trend failure, stop hit, or 2:1 reward target

The strategy is deterministic and returns a signal series plus a configured
BacktestEngine instance.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.indicators.trend.ema import calculate_ema_stack, death_cross, golden_cross

# Shared Phase 2 defaults
TRADE_STYLE_PRESETS = {
    "day_trade": {"timeframe": "1h", "duration_days": 1},
    "swing_trade": {"timeframe": "4h", "duration_days": 7},
}

# Default EMA periods for stack analysis
DEFAULT_EMA_PERIODS = [7, 25, 50, 200]


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a UTC-aware DatetimeIndex."""
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


def run_ema_stack_backtest(
    ohlcv_df: pd.DataFrame,
    ema_periods: list[int] | tuple[int, ...] = tuple(DEFAULT_EMA_PERIODS),
    rr_multiple: float = 2.0,
    trend_confirmation_bars: int = 1,
    stop_buffer_pct: float = 0.0,
    # --- Phase 2 preset/plan parameters (caller-facing, informational) ---------
    trade_style: Literal["day_trade", "swing_trade", "custom"] | None = None,
    timeframe: str | None = None,
    start_at: pd.Timestamp | None = None,
    end_at: pd.Timestamp | None = None,
    exchange: str | None = None,
    initial_capital: float = 100_000.0,
) -> tuple[pd.Series, BacktestEngine]:
    """Run the EMA Stack Trend Following backtest strategy.

    Parameters
    ----------
    ohlcv_df:
        Resolved OHLCV DataFrame with UTC-aware index and columns
        [open, high, low, close, volume]. Must include sufficient warmup
        history for the longest EMA period.
    ema_periods:
        EMA periods to calculate in the stack. Defaults to [7, 25, 50, 200].
        Must have at least 4 periods for golden/death cross detection.
    rr_multiple:
        Reward-to-risk target multiplier. ``2.0`` implements 2:1 R:R.
    trend_confirmation_bars:
        Number of consecutive bars the EMA stack must be valid before entry.
        Default is 1 (immediate entry on first confirmation).
    stop_buffer_pct:
        Optional buffer added to stop distance (0.0 = no buffer).
    trade_style / timeframe / start_at / end_at / exchange:
        Informational plan parameters kept for contract alignment; the
        strategy does not own date-range resolution.
    initial_capital:
        Starting capital for the backtest engine.

    Returns
    -------
    signals, engine:
        Signal series and configured engine after the run.

    Raises
    ------
    ValueError
        For malformed inputs, invalid periods, missing OHLCV columns,
        or insufficient warmup bars for the configured EMA periods.
    """
    # Validate required columns
    if ohlcv_df.empty:
        raise ValueError("ohlcv_df must not be empty")

    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(ohlcv_df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {', '.join(sorted(missing))}")

    # Validate strategy parameters
    if isinstance(ema_periods, list):
        ema_periods = tuple(ema_periods)

    if len(ema_periods) < 4:
        raise ValueError(
            f"ema_periods must have at least 4 periods for stack confirmation, got {len(ema_periods)}"
        )

    if rr_multiple <= 0:
        raise ValueError("rr_multiple must be positive")

    if trend_confirmation_bars <= 0:
        raise ValueError("trend_confirmation_bars must be a positive integer")

    if stop_buffer_pct < 0:
        raise ValueError("stop_buffer_pct must be non-negative")

    # Normalize index to UTC
    ohlcv_df = _ensure_utc_index(ohlcv_df)
    close = ohlcv_df["close"]
    high = ohlcv_df["high"]
    low = ohlcv_df["low"]

    # Calculate EMA stack (raises ValueError if insufficient data)
    ema_stack = calculate_ema_stack(close, periods=list(ema_periods))

    # Initialize signal series
    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # Local position state for stop/target tracking
    # None = flat, dict = in position with stop/tp info
    position: dict[str, float | str] | None = None

    # Track consecutive confirmation bars for entry
    bullish_confirmed_bars = 0
    bearish_confirmed_bars = 0

    for i in range(1, len(ohlcv_df)):
        bar_high = float(high.iloc[i])
        bar_low = float(low.iloc[i])
        bar_close = float(close.iloc[i])
        bar_open = float(ohlcv_df["open"].iloc[i])

        # Build ema_stack snapshot for this bar (all series up to index i)
        current_stack: dict[str, pd.Series] = {}
        for key, series in ema_stack.items():
            current_stack[key] = series.iloc[: i + 1]

        # Check trend direction at this bar
        is_bullish = golden_cross(current_stack)
        is_bearish = death_cross(current_stack)

        # Handle exit conditions first (while in position)
        if position is not None:
            direction = position["direction"]
            stop_price = float(position["stop_price"])
            tp_price = float(position["tp_price"])
            exit_signal: SignalAction | None = None

            if direction == "long":
                # Stop check: open or intrabar extreme
                if bar_open <= stop_price or bar_low <= stop_price:
                    exit_signal = SignalAction.LONG_EXIT
                # Target check
                elif bar_high >= tp_price:
                    exit_signal = SignalAction.LONG_EXIT
                # Trend failure: bearish stack
                elif is_bearish:
                    exit_signal = SignalAction.LONG_EXIT
            else:  # short
                # Stop check: open or intrabar extreme
                if bar_open >= stop_price or bar_high >= stop_price:
                    exit_signal = SignalAction.SHORT_EXIT
                # Target check
                elif bar_low <= tp_price:
                    exit_signal = SignalAction.SHORT_EXIT
                # Trend failure: bullish stack
                elif is_bullish:
                    exit_signal = SignalAction.SHORT_EXIT

            if exit_signal is not None:
                signals.iloc[i] = exit_signal
                position = None
                bullish_confirmed_bars = 0
                bearish_confirmed_bars = 0
                continue

        # Entry logic (when flat)
        if position is None:
            # Update confirmation bar counters
            if is_bullish:
                bullish_confirmed_bars += 1
                bearish_confirmed_bars = 0
            elif is_bearish:
                bearish_confirmed_bars += 1
                bullish_confirmed_bars = 0
            else:
                bullish_confirmed_bars = 0
                bearish_confirmed_bars = 0

            # Check for long entry
            if is_bullish and bullish_confirmed_bars >= trend_confirmation_bars:
                # Stop: signal-bar low with optional buffer
                stop_price = bar_low * (1 - stop_buffer_pct)
                risk_distance = bar_close - stop_price

                # Skip if invalid risk (shouldn't happen with positive buffer, but guard)
                if risk_distance <= 0:
                    bullish_confirmed_bars = 0
                    continue

                tp_price = bar_close + rr_multiple * risk_distance

                signals.iloc[i] = SignalAction.LONG_ENTRY
                position = {
                    "direction": "long",
                    "entry_price": bar_close,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                }
                bullish_confirmed_bars = 0  # Reset after entry

            # Check for short entry
            elif is_bearish and bearish_confirmed_bars >= trend_confirmation_bars:
                # Stop: signal-bar high with optional buffer
                stop_price = bar_high * (1 + stop_buffer_pct)
                risk_distance = stop_price - bar_close

                if risk_distance <= 0:
                    bearish_confirmed_bars = 0
                    continue

                tp_price = bar_close - rr_multiple * risk_distance

                signals.iloc[i] = SignalAction.SHORT_ENTRY
                position = {
                    "direction": "short",
                    "entry_price": bar_close,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                }
                bearish_confirmed_bars = 0  # Reset after entry

    # Run engine with final signals
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(ohlcv_df, signals)

    return signals, engine
