"""RSI Mean Reversion Backtest Strategy — ENG-20.

Generates LONG_ENTRY / SHORT_ENTRY / LONG_EXIT / SHORT_EXIT / HOLD signals
based on RSI overbought/oversold conditions with optional divergence confirmation.

Execution semantics (per ENG-20 design):
- Signals fire at bar close; engine executes on next bar open (no lookahead).
- Stop-loss and 2:1 target are SIGNAL-GENERATION triggers, not guaranteed
  intrabar fills at the stop/target price. The strategy generates a
  LONG_EXIT or SHORT_EXIT signal when the condition is met on bar N;
  the engine executes that exit on bar N+1 open.

Divergence semantics:
- divergence_window: int = 20 (lookback for local extrema in divergence detection)
  This is the sole additional RSI knob exposed per revised ENG-20 scope.
"""

from __future__ import annotations

import pandas as pd

from tempest_mcp.backtest.engine import SignalAction
from tempest_mcp.indicators.momentum.rsi import (
    CENTERLINE,
    OVERBOUGHT_THRESHOLD,
    OVERSOLD_THRESHOLD,
    RSI_DEFAULT_PERIOD,
    calculate_rsi,
    detect_rsi_cross,
    detect_rsi_divergence,
)
from tempest_mcp.indicators.volatility.atr import ATR_DEFAULT_PERIOD, calculate_atr

# ---------------------------------------------------------------------------
# Local swing detection helpers
# ---------------------------------------------------------------------------


def _detect_swing_low(prices: pd.Series) -> pd.Series:
    """Detect local swing lows using strict local minimum definition.

    A swing low at index i is lower than BOTH neighbors:
        prices[i] < prices[i-1] AND prices[i] < prices[i+1]

    Args:
        prices: Series of price values (typically close).

    Returns:
        pd.Series of boolean, True where a swing low is detected.
    """
    if len(prices) < 3:
        return pd.Series(False, index=prices.index)

    # Strict local minimum: lower than BOTH neighbors
    result = pd.Series(False, index=prices.index, dtype=bool)

    for i in range(1, len(prices) - 1):
        curr = prices.iloc[i]
        prev = prices.iloc[i - 1]
        next_val = prices.iloc[i + 1]
        # Strict local minimum: lower than BOTH neighbors
        if curr < prev and curr < next_val:
            result.iloc[i] = True

    return result


def _detect_swing_high(prices: pd.Series) -> pd.Series:
    """Detect local swing highs using strict local maximum definition.

    A swing high at index i is higher than BOTH neighbors:
        prices[i] > prices[i-1] AND prices[i] > prices[i+1]

    Args:
        prices: Series of price values (typically close).

    Returns:
        pd.Series of boolean, True where a swing high is detected.
    """
    if len(prices) < 3:
        return pd.Series(False, index=prices.index, dtype=bool)

    result = pd.Series(False, index=prices.index, dtype=bool)

    for i in range(1, len(prices) - 1):
        curr = prices.iloc[i]
        prev = prices.iloc[i - 1]
        next_val = prices.iloc[i + 1]
        # Strict local maximum: higher than BOTH neighbors
        if curr > prev and curr > next_val:
            result.iloc[i] = True

    return result


def _get_recent_swing_low(prices: pd.Series, current_idx: int, lookback: int = 10) -> float | None:
    """Get the most recent swing low before current_idx.

    Args:
        prices: Series of price values.
        current_idx: Current bar index.
        lookback: Number of bars to look back.

    Returns:
        Price value of most recent swing low, or None if no swing low found.
    """
    start_idx = max(0, current_idx - lookback)
    window_prices = prices.iloc[start_idx:current_idx]
    swing_lows = _detect_swing_low(window_prices)

    if not swing_lows.any():
        return None

    # Get the last swing low price
    last_swing_idx = swing_lows[swing_lows].index[-1]
    return float(window_prices.loc[last_swing_idx])


def _get_recent_swing_high(prices: pd.Series, current_idx: int, lookback: int = 10) -> float | None:
    """Get the most recent swing high before current_idx.

    Args:
        prices: Series of price values.
        current_idx: Current bar index.
        lookback: Number of bars to look back.

    Returns:
        Price value of most recent swing high, or None if no swing high found.
    """
    start_idx = max(0, current_idx - lookback)
    window_prices = prices.iloc[start_idx:current_idx]
    swing_highs = _detect_swing_high(window_prices)

    if not swing_highs.any():
        return None

    # Get the last swing high price
    last_swing_idx = swing_highs[swing_highs].index[-1]
    return float(window_prices.loc[last_swing_idx])


# ---------------------------------------------------------------------------
# Main strategy function
# ---------------------------------------------------------------------------


def generate_rsi_signals(
    ohlcv_df: pd.DataFrame,
    rsi_period: int = RSI_DEFAULT_PERIOD,
    confirmation_enabled: bool = True,
    oversold_threshold: int = OVERSOLD_THRESHOLD,
    overbought_threshold: int = OVERBOUGHT_THRESHOLD,
    risk_reward_ratio: float = 2.0,
    atr_stop_multiplier: float = 1.5,
    divergence_window: int = 20,
) -> pd.Series:
    """Generate RSI mean-reversion trading signals.

    Entry logic:
        LONG_ENTRY:  RSI < oversold_threshold
                    AND (confirmation_enabled=False OR bullish_divergence_detected)
        SHORT_ENTRY: RSI > overbought_threshold
                    AND (confirmation_enabled=False OR bearish_divergence_detected)

    Exit logic (signal triggers):
        LONG_EXIT:  RSI crosses centerline (50) upward from below (bullish cross)
                    OR price hits 2:1 reward-to-risk target
                    OR price hits ATR-adjusted swing-based stop
        SHORT_EXIT: RSI crosses centerline (50) downward from above (bearish cross)
                    OR price hits 2:1 reward-to-risk target
                    OR price hits ATR-adjusted swing-based stop

    Stop placement:
        Long:  recent_swing_low - atr_stop_multiplier * ATR
        Short: recent_swing_high + atr_stop_multiplier * ATR

    Args:
        ohlcv_df: DataFrame with columns [open, high, low, close, volume] and
                  DatetimeIndex.
        rsi_period: Period for RSI calculation (default 14).
        confirmation_enabled: If True, require divergence confirmation for entries
                             (default True).
        oversold_threshold: RSI level for oversold condition (default 30).
        overbought_threshold: RSI level for overbought condition (default 70).
        risk_reward_ratio: Reward-to-risk ratio for target placement (default 2.0).
        atr_stop_multiplier: ATR multiplier for stop distance (default 1.5).
        divergence_window: Lookback window for divergence detection (default 20).
                          This is the sole extra RSI knob exposed per ENG-20 scope.

    Returns:
        pd.Series of SignalAction values indexed to ohlcv_df.index.
        One signal per bar; engine executes entries/exits on next bar open.

    Raises:
        ValueError: If ohlcv_df is missing required columns, has insufficient data,
                    or parameters are invalid.

    Note:
        Stop-loss and 2:1 target are signal-generation triggers. The strategy
        generates LONG_EXIT/SHORT_EXIT when stop or target is hit on bar N;
        the BacktestEngine executes that exit on bar N+1 open. This means
        the actual exit price may differ from the stop/target price.
    """
    # ---- Input validation ----
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(ohlcv_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"OHLCV DataFrame missing required columns: {missing_list}")

    if rsi_period <= 0:
        raise ValueError(f"rsi_period must be positive, got {rsi_period}")

    if oversold_threshold >= overbought_threshold:
        raise ValueError(
            f"oversold_threshold ({oversold_threshold}) must be less than "
            f"overbought_threshold ({overbought_threshold})"
        )

    if risk_reward_ratio <= 0:
        raise ValueError(f"risk_reward_ratio must be positive, got {risk_reward_ratio}")

    if atr_stop_multiplier <= 0:
        raise ValueError(f"atr_stop_multiplier must be positive, got {atr_stop_multiplier}")

    if divergence_window <= 0:
        raise ValueError(f"divergence_window must be positive, got {divergence_window}")

    # ---- Calculate indicators ----
    close_prices = ohlcv_df["close"]
    high_prices = ohlcv_df["high"]
    low_prices = ohlcv_df["low"]

    # RSI
    rsi = calculate_rsi(close_prices, period=rsi_period)

    # ATR (using default period, aligned with RSI)
    atr_period = ATR_DEFAULT_PERIOD
    atr = calculate_atr(high_prices, low_prices, close_prices, period=atr_period)

    def _normalize_timestamp(value: object) -> object:
        """Normalize timestamp keys for robust index/date comparisons.

        - Preserve non-datetime values as-is
        - Convert timezone-aware timestamps to UTC-naive for stable set/dict lookups
        """
        if not isinstance(value, pd.Timestamp):
            converted = pd.to_datetime(value, errors="coerce")
            if pd.isna(converted):
                return value
            value = converted

        if value.tz is not None:
            return value.tz_convert("UTC").tz_localize(None)
        return value

    # Divergence detection
    divergence = detect_rsi_divergence(close_prices, rsi, window=divergence_window)

    # Build sets of normalized dates with bullish/bearish divergence for O(1) lookup
    bullish_div_dates: set[object] = set()
    bearish_div_dates: set[object] = set()
    if not divergence.empty:
        bullish_div_dates = {
            _normalize_timestamp(date)
            for date in divergence.loc[divergence["type"] == "bullish", "date"]
        }
        bearish_div_dates = {
            _normalize_timestamp(date)
            for date in divergence.loc[divergence["type"] == "bearish", "date"]
        }

    # RSI centerline crosses (for mean reversion exits), normalized for O(1) lookup
    centerline_crosses = detect_rsi_cross(rsi, threshold=CENTERLINE)
    cross_direction_by_date: dict[object, str] = {}
    if not centerline_crosses.empty:
        for row in centerline_crosses.itertuples(index=False):
            cross_direction_by_date[_normalize_timestamp(row.date)] = row.direction

    # ---- Initialize signals ----
    n = len(ohlcv_df)
    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # Track open position for stop/target management
    # position state: None, or {"entry_idx": int, "entry_price": float, "direction": "long"|"short",
    #                           "stop_price": float, "target_price": float}
    position: dict | None = None

    # RSI extrema detection for entry conditions
    rsi_oversold = rsi < oversold_threshold
    rsi_overbought = rsi > overbought_threshold

    for i in range(n):
        idx = ohlcv_df.index[i]
        idx_normalized = _normalize_timestamp(idx)
        current_signal = SignalAction.HOLD

        # ---- Entry logic ----
        if position is None:
            # Check for LONG entry
            if (
                rsi_oversold.iloc[i]
                if i < len(rsi_oversold) and not pd.isna(rsi_oversold.iloc[i])
                else False
            ):
                # RSI in oversold zone
                entry_confirmed = False
                if not confirmation_enabled:
                    entry_confirmed = True
                elif idx_normalized in bullish_div_dates:
                    entry_confirmed = True

                if entry_confirmed:
                    current_signal = SignalAction.LONG_ENTRY
                    # Set up position tracking
                    entry_price = float(ohlcv_df["close"].iloc[i])  # Use close for signal price
                    swing_low = _get_recent_swing_low(close_prices, i, lookback=10)
                    # Handle empty ATR (insufficient data)
                    if len(atr) > i and not pd.isna(atr.iloc[i]):
                        current_atr = float(atr.iloc[i])
                    else:
                        current_atr = 0.0

                    if swing_low is not None and current_atr > 0:
                        stop_price = swing_low - atr_stop_multiplier * current_atr
                    else:
                        # Fallback: ATR-adjusted entry offset; 5% buffer if ATR unavailable
                        stop_price = (
                            entry_price - atr_stop_multiplier * current_atr
                            if current_atr > 0
                            else entry_price * 0.95
                        )

                    risk = entry_price - stop_price
                    target_price = entry_price + risk_reward_ratio * risk

                    position = {
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "direction": "long",
                        "stop_price": stop_price,
                        "target_price": target_price,
                    }

            # Check for SHORT entry
            elif (
                rsi_overbought.iloc[i]
                if i < len(rsi_overbought) and not pd.isna(rsi_overbought.iloc[i])
                else False
            ):
                # RSI in overbought zone
                entry_confirmed = False
                if not confirmation_enabled:
                    entry_confirmed = True
                elif idx_normalized in bearish_div_dates:
                    entry_confirmed = True

                if entry_confirmed:
                    current_signal = SignalAction.SHORT_ENTRY
                    # Set up position tracking
                    entry_price = float(ohlcv_df["close"].iloc[i])
                    swing_high = _get_recent_swing_high(close_prices, i, lookback=10)
                    # Handle empty ATR (insufficient data)
                    if len(atr) > i and not pd.isna(atr.iloc[i]):
                        current_atr = float(atr.iloc[i])
                    else:
                        current_atr = 0.0

                    if swing_high is not None and current_atr > 0:
                        stop_price = swing_high + atr_stop_multiplier * current_atr
                    else:
                        # Fallback: ATR-adjusted entry offset; 5% buffer if ATR unavailable
                        stop_price = (
                            entry_price + atr_stop_multiplier * current_atr
                            if current_atr > 0
                            else entry_price * 1.05
                        )

                    risk = stop_price - entry_price
                    target_price = entry_price - risk_reward_ratio * risk

                    position = {
                        "entry_idx": i,
                        "entry_price": entry_price,
                        "direction": "short",
                        "stop_price": stop_price,
                        "target_price": target_price,
                    }

        # ---- Exit logic ----
        else:
            current_high = float(ohlcv_df["high"].iloc[i])
            current_low = float(ohlcv_df["low"].iloc[i])

            if position["direction"] == "long":
                # Check stop hit ( price touches stop on bar N -> signal exit for bar N+1 )
                if current_low <= position["stop_price"]:
                    current_signal = SignalAction.LONG_EXIT
                    position = None

                # Check target hit
                elif current_high >= position["target_price"]:
                    current_signal = SignalAction.LONG_EXIT
                    position = None

                # Check mean reversion: RSI crosses centerline upward from below
                else:
                    cross_direction = cross_direction_by_date.get(idx_normalized)
                    if cross_direction == "bullish":
                        current_signal = SignalAction.LONG_EXIT
                        position = None

            elif position["direction"] == "short":
                # Check stop hit
                if current_high >= position["stop_price"]:
                    current_signal = SignalAction.SHORT_EXIT
                    position = None

                # Check target hit
                elif current_low <= position["target_price"]:
                    current_signal = SignalAction.SHORT_EXIT
                    position = None

                # Check mean reversion: RSI crosses centerline downward from above
                else:
                    cross_direction = cross_direction_by_date.get(idx_normalized)
                    if cross_direction == "bearish":
                        current_signal = SignalAction.SHORT_EXIT
                        position = None

        signals.iloc[i] = current_signal

    return signals


__all__ = ["generate_rsi_signals"]
