"""Order Blocks Mean-Reversion Backtest Strategy — ENG-23.

Generates LONG_ENTRY / SHORT_ENTRY / LONG_EXIT / SHORT_EXIT / HOLD signals
based on deterministic order-block / imbalance zone detection with
mean-reversion entries confirmed by zone retest and rejection.

Execution semantics (per ENG-23 design):
- Signals fire at bar close; engine executes on next bar open (no lookahead).
- Stop-loss and target are SIGNAL-GENERATION triggers, not guaranteed
  intrabar fills. The strategy generates a LONG_EXIT or SHORT_EXIT signal
  when the condition is met on bar N; the engine executes that exit on bar N+1 open.
- No direct LONG->SHORT or SHORT->LONG flips without FLAT transition.

Zone detection semantics:
- Bullish zone: last bearish candle before bullish displacement, where
  displacement close breaks prior high and body >= impulse_atr_mult * ATR.
  Zone bounds: [low_of_ob_candle, open_of_ob_candle].
- Bearish zone: mirror rule.
  Zone bounds: [open_of_ob_candle, high_of_ob_candle].
- Imbalance/FVG (optional): bullish low[i+1] > high[i-1], bearish high[i+1] < low[i-1].
- Entry confirmation: retest intersects zone + rejection close in mean-reversion direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import pandas as pd

from tempest_mcp.backtest.engine import SignalAction
from tempest_mcp.indicators.volatility.atr import ATR_DEFAULT_PERIOD, calculate_atr

# ---------------------------------------------------------------------------
# Internal enums and dataclasses
# ---------------------------------------------------------------------------


class _ZoneDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"


class _ExitReason(Enum):
    STOP_HIT = "stop_hit"
    TARGET_HIT = "target_hit"
    ZONE_INVALIDATION = "zone_invalidation"
    STRUCTURAL_FAILURE = "structural_failure"


@dataclass
class _Zone:
    """Represents an order-block / imbalance zone."""

    direction: _ZoneDirection
    # Index of the candle that formed the zone (the OB candle)
    ob_candle_idx: int
    # Zone price bounds
    low: float
    high: float
    # ATR at zone creation (for tolerance calculations)
    atr: float
    # Whether this zone has been invalidated by subsequent price action
    invalidated: bool = False
    # Whether FVG/imbalance was present on zone creation
    has_fvg: bool = False
    # Bar index when zone was created
    created_at: int = 0


@dataclass
class _OpenPosition:
    """Tracks an open position and its associated zone."""

    direction: Literal["long", "short"]
    entry_idx: int
    entry_price: float
    stop_price: float
    target_price: float
    # The zone this position was entered from
    zone: _Zone
    # Structural failure threshold (ATR-based)
    structural_threshold: float = 0.0


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

OB_IMPULSE_ATR_MULT: float = 1.0
"""Body size multiplier for impulse candle (order block candle must be a significant move)."""

OB_RETEST_ATR_TOLERANCE: float = 0.5
"""ATR fraction tolerance for retest intersection detection."""

OB_MIN_BARS_BEFORE_ENTRY: int = 2
"""Minimum bars after zone creation before entries are allowed."""

OB_MAX_ZONE_AGE_BARS: int = 20
"""Maximum age of a zone before it expires (no entries)."""

DEFAULT_ATR_PERIOD: int = ATR_DEFAULT_PERIOD
"""Default period for ATR calculation."""

RISK_REWARD_RATIO: float = 2.0
"""Default risk-reward ratio for target placement."""

STOP_ATR_MULTIPLIER: float = 1.5
"""ATR multiplier for stop distance below/above zone for long/short entries."""

STRUCTURAL_BREAK_ATR_MULT: float = 1.0
"""ATR multiplier for structural failure threshold (break of recent swing)."""


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    ohlcv_df: pd.DataFrame,
    confirmation_enabled: bool,
    atr_period: int,
    impulse_atr_mult: float,
    retest_atr_tolerance: float,
    min_bars_before_entry: int,
    max_zone_age_bars: int,
    risk_reward_ratio: float,
    stop_atr_multiplier: float,
    structural_break_atr_mult: float,
) -> None:
    """Validate all strategy inputs; raise ValueError on invalid input."""
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(ohlcv_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"OHLCV DataFrame missing required columns: {missing_list}")

    if atr_period <= 0:
        raise ValueError(f"atr_period must be positive, got {atr_period}")

    if impulse_atr_mult <= 0:
        raise ValueError(f"impulse_atr_mult must be positive, got {impulse_atr_mult}")

    if retest_atr_tolerance < 0:
        raise ValueError(f"retest_atr_tolerance must be non-negative, got {retest_atr_tolerance}")

    if min_bars_before_entry < 0:
        raise ValueError(f"min_bars_before_entry must be non-negative, got {min_bars_before_entry}")

    if max_zone_age_bars <= 0:
        raise ValueError(f"max_zone_age_bars must be positive, got {max_zone_age_bars}")

    if risk_reward_ratio <= 0:
        raise ValueError(f"risk_reward_ratio must be positive, got {risk_reward_ratio}")

    if stop_atr_multiplier <= 0:
        raise ValueError(f"stop_atr_multiplier must be positive, got {stop_atr_multiplier}")

    if structural_break_atr_mult <= 0:
        raise ValueError(
            f"structural_break_atr_mult must be positive, got {structural_break_atr_mult}"
        )

    if len(ohlcv_df) < max(atr_period, 4):
        raise ValueError(
            f"Insufficient data for ATR({atr_period}) + zone detection: "
            f"need at least {max(atr_period, 4)} bars, got {len(ohlcv_df)}"
        )

    # Check for NaN in required columns
    for col in required_columns:
        if ohlcv_df[col].isna().any():
            raise ValueError(f"OHLCV DataFrame contains NaN values in required column: {col}")

    # Check for non-monotonic index
    if not ohlcv_df.index.is_monotonic_increasing and not ohlcv_df.index.equals(
        ohlcv_df.index.sort_values()
    ):
        raise ValueError("OHLCV DataFrame index must be monotonically increasing")

    if ohlcv_df.index.has_duplicates:
        raise ValueError("OHLCV DataFrame index must not contain duplicates")


# ---------------------------------------------------------------------------
# Context computation (ATR + derived candle metrics)
# ---------------------------------------------------------------------------


def _compute_context(
    high: pd.Series,
    low: pd.Series,
    open_series: pd.Series,
    close: pd.Series,
    atr_period: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute ATR and derived candle metrics.

    Returns:
        Tuple of (atr, body_size, upper_wick, lower_wick) series, all aligned
        to the input series index.
    """
    atr = calculate_atr(high, low, close, period=atr_period)

    body_size = (close - open_series).abs()
    upper_wick = high - body_size  # simplified: distance from body top to high
    lower_wick = body_size - low  # simplified: distance from body bottom to low

    return atr, body_size, upper_wick, lower_wick


# ---------------------------------------------------------------------------
# Order block candidate detection
# ---------------------------------------------------------------------------


def _detect_order_block_candidates(
    high: pd.Series,
    low: pd.Series,
    open: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    impulse_atr_mult: float,
) -> list[_Zone]:
    """Detect all order-block / imbalance zone candidates.

    Scans for:
    - Bullish OB: last bearish candle before bullish displacement.
      Displacement: close breaks prior high AND body >= impulse_atr_mult * ATR.
      Zone: [low_of_ob, open_of_ob].
    - Bearish OB: mirror rule.
      Zone: [open_of_ob, high_of_ob].

    Returns:
        List of _Zone objects (may be empty).
    """
    zones: list[_Zone] = []
    n = len(close)

    for i in range(2, n - 1):
        curr_open = float(open.iloc[i])
        curr_close = float(close.iloc[i])
        prev_open = float(open.iloc[i - 1])
        prev_close = float(close.iloc[i - 1])
        prev_high = float(high.iloc[i - 1])
        prev_low = float(low.iloc[i - 1])

        # Current bar ATR (may be NaN for early bars)
        curr_atr = float(atr.iloc[i]) if not pd.isna(atr.iloc[i]) else 0.0

        # Determine if current bar is bullish or bearish
        curr_is_bullish = curr_close > curr_open
        curr_is_bearish = curr_close < curr_open

        # Determine if prior bar is bullish or bearish
        prev_is_bullish = prev_close > prev_open
        prev_is_bearish = prev_close < prev_open

        if curr_is_bullish and prev_is_bearish:
            # Bullish displacement: current close breaks prior high
            # Body must be significant: body >= impulse_atr_mult * ATR
            body_size = abs(curr_close - curr_open)
            if body_size >= impulse_atr_mult * curr_atr and curr_close > prev_high:
                # Bullish OB: zone is [low_of_bearish_candle, open_of_bearish_candle]
                zone = _Zone(
                    direction=_ZoneDirection.BULLISH,
                    ob_candle_idx=i - 1,  # the bearish candle is the OB
                    low=prev_low,
                    high=prev_open,
                    atr=curr_atr,
                    invalidated=False,
                    has_fvg=_check_fvg_bullish(high, low, i),
                    created_at=i,
                )
                zones.append(zone)

        elif curr_is_bearish and prev_is_bullish:
            # Bearish displacement: current close breaks prior low
            body_size = abs(curr_close - curr_open)
            if body_size >= impulse_atr_mult * curr_atr and curr_close < prev_low:
                # Bearish OB: zone is [open_of_bullish_candle, high_of_bullish_candle]
                zone = _Zone(
                    direction=_ZoneDirection.BEARISH,
                    ob_candle_idx=i - 1,  # the bullish candle is the OB
                    low=prev_open,
                    high=prev_high,
                    atr=curr_atr,
                    invalidated=False,
                    has_fvg=_check_fvg_bearish(high, low, i),
                    created_at=i,
                )
                zones.append(zone)

    return zones


def _check_fvg_bullish(high: pd.Series, low: pd.Series, idx: int) -> bool:
    """Check for bullish FVG (Fair Value Gap) at idx.

    Bullish FVG: low[i+1] > high[i-1] (gap between prior high and current low).
    Note: idx here is the displacement candle index; check FVG around it.
    """
    if idx < 2 or idx >= len(high) - 1:
        return False
    try:
        return float(low.iloc[idx + 1]) > float(high.iloc[idx - 1])
    except (IndexError, ValueError):
        return False


def _check_fvg_bearish(high: pd.Series, low: pd.Series, idx: int) -> bool:
    """Check for bearish FVG (Fair Value Gap) at idx.

    Bearish FVG: high[i+1] < low[i-1] (gap between prior low and current high).
    """
    if idx < 2 or idx >= len(high) - 1:
        return False
    try:
        return float(high.iloc[idx + 1]) < float(low.iloc[idx - 1])
    except (IndexError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Active zone selection
# ---------------------------------------------------------------------------


def _select_active_zones(
    zones: list[_Zone],
    current_idx: int,
    max_zone_age_bars: int,
) -> list[tuple[_Zone, int]]:
    """Select eligible active zones at current bar index.

    Filters:
    - Zone must not be invalidated
    - Zone age must be within max_zone_age_bars
    - Zone ob_candle_idx must be before current_idx

    Returns:
        List of (zone, bars_since_creation) tuples for eligible zones.
    """
    active: list[tuple[_Zone, int]] = []
    for zone in zones:
        if zone.invalidated:
            continue
        age = current_idx - zone.created_at
        if age < 0:
            continue
        if age > max_zone_age_bars:
            continue
        active.append((zone, age))
    return active


# ---------------------------------------------------------------------------
# Retest and confirmation
# ---------------------------------------------------------------------------


def _find_zone_retest(
    zone: _Zone,
    high: pd.Series,
    low: pd.Series,
    open: pd.Series,
    close: pd.Series,
    current_idx: int,
    retest_atr_tolerance: float,
) -> bool:
    """Detect if price has retested the zone between ob_candle_idx+1 and current_idx.

    Retest detection: any bar's low (for bullish) or high (for bearish)
    intersects zone within ATR tolerance.
    """
    start = zone.ob_candle_idx + 1
    end = current_idx

    if end <= start:
        return False

    tolerance = zone.atr * retest_atr_tolerance

    for i in range(start, min(end + 1, len(high))):
        bar_high = float(high.iloc[i])
        bar_low = float(low.iloc[i])

        if zone.direction == _ZoneDirection.BULLISH:
            # Bullish zone: [zone.low, zone.high]
            # Retest = price comes down to zone area
            # Intersection if bar_low <= zone.high + tolerance and bar_high >= zone.low - tolerance
            if bar_low <= zone.high + tolerance and bar_high >= zone.low - tolerance:
                return True
        else:  # BEARISH
            # Bearish zone: [zone.low, zone.high]
            # Retest = price comes up to zone area
            if bar_high >= zone.low - tolerance and bar_low <= zone.high + tolerance:
                return True

    return False


def _entry_confirmation_passes(
    zone: _Zone,
    high: pd.Series,
    low: pd.Series,
    open: pd.Series,
    close: pd.Series,
    current_idx: int,
) -> bool:
    """Check if entry confirmation passes at current bar.

    For bullish: rejection close above zone high.
    For bearish: rejection close below zone low.
    """
    if current_idx >= len(close):
        return False

    bar_close = float(close.iloc[current_idx])

    if zone.direction == _ZoneDirection.BULLISH:
        # Bullish confirmation: close above zone high (rejection of lower prices)
        return bar_close > zone.high
    else:
        # Bearish confirmation: close below zone low (rejection of higher prices)
        return bar_close < zone.low


# ---------------------------------------------------------------------------
# Risk levels
# ---------------------------------------------------------------------------


def _build_risk_levels(
    zone: _Zone,
    entry_price: float,
    risk_reward_ratio: float,
    stop_atr_multiplier: float,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[float, float, float]:
    """Build stop price, target price, and structural threshold.

    Returns:
        Tuple of (stop_price, target_price, structural_threshold).
    """
    atr = zone.atr
    stop_distance = stop_atr_multiplier * atr

    if zone.direction == _ZoneDirection.BULLISH:
        # Stop below zone low
        stop_price = zone.low - stop_distance
        risk = entry_price - stop_price
        target_price = entry_price + risk_reward_ratio * risk
        # Structural failure: break of recent swing low
        structural_threshold = zone.low - STRUCTURAL_BREAK_ATR_MULT * atr
    else:
        # Stop above zone high
        stop_price = zone.high + stop_distance
        risk = stop_price - entry_price
        target_price = entry_price - risk_reward_ratio * risk
        # Structural failure: break of recent swing high
        structural_threshold = zone.high + STRUCTURAL_BREAK_ATR_MULT * atr

    return stop_price, target_price, structural_threshold


# ---------------------------------------------------------------------------
# Exit reason detection
# ---------------------------------------------------------------------------


def _exit_reason(
    position: _OpenPosition,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    current_idx: int,
) -> _ExitReason | None:
    """Determine exit reason if any exit condition is met.

    Precedence: stop -> target -> zone invalidation -> structural failure.

    Returns:
        Exit reason enum if exit triggered, None otherwise.
    """
    bar_high = float(high.iloc[current_idx])
    bar_low = float(low.iloc[current_idx])
    bar_close = float(close.iloc[current_idx])
    zone = position.zone

    if position.direction == "long":
        # Stop hit
        if bar_low <= position.stop_price:
            return _ExitReason.STOP_HIT
        # Target hit
        if bar_high >= position.target_price:
            return _ExitReason.TARGET_HIT
        # Zone invalidation: price closes below zone low
        if bar_close < zone.low:
            return _ExitReason.ZONE_INVALIDATION
        # Structural failure
        if bar_close < position.structural_threshold:
            return _ExitReason.STRUCTURAL_FAILURE

    else:  # short
        # Stop hit
        if bar_high >= position.stop_price:
            return _ExitReason.STOP_HIT
        # Target hit
        if bar_low <= position.target_price:
            return _ExitReason.TARGET_HIT
        # Zone invalidation: price closes above zone high
        if bar_close > zone.high:
            return _ExitReason.ZONE_INVALIDATION
        # Structural failure
        if bar_close > position.structural_threshold:
            return _ExitReason.STRUCTURAL_FAILURE

    return None


# ---------------------------------------------------------------------------
# Invalidate zones helper
# ---------------------------------------------------------------------------


def _invalidate_zones_by_retest(
    zones: list[_Zone],
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    current_idx: int,
) -> list[_Zone]:
    """Mark zones as invalidated if price has moved through them significantly.

    A zone is invalidated if price closes beyond it without a valid entry.
    """
    for zone in zones:
        if zone.invalidated:
            continue
        if current_idx <= zone.ob_candle_idx:
            continue

        # Check if price has closed beyond zone
        bar_close = float(close.iloc[current_idx]) if current_idx < len(close) else 0.0

        if zone.direction == _ZoneDirection.BULLISH:
            # Bullish zone invalidated if price closes below zone entirely
            if bar_close < zone.low:
                zone.invalidated = True
        else:
            # Bearish zone invalidated if price closes above zone entirely
            if bar_close > zone.high:
                zone.invalidated = True

    return zones


# ---------------------------------------------------------------------------
# Main strategy function
# ---------------------------------------------------------------------------


def generate_order_block_signals(
    ohlcv_df: pd.DataFrame,
    confirmation_enabled: bool = True,
    atr_period: int = DEFAULT_ATR_PERIOD,
    impulse_atr_mult: float = OB_IMPULSE_ATR_MULT,
    retest_atr_tolerance: float = OB_RETEST_ATR_TOLERANCE,
    min_bars_before_entry: int = OB_MIN_BARS_BEFORE_ENTRY,
    max_zone_age_bars: int = OB_MAX_ZONE_AGE_BARS,
    risk_reward_ratio: float = RISK_REWARD_RATIO,
    stop_atr_multiplier: float = STOP_ATR_MULTIPLIER,
    structural_break_atr_mult: float = STRUCTURAL_BREAK_ATR_MULT,
) -> pd.Series:
    """Generate order-block mean-reversion trading signals.

    Entry logic (bullish):
        - Bullish OB zone detected and not invalidated
        - Zone age >= min_bars_before_entry
        - Zone retest detected (price touched zone within ATR tolerance)
        - (confirmation_enabled=False OR rejection close above zone high)

    Entry logic (bearish):
        - Bearish OB zone detected and not invalidated
        - Zone age >= min_bars_before_entry
        - Zone retest detected
        - (confirmation_enabled=False OR rejection close below zone low)

    Exit logic (signal triggers, in precedence order):
        1. Stop hit (price crosses stop level)
        2. Target hit (price reaches 2:1 reward-to-risk target)
        3. Zone invalidation (price closes beyond zone without hitting target)
        4. Structural failure (price breaks structural threshold)

    Args:
        ohlcv_df: DataFrame with columns [open, high, low, close, volume] and
                  UTC-aware DatetimeIndex.
        confirmation_enabled: If True, require rejection close confirmation
                              before entries (default True).
        atr_period: Period for ATR calculation (default 14).
        impulse_atr_mult: Body size must be >= impulse_atr_mult * ATR (default 1.0).
        retest_atr_tolerance: ATR fraction tolerance for retest detection (default 0.5).
        min_bars_before_entry: Min bars after zone creation before entries (default 2).
        max_zone_age_bars: Max age of zone before expiration (default 20).
        risk_reward_ratio: Reward-to-risk ratio for target (default 2.0).
        stop_atr_multiplier: ATR multiplier for stop distance (default 1.5).
        structural_break_atr_mult: ATR multiplier for structural threshold (default 1.0).

    Returns:
        pd.Series of SignalAction values indexed to ohlcv_df.index.
        Default is HOLD on all bars; entries/exits emitted per above logic.
        Engine executes entries/exits on next bar open.

    Raises:
        ValueError: If ohlcv_df is missing required columns, has insufficient data,
                    contains NaN values, has non-monotonic index, or parameters are invalid.

    Note:
        Signals fire at bar close; engine executes on next bar open. Stop and target
        are signal-generation triggers, not guaranteed fill prices.
    """
    # ---- Input validation ----
    _validate_inputs(
        ohlcv_df=ohlcv_df,
        confirmation_enabled=confirmation_enabled,
        atr_period=atr_period,
        impulse_atr_mult=impulse_atr_mult,
        retest_atr_tolerance=retest_atr_tolerance,
        min_bars_before_entry=min_bars_before_entry,
        max_zone_age_bars=max_zone_age_bars,
        risk_reward_ratio=risk_reward_ratio,
        stop_atr_multiplier=stop_atr_multiplier,
        structural_break_atr_mult=structural_break_atr_mult,
    )

    # ---- Extract series ----
    open_series = ohlcv_df["open"]
    high_series = ohlcv_df["high"]
    low_series = ohlcv_df["low"]
    close_series = ohlcv_df["close"]

    # ---- Compute context (ATR + derived metrics) ----
    atr, body_size, upper_wick, lower_wick = _compute_context(
        high=high_series,
        low=low_series,
        open_series=open_series,
        close=close_series,
        atr_period=atr_period,
    )

    # ---- Initialize signals ----
    n = len(ohlcv_df)
    signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)

    # ---- Detect all order-block candidates upfront ----
    all_zones: list[_Zone] = _detect_order_block_candidates(
        high=high_series,
        low=low_series,
        open=open_series,
        close=close_series,
        atr=atr,
        impulse_atr_mult=impulse_atr_mult,
    )

    # ---- Track open position ----
    position: _OpenPosition | None = None

    # ---- Iterate bars left-to-right ----
    for i in range(n):
        current_signal = SignalAction.HOLD

        # ---- Zone invalidation pass: update zone states based on price action ----
        if i > 0:
            all_zones = _invalidate_zones_by_retest(
                all_zones, high_series, low_series, close_series, i
            )

        # ---- Exit logic ----
        if position is not None:
            exit_reason = _exit_reason(
                position=position,
                high=high_series,
                low=low_series,
                close=close_series,
                current_idx=i,
            )
            if exit_reason is not None:
                if position.direction == "long":
                    current_signal = SignalAction.LONG_EXIT
                else:
                    current_signal = SignalAction.SHORT_EXIT
                position = None

        # ---- Entry logic ----
        if position is None:
            # Select eligible active zones at this bar
            eligible = _select_active_zones(
                zones=all_zones,
                current_idx=i,
                max_zone_age_bars=max_zone_age_bars,
            )

            # Filter by minimum age
            eligible = [(z, age) for z, age in eligible if age >= min_bars_before_entry]

            for zone, _age in eligible:
                # Check retest
                has_retest = _find_zone_retest(
                    zone=zone,
                    high=high_series,
                    low=low_series,
                    open=open_series,
                    close=close_series,
                    current_idx=i,
                    retest_atr_tolerance=retest_atr_tolerance,
                )
                if not has_retest:
                    continue

                # Check confirmation
                if confirmation_enabled:
                    if not _entry_confirmation_passes(
                        zone=zone,
                        high=high_series,
                        low=low_series,
                        open=open_series,
                        close=close_series,
                        current_idx=i,
                    ):
                        continue

                # Entry price = close of signal bar
                entry_price = float(close_series.iloc[i])

                # Build risk levels
                stop_price, target_price, structural_threshold = _build_risk_levels(
                    zone=zone,
                    entry_price=entry_price,
                    risk_reward_ratio=risk_reward_ratio,
                    stop_atr_multiplier=stop_atr_multiplier,
                    high=high_series,
                    low=low_series,
                    close=close_series,
                )

                # Emit entry signal
                if zone.direction == _ZoneDirection.BULLISH:
                    current_signal = SignalAction.LONG_ENTRY
                    position = _OpenPosition(
                        direction="long",
                        entry_idx=i,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        zone=zone,
                        structural_threshold=structural_threshold,
                    )
                else:
                    current_signal = SignalAction.SHORT_ENTRY
                    position = _OpenPosition(
                        direction="short",
                        entry_idx=i,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        zone=zone,
                        structural_threshold=structural_threshold,
                    )

                # Only take the first eligible zone for entry
                break

        signals.iloc[i] = current_signal

    return signals


# ---------------------------------------------------------------------------
# Standalone analytical detection helper (ENG-28)
# ---------------------------------------------------------------------------


def detect_active_order_blocks(
    ohlcv_df: pd.DataFrame,
    *,
    atr_period: int = DEFAULT_ATR_PERIOD,
    impulse_atr_mult: float = OB_IMPULSE_ATR_MULT,
    max_zone_age_bars: int = OB_MAX_ZONE_AGE_BARS,
) -> list[dict]:
    """Detect active, uninvalidated order-block zones as of the final bar in the window.

    This is a **read-only analytical** helper that exposes only the detection-stage
    boundary: candidate detection → invalidation filtering → age filtering.
    It **does not** produce retest confirmation, entry signals, or PnL output.

    The function processes the full window left-to-right, progressively applying
    invalidation so that zones invalidated before the end-of-window are excluded.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        DataFrame with columns [open, high, low, close, volume] and
        UTC-aware DatetimeIndex. Must be non-empty, monotonic, non-duplicated.
    atr_period : int, default 14
        Period for ATR calculation.
    impulse_atr_mult : float, default 1.0
        Body size must be >= impulse_atr_mult * ATR for a valid displacement candle.
    max_zone_age_bars : int, default 20
        Maximum age of a zone before it is excluded from the active set.

    Returns
    -------
    list[dict]
        List of active zone dicts sorted by creation order. Each dict contains:
        - ``date``: ISO 8601 timestamp of the zone's displacement candle (UTC)
        - ``type``: ``"bullish"`` or ``"bearish"``
        - ``zone_high``: Upper boundary of the zone
        - ``zone_low``: Lower boundary of the zone
        - ``freshness_candles``: Bars from zone creation to final window bar
          (after invalidation and age filtering)

    Raises
    ------
    ValueError
        If ohlcv_df is missing required columns, has insufficient data,
        contains NaN values, has non-monotonic index, or parameters are invalid.

    Note
    ----
    ``freshness_candles`` is computed as ``final_bar_index - zone.created_at``
    after invalidation and age filtering have been applied at the end of the window.
    A zone that was invalidated before the final bar will not appear in the result.
    """
    # Validate detection-stage inputs only (subset of full strategy validation)
    _validate_detection_inputs(
        ohlcv_df=ohlcv_df,
        atr_period=atr_period,
        impulse_atr_mult=impulse_atr_mult,
        max_zone_age_bars=max_zone_age_bars,
    )

    open_series = ohlcv_df["open"]
    high_series = ohlcv_df["high"]
    low_series = ohlcv_df["low"]
    close_series = ohlcv_df["close"]

    # Compute ATR context
    atr, _, _, _ = _compute_context(
        high=high_series,
        low=low_series,
        open_series=open_series,
        close=close_series,
        atr_period=atr_period,
    )

    # Detect all candidate zones upfront
    all_zones: list[_Zone] = _detect_order_block_candidates(
        high=high_series,
        low=low_series,
        open=open_series,
        close=close_series,
        atr=atr,
        impulse_atr_mult=impulse_atr_mult,
    )

    n = len(ohlcv_df)
    final_idx = n - 1

    # Progressively apply invalidation across the window
    for i in range(n):
        if i > 0:
            all_zones = _invalidate_zones_by_retest(
                all_zones, high_series, low_series, close_series, i
            )

    # Select active zones at the final bar
    active_at_end = _select_active_zones(
        zones=all_zones,
        current_idx=final_idx,
        max_zone_age_bars=max_zone_age_bars,
    )

    # Build result sorted by creation order (stable, deterministic)
    result = []
    for zone, _age in sorted(active_at_end, key=lambda item: item[0].ob_candle_idx):
        # freshness_candles = bars from creation to final bar
        freshness = final_idx - zone.created_at
        zone_ts = ohlcv_df.index[zone.ob_candle_idx]
        result.append(
            {
                "date": zone_ts.isoformat(),
                "type": zone.direction.value,
                "zone_high": zone.high,
                "zone_low": zone.low,
                "freshness_candles": freshness,
            }
        )

    return result


def _validate_detection_inputs(
    ohlcv_df: pd.DataFrame,
    atr_period: int,
    impulse_atr_mult: float,
    max_zone_age_bars: int,
) -> None:
    """Validate detection-stage inputs for detect_active_order_blocks."""
    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(ohlcv_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"OHLCV DataFrame missing required columns: {missing_list}")

    if atr_period <= 0:
        raise ValueError(f"atr_period must be positive, got {atr_period}")

    if impulse_atr_mult <= 0:
        raise ValueError(f"impulse_atr_mult must be positive, got {impulse_atr_mult}")

    if max_zone_age_bars <= 0:
        raise ValueError(f"max_zone_age_bars must be positive, got {max_zone_age_bars}")

    if len(ohlcv_df) < max(atr_period, 4):
        raise ValueError(
            f"Insufficient data for ATR({atr_period}) + zone detection: "
            f"need at least {max(atr_period, 4)} bars, got {len(ohlcv_df)}"
        )

    for col in required_columns:
        if ohlcv_df[col].isna().any():
            raise ValueError(f"OHLCV DataFrame contains NaN values in required column: {col}")

    if not ohlcv_df.index.is_monotonic_increasing and not ohlcv_df.index.equals(
        ohlcv_df.index.sort_values()
    ):
        raise ValueError("OHLCV DataFrame index must be monotonically increasing")

    if ohlcv_df.index.has_duplicates:
        raise ValueError("OHLCV DataFrame index must not contain duplicates")


__all__ = ["generate_order_block_signals", "detect_active_order_blocks"]
