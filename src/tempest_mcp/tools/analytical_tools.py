"""Analytical MCP tool handlers — ENG-37.

Public MCP handlers (4):
    calculate_fibonacci
    calculate_tpo
    detect_elliot_wave
    get_market_structure

All tools expose deterministic analytical outputs using the shared backtest window
resolution contract (BacktestWindowRequest + resolve_and_fetch_backtest_ohlcv).

Architecture:
    - Tool boundary resolves/fetches OHLCV exactly once via backtest_window.py
    - calculate_fibonacci reuses calculate_fib_retracements/calculate_fib_extensions
    - calculate_tpo reuses calculate_tpo_chart
    - detect_elliot_wave reuses detect_elliott_waves
    - get_market_structure reuses summarize_market_structure
    - Neither tool produces trading signals, entries/exits, or PnL output
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
from structlog import get_logger

from tempest_mcp.indicators.structure import (
    calculate_fib_extensions,
    calculate_fib_retracements,
    detect_elliott_waves,
    summarize_market_structure,
)
from tempest_mcp.indicators.volume.tpo import calculate_tpo_chart
from tempest_mcp.tools._ohlcv_lifecycle import (
    OhlcvLifecycleRequest,
    analysis_window_payload,
    min_bars_check,
    run_ohlcv_lifecycle,
)
from tempest_mcp.tools._ohlcv_lifecycle import (
    internal_error as _internal_error,
)
from tempest_mcp.tools._ohlcv_lifecycle import (
    parse_iso_datetime as _parse_iso_datetime,
)
from tempest_mcp.tools._ohlcv_lifecycle import (
    validation_error as _validation_error,
)
from tempest_mcp.tools.backtest_window import (
    resolve_and_fetch_backtest_ohlcv,
    validate_max_bars,
    validate_timeframe,
)

logger = get_logger(__name__)
# Compatibility alias retained for existing private-helper tests/imports.
_INTERNAL_ERROR_ALIAS = _internal_error

# Valid output modes for calculate_fibonacci
VALID_FIB_OUTPUT_MODES = ("retracement", "extension")
VALID_TREND_DIRECTIONS = ("bullish", "bearish")

# Session detection helper for TPO single-session enforcement
_SESSION_GAP_HOURS = 4  # If gap > 4 hours between bars, consider it a new session


# ── Validation helpers ──────────────────────────────────────────────────────────


def _validate_symbol(symbol: str) -> None:
    """Validate symbol format defensively before fetch."""
    import re

    if not isinstance(symbol, str) or not symbol:
        raise ValueError("Symbol cannot be empty")
    pattern = re.compile(r"^[A-Za-z0-9]+([/-][A-Za-z0-9]+)?$")
    if not pattern.match(symbol):
        raise ValueError(
            f"Invalid symbol format: {symbol!r} — expected alphanumeric with optional '/' or '-'"
        )
    if symbol.startswith(("/", "-")) or symbol.endswith(("/", "-")):
        raise ValueError(
            f"Invalid symbol format: {symbol!r} — separator cannot be leading or trailing"
        )


def _validate_swing_high_low(swing_high: float, swing_low: float) -> None:
    """Validate swing_high and swing_low are valid numeric values."""
    if not isinstance(swing_high, (int, float)) or not isinstance(swing_low, (int, float)):
        raise ValueError("swing_high and swing_low must be numeric")
    if not math.isfinite(swing_high) or not math.isfinite(swing_low):
        raise ValueError("swing_high and swing_low must be finite numbers")
    if swing_high <= swing_low:
        raise ValueError("swing_high must be greater than swing_low")


def _detect_sessions(ohlcv: pd.DataFrame) -> int:
    """Detect number of sessions in OHLCV data based on time gaps.

    Returns the number of distinct sessions detected.
    A new session is detected when there's a gap > _SESSION_GAP_HOURS between bars.
    """
    if len(ohlcv) < 2:
        return 1

    # Calculate time differences between consecutive bars
    time_diffs = ohlcv.index.to_series().diff()
    time_diffs = time_diffs.dropna()

    # Count sessions: each gap > 4 hours counts as a new session
    session_count = 1
    for diff in time_diffs:
        hours = diff.total_seconds() / 3600
        if hours > _SESSION_GAP_HOURS:
            session_count += 1

    return session_count


def _analytical_lifecycle_request(
    *,
    tool_name: str,
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    exchange: str,
    max_bars: int | None,
) -> OhlcvLifecycleRequest:
    """Build the shared C3 lifecycle request for analytical custom windows."""
    parsed_start_at = _parse_iso_datetime("start_at", start_at)
    parsed_end_at = _parse_iso_datetime("end_at", end_at)
    validated_timeframe = validate_timeframe(timeframe)
    validated_max_bars = validate_max_bars(max_bars)

    return OhlcvLifecycleRequest(
        tool_name=tool_name,
        symbol=symbol,
        trade_style="custom",
        timeframe=validated_timeframe,
        start_at=parsed_start_at,
        end_at=parsed_end_at,
        exchange=exchange,
        max_bars=validated_max_bars,
    )


# ── calculate_fibonacci handler ─────────────────────────────────────────────────


async def calculate_fibonacci(
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    swing_high: float,
    swing_low: float,
    exchange: str = "binance",
    output_mode: str = "retracement",
    trend_direction: str | None = None,
    levels: list[float] | None = None,
    max_bars: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate Fibonacci retracement or extension levels using deterministic anchors.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "BTC/USDT".
    timeframe : str
        OHLCV timeframe, e.g. "1h", "4h".
    start_at : str
        ISO 8601 start datetime.
    end_at : str
        ISO 8601 end datetime.
    swing_high : float
        The swing high price anchor (must be greater than swing_low).
    swing_low : float
        The swing low price anchor (must be less than swing_high).
    exchange : str, default "binance"
        Exchange identifier.
    output_mode : str, default "retracement"
        Output mode: "retracement" or "extension".
        When output_mode="extension", trend_direction is required.
    trend_direction : str, optional
        Required when output_mode="extension". Must be "bullish" or "bearish".
    levels : list[float], optional
        Custom Fibonacci levels. For retracement: values in [0, 1].
        For extension: values > 1. Defaults to standard levels.
    max_bars : int, optional
        Safety cap on estimated candle count.

    Returns
    -------
    dict
        Success envelope with tool name, symbol, window metadata,
        and serialized fib_levels list.
    """
    # 0. Defensive symbol validation
    try:
        _validate_symbol(symbol)
    except ValueError as e:
        return _validation_error(str(e))

    # 1. Validate swing anchors
    try:
        _validate_swing_high_low(swing_high, swing_low)
    except ValueError as e:
        return _validation_error(str(e))

    # 2. Validate output_mode and trend_direction
    if output_mode not in VALID_FIB_OUTPUT_MODES:
        return _validation_error(
            f"output_mode must be one of {VALID_FIB_OUTPUT_MODES}, got {output_mode!r}"
        )

    if output_mode == "extension":
        if trend_direction is None:
            return _validation_error(
                "trend_direction is required when output_mode='extension'"
            )
        if trend_direction not in VALID_TREND_DIRECTIONS:
            return _validation_error(
                f"trend_direction must be one of {VALID_TREND_DIRECTIONS}, got {trend_direction!r}"
            )
    else:
        # For retracement mode, trend_direction should not be provided or is ignored
        if trend_direction is not None and trend_direction not in VALID_TREND_DIRECTIONS:
            return _validation_error(
                f"trend_direction must be one of {VALID_TREND_DIRECTIONS}, got {trend_direction!r}"
            )

    # 3. Build shared lifecycle request (needed for metadata)
    try:
        request = _analytical_lifecycle_request(
            tool_name="calculate_fibonacci",
            symbol=symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
            exchange=exchange,
            max_bars=max_bars,
        )
    except ValueError as e:
        return _validation_error(str(e))

    def _callback(_ohlcv_df: pd.DataFrame, resolved_window: Any) -> dict[str, Any]:
        # Fibonacci remains deterministic anchor-based; OHLCV is fetched for existing window metadata.
        if output_mode == "retracement":
            fib_df = calculate_fib_retracements(
                swing_high=swing_high,
                swing_low=swing_low,
                levels=levels,
            )
        else:  # extension
            fib_df = calculate_fib_extensions(
                swing_high=swing_high,
                swing_low=swing_low,
                trend_direction=trend_direction,
                levels=levels,
            )

        fib_levels = []
        for row in fib_df.itertuples(index=False):
            fib_levels.append(
                {
                    "level_type": row.level_type,
                    "level_ratio": float(row.level_ratio),
                    "price": float(row.price),
                    "swing_high": float(row.swing_high),
                    "swing_low": float(row.swing_low),
                    "trend_direction": row.trend_direction,
                }
            )

        return {
            "tool": "calculate_fibonacci",
            "symbol": symbol,
            "timeframe": resolved_window.timeframe,
            "window": analysis_window_payload(resolved_window),
            "output_mode": output_mode,
            "swing_high": float(swing_high),
            "swing_low": float(swing_low),
            "trend_direction": trend_direction if output_mode == "extension" else None,
            "fib_levels": fib_levels,
            "count": len(fib_levels),
        }

    return run_ohlcv_lifecycle(
        request,
        logger=logger,
        callback=_callback,
        calculation_error_message="Fibonacci calculation failed",
        fetch_ohlcv=resolve_and_fetch_backtest_ohlcv,
    )

# ── calculate_tpo handler ───────────────────────────────────────────────────────


async def calculate_tpo(
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    row_size: float,
    exchange: str = "binance",
    value_area_pct: float = 0.70,
    max_bars: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate TPO (Time-Price Opportunity) chart for a single session.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "BTC/USDT".
    timeframe : str
        OHLCV timeframe, e.g. "1h", "4h".
    start_at : str
        ISO 8601 start datetime.
    end_at : str
        ISO 8601 end datetime.
    row_size : float
        Required price increment for row buckets. Must be positive.
    exchange : str, default "binance"
        Exchange identifier.
    value_area_pct : float, default 0.70
        Percentage of total TPOs for Value Area.
    max_bars : int, optional
        Safety cap on estimated candle count.

    Returns
    -------
    dict
        Success envelope with tool name, symbol, window metadata,
        TPO rows, and session metadata.
    """
    # 0. Defensive symbol validation
    try:
        _validate_symbol(symbol)
    except ValueError as e:
        return _validation_error(str(e))

    # 1. Validate row_size
    if not isinstance(row_size, (int, float)) or row_size <= 0:
        return _validation_error("row_size must be a positive number")
    if not math.isfinite(row_size):
        return _validation_error("row_size must be a finite number")

    # 2. Validate value_area_pct
    if not (0 < value_area_pct <= 1):
        return _validation_error("value_area_pct must be in range (0, 1]")

    # 3. Build shared lifecycle request
    try:
        request = _analytical_lifecycle_request(
            tool_name="calculate_tpo",
            symbol=symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
            exchange=exchange,
            max_bars=max_bars,
        )
    except ValueError as e:
        return _validation_error(str(e))

    def _tpo_sufficiency(ohlcv_df: pd.DataFrame, _resolved_window: Any) -> str | None:
        session_count = _detect_sessions(ohlcv_df)
        if session_count > 1:
            return (
                f"TPO chart requires a single session, but window spans {session_count} sessions. "
                "Please narrow the date range to contain only one continuous session."
            )
        return min_bars_check(2)(ohlcv_df, _resolved_window)

    def _callback(ohlcv_df: pd.DataFrame, resolved_window: Any) -> dict[str, Any]:
        tpo_df = calculate_tpo_chart(
            ohlcv=ohlcv_df,
            row_size=row_size,
            value_area_pct=value_area_pct,
        )

        tpo_rows = []
        for row in tpo_df.itertuples(index=True):
            tpo_rows.append(
                {
                    "row_index": row.Index,
                    "row_low": float(row.row_low),
                    "row_high": float(row.row_high),
                    "row_mid": float(row.row_mid),
                    "tpo_count": int(row.tpo_count),
                    "period_markers": list(row.period_markers),
                    "period_count": int(row.period_count),
                    "in_value_area": bool(row.in_value_area),
                }
            )

        attrs = tpo_df.attrs
        return {
            "tool": "calculate_tpo",
            "symbol": symbol,
            "timeframe": resolved_window.timeframe,
            "window": analysis_window_payload(resolved_window),
            "session": {
                "row_size": float(attrs.get("row_size", row_size)),
                "marker_count": int(attrs.get("marker_count", 0)),
                "poc_price": float(attrs.get("poc_price", 0)),
                "vah_price": float(attrs.get("vah_price", 0)),
                "val_price": float(attrs.get("val_price", 0)),
                "initial_balance_low": float(attrs.get("initial_balance_low", 0)),
                "initial_balance_high": float(attrs.get("initial_balance_high", 0)),
                "range_expanded_up": bool(attrs.get("range_expanded_up", False)),
                "range_expanded_down": bool(attrs.get("range_expanded_down", False)),
            },
            "tpo_rows": tpo_rows,
            "count": len(tpo_rows),
        }

    return run_ohlcv_lifecycle(
        request,
        logger=logger,
        callback=_callback,
        sufficiency_check=_tpo_sufficiency,
        calculation_error_message="TPO chart calculation failed",
        fetch_ohlcv=resolve_and_fetch_backtest_ohlcv,
    )

# ── detect_elliot_wave handler ─────────────────────────────────────────────────


async def detect_elliot_wave(
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    exchange: str = "binance",
    swing_window: int = 2,
    min_swing_pct: float = 0.05,
    wave2_retrace_band: tuple[float, float] | None = None,
    wave3_extension_min: float = 1.0,
    wave4_retrace_max: float = 0.618,
    waveb_retrace_band: tuple[float, float] | None = None,
    wavec_extension_min: float = 1.0,
    degree_thresholds: tuple[float, float] | None = None,
    include_rejected: bool = True,
    max_bars: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect Elliott Wave patterns from OHLCV data.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "BTC/USDT".
    timeframe : str
        OHLCV timeframe, e.g. "1h", "4h".
    start_at : str
        ISO 8601 start datetime.
    end_at : str
        ISO 8601 end datetime.
    exchange : str, default "binance"
        Exchange identifier.
    swing_window : int, default 2
        Window size for swing detection.
    min_swing_pct : float, default 0.05
        Minimum percentage move to qualify as a swing.
    wave2_retrace_band : tuple, optional
        Acceptable retracement range for wave 2 as (min, max).
    wave3_extension_min : float, default 1.0
        Minimum extension ratio for wave 3 relative to wave 1.
    wave4_retrace_max : float, default 0.618
        Maximum retracement for wave 4 relative to wave 3.
    waveb_retrace_band : tuple, optional
        Acceptable retracement range for wave B as (min, max).
    wavec_extension_min : float, default 1.0
        Minimum extension ratio for wave C relative to wave A.
    degree_thresholds : tuple, optional
        Thresholds for degree classification as (micro_max, minor_max).
    include_rejected : bool, default True
        If True, include rejected candidates in output.
    max_bars : int, optional
        Safety cap on estimated candle count.

    Returns
    -------
    dict
        Success envelope with tool name, symbol, window metadata,
        and serialized wave sequences.
    """
    # 0. Defensive symbol validation
    try:
        _validate_symbol(symbol)
    except ValueError as e:
        return _validation_error(str(e))

    # 1. Validate swing detection params
    if not isinstance(swing_window, int) or swing_window < 1:
        return _validation_error("swing_window must be an integer >= 1")
    if not isinstance(min_swing_pct, (int, float)) or not (0.0 < min_swing_pct < 1.0):
        return _validation_error("min_swing_pct must be a float in (0.0, 1.0)")

    # 2. Validate and coerce wave band/threshold params (accept list or tuple, coerce to tuple)
    def _coerce_pair(
        value: Any,
        field_name: str,
        validate_range: tuple[float, float] | None = None,
    ) -> tuple[float, float] | None:
        """Coerce list-or-tuple to tuple, validate length==2 and optional numeric range."""
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            return _validation_error(f"{field_name} must be a list or tuple of (min, max)")
        if len(value) != 2:
            return _validation_error(f"{field_name} must have exactly 2 elements (min, max)")
        try:
            coerced = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return _validation_error(f"{field_name} values must be numeric")
        if validate_range is not None:
            lo, hi = validate_range
            if not (lo < coerced[0] < coerced[1] < hi):
                return _validation_error(
                    f"{field_name} values must satisfy {lo} < min < max < {hi}"
                )
        return coerced

    wave2_retrace_band = _coerce_pair(wave2_retrace_band, "wave2_retrace_band", validate_range=(0.0, 1.0))
    if isinstance(wave2_retrace_band, dict):  # error envelope returned
        return wave2_retrace_band

    if not isinstance(wave3_extension_min, (int, float)) or wave3_extension_min < 0.0:
        return _validation_error("wave3_extension_min must be a non-negative number")

    if not isinstance(wave4_retrace_max, (int, float)) or not (0.0 < wave4_retrace_max < 1.0):
        return _validation_error("wave4_retrace_max must be a float in (0.0, 1.0)")

    waveb_retrace_band = _coerce_pair(waveb_retrace_band, "waveb_retrace_band", validate_range=(0.0, 1.0))
    if isinstance(waveb_retrace_band, dict):  # error envelope returned
        return waveb_retrace_band

    if not isinstance(wavec_extension_min, (int, float)) or wavec_extension_min < 0.0:
        return _validation_error("wavec_extension_min must be a non-negative number")

    degree_thresholds = _coerce_pair(degree_thresholds, "degree_thresholds", validate_range=(0.0, float("inf")))
    if isinstance(degree_thresholds, dict):  # error envelope returned
        return degree_thresholds
    if degree_thresholds is not None and not (0.0 < degree_thresholds[0] < degree_thresholds[1]):
        return _validation_error(
            "degree_thresholds must satisfy 0 < micro_max < minor_max"
        )

    # 3. Build shared lifecycle request
    try:
        request = _analytical_lifecycle_request(
            tool_name="detect_elliot_wave",
            symbol=symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
            exchange=exchange,
            max_bars=max_bars,
        )
    except ValueError as e:
        return _validation_error(str(e))

    # 4. Preserve existing default band values before callback execution.
    if wave2_retrace_band is None:
        wave2_retrace_band = (0.382, 0.786)
    if waveb_retrace_band is None:
        waveb_retrace_band = (0.382, 0.886)
    if degree_thresholds is None:
        degree_thresholds = (0.02, 0.08)

    def _callback(ohlcv_df: pd.DataFrame, resolved_window: Any) -> dict[str, Any]:
        waves_df = detect_elliott_waves(
            ohlcv=ohlcv_df,
            swing_window=swing_window,
            min_swing_pct=min_swing_pct,
            wave2_retrace_band=wave2_retrace_band,
            wave3_extension_min=wave3_extension_min,
            wave4_retrace_max=wave4_retrace_max,
            waveb_retrace_band=waveb_retrace_band,
            wavec_extension_min=wavec_extension_min,
            degree_thresholds=degree_thresholds,
            include_rejected=include_rejected,
        )

        wave_sequences = []
        for row in waves_df.itertuples(index=False):
            wave_sequences.append(
                {
                    "sequence_id": row.sequence_id,
                    "sequence_type": row.sequence_type,
                    "wave_label": row.wave_label,
                    "segment_order": int(row.segment_order),
                    "direction": row.direction,
                    "degree": row.degree,
                    "start_ts": str(row.start_ts) if pd.notna(row.start_ts) else None,
                    "end_ts": str(row.end_ts) if pd.notna(row.end_ts) else None,
                    "start_price": float(row.start_price) if pd.notna(row.start_price) else None,
                    "end_price": float(row.end_price) if pd.notna(row.end_price) else None,
                    "price_delta": float(row.price_delta) if pd.notna(row.price_delta) else None,
                    "retrace_ratio": float(row.retrace_ratio) if pd.notna(row.retrace_ratio) else None,
                    "extension_ratio": float(row.extension_ratio) if pd.notna(row.extension_ratio) else None,
                    "overlap_violation": bool(row.overlap_violation),
                    "invalidation_violation": bool(row.invalidation_violation),
                    "is_rule_compliant": bool(row.is_rule_compliant),
                    "is_accepted_sequence": bool(row.is_accepted_sequence),
                    "rejection_reason": row.rejection_reason if pd.notna(row.rejection_reason) else None,
                }
            )

        return {
            "tool": "detect_elliot_wave",
            "symbol": symbol,
            "timeframe": resolved_window.timeframe,
            "window": analysis_window_payload(resolved_window),
            "parameters": {
                "swing_window": swing_window,
                "min_swing_pct": float(min_swing_pct),
                "wave2_retrace_band": list(wave2_retrace_band),
                "wave3_extension_min": float(wave3_extension_min),
                "wave4_retrace_max": float(wave4_retrace_max),
                "waveb_retrace_band": list(waveb_retrace_band),
                "wavec_extension_min": float(wavec_extension_min),
                "degree_thresholds": list(degree_thresholds),
                "include_rejected": include_rejected,
            },
            "wave_sequences": wave_sequences,
            "count": len(wave_sequences),
        }

    return run_ohlcv_lifecycle(
        request,
        logger=logger,
        callback=_callback,
        sufficiency_check=min_bars_check(10, "for Elliott Wave detection"),
        calculation_error_message="Elliott Wave detection failed",
        fetch_ohlcv=resolve_and_fetch_backtest_ohlcv,
    )

# ── get_market_structure handler ───────────────────────────────────────────────


async def get_market_structure(
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    exchange: str = "binance",
    swing_window: int = 2,
    min_swing_pct: float = 0.02,
    range_lookback: int = 20,
    max_range_pct: float = 0.03,
    breakout_confirm_bars: int = 1,
    adx_period: int = 14,
    adx_trend_threshold: float = 25.0,
    adx_range_ceiling: float = 20.0,
    di_spread_min: float = 2.0,
    breakout_recency_bars: int = 3,
    max_bars: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Get deterministic market structure summary for a time window.

    Parameters
    ----------
    symbol : str
        Trading symbol, e.g. "BTC/USDT".
    timeframe : str
        OHLCV timeframe, e.g. "1h", "4h".
    start_at : str
        ISO 8601 start datetime.
    end_at : str
        ISO 8601 end datetime.
    exchange : str, default "binance"
        Exchange identifier.
    swing_window : int, default 2
        Window size for swing detection.
    min_swing_pct : float, default 0.02
        Minimum percentage move to qualify as a swing.
    range_lookback : int, default 20
        Number of bars to evaluate for range detection.
    max_range_pct : float, default 0.03
        Maximum range width as percentage of range midpoint.
    breakout_confirm_bars : int, default 1
        Number of consecutive bars required to confirm breakout.
    adx_period : int, default 14
        ADX smoothing period.
    adx_trend_threshold : float, default 25.0
        Minimum ADX level to confirm a trending regime.
    adx_range_ceiling : float, default 20.0
        Maximum ADX level below which a ranging regime is confirmed.
    di_spread_min : float, default 2.0
        Minimum DI+ / DI- spread to confirm directional trend.
    breakout_recency_bars : int, default 3
        Maximum bars since breakout to count as recent.
    max_bars : int, optional
        Safety cap on estimated candle count.

    Returns
    -------
    dict
        Success envelope with tool name, symbol, window metadata,
        and the single-row market structure summary.
    """
    # 0. Defensive symbol validation
    try:
        _validate_symbol(symbol)
    except ValueError as e:
        return _validation_error(str(e))

    # 1. Validate swing detection params
    if not isinstance(swing_window, int) or swing_window < 1:
        return _validation_error("swing_window must be an integer >= 1")
    if not isinstance(min_swing_pct, (int, float)) or not (0.0 <= min_swing_pct < 1.0):
        return _validation_error("min_swing_pct must be a float in [0.0, 1.0)")

    # 2. Validate range params
    if not isinstance(range_lookback, int) or range_lookback < 2:
        return _validation_error("range_lookback must be an integer >= 2")
    if not isinstance(max_range_pct, (int, float)) or not (0.0 < max_range_pct < 1.0):
        return _validation_error("max_range_pct must be a float in (0.0, 1.0)")

    # 3. Validate breakout params
    if not isinstance(breakout_confirm_bars, int) or breakout_confirm_bars < 1:
        return _validation_error("breakout_confirm_bars must be an integer >= 1")

    # 4. Validate ADX params
    if not isinstance(adx_period, int) or adx_period < 1:
        return _validation_error("adx_period must be an integer >= 1")
    if not isinstance(adx_trend_threshold, (int, float)) or not (0.0 <= adx_trend_threshold <= 100.0):
        return _validation_error("adx_trend_threshold must be a float in [0.0, 100.0]")
    if not isinstance(adx_range_ceiling, (int, float)) or not (0.0 <= adx_range_ceiling <= 100.0):
        return _validation_error("adx_range_ceiling must be a float in [0.0, 100.0]")
    if not isinstance(di_spread_min, (int, float)) or not (0.0 <= di_spread_min <= 100.0):
        return _validation_error("di_spread_min must be a float in [0.0, 100.0]")

    if not isinstance(breakout_recency_bars, int) or breakout_recency_bars < 1:
        return _validation_error("breakout_recency_bars must be an integer >= 1")

    # 5. Build shared lifecycle request
    try:
        request = _analytical_lifecycle_request(
            tool_name="get_market_structure",
            symbol=symbol,
            timeframe=timeframe,
            start_at=start_at,
            end_at=end_at,
            exchange=exchange,
            max_bars=max_bars,
        )
    except ValueError as e:
        return _validation_error(str(e))

    min_engine_bars = max(adx_period * 2, range_lookback, 2 * swing_window + 3)

    def _callback(ohlcv_df: pd.DataFrame, resolved_window: Any) -> dict[str, Any]:
        summary_df = summarize_market_structure(
            ohlcv=ohlcv_df,
            swing_window=swing_window,
            min_swing_pct=min_swing_pct,
            range_lookback=range_lookback,
            max_range_pct=max_range_pct,
            breakout_confirm_bars=breakout_confirm_bars,
            adx_period=adx_period,
            adx_trend_threshold=adx_trend_threshold,
            adx_range_ceiling=adx_range_ceiling,
            di_spread_min=di_spread_min,
            breakout_recency_bars=breakout_recency_bars,
        )

        if len(summary_df) == 0:
            return {
                "tool": "get_market_structure",
                "symbol": symbol,
                "timeframe": resolved_window.timeframe,
                "window": analysis_window_payload(resolved_window),
                "summary": None,
                "insufficient_data": True,
            }

        row = summary_df.iloc[0]

        def _safe_val(val):
            """Convert value to JSON-safe type."""
            if val is None or (isinstance(val, float) and math.isnan(val)):
                return None
            if isinstance(val, pd.Timestamp):
                return str(val)
            if isinstance(val, (int, float)):
                return float(val)
            if pd.isna(val):
                return None
            return val

        return {
            "tool": "get_market_structure",
            "symbol": symbol,
            "timeframe": resolved_window.timeframe,
            "window": analysis_window_payload(resolved_window),
            "summary": {
                "analysis_ts": _safe_val(row.analysis_ts),
                "window_start_ts": _safe_val(row.window_start_ts),
                "window_end_ts": _safe_val(row.window_end_ts),
                "summary_label": row.summary_label if pd.notna(row.summary_label) else None,
                "decision_rule": row.decision_rule if pd.notna(row.decision_rule) else None,
                "structure_event_ts": _safe_val(row.structure_event_ts),
                "structure_classification": row.structure_classification if pd.notna(row.structure_classification) else None,
                "structure_trend_state": row.structure_trend_state if pd.notna(row.structure_trend_state) else None,
                "adx": _safe_val(row.adx),
                "plus_di": _safe_val(row.plus_di),
                "minus_di": _safe_val(row.minus_di),
                "di_spread": _safe_val(row.di_spread),
                "range_id": int(row.range_id) if pd.notna(row.range_id) else None,
                "range_status": row.range_status if pd.notna(row.range_status) else None,
                "range_high": _safe_val(row.range_high),
                "range_low": _safe_val(row.range_low),
                "breakout_id": int(row.breakout_id) if pd.notna(row.breakout_id) else None,
                "breakout_ts": _safe_val(row.breakout_ts),
                "breakout_direction": row.breakout_direction if pd.notna(row.breakout_direction) else None,
                "breakout_distance_pct": _safe_val(row.breakout_distance_pct),
                "regime_strength": _safe_val(row.regime_strength),
                "confidence": _safe_val(row.confidence),
            },
            "insufficient_data": False,
        }

    return run_ohlcv_lifecycle(
        request,
        logger=logger,
        callback=_callback,
        sufficiency_check=min_bars_check(min_engine_bars, "for market structure analysis"),
        calculation_error_message="Market structure analysis failed",
        fetch_ohlcv=resolve_and_fetch_backtest_ohlcv,
    )
