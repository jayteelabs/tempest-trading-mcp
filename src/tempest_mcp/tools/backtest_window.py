"""Shared backtest window resolver and OHLCV fetcher — ENG-17 Phase 2 contract.

This module is the single canonical place for:
1. Validating window arguments (trade_style, start_at, end_at)
2. Resolving preset/custom date ranges
3. Estimating candle count before fetch
4. Fetching OHLCV exactly once
5. Returning a normalized DataFrame + resolved window metadata

Strategy modules remain pure (signals/backtest logic only) and do NOT own
date-range resolution or data fetching.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd
from structlog import get_logger

from tempest_mcp.data._hist import HistoricalDataSource
from tempest_mcp.time_utils import coerce_window_datetime_to_utc

logger = get_logger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────────

TradeStyle = Literal["day_trade", "swing_trade", "custom"]
Timeframe = Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"]

SUPPORTED_TRADE_STYLES: tuple[TradeStyle, ...] = (
    "day_trade",
    "swing_trade",
    "custom",
)
SUPPORTED_TIMEFRAMES: tuple[Timeframe, ...] = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "4h",
    "1d",
    "1wk",
    "1mo",
)
TIMEFRAME_SECONDS: dict[Timeframe, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1wk": 604800,
    "1mo": 2592000,  # approximate
}

# ── Preset defaults ─────────────────────────────────────────────────────────────

TRADE_STYLE_PRESETS: dict[TradeStyle, dict] = {
    "day_trade": {"timeframe": "1h", "duration_days": 1},
    "swing_trade": {"timeframe": "4h", "duration_days": 7},
}

# Hard safety cap for estimated candle count (aligned with historical adapter behavior)
MAX_BARS_HARD_CAP = 1000


@dataclass(frozen=True)
class BacktestWindowRequest:
    """Request payload for backtest window resolution."""

    symbol: str
    trade_style: TradeStyle = "day_trade"
    timeframe: Timeframe | None = None
    start_at: datetime | None = None  # naive → BUSINESS_TZ_NAME, then UTC
    end_at: datetime | None = None  # naive → BUSINESS_TZ_NAME, then UTC
    exchange: str = "binance"
    max_bars: int | None = None  # optional caller cap


@dataclass(frozen=True)
class ResolvedBacktestWindow:
    """Resolved backtest window with metadata after validation + range resolution."""

    symbol: str
    trade_style: TradeStyle
    timeframe: str
    start_at_utc: pd.Timestamp
    end_at_utc: pd.Timestamp
    estimated_bars: int
    exchange: str
    max_bars: int | None = None


# ── Core resolution ─────────────────────────────────────────────────────────────


def resolve_backtest_window(request: BacktestWindowRequest) -> ResolvedBacktestWindow:
    """Resolve a BacktestWindowRequest into a ResolvedBacktestWindow.

    Applies preset defaults for day_trade/swing_trade, requires explicit
    start_at/end_at for custom. Validates window integrity.

    Parameters
    ----------
    request : BacktestWindowRequest
        Raw window request from the MCP tool layer.

    Returns
    -------
    ResolvedBacktestWindow

    Raises
    ------
    ValueError
        When window parameters are invalid (e.g., start_at >= end_at,
        custom without both timestamps, oversized window).
    """
    trade_style = validate_trade_style(request.trade_style)
    timeframe_override = validate_timeframe(request.timeframe)
    caller_cap = validate_max_bars(request.max_bars)
    preset = TRADE_STYLE_PRESETS.get(trade_style)

    if trade_style == "custom":
        # Custom requires explicit timestamps
        if request.start_at is None or request.end_at is None:
            raise ValueError(
                "trade_style='custom' requires both start_at and end_at to be specified"
            )
        # Reject start_at/end_at for non-custom trade styles (strict reject)
        start_utc = coerce_window_datetime_to_utc(request.start_at)
        end_utc = coerce_window_datetime_to_utc(request.end_at)
        timeframe = timeframe_override or "1h"
    else:
        # Preset styles: reject explicit start_at/end_at (strict reject)
        if request.start_at is not None or request.end_at is not None:
            raise ValueError(
                f"trade_style='{trade_style}' does not support start_at or end_at. "
                "Use trade_style='custom' with explicit start_at and end_at."
            )
        duration_days = preset["duration_days"]  # type: ignore[index]
        timeframe = timeframe_override or preset["timeframe"]  # type: ignore[assignment]

        end_utc = datetime.now(timezone.utc)
        start_utc = end_utc - timedelta(days=duration_days)

    # Normalize to UTC timestamps
    if start_utc.tzinfo is None:
        start_utc = start_utc.replace(tzinfo=timezone.utc)
    if end_utc.tzinfo is None:
        end_utc = end_utc.replace(tzinfo=timezone.utc)

    # Window integrity validation
    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)

    if start_ts >= end_ts:
        raise ValueError(f"Invalid window: start_at ({start_ts}) must be before end_at ({end_ts})")

    # Estimate candle count
    estimated_bars = _estimate_candle_count(start_ts, end_ts, timeframe)

    # Safety cap validation
    if estimated_bars > MAX_BARS_HARD_CAP:
        raise ValueError(
            f"Window too large: estimated {estimated_bars} bars exceeds hard safety cap "
            f"of {MAX_BARS_HARD_CAP} bars. Use a shorter date range or increase timeframe."
        )

    if caller_cap is not None and estimated_bars > caller_cap:
        raise ValueError(
            f"Window too large: estimated {estimated_bars} bars exceeds caller-supplied "
            f"max_bars={caller_cap}. Reduce date range or increase max_bars."
        )

    return ResolvedBacktestWindow(
        symbol=request.symbol,
        trade_style=trade_style,
        timeframe=timeframe,
        start_at_utc=start_ts,
        end_at_utc=end_ts,
        estimated_bars=estimated_bars,
        exchange=request.exchange,
        max_bars=caller_cap,
    )


def validate_trade_style(trade_style: str) -> TradeStyle:
    """Validate supported backtest trade styles."""
    if trade_style not in SUPPORTED_TRADE_STYLES:
        supported = ", ".join(SUPPORTED_TRADE_STYLES)
        raise ValueError(f"trade_style must be one of: {supported}")
    return trade_style


def validate_timeframe(timeframe: str | None) -> Timeframe | None:
    """Validate supported backtest timeframes."""
    if timeframe is None:
        return None
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(f"timeframe must be one of: {supported}")
    return timeframe


def validate_max_bars(max_bars: int | None) -> int | None:
    """Validate optional caller-supplied bar cap."""
    if max_bars is None:
        return None
    if isinstance(max_bars, bool) or not isinstance(max_bars, int) or max_bars <= 0:
        raise ValueError("max_bars must be an integer greater than 0")
    return max_bars


def _estimate_candle_count(start: pd.Timestamp, end: pd.Timestamp, timeframe: str) -> int:
    """Estimate the number of candles for a time range at a given timeframe.

    Returns a rough candle count used for safety guardrails before actual fetch.
    """
    duration = end - start
    seconds = duration.total_seconds()

    validated_timeframe = validate_timeframe(timeframe)
    if validated_timeframe is None:
        raise ValueError("timeframe is required")

    tf_sec = TIMEFRAME_SECONDS[validated_timeframe]
    return max(1, int(seconds / tf_sec))


def fetch_resolved_ohlcv(window: ResolvedBacktestWindow) -> pd.DataFrame:
    """Fetch OHLCV data for a resolved backtest window.

    Uses HistoricalDataSource (CCXT primary, yfinance fallback).

    Parameters
    ----------
    window : ResolvedBacktestWindow
        Resolved window with start/end timestamps and timeframe.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with UTC-aware DatetimeIndex and columns
        [open, high, low, close, volume].

    Raises
    ------
    DataSourceError
        When data fetch fails after retries.
    """
    source = HistoricalDataSource()

    try:
        df = source.fetch_ohlcv(
            symbol=window.symbol,
            interval=window.timeframe,
            start=window.start_at_utc.to_pydatetime(),
            end=window.end_at_utc.to_pydatetime(),
        )
    except Exception as e:
        logger.error("OHLCV fetch failed", symbol=window.symbol, error=str(e))
        raise

    if df.empty:
        logger.warning("OHLCV fetch returned empty", symbol=window.symbol)

    return df


def resolve_and_fetch_backtest_ohlcv(
    request: BacktestWindowRequest,
) -> tuple[pd.DataFrame, ResolvedBacktestWindow]:
    """Convenience wrapper: resolve window then fetch OHLCV in one call.

    This is the canonical entry point for the tool layer — it handles
    both resolution and fetch, returning the DataFrame and metadata.

    Parameters
    ----------
    request : BacktestWindowRequest

    Returns
    -------
    tuple[pd.DataFrame, ResolvedBacktestWindow]

    Raises
    ------
    ValueError
        Window validation failures.
    DataSourceError
        OHLCV fetch failures.
    """
    resolved = resolve_backtest_window(request)
    df = fetch_resolved_ohlcv(resolved)
    return df, resolved
