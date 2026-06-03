"""Internal OHLCV-backed MCP tool lifecycle helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from structlog.stdlib import BoundLogger

from tempest_mcp.config import ErrorCodes
from tempest_mcp.tools.backtest_window import (
    BacktestWindowRequest,
    ResolvedBacktestWindow,
    TradeStyle,
    resolve_and_fetch_backtest_ohlcv,
)


@dataclass(frozen=True)
class OhlcvLifecycleRequest:
    """Validated public window arguments for an OHLCV-backed tool."""

    tool_name: str
    symbol: str
    trade_style: TradeStyle
    timeframe: str | None
    start_at: datetime | None
    end_at: datetime | None
    exchange: str
    max_bars: int | None


def parse_iso_datetime(field_name: str, value: Any) -> datetime | None:
    """Parse an optional ISO-8601 datetime string using existing tool semantics."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a valid ISO 8601 datetime") from exc


def validation_error(message: str) -> dict[str, Any]:
    """Return the existing deterministic validation error envelope."""
    return {
        "success": False,
        "error": {"code": ErrorCodes.INVALID_PARAMETER, "message": message},
    }


def internal_error(message: str) -> dict[str, Any]:
    """Return the existing deterministic internal error envelope."""
    return {
        "success": False,
        "error": {"code": ErrorCodes.INTERNAL_ERROR, "message": message},
    }


def success_envelope(data: dict[str, Any]) -> dict[str, Any]:
    """Wrap a tool-specific data payload in the existing success envelope."""
    return {"success": True, "data": data}


def analysis_window_payload(window: ResolvedBacktestWindow) -> dict[str, Any]:
    """Serialize analysis/analytical window metadata without trade_style."""
    return {
        "start_at_utc": window.start_at_utc.isoformat(),
        "end_at_utc": window.end_at_utc.isoformat(),
        "estimated_bars": window.estimated_bars,
        "exchange": window.exchange,
    }


def backtest_window_payload(window: ResolvedBacktestWindow) -> dict[str, Any]:
    """Serialize backtest window metadata with trade_style."""
    return {
        "trade_style": window.trade_style,
        "timeframe": window.timeframe,
        "start_at_utc": window.start_at_utc.isoformat(),
        "end_at_utc": window.end_at_utc.isoformat(),
        "estimated_bars": window.estimated_bars,
        "exchange": window.exchange,
    }


def build_backtest_window_request(request: OhlcvLifecycleRequest) -> BacktestWindowRequest:
    """Construct the existing C2 window request contract."""
    return BacktestWindowRequest(
        symbol=request.symbol,
        trade_style=request.trade_style,
        timeframe=request.timeframe,
        start_at=request.start_at,
        end_at=request.end_at,
        exchange=request.exchange,
        max_bars=request.max_bars,
    )


SufficiencyCheck = Callable[[pd.DataFrame, ResolvedBacktestWindow], str | None]
LifecycleCallback = Callable[[pd.DataFrame, ResolvedBacktestWindow], dict[str, Any]]


def min_bars_check(minimum: int, suffix: str | None = None) -> SufficiencyCheck:
    """Create an existing-style insufficient-data validation hook."""

    def check(ohlcv_df: pd.DataFrame, _window: ResolvedBacktestWindow) -> str | None:
        if len(ohlcv_df) >= minimum:
            return None
        if suffix:
            return (
                f"Insufficient data: only {len(ohlcv_df)} bars returned "
                f"(minimum {minimum} required {suffix})"
            )
        return f"Insufficient data: only {len(ohlcv_df)} bars returned (minimum {minimum} required)"

    return check


def run_ohlcv_lifecycle(
    request: OhlcvLifecycleRequest,
    *,
    logger: BoundLogger,
    callback: LifecycleCallback,
    sufficiency_check: SufficiencyCheck | None = None,
    calculation_error_message: str,
    fetch_error_message: str = "Data fetch failed",
    fetch_ohlcv: Callable[[BacktestWindowRequest], tuple[pd.DataFrame, ResolvedBacktestWindow]] = resolve_and_fetch_backtest_ohlcv,
) -> dict[str, Any]:
    """Run shared OHLCV fetch/sufficiency/callback/envelope policy."""
    try:
        ohlcv_df, resolved_window = fetch_ohlcv(build_backtest_window_request(request))
    except ValueError as e:
        return validation_error(str(e))
    except Exception as e:
        logger.error("Window resolution/fetch failed", tool=request.tool_name, error=str(e))
        return internal_error(fetch_error_message)

    if sufficiency_check is not None:
        insufficiency_message = sufficiency_check(ohlcv_df, resolved_window)
        if insufficiency_message is not None:
            return validation_error(insufficiency_message)

    try:
        data = callback(ohlcv_df, resolved_window)
    except ValueError as e:
        return validation_error(str(e))
    except Exception as e:
        logger.error(calculation_error_message, tool=request.tool_name, error=str(e))
        return internal_error(calculation_error_message)

    return success_envelope(data)
