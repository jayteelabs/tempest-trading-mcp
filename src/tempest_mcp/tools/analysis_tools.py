"""Analysis MCP tool handlers — ENG-28 Phase 2.

Public MCP handlers (2):
    calculate_volume_profile
    detect_order_blocks

Both tools expose read-only analytical outputs using the shared backtest window
resolution contract (BacktestWindowRequest + resolve_and_fetch_backtest_ohlcv).

Architecture:
    - Tool boundary resolves/fetches OHLCV exactly once via backtest_window.py
    - calculate_volume_profile reuses the existing calculate_volume_profile indicator
    - detect_order_blocks reuses detect_active_order_blocks from backtest_order_blocks.py
    - Neither tool produces trading signals, entries/exits, or PnL output
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from structlog import get_logger

from tempest_mcp.config import ErrorCodes
from tempest_mcp.indicators.volume.volume_profile import (
    calculate_volume_profile as _calculate_volume_profile_indicator,
)
from tempest_mcp.strategies.backtest_order_blocks import detect_active_order_blocks
from tempest_mcp.tools.backtest_window import (
    BacktestWindowRequest,
    resolve_and_fetch_backtest_ohlcv,
    validate_max_bars,
    validate_timeframe,
)

# Symbol format pattern — matches alphanumeric base/quote with optional / or -
_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9]+([/-][A-Za-z0-9]+)?$")

logger = get_logger(__name__)

# Valid profile types and dynamic modes (shared with indicator)
VALID_PROFILE_TYPES = ("fixed", "dynamic")
VALID_DYNAMIC_MODES = ("atr", "pct")

# Conservative upper bound for bin_count — prevents CPU-heavy computation
MAX_BIN_COUNT = 500


# ── Validation helpers ──────────────────────────────────────────────────────────


def _parse_iso_datetime(field_name: str, value: Any) -> datetime | None:
    """Parse an optional ISO-8601 datetime string."""
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


def _validation_error(message: str) -> dict[str, Any]:
    """Return a deterministic validation error envelope."""
    return {
        "success": False,
        "error": {
            "code": ErrorCodes.INVALID_PARAMETER,
            "message": message,
        },
    }


def _internal_error(message: str) -> dict[str, Any]:
    """Return a deterministic internal error envelope."""
    return {
        "success": False,
        "error": {
            "code": ErrorCodes.INTERNAL_ERROR,
            "message": message,
        },
    }


def _validate_symbol(symbol: str) -> None:
    """Validate symbol format defensively before fetch.

    Raises ValueError if symbol is empty or format is invalid.
    """
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("Symbol cannot be empty")
    if not _SYMBOL_PATTERN.match(symbol):
        raise ValueError(
            f"Invalid symbol format: {symbol!r} — expected alphanumeric with optional '/' or '-'"
        )
    if symbol.startswith(("/", "-")) or symbol.endswith(("/", "-")):
        raise ValueError(
            f"Invalid symbol format: {symbol!r} — separator cannot be leading or trailing"
        )
    if "//" in symbol or "--" in symbol or "/-" in symbol or "-/" in symbol:
        raise ValueError(f"Invalid symbol format: {symbol!r} — malformed separators")


# ── calculate_volume_profile handler ────────────────────────────────────────────


async def calculate_volume_profile(
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    exchange: str = "binance",
    bin_count: int = 100,
    profile_type: str = "fixed",
    dynamic_mode: str | None = None,
    atr_period: int = 14,
    atr_mult: float = 1.0,
    range_pct: float | None = None,
    value_area_pct: float = 0.70,
    max_bars: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Calculate volume profile for a symbol over a time window.

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
    bin_count : int, default 100
        Number of bins for fixed mode.
    profile_type : str, default "fixed"
        "fixed" or "dynamic" bin sizing.
    dynamic_mode : str, optional
        "atr" or "pct". Required when profile_type="dynamic".
    atr_period : int, default 14
        ATR period for dynamic ATR mode.
    atr_mult : float, default 1.0
        ATR multiplier for dynamic ATR mode.
    range_pct : float, optional
        Percentage of close price for bin width. Required when dynamic_mode="pct".
    value_area_pct : float, default 0.70
        Value area coverage percentage.
    max_bars : int, optional
        Safety cap on estimated candle count.

    Returns
    -------
    dict
        Success envelope with tool name, symbol, timeframe, window metadata,
        scalar summary, and serialized profile_rows list.
    """
    # 0. Defensive symbol validation before fetch
    try:
        _validate_symbol(symbol)
    except ValueError as e:
        return _validation_error(str(e))

    # 1. Parse shared window args
    try:
        parsed_start_at = _parse_iso_datetime("start_at", start_at)
        parsed_end_at = _parse_iso_datetime("end_at", end_at)
        validated_timeframe = validate_timeframe(timeframe)
        validated_max_bars = validate_max_bars(max_bars)
    except ValueError as e:
        return _validation_error(str(e))

    # 2. Validate profile-specific inputs
    if profile_type not in VALID_PROFILE_TYPES:
        return _validation_error(
            f"profile_type must be one of {VALID_PROFILE_TYPES}, got {profile_type!r}"
        )
    if profile_type == "dynamic":
        if dynamic_mode is None:
            return _validation_error("dynamic_mode is required when profile_type='dynamic'")
        if dynamic_mode not in VALID_DYNAMIC_MODES:
            return _validation_error(
                f"dynamic_mode must be one of {VALID_DYNAMIC_MODES}, got {dynamic_mode!r}"
            )
        if dynamic_mode == "atr":
            if atr_period <= 0:
                return _validation_error("atr_period must be positive")
            if atr_mult <= 0:
                return _validation_error("atr_mult must be positive")
        if dynamic_mode == "pct":
            if range_pct is None:
                return _validation_error("range_pct is required when dynamic_mode='pct'")
            if range_pct <= 0:
                return _validation_error("range_pct must be positive")
    if not (0 < bin_count <= MAX_BIN_COUNT):
        return _validation_error(f"bin_count must be a positive integer <= {MAX_BIN_COUNT}")
    if not (0 < value_area_pct <= 1):
        return _validation_error("value_area_pct must be in range (0, 1]")

    # 3. Resolve window + fetch OHLCV (trade_style='custom' forces explicit timestamps)
    request = BacktestWindowRequest(
        symbol=symbol,
        trade_style="custom",
        timeframe=validated_timeframe,
        start_at=parsed_start_at,
        end_at=parsed_end_at,
        exchange=exchange,
        max_bars=validated_max_bars,
    )

    try:
        ohlcv_df, resolved_window = resolve_and_fetch_backtest_ohlcv(request)
    except ValueError as e:
        return _validation_error(str(e))
    except Exception as e:
        logger.error(
            "Window resolution/fetch failed", tool="calculate_volume_profile", error=str(e)
        )
        return _internal_error("Data fetch failed")

    # 4. Validate fetched data
    if len(ohlcv_df) < 2:
        return _validation_error(
            f"Insufficient data: only {len(ohlcv_df)} bars returned (minimum 2 required)"
        )

    # 5. Call indicator (reuses existing implementation)
    try:
        profile_df = _calculate_volume_profile_indicator(
            ohlcv_df,
            bin_count=bin_count,
            profile_type=profile_type,
            dynamic_mode=dynamic_mode,
            atr_period=atr_period,
            atr_mult=atr_mult,
            range_pct=range_pct,
            value_area_pct=value_area_pct,
        )
    except ValueError as e:
        return _validation_error(str(e))
    except Exception as e:
        logger.error(
            "Volume profile calculation failed", tool="calculate_volume_profile", error=str(e)
        )
        return _internal_error("Volume profile calculation failed")

    # 6. Serialize result
    profile_rows = []
    for row in profile_df.itertuples(index=True):
        profile_rows.append(
            {
                "bin_index": row.Index,
                "bin_low": float(row.bin_low),
                "bin_high": float(row.bin_high),
                "bin_mid": float(row.bin_mid),
                "bin_volume": float(row.bin_volume),
                "bin_candle_count": int(row.bin_candle_count),
                "is_hvn": bool(row.is_hvn),
                "is_lvn": bool(row.is_lvn),
                "in_value_area": bool(row.in_value_area),
            }
        )

    return {
        "success": True,
        "data": {
            "tool": "calculate_volume_profile",
            "symbol": symbol,
            "timeframe": resolved_window.timeframe,
            "window": {
                "start_at_utc": resolved_window.start_at_utc.isoformat(),
                "end_at_utc": resolved_window.end_at_utc.isoformat(),
                "estimated_bars": resolved_window.estimated_bars,
                "exchange": resolved_window.exchange,
            },
            "summary": {
                "poc_price": _safe_float(profile_df.attrs.get("poc_price")),
                "vah_price": _safe_float(profile_df.attrs.get("vah_price")),
                "val_price": _safe_float(profile_df.attrs.get("val_price")),
                "profile_shape": profile_df.attrs.get("profile_shape"),
                "total_volume": _safe_float(profile_df.attrs.get("total_volume")),
                "bin_count": profile_df.attrs.get("bin_count"),
                "profile_type": profile_df.attrs.get("profile_type"),
            },
            "profile_rows": profile_rows,
        },
    }


# ── detect_order_blocks handler ─────────────────────────────────────────────────


async def detect_order_blocks(
    symbol: str,
    timeframe: str,
    start_at: str,
    end_at: str,
    exchange: str = "binance",
    atr_period: int = 14,
    impulse_atr_mult: float = 1.0,
    max_zone_age_bars: int = 20,
    max_bars: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Detect active order-block zones as of the end of the requested window.

    This is a **read-only analytical** tool. It returns only active, uninvalidated
    zones after candidate detection + invalidation + age filtering. It does NOT
    produce retest confirmation, entry signals, or PnL output.

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
    atr_period : int, default 14
        Period for ATR calculation.
    impulse_atr_mult : float, default 1.0
        Body size must be >= impulse_atr_mult * ATR for a valid displacement candle.
    max_zone_age_bars : int, default 20
        Maximum age of a zone before it is excluded.
    max_bars : int, optional
        Safety cap on estimated candle count.

    Returns
    -------
    dict
        Success envelope with tool name, symbol, timeframe, window metadata,
        serialized order_blocks list, and count.
    """
    # 0. Defensive symbol validation before fetch
    try:
        _validate_symbol(symbol)
    except ValueError as e:
        return _validation_error(str(e))

    # 1. Parse shared window args
    try:
        parsed_start_at = _parse_iso_datetime("start_at", start_at)
        parsed_end_at = _parse_iso_datetime("end_at", end_at)
        validated_timeframe = validate_timeframe(timeframe)
        validated_max_bars = validate_max_bars(max_bars)
    except ValueError as e:
        return _validation_error(str(e))

    # 2. Validate detection-specific inputs
    if atr_period <= 0:
        return _validation_error("atr_period must be positive")
    if impulse_atr_mult <= 0:
        return _validation_error("impulse_atr_mult must be positive")
    if max_zone_age_bars <= 0:
        return _validation_error("max_zone_age_bars must be positive")

    # 3. Resolve window + fetch OHLCV (trade_style='custom' forces explicit timestamps)
    request = BacktestWindowRequest(
        symbol=symbol,
        trade_style="custom",
        timeframe=validated_timeframe,
        start_at=parsed_start_at,
        end_at=parsed_end_at,
        exchange=exchange,
        max_bars=validated_max_bars,
    )

    try:
        ohlcv_df, resolved_window = resolve_and_fetch_backtest_ohlcv(request)
    except ValueError as e:
        return _validation_error(str(e))
    except Exception as e:
        logger.error("Window resolution/fetch failed", tool="detect_order_blocks", error=str(e))
        return _internal_error("Data fetch failed")

    # 4. Validate fetched data
    if len(ohlcv_df) < max(atr_period, 4):
        return _validation_error(
            f"Insufficient data: only {len(ohlcv_df)} bars returned "
            f"(minimum {max(atr_period, 4)} required for ATR + zone detection)"
        )

    # 5. Call detection helper (read-only analytical boundary)
    try:
        order_blocks = detect_active_order_blocks(
            ohlcv_df,
            atr_period=atr_period,
            impulse_atr_mult=impulse_atr_mult,
            max_zone_age_bars=max_zone_age_bars,
        )
    except ValueError as e:
        return _validation_error(str(e))
    except Exception as e:
        logger.error("Order block detection failed", tool="detect_order_blocks", error=str(e))
        return _internal_error("Order block detection failed")

    # 6. Serialize result
    return {
        "success": True,
        "data": {
            "tool": "detect_order_blocks",
            "symbol": symbol,
            "timeframe": resolved_window.timeframe,
            "window": {
                "start_at_utc": resolved_window.start_at_utc.isoformat(),
                "end_at_utc": resolved_window.end_at_utc.isoformat(),
                "estimated_bars": resolved_window.estimated_bars,
                "exchange": resolved_window.exchange,
            },
            "order_blocks": order_blocks,
            "count": len(order_blocks),
        },
    }


# ── Serialization helpers ───────────────────────────────────────────────────────


def _safe_float(value: Any) -> float | None:
    """Convert value to float, returning None for non-finite values."""
    if value is None:
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None
