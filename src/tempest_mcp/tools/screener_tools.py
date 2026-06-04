"""Screener MCP tool implementation — ENG-34/ENG-35."""

import math
from typing import Any

from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.indicator import SessionType
from tempest_mcp.screener._jobs import (
    screening_success,
    serialize_order_block_candidate,
    serialize_order_block_failure,
    serialize_scan_failure,
    serialize_scan_result,
    sort_scan_failures_for_session,
)
from tempest_mcp.screener.scanner import (
    DEFAULT_FILTER_PRESET,
    ORDER_BLOCK_HORIZONS,
    OrderBlockCandidate,
    OrderBlockFailure,
    ScanFailure,
    ScanFilter,
    Screener,
)

logger = get_logger(__name__)

SUPPORTED_EXCHANGES = frozenset({"binance", "bybit", "coinbase", "kraken"})
MAX_SCAN_SYMBOLS = 25

# Session types for session_breakout_scan
SUPPORTED_SESSIONS = frozenset({"asia", "london", "ny"})


# Map of filter string values to ScanFilter enums
FILTER_VALUE_MAP: dict[str, ScanFilter] = {
    "rsi_oversold": ScanFilter.RSI_OVERSOLD,
    "rsi_overbought": ScanFilter.RSI_OVERBOUGHT,
    "trend_bullish": ScanFilter.TREND_BULLISH,
    "trend_bearish": ScanFilter.TREND_BEARISH,
    "high_volatility": ScanFilter.HIGH_VOLATILITY,
    "low_volatility": ScanFilter.LOW_VOLATILITY,
    "volume_spike": ScanFilter.VOLUME_SPIKE,
}


def _parse_filters(filter_strings: list[str] | None) -> list[ScanFilter]:
    """Parse filter string list into ScanFilter enums.

    Args:
        filter_strings: List of filter string values (e.g., ["rsi_oversold", "trend_bullish"])

    Returns:
        List of ScanFilter enums

    Raises:
        ValueError: If any filter string is invalid
    """
    if not filter_strings:
        return []
    filters: list[ScanFilter] = []
    for f in filter_strings:
        if f not in FILTER_VALUE_MAP:
            raise ValueError(
                f"Invalid filter: {f!r}. Valid values: {list(FILTER_VALUE_MAP.keys())}"
            )
        parsed_filter = FILTER_VALUE_MAP[f]
        if parsed_filter not in filters:
            filters.append(parsed_filter)
    return filters


def _error_response(code: int, message: str) -> dict[str, Any]:
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message,
        },
    }


def _validate_symbols(symbols: list[str] | None) -> str | None:
    if symbols is None:
        return None
    if not isinstance(symbols, list):
        return "symbols must be an array of strings"
    if len(symbols) == 0:
        return "symbols must contain at least 1 entry"
    if len(symbols) > MAX_SCAN_SYMBOLS:
        return f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries"
    return None


def _normalize_symbols(symbols: list[str] | None) -> list[str] | None:
    """Deduplicate explicit symbol lists while preserving request order."""
    if symbols is None:
        return None
    return list(dict.fromkeys(symbols))


def _validate_min_score(min_score: float) -> str | None:
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        return "min_score must be a number"
    if not math.isfinite(min_score):
        return "min_score must be finite"
    if min_score < 0 or min_score > 100:
        return "min_score must be between 0 and 100"
    return None


def _validate_exchange(exchange: str) -> tuple[str | None, str | None]:
    if not isinstance(exchange, str):
        return None, "exchange must be a string"
    normalized_exchange = exchange.lower()
    if normalized_exchange not in SUPPORTED_EXCHANGES:
        return None, f"exchange must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}"
    return normalized_exchange, None


def _validate_session(session: str) -> tuple[str | None, str | None]:
    """Validate and normalize session type.

    Args:
        session: Session string (asia, london, ny, or new_york alias)

    Returns:
        Tuple of (normalized_session, error_message)
        - normalized_session: 'asia', 'london', or 'ny' (new_york normalized)
        - error_message: None if valid, error string if invalid
    """
    if not isinstance(session, str):
        return None, "session must be a string"
    normalized = session.lower()
    # Handle new_york alias
    if normalized == "new_york":
        normalized = "ny"
    if normalized not in SUPPORTED_SESSIONS:
        return None, f"session must be one of: {', '.join(sorted(SUPPORTED_SESSIONS))}"
    return normalized, None


def _validate_proximity_pct(proximity_pct: float) -> str | None:
    """Validate proximity_pct parameter."""
    if isinstance(proximity_pct, bool) or not isinstance(proximity_pct, (int, float)):
        return "proximity_pct must be a number"
    if not math.isfinite(proximity_pct):
        return "proximity_pct must be finite"
    if proximity_pct < 0 or proximity_pct > 100:
        return "proximity_pct must be between 0 and 100"
    return None


def _validate_volume_multiplier(volume_multiplier: float) -> str | None:
    """Validate volume_multiplier parameter."""
    if isinstance(volume_multiplier, bool) or not isinstance(volume_multiplier, (int, float)):
        return "volume_multiplier must be a number"
    if not math.isfinite(volume_multiplier):
        return "volume_multiplier must be finite"
    if volume_multiplier < 0:
        return "volume_multiplier must be non-negative"
    return None


def _serialize_scan_result(result: Any) -> dict[str, Any]:
    """Serialize a ScanResult to a JSON-serializable dict."""
    return serialize_scan_result(result)


def _serialize_scan_failure(failure: ScanFailure) -> dict[str, Any]:
    """Serialize a ScanFailure to a JSON-serializable dict."""
    return serialize_scan_failure(failure)


def _validate_atr_period(atr_period: int) -> str | None:
    """Validate atr_period parameter."""
    if isinstance(atr_period, bool) or not isinstance(atr_period, int):
        return "atr_period must be an integer"
    if atr_period < 2 or atr_period > 200:
        return "atr_period must be between 2 and 200"
    return None


def _validate_impulse_atr_mult(impulse_atr_mult: float) -> str | None:
    """Validate impulse_atr_mult parameter."""
    if isinstance(impulse_atr_mult, bool) or not isinstance(impulse_atr_mult, (int, float)):
        return "impulse_atr_mult must be a number"
    if not math.isfinite(impulse_atr_mult):
        return "impulse_atr_mult must be finite"
    if impulse_atr_mult <= 0 or impulse_atr_mult > 10:
        return "impulse_atr_mult must be greater than 0 and at most 10"
    return None


def _validate_max_zone_age_bars(max_zone_age_bars: int) -> str | None:
    """Validate max_zone_age_bars parameter."""
    if isinstance(max_zone_age_bars, bool) or not isinstance(max_zone_age_bars, int):
        return "max_zone_age_bars must be an integer"
    if max_zone_age_bars < 1 or max_zone_age_bars > 500:
        return "max_zone_age_bars must be between 1 and 500"
    return None


def _serialize_order_block_candidate(candidate: OrderBlockCandidate) -> dict[str, Any]:
    """Serialize an OrderBlockCandidate to a JSON-serializable dict."""
    return serialize_order_block_candidate(candidate)


def _serialize_order_block_failure(failure: OrderBlockFailure) -> dict[str, Any]:
    """Serialize an OrderBlockFailure to a JSON-serializable dict."""
    return serialize_order_block_failure(failure)


async def screener_scan(
    symbols: list[str] | None = None,
    filters: list[str] | None = None,
    min_score: float = 0.0,
    exchange: str = "binance",
) -> dict[str, Any]:
    """Multi-factor crypto screener — scan symbols against technical filters.

    Args:
        symbols: List of trading symbols to scan (e.g., ["BTC/USDT", "ETH/USDT"]).
                 If None, uses default symbols from config.
        filters: List of filter names to apply. If None, uses default preset.
                 Valid filters: rsi_oversold, rsi_overbought, trend_bullish,
                 trend_bearish, high_volatility, low_volatility, volume_spike
        min_score: Minimum score threshold (0-100). Only results with score >= min_score
                  are returned. Default 0.0.
        exchange: Exchange to scan (default: binance)

    Returns:
        Dict with success/error envelope:
        - success: True if at least one symbol was scanned successfully
        - data: Contains tool, exchange, applied_config, results, failures (on success)
        - error: Contains code and message (on failure)
    """
    logger.info(
        "Tool invoked: screener_scan", symbols=symbols, filters=filters, min_score=min_score
    )

    # ── Validate request ─────────────────────────────────────────────────────
    if symbol_error := _validate_symbols(symbols):
        return _error_response(ErrorCodes.INVALID_PARAMETER, symbol_error)

    if min_score_error := _validate_min_score(min_score):
        return _error_response(ErrorCodes.INVALID_PARAMETER, min_score_error)

    normalized_exchange, exchange_error = _validate_exchange(exchange)
    if exchange_error:
        return _error_response(ErrorCodes.INVALID_PARAMETER, exchange_error)

    # ── Parse filters ────────────────────────────────────────────────────────
    try:
        parsed_filters = _parse_filters(filters)
    except ValueError as e:
        logger.warning("Invalid filter value", error=str(e))
        return _error_response(ErrorCodes.INVALID_PARAMETER, str(e))

    # ── Execute scan ─────────────────────────────────────────────────────────
    try:
        normalized_symbols = _normalize_symbols(symbols)
        screener = Screener(
            symbols=tuple(normalized_symbols)
            if normalized_symbols is not None
            else ("BTC/USDT", "ETH/USDT", "DOGE/USDT"),
            exchange=normalized_exchange,
            filters=parsed_filters,
            min_score=min_score,
        )
        effective_exchange = getattr(screener, "exchange", normalized_exchange)

        results, failures = screener.scan()

    except Exception as e:
        logger.error("Screener scan failed", error=str(e))
        return _error_response(
            ErrorCodes.INTERNAL_ERROR,
            "Unable to complete screener scan",
        )

    # ── Determine success/failure ────────────────────────────────────────────
    # Partial success: at least one symbol returned usable data
    # Full failure: nothing usable came back (all symbols failed)
    # Success if we got at least some results OR if all symbols failed but no exceptions
    success = screening_success(results, failures)

    # Build applied config (shows what filters were actually used)
    applied_filters = [
        f.value for f in (parsed_filters if parsed_filters else DEFAULT_FILTER_PRESET)
    ]

    response: dict[str, Any] = {
        "success": success,
    }

    if success:
        response["data"] = {
            "tool": "screener_scan",
            "exchange": effective_exchange,
            "applied_config": {
                "filters": applied_filters,
                "min_score": min_score,
            },
            "results": [_serialize_scan_result(r) for r in results],
            "failures": [_serialize_scan_failure(f) for f in failures],
        }
    else:
        # Full failure - return error envelope
        response["error"] = {
            "code": ErrorCodes.DATA_SOURCE_ERROR,
            "message": "All symbols failed to scan",
        }

    logger.info(
        "Tool completed: screener_scan",
        success=success,
        result_count=len(results),
        failure_count=len(failures),
    )

    return response


async def session_breakout_scan(
    session: str,
    symbols: list[str] | None = None,
    exchange: str = "binance",
    proximity_pct: float = 1.0,
    volume_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Session breakout screener — scan symbols for session breakout patterns.

    Evaluates symbols against the requested session (asia, london, ny) using
    detect_session_levels() for session high/low and detect_pdh_pdl() for
    previous-day context. Breakout/proximity flags are computed against both
    session and PDH/PDL levels, with volume confirmation.

    Args:
        session: Session type (required). Valid values: asia, london, ny.
                 Accepts 'new_york' as an alias for 'ny'.
        symbols: List of trading symbols to scan (e.g., ["BTC/USDT", "ETH/USDT"]).
                 If None, uses default symbols from config.
        exchange: Exchange to scan (default: binance).
        proximity_pct: Percentage threshold for near-breakout detection (default: 1.0).
                       Price within proximity_pct% of session high/low is flagged
                       as near-breakout.
        volume_multiplier: Volume threshold multiplier for confirmation (default: 2.0).
                           Current volume must be >= volume_multiplier * prior_window_avg.

    Returns:
        Dict with success/error envelope:
        - success: True if at least one symbol was scanned successfully
        - data: Contains tool, exchange, applied_config, results, failures (on success)
        - error: Contains code and message (on failure)
    """
    logger.info(
        "Tool invoked: session_breakout_scan",
        session=session,
        symbols=symbols,
        exchange=exchange,
        proximity_pct=proximity_pct,
        volume_multiplier=volume_multiplier,
    )

    # ── Validate session ─────────────────────────────────────────────────────
    normalized_session, session_error = _validate_session(session)
    if session_error:
        return _error_response(ErrorCodes.INVALID_PARAMETER, session_error)

    # ── Validate symbols ─────────────────────────────────────────────────────
    if symbol_error := _validate_symbols(symbols):
        return _error_response(ErrorCodes.INVALID_PARAMETER, symbol_error)

    # ── Validate exchange ────────────────────────────────────────────────────
    normalized_exchange, exchange_error = _validate_exchange(exchange)
    if exchange_error:
        return _error_response(ErrorCodes.INVALID_PARAMETER, exchange_error)

    # ── Validate proximity_pct ───────────────────────────────────────────────
    if proximity_error := _validate_proximity_pct(proximity_pct):
        return _error_response(ErrorCodes.INVALID_PARAMETER, proximity_error)

    # ── Validate volume_multiplier ──────────────────────────────────────────
    if volume_error := _validate_volume_multiplier(volume_multiplier):
        return _error_response(ErrorCodes.INVALID_PARAMETER, volume_error)

    # ── Map session string to SessionType enum ──────────────────────────────
    session_type_map = {
        "asia": SessionType.ASIA,
        "london": SessionType.LONDON,
        "ny": SessionType.NEW_YORK,
    }
    session_type = session_type_map[normalized_session]

    # ── Execute scan ─────────────────────────────────────────────────────────
    try:
        normalized_symbols = _normalize_symbols(symbols)
        screener = Screener(
            symbols=tuple(normalized_symbols)
            if normalized_symbols is not None
            else ("BTC/USDT", "ETH/USDT", "DOGE/USDT"),
            exchange=normalized_exchange,
        )
        effective_exchange = getattr(screener, "exchange", normalized_exchange)

        results, failures = screener.session_breakout_scan(
            session=session_type,
            symbols=normalized_symbols,
            proximity_pct=proximity_pct,
            volume_multiplier=volume_multiplier,
        )

    except Exception as e:
        logger.error("Session breakout scan failed", error=str(e))
        return _error_response(
            ErrorCodes.INTERNAL_ERROR,
            "Unable to complete session breakout scan",
        )

    # ── Determine success/failure ────────────────────────────────────────────
    success = screening_success(results, failures)

    response: dict[str, Any] = {
        "success": success,
    }

    if success:
        response["data"] = {
            "tool": "session_breakout_scan",
            "exchange": effective_exchange,
            "applied_config": {
                "session": normalized_session,
                "symbols": normalized_symbols,
                "proximity_pct": proximity_pct,
                "volume_multiplier": volume_multiplier,
            },
            "results": [_serialize_scan_result(r) for r in results],
            "failures": [_serialize_scan_failure(f) for f in failures],
        }
    else:
        # Full failure - deterministic error response
        response["error"] = {
            "code": ErrorCodes.DATA_SOURCE_ERROR,
            "message": "All symbols failed to scan",
        }
        # Include failures in deterministic order
        response["data"] = {
            "tool": "session_breakout_scan",
            "exchange": effective_exchange,
            "applied_config": {
                "session": normalized_session,
                "symbols": normalized_symbols,
                "proximity_pct": proximity_pct,
                "volume_multiplier": volume_multiplier,
            },
            "results": [],
            "failures": [_serialize_scan_failure(f) for f in sort_scan_failures_for_session(failures)],
        }

    logger.info(
        "Tool completed: session_breakout_scan",
        success=success,
        result_count=len(results),
        failure_count=len(failures),
    )

    return response


async def order_block_screener_scan(
    symbols: list[str] | None = None,
    exchange: str = "binance",
    atr_period: int = 14,
    impulse_atr_mult: float = 1.0,
    max_zone_age_bars: int = 20,
) -> dict[str, Any]:
    """Order-block screener — scan symbols for active order-block zones.

    Evaluates symbols across two fixed horizons:
    - day_trade pass: 1h timeframe over 1d window
    - swing_trade pass: 4h timeframe over 7d window

    Each (symbol, horizon) job emits one best candidate or one failure record.

    Args:
        symbols: List of trading symbols to scan (e.g., ["BTC/USDT", "ETH/USDT"]).
                 If None, uses default symbols from config.
        exchange: Exchange to scan (default: binance).
        atr_period: ATR period for order-block detection (default: 14, range 2-200).
        impulse_atr_mult: Body size must be >= impulse_atr_mult * ATR (default: 1.0, range >0 to 10).
        max_zone_age_bars: Max age of zone before expiration (default: 20, range 1-500).

    Returns:
        Dict with success/error envelope:
        - success: True if at least one candidate exists
        - data: Contains tool, exchange, applied_config, candidates, failures (on success)
        - error: Contains code and message (on full failure)
    """
    logger.info(
        "Tool invoked: order_block_screener_scan",
        symbols=symbols,
        exchange=exchange,
        atr_period=atr_period,
        impulse_atr_mult=impulse_atr_mult,
        max_zone_age_bars=max_zone_age_bars,
    )

    # ── Validate symbols ─────────────────────────────────────────────────────
    if symbol_error := _validate_symbols(symbols):
        return _error_response(ErrorCodes.INVALID_PARAMETER, symbol_error)

    # ── Validate exchange ────────────────────────────────────────────────────
    normalized_exchange, exchange_error = _validate_exchange(exchange)
    if exchange_error:
        return _error_response(ErrorCodes.INVALID_PARAMETER, exchange_error)

    # ── Validate atr_period ──────────────────────────────────────────────────
    if atr_error := _validate_atr_period(atr_period):
        return _error_response(ErrorCodes.INVALID_PARAMETER, atr_error)

    # ── Validate impulse_atr_mult ─────────────────────────────────────────────
    if impulse_error := _validate_impulse_atr_mult(impulse_atr_mult):
        return _error_response(ErrorCodes.INVALID_PARAMETER, impulse_error)

    # ── Validate max_zone_age_bars ────────────────────────────────────────────
    if age_error := _validate_max_zone_age_bars(max_zone_age_bars):
        return _error_response(ErrorCodes.INVALID_PARAMETER, age_error)

    # ── Execute scan ─────────────────────────────────────────────────────────
    try:
        normalized_symbols = _normalize_symbols(symbols)
        screener = Screener(
            symbols=tuple(normalized_symbols)
            if normalized_symbols is not None
            else ("BTC/USDT", "ETH/USDT", "DOGE/USDT"),
            exchange=normalized_exchange,
        )
        effective_exchange = getattr(screener, "exchange", normalized_exchange)

        candidates, failures = screener.order_block_scan(
            symbols=normalized_symbols,
            atr_period=atr_period,
            impulse_atr_mult=impulse_atr_mult,
            max_zone_age_bars=max_zone_age_bars,
        )

    except Exception as e:
        logger.error("Order-block screener scan failed", error=str(e))
        return _error_response(
            ErrorCodes.INTERNAL_ERROR,
            "Unable to complete order-block screener scan",
        )

    # ── Determine success/failure ────────────────────────────────────────────
    success = screening_success(candidates, failures)

    # Build applied config showing fixed horizons
    applied_horizons = [{"timeframe": tf, "window_days": wd} for tf, wd in ORDER_BLOCK_HORIZONS]

    response: dict[str, Any] = {
        "success": success,
    }

    if success:
        response["data"] = {
            "tool": "order_block_screener_scan",
            "exchange": effective_exchange,
            "applied_config": {
                "symbols": normalized_symbols,
                "atr_period": atr_period,
                "impulse_atr_mult": impulse_atr_mult,
                "max_zone_age_bars": max_zone_age_bars,
                "horizons": applied_horizons,
            },
            "candidates": [_serialize_order_block_candidate(c) for c in candidates],
            "failures": [_serialize_order_block_failure(f) for f in failures],
        }
    else:
        # Full failure - deterministic error response
        response["error"] = {
            "code": ErrorCodes.DATA_SOURCE_ERROR,
            "message": "All symbol/horizon jobs failed",
        }
        response["data"] = {
            "tool": "order_block_screener_scan",
            "exchange": effective_exchange,
            "applied_config": {
                "symbols": normalized_symbols,
                "atr_period": atr_period,
                "impulse_atr_mult": impulse_atr_mult,
                "max_zone_age_bars": max_zone_age_bars,
                "horizons": applied_horizons,
            },
            "candidates": [],
            "failures": [_serialize_order_block_failure(f) for f in failures],
        }

    logger.info(
        "Tool completed: order_block_screener_scan",
        success=success,
        candidate_count=len(candidates),
        failure_count=len(failures),
    )

    return response
