"""Screener MCP tool implementation — ENG-34."""

import math
from typing import Any

from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger
from tempest_mcp.screener.scanner import DEFAULT_FILTER_PRESET, ScanFailure, ScanFilter, Screener

logger = get_logger(__name__)

SUPPORTED_EXCHANGES = frozenset({"binance", "bybit", "coinbase", "kraken"})
MAX_SCAN_SYMBOLS = 25


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
    if len(symbols) > MAX_SCAN_SYMBOLS:
        return f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries"
    return None


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


def _serialize_scan_result(result: Any) -> dict[str, Any]:
    """Serialize a ScanResult to a JSON-serializable dict."""
    return {
        "symbol": result.symbol,
        "exchange": result.exchange,
        "timestamp": result.timestamp,
        "price": result.price,
        "filters_matched": result.filters_matched,
        "indicator_values": result.indicator_values,
        "score": result.score,
    }


def _serialize_scan_failure(failure: ScanFailure) -> dict[str, Any]:
    """Serialize a ScanFailure to a JSON-serializable dict."""
    return {
        "symbol": failure.symbol,
        "exchange": failure.exchange,
        "reason": failure.reason,
    }


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
        screener = Screener(
            symbols=tuple(symbols) if symbols else ("BTC/USDT", "ETH/USDT", "DOGE/USDT"),
            exchange=normalized_exchange,
            filters=parsed_filters,
            min_score=min_score,
        )

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
    has_results = len(results) > 0
    has_failures = len(failures) > 0

    # Success if we got at least some results OR if all symbols failed but no exceptions
    success = has_results or not has_failures

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
            "exchange": normalized_exchange,
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
