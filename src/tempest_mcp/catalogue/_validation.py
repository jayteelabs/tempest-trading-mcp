"""Argument validation routing for MCP tools."""

import math
import re
from datetime import datetime
from typing import Any

from tempest_mcp.tools.screener_tools import MAX_SCAN_SYMBOLS, SUPPORTED_EXCHANGES

# Symbol format: alphanumeric with optional single separator (/, -)
SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9]+([/-][A-Za-z0-9]+)?$")

def validate_symbol(symbol: str, field_name: str = "symbol") -> str | None:
    """Validate symbol format. Returns None if valid, error message if invalid."""
    if not isinstance(symbol, str):
        return f"{field_name} must be a string"
    if not symbol:
        return f"{field_name} cannot be empty"
    if len(symbol) < 2 or len(symbol) > 20:
        return f"Invalid {field_name} length: {symbol!r} — must be 2-20 characters"
    if not SYMBOL_PATTERN.match(symbol):
        return f"Invalid {field_name} format: {symbol!r} — expected alphanumeric symbols with an optional single '/' or '-'"
    if symbol.startswith(("/", "-")) or symbol.endswith(("/", "-")):
        return f"Invalid {field_name} format: {symbol!r} — separator cannot be leading or trailing"
    if "//" in symbol or "--" in symbol or "/-" in symbol or "-/" in symbol:
        return f"Invalid {field_name} format: {symbol!r} — malformed separators"
    return None


def _validate_exchange(exchange: Any) -> str | None:
    """Validate exchange parameter."""
    if exchange is None:
        return None
    if not isinstance(exchange, str):
        return "exchange must be a string"
    if exchange.lower() not in {"binance", "bybit", "coinbase", "kraken"}:
        return "exchange must be one of: binance, bybit, coinbase, kraken"
    return None


def _validate_timeframe(timeframe: Any) -> str | None:
    """Validate timeframe parameter."""
    if timeframe is None:
        return None
    if not isinstance(timeframe, str):
        return "timeframe must be a string"
    if timeframe not in {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"}:
        return "timeframe must be one of: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo"
    return None


def _validate_limit(limit: Any, field_name: str = "limit", min_val: int = 1, max_val: int = 1000) -> str | None:
    """Validate limit parameter."""
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int):
        return f"{field_name} must be an integer"
    if limit < min_val or limit > max_val:
        return f"{field_name} must be between {min_val} and {max_val}"
    return None


def _validate_iso8601_datetime(value: Any, field_name: str = "since") -> str | None:
    """Validate ISO-8601 datetime string parameters."""
    if value is None:
        return None
    if not isinstance(value, str):
        return f"{field_name} must be a string"
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return f"{field_name} must be a valid ISO-8601 datetime string"
    return None


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> str | None:
    """Validate tool arguments. Returns error message or None if valid."""
    if name == "fetch_ticker":
        if err := validate_symbol(arguments.get("symbol", ""), "symbol"):
            return err
        return _validate_exchange(arguments.get("exchange"))
    if name == "fetch_klines":
        if err := validate_symbol(arguments.get("symbol", ""), "symbol"):
            return err
        if err := _validate_timeframe(arguments.get("timeframe")):
            return err
        if err := _validate_limit(arguments.get("limit"), "limit", 1, 1000):
            return err
        if err := _validate_iso8601_datetime(arguments.get("since"), "since"):
            return err
        if err := _validate_exchange(arguments.get("exchange")):
            return err
        source = arguments.get("source")
        if source is not None and source != "ccxt":
            return 'source must be "ccxt" (historical routing is CCXT+yfinance fallback)'
        return None
    if name == "fetch_orderbook":
        if err := validate_symbol(arguments.get("symbol", ""), "symbol"):
            return err
        if err := _validate_limit(arguments.get("limit"), "limit", 1, 100):
            return err
        return _validate_exchange(arguments.get("exchange"))
    if name == "indicator_rsi":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    # Phase 2 backtest tools — validate symbol
    if name in (
        "backtest_pdh_session",
        "backtest_rsi",
        "backtest_vwap",
        "backtest_ema_stack",
        "backtest_order_blocks",
        "backtest_elliot_wave",
        "compare_strategies",
    ):
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    # Phase 2 analysis tools (ENG-28) — validate symbol
    if name in ("calculate_volume_profile", "detect_order_blocks"):
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    # ENG-37 analytical tools — validate symbol
    if name in ("calculate_fibonacci", "calculate_tpo", "detect_elliot_wave", "get_market_structure"):
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    # ENG-41 sentiment tool — validate symbol and price_bias
    if name == "get_combined_sentiment_dashboard":
        if err := validate_symbol(arguments.get("symbol", ""), "symbol"):
            return err
        price_bias = arguments.get("price_bias")
        if price_bias is None:
            return "price_bias is required"
        if not isinstance(price_bias, str):
            return "price_bias must be a string"
        if price_bias not in ("bullish", "bearish", "neutral"):
            return "price_bias must be one of: bullish, bearish, neutral"
        return None
    # Legacy deprecated tool — still validate symbol for completeness
    if name == "backtest_strategy":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    if name == "screener_scan":
        symbols = arguments.get("symbols")
        if symbols is None:
            pass
        elif not isinstance(symbols, list):
            return "symbols must be an array of strings"
        elif len(symbols) == 0:
            return "symbols must contain at least 1 entry"
        elif len(symbols) > MAX_SCAN_SYMBOLS:
            return f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries"
        else:
            for i, sym in enumerate(symbols):
                if err := validate_symbol(sym, f"symbols[{i}]"):
                    return err

        min_score = arguments.get("min_score", 0.0)
        if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
            return "min_score must be a number"
        if not math.isfinite(min_score):
            return "min_score must be finite"
        if min_score < 0 or min_score > 100:
            return "min_score must be between 0 and 100"

        exchange = arguments.get("exchange", "binance")
        if not isinstance(exchange, str):
            return "exchange must be a string"
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            return f"exchange must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}"

        return None
    # Session breakout screener (ENG-35)
    if name == "session_breakout_scan":
        session = arguments.get("session")
        if session is None:
            return "session is required"
        if not isinstance(session, str):
            return "session must be a string"
        normalized_session = session.lower()
        if normalized_session == "new_york":
            normalized_session = "ny"
        if normalized_session not in ("asia", "london", "ny"):
            return "session must be one of: asia, london, ny"

        symbols = arguments.get("symbols")
        if symbols is not None:
            if not isinstance(symbols, list):
                return "symbols must be an array of strings"
            if len(symbols) == 0:
                return "symbols must contain at least 1 entry"
            if len(symbols) > MAX_SCAN_SYMBOLS:
                return f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries"
            for i, sym in enumerate(symbols):
                if err := validate_symbol(sym, f"symbols[{i}]"):
                    return err

        exchange = arguments.get("exchange", "binance")
        if not isinstance(exchange, str):
            return "exchange must be a string"
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            return f"exchange must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}"

        proximity_pct = arguments.get("proximity_pct", 1.0)
        if isinstance(proximity_pct, bool) or not isinstance(proximity_pct, (int, float)):
            return "proximity_pct must be a number"
        if not math.isfinite(proximity_pct):
            return "proximity_pct must be finite"
        if proximity_pct < 0 or proximity_pct > 100:
            return "proximity_pct must be between 0 and 100"

        volume_multiplier = arguments.get("volume_multiplier", 2.0)
        if isinstance(volume_multiplier, bool) or not isinstance(volume_multiplier, (int, float)):
            return "volume_multiplier must be a number"
        if not math.isfinite(volume_multiplier):
            return "volume_multiplier must be finite"
        if volume_multiplier < 0:
            return "volume_multiplier must be non-negative"

        return None
    # Order-block screener (ENG-36)
    if name == "order_block_screener_scan":
        symbols = arguments.get("symbols")
        if symbols is not None:
            if not isinstance(symbols, list):
                return "symbols must be an array of strings"
            if len(symbols) == 0:
                return "symbols must contain at least 1 entry"
            if len(symbols) > MAX_SCAN_SYMBOLS:
                return f"symbols must contain at most {MAX_SCAN_SYMBOLS} entries"
            for i, sym in enumerate(symbols):
                if err := validate_symbol(sym, f"symbols[{i}]"):
                    return err

        exchange = arguments.get("exchange", "binance")
        if not isinstance(exchange, str):
            return "exchange must be a string"
        if exchange.lower() not in SUPPORTED_EXCHANGES:
            return f"exchange must be one of: {', '.join(sorted(SUPPORTED_EXCHANGES))}"

        atr_period = arguments.get("atr_period", 14)
        if isinstance(atr_period, bool) or not isinstance(atr_period, int):
            return "atr_period must be an integer"
        if atr_period < 2 or atr_period > 200:
            return "atr_period must be between 2 and 200"

        impulse_atr_mult = arguments.get("impulse_atr_mult", 1.0)
        if isinstance(impulse_atr_mult, bool) or not isinstance(impulse_atr_mult, (int, float)):
            return "impulse_atr_mult must be a number"
        if not math.isfinite(impulse_atr_mult):
            return "impulse_atr_mult must be finite"
        if impulse_atr_mult <= 0 or impulse_atr_mult > 10:
            return "impulse_atr_mult must be greater than 0 and at most 10"

        max_zone_age_bars = arguments.get("max_zone_age_bars", 20)
        if isinstance(max_zone_age_bars, bool) or not isinstance(max_zone_age_bars, int):
            return "max_zone_age_bars must be an integer"
        if max_zone_age_bars < 1 or max_zone_age_bars > 500:
            return "max_zone_age_bars must be between 1 and 500"

        return None
    return None


