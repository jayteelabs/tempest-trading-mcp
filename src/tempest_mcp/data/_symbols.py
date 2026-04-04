"""
Symbol conversion utilities for cross-exchange compatibility.

This module is load-bearing - it provides the canonical symbol format conversion
between TradingView (BTCUSD) and CCXT/Binance (BTCUSDT) formats.

Design Decisions (D11, D12):
- D11: Symbol conversion via _symbols.py - canonical format is CCXT/Binance (BTCUSDT)
- D12: TV uses BTCUSD, CCXT uses BTCUSDT - adapter normalizes

WARNING: BTCUSD and BTCUSDT are DIFFERENT instruments on some exchanges.
Never silently convert between them without being explicit about the intent.

The `exchange` param on `fetch_live_price` is a future extension point for
multi-exchange support.
"""

from typing import Literal, TypedDict

import structlog

logger = structlog.get_logger()

# Supported exchange types
ExchangeName = Literal["binance", "bybit", "coinbase", "kraken"]

# Symbol format type
SymbolFormat = Literal["tradingview", "ccxt"]

# Supported timeframe strings
SUPPORTED_TIMEFRAMES: frozenset[str] = frozenset(
    {
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "4h",
        "1d",
        "1wk",
        "1mo",
    }
)

# Limit bounds
MIN_LIMIT: int = 1
MAX_LIMIT: int = 1000


class SymbolMapping(TypedDict):
    """Mapping between TradingView and CCXT symbol formats."""

    tradingview: str
    ccxt: str
    base: str
    quote: str


# Canonical symbol mappings for v1 crypto scope
# Design: TV uses BTCUSD, CCXT/Binance uses BTCUSDT
# WARNING: BTCUSD ≠ BTCUSDT — they are different instruments
SYMBOL_MAPPINGS: dict[str, SymbolMapping] = {
    "BTCUSDT": {
        "tradingview": "BTCUSD",
        "ccxt": "BTCUSDT",
        "base": "BTC",
        "quote": "USDT",
    },
    "BTCUSD": {
        "tradingview": "BTCUSD",
        "ccxt": "BTCUSDT",
        "base": "BTC",
        "quote": "USD",
    },
    "ETHUSDT": {
        "tradingview": "ETHUSD",
        "ccxt": "ETHUSDT",
        "base": "ETH",
        "quote": "USDT",
    },
    "ETHUSD": {
        "tradingview": "ETHUSD",
        "ccxt": "ETHUSDT",
        "base": "ETH",
        "quote": "USD",
    },
    "DOGEUSDT": {
        "tradingview": "DOGEUSD",
        "ccxt": "DOGEUSDT",
        "base": "DOGE",
        "quote": "USDT",
    },
    "DOGEUSD": {
        "tradingview": "DOGEUSD",
        "ccxt": "DOGEUSDT",
        "base": "DOGE",
        "quote": "USD",
    },
}


def _sanitize_symbol(symbol: str) -> str:
    """Strip whitespace and control characters from symbol for safe logging."""
    return "".join(c for c in symbol if c.isprintable())


def _validate_timeframe(timeframe: str) -> bool:
    """Check if timeframe is supported."""
    return timeframe in SUPPORTED_TIMEFRAMES


def _validate_limit(limit: int) -> int:
    """Clamp limit to valid range."""
    return max(MIN_LIMIT, min(MAX_LIMIT, limit))


def normalize_to_ccxt(symbol: str, exchange: ExchangeName = "binance") -> str:
    """Convert a symbol to CCXT canonical format.

    CCXT format is the canonical format (D11). This function accepts either
    TradingView format (BTCUSD) or CCXT format (BTCUSDT) and returns the
    CCXT format suitable for exchange API calls.

    Args:
        symbol: Symbol in any supported format (e.g., "BTCUSD", "BTCUSDT")
        exchange: Target exchange (default: "binance"). Future extension point
            for exchange-specific symbol formats.

    Returns:
        Symbol in CCXT canonical format (e.g., "BTCUSDT")

    Raises:
        ValueError: If symbol is not recognized or is invalid

    Example:
        >>> normalize_to_ccxt("BTCUSD")
        'BTCUSDT'
        >>> normalize_to_ccxt("BTCUSDT")
        'BTCUSDT'
        >>> normalize_to_ccxt("ethusdt")
        'ETHUSDT'
    """
    # Reject empty/whitespace-only symbols early
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty or whitespace")

    symbol_upper = symbol.strip().upper()

    # Direct lookup in mappings
    if symbol_upper in SYMBOL_MAPPINGS:
        return SYMBOL_MAPPINGS[symbol_upper]["ccxt"]

    # Try to infer format
    # If ends with USDT, already in CCXT format
    if symbol_upper.endswith("USDT"):
        return symbol_upper

    # If ends with USD but not USDT, likely TradingView format
    # WARNING: BTCUSD and BTCUSDT are DIFFERENT instruments
    if symbol_upper.endswith("USD") and not symbol_upper.endswith("USDT"):
        base = symbol_upper[:-3]  # Remove USD
        ccxt_symbol = f"{base}USDT"
        logger.warning(
            "symbol_converted_tv_to_ccxt",
            symbol=symbol_upper,
            ccxt_symbol=ccxt_symbol,
            warning="BTCUSD and BTCUSDT are different instruments - confirm this is intentional",
        )
        return ccxt_symbol

    # Unknown format
    safe_symbol = _sanitize_symbol(symbol)
    raise ValueError(
        f"Unrecognized symbol format: '{safe_symbol}'. "
        f"Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
    )


def normalize_to_tradingview(symbol: str) -> str:
    """Convert a symbol to TradingView format.

    TradingView uses the format without T (e.g., BTCUSD instead of BTCUSDT).

    Args:
        symbol: Symbol in any supported format

    Returns:
        Symbol in TradingView format (e.g., "BTCUSD")

    Raises:
        ValueError: If symbol is not recognized or is invalid

    Example:
        >>> normalize_to_tradingview("BTCUSDT")
        'BTCUSD'
        >>> normalize_to_tradingview("BTCUSD")
        'BTCUSD'
    """
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty or whitespace")

    symbol_upper = symbol.strip().upper()

    # Direct lookup in mappings
    if symbol_upper in SYMBOL_MAPPINGS:
        return SYMBOL_MAPPINGS[symbol_upper]["tradingview"]

    # If ends with USDT, convert to TV format
    if symbol_upper.endswith("USDT"):
        base = symbol_upper[:-4]  # Remove USDT
        tv_symbol = f"{base}USD"
        logger.warning(
            "symbol_converted_ccxt_to_tv",
            symbol=symbol_upper,
            tv_symbol=tv_symbol,
            warning="BTCUSDT and BTCUSD are different instruments - confirm this is intentional",
        )
        return tv_symbol

    # If already in USD format, return as-is
    if symbol_upper.endswith("USD"):
        return symbol_upper

    safe_symbol = _sanitize_symbol(symbol)
    raise ValueError(
        f"Unrecognized symbol format: '{safe_symbol}'. "
        f"Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
    )


def get_base_currency(symbol: str) -> str:
    """Extract the base currency from a symbol.

    Args:
        symbol: Symbol in any supported format

    Returns:
        Base currency (e.g., "BTC" from "BTCUSDT")

    Example:
        >>> get_base_currency("BTCUSDT")
        'BTC'
        >>> get_base_currency("BTCUSD")
        'BTC'
    """
    if not symbol or not symbol.strip():
        return ""

    symbol_upper = symbol.strip().upper()

    if symbol_upper in SYMBOL_MAPPINGS:
        return SYMBOL_MAPPINGS[symbol_upper]["base"]

    # Fallback: strip quote currency
    for quote in ["USDT", "USD", "BTC", "ETH", "BUSD"]:
        if symbol_upper.endswith(quote):
            return symbol_upper[: -len(quote)]

    return symbol_upper


def validate_symbol(symbol: str) -> bool:
    """Validate that a symbol is in a recognized format.

    Args:
        symbol: Symbol to validate

    Returns:
        True if symbol is recognized, False otherwise

    Example:
        >>> validate_symbol("BTCUSDT")
        True
        >>> validate_symbol("BTCUSD")
        True
        >>> validate_symbol("")
        False
        >>> validate_symbol("INVALID")
        False
    """
    if not symbol or not symbol.strip():
        return False

    symbol_upper = symbol.strip().upper()

    # Check direct mapping
    if symbol_upper in SYMBOL_MAPPINGS:
        return True

    # Check inferred formats
    if symbol_upper.endswith("USDT") or symbol_upper.endswith("USD"):
        return True

    return False
