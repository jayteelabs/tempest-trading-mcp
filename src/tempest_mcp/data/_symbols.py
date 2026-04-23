"""Symbol normalization utilities for the data layer.

This module defines the active canonical symbol handling used by the data
adapters:
- CCXT-style symbols are the internal canonical format for live/exchange data
- yfinance symbols are derived from the canonical format for historical fallback
- TradingView-era conversions are retained only for backward compatibility and
  migration of legacy callers; they are not the preferred path for new code

Design decisions (D11, D12):
- D11: Symbol conversion is centralized in _symbols.py
- D12: Legacy callers may still provide TradingView-style BTCUSD while the
  active exchange-backed path uses CCXT-style BTCUSDT

WARNING: BTCUSD and BTCUSDT are different instruments on some venues. Any
conversion between them should be treated as an explicit compatibility choice,
not as proof of instrument equivalence.
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

CCXT_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1w",
    "1mo": "1M",
}

YF_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}

# Limit bounds
MIN_LIMIT: int = 1
MAX_LIMIT: int = 1000


class SymbolMapping(TypedDict):
    """Mapping between legacy input aliases and canonical data-layer formats."""

    tradingview: str
    ccxt: str
    base: str
    quote: str


# Canonical symbol mappings for v1 crypto scope.
# CCXT-style symbols are the active canonical representation; TradingView-style
# symbols remain as compatibility aliases only.
# WARNING: BTCUSD ≠ BTCUSDT — they are different instruments.
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
    """Strip non-alphanumeric characters from symbol for safe logging.

    Uses strict allowlist (alphanumeric only) to prevent log injection
    and ensure well-formed symbol output.
    """
    return "".join(c for c in symbol if c.isalnum())


def _validate_timeframe(timeframe: str) -> bool:
    """Check if timeframe is supported."""
    return timeframe in SUPPORTED_TIMEFRAMES


def _validate_limit(limit: int) -> int:
    """Clamp limit to valid range."""
    return max(MIN_LIMIT, min(MAX_LIMIT, limit))


def normalize_to_ccxt_timeframe(interval: str) -> str:
    """Convert a generic interval string to a CCXT timeframe string."""
    return CCXT_TIMEFRAME_MAP.get(interval, "1d")


def normalize_to_yf_interval(interval: str) -> str | None:
    """Convert a generic interval string to a yfinance interval string."""
    return YF_INTERVAL_MAP.get(interval)


def normalize_to_ccxt(symbol: str, exchange: ExchangeName = "binance") -> str:
    """Convert a symbol to the canonical CCXT-style format.

    CCXT format is the active internal representation for exchange-backed data.
    This function accepts canonical CCXT-style symbols and a small set of
    legacy aliases, then returns the normalized CCXT-style symbol suitable for
    adapter calls.

    Args:
        symbol: Symbol in any supported format (e.g., "BTCUSD", "BTCUSDT")
        exchange: Target exchange (default: "binance"). Future extension point
            for exchange-specific symbol formats.

    Returns:
        Symbol in CCXT format (e.g., "BTCUSDT" or "BTC/USDT")

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

    # If already slash-separated, assume CCXT market format
    if "/" in symbol_upper:
        base, _, quote = symbol_upper.partition("/")
        if not base.isalnum() or not quote.isalnum():
            raise ValueError(
                f"Invalid symbol format: '{symbol}'. Expected alphanumeric base/quote."
            )
        return symbol_upper

    # Handle single hyphen-delimited symbols FIRST (e.g., BTC-USDT → BTC/USDT).
    # This must come before endswith("USDT") check so BTC-USDT is normalized
    # to BTC/USDT before that check can return it unchanged.
    # Malformed cases (multiple hyphens, backslash, colon) are still rejected.
    if "-" in symbol_upper:
        if symbol_upper.count("-") == 1:
            base, _, quote = symbol_upper.partition("-")
            if base.isalnum() and quote.isalnum():
                # Valid single-hyphen pattern - convert to slash format
                return f"{base}/{quote}"
        # Malformed hyphen pattern (multiple hyphens, or invalid base/quote)
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
        )

    # If ends with USDT, already in CCXT format — but validate base is alphanumeric
    if symbol_upper.endswith("USDT"):
        base = symbol_upper[:-4]
        if base.isalnum():
            return symbol_upper
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
        )

    if symbol_upper.endswith("USDC"):
        base = symbol_upper[:-4]
        if base.isalnum():
            return symbol_upper
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
        )

    # If ends with USD but not USDT, treat it as a legacy alias and normalize
    # into the canonical CCXT-style symbol. This is compatibility behavior,
    # not a statement that USD and USDT instruments are identical.
    if symbol_upper.endswith("USD") and not symbol_upper.endswith("USDT"):
        # Reject hyphenated USD symbols to avoid rewriting BTC-USD into BTCUSDT.
        if "-" in symbol_upper:
            raise ValueError(
                f"Invalid symbol format: '{symbol}'. Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
            )
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


def normalize_to_ccxt_market(symbol: str, exchange: ExchangeName = "binance") -> str:
    """Convert a symbol to the CCXT market pair format used by exchange APIs."""
    symbol_upper = symbol.strip().upper()
    if "/" in symbol_upper:
        # Already in market format - validate base/quote are alphanumeric
        base, _, quote = symbol_upper.partition("/")
        if base.isalnum() and quote.isalnum():
            return symbol_upper
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. Expected alphanumeric base/quote."
        )
    ccxt_symbol = normalize_to_ccxt(symbol, exchange=exchange)

    # If normalize_to_ccxt returned a market-formatted symbol (contains slash),
    # validate and return it directly without further suffix stripping.
    # This prevents double-slash issues when normalize_to_ccxt converts
    # hyphenated symbols like BTC-USDT to BTC/USDT.
    if "/" in ccxt_symbol:
        base, _, quote = ccxt_symbol.partition("/")
        if base.isalnum() and quote.isalnum():
            return ccxt_symbol
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. Expected alphanumeric base/quote."
        )

    for quote in ("USDT", "USDC", "USD", "BTC", "ETH", "BUSD"):
        if ccxt_symbol.endswith(quote):
            base = ccxt_symbol[: -len(quote)]
            if base:
                return f"{base}/{quote}"

    raise ValueError(f"Unable to derive CCXT market symbol from '{symbol}'")


def normalize_to_ccxt_exchange(symbol: str, exchange: ExchangeName = "binance") -> str:
    """Backward-compatible alias for normalize_to_ccxt_market."""
    return normalize_to_ccxt_market(symbol, exchange=exchange)


def normalize_to_tradingview(symbol: str) -> str:
    """Convert a symbol to a legacy TradingView-style alias.

    This helper is retained for backward compatibility only. New data-layer
    code should prefer canonical CCXT-style symbols internally.

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


def normalize_to_yf(symbol: str) -> str:
    """Convert a canonical/live symbol into yfinance fallback format.

    yfinance uses BTC-USD style symbols while the active internal format is
    CCXT-style (for example, BTCUSDT). This helper translates canonical or
    legacy-compatible inputs into the historical fallback format.

    Args:
        symbol: Symbol in CCXT format (e.g., "BTCUSDT", "ETHUSDT")

    Returns:
        Symbol in yfinance format (e.g., "BTC-USD", "ETH-USD")

    Raises:
        ValueError: If symbol is not recognized or is invalid

    Example:
        >>> normalize_to_yf("BTCUSDT")
        'BTC-USD'
        >>> normalize_to_yf("ETHUSDT")
        'ETH-USD'
    """
    # Normalize first, then check once
    symbol_normalized = symbol.strip().upper()
    if not symbol_normalized:
        raise ValueError("Symbol cannot be empty or whitespace")

    # Direct lookup in mappings for known pairs/aliases.
    if symbol_normalized in SYMBOL_MAPPINGS:
        mapping = SYMBOL_MAPPINGS[symbol_normalized]
        # Use the legacy USD alias as an intermediate, then convert hyphen.
        tv_symbol = mapping["tradingview"]
        # tradingview is BTCUSD, yfinance is BTC-USD
        # Insert hyphen before USD
        if tv_symbol.endswith("USD"):
            base = tv_symbol[:-3]
            if not base:
                raise ValueError(f"Invalid symbol: cannot determine base currency for '{symbol}'")
            return f"{base}-USD"
        return tv_symbol
    if "/" in symbol_normalized:
        base, _, quote = symbol_normalized.partition("/")
        if not base.isalnum() or not quote.isalnum():
            raise ValueError(
                f"Invalid symbol format: '{symbol}'. Expected alphanumeric base/quote."
            )
        if quote in ("USDT", "USD", "USDC", "US"):
            return f"{base}-USD"
        logger.warning(
            "normalize_to_yf_non_usd_quote",
            symbol=symbol_normalized,
            quote=quote,
        )
        return f"{base}-{quote}"

    # If ends with USDT, convert: BTCUSDT → BTC-USD
    if symbol_normalized.endswith("USDT"):
        base = symbol_normalized[:-4]  # Remove USDT
        if not base:
            raise ValueError(f"Invalid symbol: empty base currency in '{symbol}'")
        # Validate base is alphanumeric-only (no hyphens, slashes, etc.)
        if not base.isalnum():
            raise ValueError(
                f"Invalid symbol format: '{symbol}' contains non-alphanumeric base '{base}'. "
                f"Expected clean base currency (e.g., BTC from BTCUSDT)."
            )
        return f"{base}-USD"

    # If ends with USD but not USDT, assume already in some yfinance-like format
    if symbol_normalized.endswith("USD"):
        # Could be BTCUSD (TV) or BTC-USD (yfinance)
        # If no hyphen, insert one
        if "-" not in symbol_normalized:
            base = symbol_normalized[:-3]
            if not base:
                raise ValueError(f"Invalid symbol: empty base currency in '{symbol}'")
            if not base.isalnum():
                raise ValueError(
                    f"Invalid symbol format: '{symbol}' contains non-alphanumeric base '{base}'."
                )
            return f"{base}-USD"
        # Already hyphenated - validate it's well-formed (e.g., BTC-USD, not BTC--USD)
        if symbol_normalized.count("-") != 1:
            raise ValueError(f"Invalid yfinance format: '{symbol}'. Expected BTC-USD or ETH-USD.")
        base, _, quote = symbol_normalized.partition("-")
        if not base.isalnum() or quote != "USD":
            raise ValueError(f"Invalid yfinance format: '{symbol}'. Expected BTC-USD or ETH-USD.")
        return symbol_normalized

    # SECURITY NOTE (for Haga): The fallback below accepts arbitrary strings
    # but requires base currency to be non-empty after sanitization.
    safe_symbol = _sanitize_symbol(symbol_normalized)
    if not safe_symbol:
        raise ValueError(
            f"Unrecognized symbol format: '{symbol}'. Expected TradingView (BTCUSD) or CCXT (BTCUSDT) format."
        )
    logger.warning(
        "normalize_to_yf_unrecognized_symbol",
        symbol=safe_symbol,
        fallback=f"{safe_symbol}-USD",
    )
    return f"{safe_symbol}-USD"


def normalize_to_yf_fallback(symbol: str) -> str:
    """Convert a symbol into the yfinance fallback format.

    Crypto-style pairs are normalized into yfinance-compatible symbols, while
    plain symbols such as stock tickers pass through unchanged.
    """
    symbol_normalized = symbol.strip().upper()
    if not symbol_normalized:
        raise ValueError("Symbol cannot be empty or whitespace")

    if "/" in symbol_normalized:
        base, _, quote = symbol_normalized.partition("/")
        if not base or not quote:
            raise ValueError(f"Invalid symbol format: '{symbol}'.")
        if quote not in {"USD", "USDT", "USDC", "US"}:
            raise ValueError(f"Unsupported quote currency for yfinance fallback: '{quote}'.")
        return normalize_to_yf(symbol_normalized)

    if symbol_normalized.endswith(("USDT", "USD", "USDC")):
        return normalize_to_yf(symbol_normalized)
    return symbol_normalized


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

    if symbol_upper in SYMBOL_MAPPINGS:
        return True

    if "/" in symbol_upper:
        base, _, quote = symbol_upper.partition("/")
        return (
            bool(base)
            and bool(quote)
            and base.isalnum()
            and quote.isalnum()
            and quote in {"USDT", "USD", "USDC", "US"}
        )

    if "-" in symbol_upper:
        if symbol_upper.count("-") != 1:
            return False
        base, _, quote = symbol_upper.partition("-")
        return (
            bool(base)
            and bool(quote)
            and base.isalnum()
            and quote.isalnum()
            and quote in {"USDT", "USD", "USDC", "US"}
        )

    if symbol_upper.endswith(("USDT", "USDC")):
        base = symbol_upper[:-4]
        return bool(base) and base.isalnum()

    if symbol_upper.endswith("USD") and not symbol_upper.endswith("USDT"):
        base = symbol_upper[:-3]
        return bool(base) and base.isalnum()

    return False
