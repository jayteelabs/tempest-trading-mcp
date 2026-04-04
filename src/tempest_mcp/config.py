"""Configuration management for Tempest MCP Server.

Environment variables with TEMPEST_ prefix, SCREAMING_SNAKE_CASE.
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Final

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    log_level: str = "INFO"
    log_format: str = "json"
    yf_cache_ttl: int = 300
    yf_timeout: int = 30
    yf_retries: int = 3
    ccxt_timeout: int = 30
    default_exchange: str = "binance"
    mcp_server_name: str = "tempest-tradingview-mcp"
    mcp_server_version: str = "0.1.0"
    default_commission: float = 0.001
    default_slippage: float = 0.0005
    screener_symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "DOGE/USDT")
    screener_interval: int = 60


def _get_str(key: str, default: str) -> str:
    return os.getenv(f"TEMPEST_{key}", default)


def _get_int(key: str, default: int) -> int:
    value = os.getenv(f"TEMPEST_{key}")
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "config: invalid integer for %s, using default %s",
            f"TEMPEST_{key}",
            default,
        )
        return default


def _get_float(key: str, default: float) -> float:
    value = os.getenv(f"TEMPEST_{key}")
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning(
            "config: invalid float for %s, using default %s",
            f"TEMPEST_{key}",
            default,
        )
        return default


def _get_tuple(key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(f"TEMPEST_{key}")
    if value is None:
        return default
    return tuple(s.strip() for s in value.split(",") if s.strip())


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config(
        log_level=_get_str("LOG_LEVEL", "INFO"),
        log_format=_get_str("LOG_FORMAT", "json"),
        yf_cache_ttl=_get_int("YF_CACHE_TTL", 300),
        yf_timeout=_get_int("YF_TIMEOUT", 30),
        yf_retries=_get_int("YF_RETRIES", 3),
        ccxt_timeout=_get_int("CCXT_TIMEOUT", 30),
        default_exchange=_get_str("DEFAULT_EXCHANGE", "binance"),
        mcp_server_name=_get_str("MCP_SERVER_NAME", "tempest-tradingview-mcp"),
        mcp_server_version=_get_str("MCP_SERVER_VERSION", "0.1.0"),
        default_commission=_get_float("DEFAULT_COMMISSION", 0.001),
        default_slippage=_get_float("DEFAULT_SLIPPAGE", 0.0005),
        screener_symbols=_get_tuple("SCREENER_SYMBOLS", ("BTC/USDT", "ETH/USDT", "DOGE/USDT")),
        screener_interval=_get_int("SCREENER_INTERVAL", 60),
    )


class ErrorCodes:
    VALIDATION_ERROR: Final[int] = 1000
    INVALID_SYMBOL: Final[int] = 1001
    INVALID_TIMEFRAME: Final[int] = 1002
    INVALID_EXCHANGE: Final[int] = 1003
    INVALID_PARAMETER: Final[int] = 1004
    MISSING_PARAMETER: Final[int] = 1005
    DATA_SOURCE_ERROR: Final[int] = 3000
    YFINANCE_ERROR: Final[int] = 3001
    CCXT_ERROR: Final[int] = 3002
    DATA_NOT_FOUND: Final[int] = 3003
    RATE_LIMIT_ERROR: Final[int] = 3004
    NETWORK_ERROR: Final[int] = 3005
    INDICATOR_ERROR: Final[int] = 5000
    INSUFFICIENT_DATA: Final[int] = 5001
    CALCULATION_ERROR: Final[int] = 5002
    INTERNAL_ERROR: Final[int] = 9000
    UNEXPECTED_ERROR: Final[int] = 9001