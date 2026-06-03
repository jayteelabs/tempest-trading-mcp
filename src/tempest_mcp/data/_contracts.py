"""Shared data-layer contracts and normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd

OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]
SUPPORTED_EXCHANGES: Final[tuple[str, ...]] = ("binance", "bybit", "coinbase", "kraken")
SUPPORTED_TIMEFRAMES: Final[frozenset[str]] = frozenset(
    {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"}
)
TIMEFRAME_SECONDS: Final[dict[str, int]] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1wk": 604800,
    "1mo": 2592000,
}
CCXT_TIMEFRAME_MAP: Final[dict[str, str]] = {
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
YF_INTERVAL_MAP: Final[dict[str, str]] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1wk": "1wk",
    "1mo": "1mo",
}
MIN_LIMIT: Final[int] = 1
MAX_LIMIT: Final[int] = 1000


def empty_ohlcv_frame() -> pd.DataFrame:
    """Return an empty OHLCV DataFrame with the canonical column order."""
    return pd.DataFrame(columns=OHLCV_COLUMNS)


def empty_orderbook_snapshot() -> dict:
    """Return the canonical empty order book payload."""
    return {
        "bids": [],
        "asks": [],
        "timestamp": None,
    }


def ohlcv_frame_from_records(
    records: Iterable,
    *,
    timestamp_column: str = "timestamp",
    timestamp_unit: str = "ms",
) -> pd.DataFrame:
    """Build a canonical OHLCV DataFrame from timestamped records."""
    df = pd.DataFrame(
        records,
        columns=[timestamp_column, *OHLCV_COLUMNS],
    )
    if df.empty:
        return empty_ohlcv_frame()

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], unit=timestamp_unit, utc=True)
    df.set_index(timestamp_column, inplace=True)
    return df[OHLCV_COLUMNS]


def canonicalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a provider DataFrame to the canonical OHLCV schema."""
    if df.empty:
        return empty_ohlcv_frame()

    result = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    result.columns = [str(col).lower().replace(" ", "_") for col in result.columns]

    for column in OHLCV_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0

    result = result[OHLCV_COLUMNS]

    if isinstance(result.index, pd.DatetimeIndex):
        if result.index.tz is None:
            result.index = result.index.tz_localize("UTC")
        else:
            result.index = result.index.tz_convert("UTC")

    return result
