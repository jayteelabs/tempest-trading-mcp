"""Shared data-layer contracts and normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

import pandas as pd

OHLCV_COLUMNS: Final[list[str]] = ["open", "high", "low", "close", "volume"]


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
