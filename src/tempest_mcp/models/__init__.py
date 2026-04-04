"""
Data models for Tempest MCP.

Using dataclasses over pydantic for lower dependencies (D6).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Ticker:
    """Real-time ticker data."""

    symbol: str
    price: float
    bid: float
    ask: float
    volume_24h: float
    timestamp: pd.Timestamp

    def __post_init__(self) -> None:
        """Ensure timestamp is UTC-aware."""
        if self.timestamp.tz is None:
            self.timestamp = self.timestamp.tz_localize("UTC")


@dataclass
class Kline:
    """OHLCV candlestick data point."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float

    def __post_init__(self) -> None:
        """Ensure timestamp is UTC-aware."""
        if self.timestamp.tz is None:
            self.timestamp = self.timestamp.tz_localize("UTC")


@dataclass
class OrderBookLevel:
    """Single order book level."""

    price: float
    amount: float


@dataclass
class OrderBook:
    """Order book snapshot."""

    symbol: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: pd.Timestamp | None

    def __post_init__(self) -> None:
        """Ensure timestamp is UTC-aware if present."""
        if self.timestamp is not None and self.timestamp.tz is None:
            self.timestamp = self.timestamp.tz_localize("UTC")


__all__ = [
    "Ticker",
    "Kline",
    "OrderBook",
    "OrderBookLevel",
]
