"""Market data models for Ticker, Klines, and OrderBook."""

from dataclasses import dataclass
from typing import TypeAlias

Price: TypeAlias = float
Volume: TypeAlias = float
Timestamp: TypeAlias = float


@dataclass(frozen=True)
class Ticker:
    symbol: str
    exchange: str
    price: Price
    timestamp: Timestamp
    volume_24h: Volume | None = None
    high_24h: Price | None = None
    low_24h: Price | None = None
    change_percent_24h: float | None = None
    bid: Price | None = None
    ask: Price | None = None


@dataclass(frozen=True)
class Kline:
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Volume


@dataclass(frozen=True)
class KlineData:
    symbol: str
    timeframe: str
    exchange: str
    klines: list[Kline]
    start_time: Timestamp | None = None
    end_time: Timestamp | None = None

    def __post_init__(self) -> None:
        if self.klines:
            object.__setattr__(self, "start_time", self.klines[0].timestamp)
            object.__setattr__(self, "end_time", self.klines[-1].timestamp)


@dataclass(frozen=True)
class OrderBookLevel:
    price: Price
    volume: Volume


@dataclass(frozen=True)
class OrderBook:
    symbol: str
    exchange: str
    bids: list[OrderBookLevel]
    asks: list[OrderBookLevel]
    timestamp: Timestamp


def klines_to_dataframe(klines: list[Kline]) -> dict:
    return {
        "timestamp": [k.timestamp for k in klines],
        "open": [k.open for k in klines],
        "high": [k.high for k in klines],
        "low": [k.low for k in klines],
        "close": [k.close for k in klines],
        "volume": [k.volume for k in klines],
    }


def dataframe_to_klines(df_dict: dict) -> list[Kline]:
    return [
        Kline(
            timestamp=df_dict["timestamp"][i],
            open=df_dict["open"][i],
            high=df_dict["high"][i],
            low=df_dict["low"][i],
            close=df_dict["close"][i],
            volume=df_dict["volume"][i],
        )
        for i in range(len(df_dict["timestamp"]))
    ]
