"""Indicator result models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

Value: TypeAlias = float
Timestamp: TypeAlias = float


class SessionType(Enum):
    ASIA = "asia"
    LONDON = "london"
    NEW_YORK = "new_york"


@dataclass(frozen=True)
class IndicatorResult:
    symbol: str
    timeframe: str
    timestamp: Timestamp
    values: dict[str, Value] = field(default_factory=dict)


@dataclass(frozen=True)
class EMAResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class VWAPResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class RSIResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class MACDResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class ATRResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class SupertrendResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class SessionLevels(IndicatorResult):
    pass


@dataclass(frozen=True)
class ADXResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class StochasticResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class CCIResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class WilliamsRResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class ROCResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class BollingerWidthResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class OBVResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class MFIResult(IndicatorResult):
    pass


@dataclass(frozen=True)
class HistoricalVolatilityResult(IndicatorResult):
    pass
