"""Multi-factor crypto screener."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from tempest_mcp.config import get_config
from tempest_mcp.data.ccxt_adapter import CCXTAdapter
from tempest_mcp.indicators.momentum import calculate_rsi_result
from tempest_mcp.indicators.trend import calculate_ema_result
from tempest_mcp.logging_config import get_logger
from tempest_mcp.models.indicator import SessionType

logger = get_logger(__name__)


class ScanFilter(Enum):
    RSI_OVERSOLD = "rsi_oversold"
    RSI_OVERBOUGHT = "rsi_overbought"
    TREND_BULLISH = "trend_bullish"
    TREND_BEARISH = "trend_bearish"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    VOLUME_SPIKE = "volume_spike"


@dataclass
class ScanResult:
    symbol: str
    exchange: str
    timestamp: float
    price: float
    filters_matched: list[str]
    indicator_values: dict[str, float]
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Screener:
    symbols: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "DOGE/USDT")
    exchange: str = "binance"
    filters: list[ScanFilter] = field(default_factory=list)
    min_score: float = 0.0
    _adapter: CCXTAdapter | None = field(default=None, init=False)

    def __post_init__(self):
        config = get_config()
        if self.symbols == ("BTC/USDT", "ETH/USDT", "DOGE/USDT"):
            self.symbols = config.screener_symbols
        if self.exchange == "binance":
            self.exchange = config.default_exchange

    @property
    def adapter(self) -> CCXTAdapter:
        if self._adapter is None:
            self._adapter = CCXTAdapter(exchange_name=self.exchange)
        return self._adapter

    def scan(self, symbols: list[str] | None = None) -> list[ScanResult]:
        symbols_to_scan = symbols or list(self.symbols)
        results = []
        logger.info(
            "Starting scan", symbols=len(symbols_to_scan), filters=[f.value for f in self.filters]
        )
        for symbol in symbols_to_scan:
            try:
                result = self._scan_symbol(symbol)
                if result and result.score >= self.min_score:
                    results.append(result)
            except Exception as e:
                logger.warning("Scan failed for symbol", symbol=symbol, error=str(e))
        results.sort(key=lambda r: r.score, reverse=True)
        logger.info("Scan complete", results=len(results))
        return results

    def _scan_symbol(self, symbol: str) -> ScanResult | None:
        df = self.adapter.fetch_ohlcv_live(symbol, timeframe="1h", limit=100)
        if df.empty:
            return None
        close = df["close"].tolist()
        indicator_values = {}
        filters_matched = []
        try:
            rsi_result = calculate_rsi_result(close)
            indicator_values["rsi"] = rsi_result.values["rsi"]
            if ScanFilter.RSI_OVERSOLD in self.filters and rsi_result.values.get("oversold"):
                filters_matched.append("rsi_oversold")
        except Exception:
            indicator_values["rsi"] = 50.0
        try:
            ema_result = calculate_ema_result(close, periods=[20, 50])
            indicator_values["ema_20"] = ema_result.values.get("ema_20", close[-1])
            indicator_values["ema_50"] = ema_result.values.get("ema_50", close[-1])
        except Exception:
            indicator_values["ema_20"] = close[-1]
            indicator_values["ema_50"] = close[-1]
        score = self._calculate_score(filters_matched, indicator_values)
        latest_ts = df.index[-1]
        return ScanResult(
            symbol=symbol,
            exchange=self.exchange,
            timestamp=latest_ts.timestamp()
            if isinstance(latest_ts, pd.Timestamp)
            else float(latest_ts),
            price=close[-1],
            filters_matched=filters_matched,
            indicator_values=indicator_values,
            score=score,
        )

    def _calculate_score(self, filters_matched, indicator_values):
        if not self.filters:
            score = 50.0
            rsi = indicator_values.get("rsi", 50)
            if rsi < 30:
                score += 20
            elif rsi > 70:
                score -= 20
            return min(100, max(0, score))
        match_ratio = len(filters_matched) / len(self.filters)
        return min(100, match_ratio * 100)

    def session_breakout_scan(
        self, session: SessionType, symbols: list[str] | None = None
    ) -> list[ScanResult]:
        from tempest_mcp.indicators.session_levels import calculate_session_levels

        symbols_to_scan = symbols or list(self.symbols)
        results = []
        for symbol in symbols_to_scan:
            try:
                df = self.adapter.fetch_ohlcv_live(symbol, timeframe="1h", limit=48)
                if df.empty:
                    continue
                timestamps = [
                    ts.timestamp() if isinstance(ts, pd.Timestamp) else float(ts) for ts in df.index
                ]
                high = df["high"].tolist()
                low = df["low"].tolist()
                close = df["close"].tolist()
                session_result = calculate_session_levels(timestamps, high, low)
                values = session_result.values
                current_price = close[-1]
                filters_matched = []
                session_key = session.value
                session_high = values.get(f"{session_key}_high", 0)
                session_low = values.get(f"{session_key}_low", 0)
                if session_high > 0 and current_price > session_high:
                    filters_matched.append(f"{session_key}_high_breakout")
                if session_low > 0 and current_price < session_low:
                    filters_matched.append(f"{session_key}_low_breakout")
                score = 80.0 if filters_matched else 0.0
                results.append(
                    ScanResult(
                        symbol=symbol,
                        exchange=self.exchange,
                        timestamp=timestamps[-1],
                        price=current_price,
                        filters_matched=filters_matched,
                        indicator_values={
                            "session_high": session_high,
                            "session_low": session_low,
                            "current_price": current_price,
                        },
                        score=score,
                    )
                )
            except Exception as e:
                logger.warning("Session scan failed", symbol=symbol, error=str(e))
        results.sort(key=lambda r: r.score, reverse=True)
        return results
