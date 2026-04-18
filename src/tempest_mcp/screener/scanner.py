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


# Default filter preset when filters is omitted
# Provides balanced multi-factor screening out of the box
DEFAULT_FILTER_PRESET: list[ScanFilter] = [
    ScanFilter.RSI_OVERSOLD,
    ScanFilter.TREND_BULLISH,
]


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
class ScanFailure:
    """Represents a per-symbol failure during scanning."""

    symbol: str
    exchange: str
    reason: str  # e.g., "empty_ohlcv", "indicator_error", "fetch_error"


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

    def scan(
        self, symbols: list[str] | None = None
    ) -> tuple[list[ScanResult], list[ScanFailure]]:
        """Execute multi-factor scan across symbols.

        Returns:
            Tuple of (results, failures) where:
            - results: List of ScanResult sorted by (-score, -len(filters_matched), symbol, exchange)
            - failures: List of ScanFailure for symbols that could not be scanned

        Deterministic partial success: If at least one symbol returns usable data,
        returns partial success with results + failures. Full failure (empty results
        with failures) only when nothing usable comes back.
        """
        symbols_to_scan = symbols or list(self.symbols)

        # Use default preset when filters is empty
        effective_filters = self.filters if self.filters else DEFAULT_FILTER_PRESET

        logger.info(
            "Starting scan",
            symbols=len(symbols_to_scan),
            filters=[f.value for f in effective_filters],
        )

        results: list[ScanResult] = []
        failures: list[ScanFailure] = []

        for symbol in symbols_to_scan:
            result, failure = self._scan_symbol(symbol, effective_filters)
            if failure is not None:
                failures.append(failure)
            elif result is not None and result.score >= self.min_score:
                results.append(result)

        # Deterministic sorting: (-score, -len(filters_matched), symbol, exchange)
        results.sort(
            key=lambda r: (-r.score, -len(r.filters_matched), r.symbol, r.exchange)
        )

        logger.info(
            "Scan complete",
            results=len(results),
            failures=len(failures),
        )

        return results, failures

    def _scan_symbol(
        self, symbol: str, filters: list[ScanFilter]
    ) -> tuple[ScanResult | None, ScanFailure | None]:
        """Scan a single symbol against filters.

        Returns:
            Tuple of (result, failure):
            - (result, None) on success
            - (None, failure) on failure
        """
        try:
            df = self.adapter.fetch_ohlcv_live(symbol, timeframe="1h", limit=100)
            if df.empty:
                return None, ScanFailure(
                    symbol=symbol,
                    exchange=self.exchange,
                    reason="empty_ohlcv",
                )
        except Exception as e:
            logger.warning("Fetch failed for symbol", symbol=symbol, error=str(e))
            return None, ScanFailure(
                symbol=symbol,
                exchange=self.exchange,
                reason=f"fetch_error: {e}",
            )

        close = df["close"].tolist()
        volume = df["volume"].tolist()

        indicator_values: dict[str, float] = {}
        filters_matched: list[str] = []

        # ── RSI Evaluation ───────────────────────────────────────────────────
        try:
            rsi_result = calculate_rsi_result(close)
            rsi = rsi_result.values.get("rsi", 50.0)
            indicator_values["rsi"] = rsi
            indicator_values["rsi_oversold"] = float(rsi_result.values.get("oversold", False))
            indicator_values["rsi_overbought"] = float(rsi_result.values.get("overbought", False))

            if ScanFilter.RSI_OVERSOLD in filters and indicator_values["rsi_oversold"]:
                filters_matched.append("rsi_oversold")
            if ScanFilter.RSI_OVERBOUGHT in filters and indicator_values["rsi_overbought"]:
                filters_matched.append("rsi_overbought")
        except Exception as e:
            logger.warning("RSI calculation failed", symbol=symbol, error=str(e))
            indicator_values["rsi"] = 50.0
            indicator_values["rsi_oversold"] = 0.0
            indicator_values["rsi_overbought"] = 0.0

        # ── Trend Evaluation (EMA alignment) ─────────────────────────────────
        try:
            ema_result = calculate_ema_result(close, periods=[7, 25, 50])
            ema_7 = ema_result.values.get("ema_7", close[-1])
            ema_25 = ema_result.values.get("ema_25", close[-1])
            ema_50 = ema_result.values.get("ema_50", close[-1])
            indicator_values["ema_7"] = ema_7
            indicator_values["ema_25"] = ema_25
            indicator_values["ema_50"] = ema_50

            # Bullish: short EMA > long EMA
            is_bullish = ema_7 > ema_25 > ema_50
            # Bearish: short EMA < long EMA
            is_bearish = ema_7 < ema_25 < ema_50

            if ScanFilter.TREND_BULLISH in filters and is_bullish:
                filters_matched.append("trend_bullish")
            if ScanFilter.TREND_BEARISH in filters and is_bearish:
                filters_matched.append("trend_bearish")
        except Exception as e:
            logger.warning("EMA calculation failed", symbol=symbol, error=str(e))
            indicator_values["ema_7"] = close[-1]
            indicator_values["ema_25"] = close[-1]
            indicator_values["ema_50"] = close[-1]

        # ── Volume Evaluation ────────────────────────────────────────────────
        try:
            # Calculate volume spike: current volume vs 20-bar average
            lookback = min(20, len(volume))
            if lookback >= 5:
                avg_volume = sum(volume[-lookback:-1]) / (lookback - 1) if lookback > 1 else volume[-1]
                current_volume = volume[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                indicator_values["volume_ratio"] = volume_ratio
                # Volume spike threshold: 1.5x average
                if ScanFilter.VOLUME_SPIKE in filters and volume_ratio >= 1.5:
                    filters_matched.append("volume_spike")
            else:
                indicator_values["volume_ratio"] = 1.0
        except Exception as e:
            logger.warning("Volume calculation failed", symbol=symbol, error=str(e))
            indicator_values["volume_ratio"] = 1.0

        # ── Momentum Evaluation ─────────────────────────────────────────────
        try:
            # Simple momentum: % change over lookback period
            lookback = min(14, len(close) - 1)
            if lookback > 0:
                momentum = ((close[-1] - close[-lookback - 1]) / close[-lookback - 1]) * 100
                indicator_values["momentum_pct"] = momentum
                # Positive momentum = bullish, negative = bearish
                # This is a simplified momentum check; more complex could use MACD histogram
                if momentum > 0 and ScanFilter.TREND_BULLISH in filters:
                    filters_matched.append("trend_bullish")
                elif momentum < 0 and ScanFilter.TREND_BEARISH in filters:
                    filters_matched.append("trend_bearish")
            else:
                indicator_values["momentum_pct"] = 0.0
        except Exception as e:
            logger.warning("Momentum calculation failed", symbol=symbol, error=str(e))
            indicator_values["momentum_pct"] = 0.0

        # ── Score Calculation ────────────────────────────────────────────────
        score = self._calculate_score(filters, filters_matched, indicator_values)

        latest_ts = df.index[-1]
        result = ScanResult(
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

        return result, None

    def _calculate_score(
        self,
        filters: list[ScanFilter],
        filters_matched: list[str],
        indicator_values: dict[str, float],
    ) -> float:
        """Calculate deterministic score based on filters matched.

        When no filters specified (empty list), uses default scoring:
        - Base score 50
        - RSI oversold adds +20
        - RSI overbought subtracts -20
        - Trend bullish adds +15
        - Volume spike adds +10

        When filters are specified, score = (matched / total) * 100
        """
        if not filters:
            # Default scoring when no filters specified
            score = 50.0
            rsi = indicator_values.get("rsi", 50)
            if rsi < 30:
                score += 20
            elif rsi > 70:
                score -= 20

            # Trend contribution
            ema_7 = indicator_values.get("ema_7", 0)
            ema_25 = indicator_values.get("ema_25", 0)
            ema_50 = indicator_values.get("ema_50", 0)
            if ema_7 > ema_25 > ema_50:
                score += 15
            elif ema_7 < ema_25 < ema_50:
                score -= 10

            # Volume spike contribution
            volume_ratio = indicator_values.get("volume_ratio", 1.0)
            if volume_ratio >= 1.5:
                score += 10

            return min(100, max(0, score))
        else:
            # Filter-based scoring
            # Map filter enums to their string values for matching
            filter_str_map = {
                ScanFilter.RSI_OVERSOLD: "rsi_oversold",
                ScanFilter.RSI_OVERBOUGHT: "rsi_overbought",
                ScanFilter.TREND_BULLISH: "trend_bullish",
                ScanFilter.TREND_BEARISH: "trend_bearish",
                ScanFilter.HIGH_VOLATILITY: "high_volatility",
                ScanFilter.LOW_VOLATILITY: "low_volatility",
                ScanFilter.VOLUME_SPIKE: "volume_spike",
            }

            # Count how many specified filters were matched
            matched_count = sum(
                1 for f in filters if filter_str_map.get(f) in filters_matched
            )
            match_ratio = matched_count / len(filters) if filters else 0
            return min(100, match_ratio * 100)

    def session_breakout_scan(
        self, session: SessionType, symbols: list[str] | None = None
    ) -> list[ScanResult]:
        from tempest_mcp.indicators.session_levels import detect_session_levels

        symbols_to_scan = symbols or list(self.symbols)
        results = []
        for symbol in symbols_to_scan:
            try:
                df = self.adapter.fetch_ohlcv_live(symbol, timeframe="1h", limit=48)
                if df.empty:
                    continue
                close = df["close"].tolist()
                session_key = session.value
                session_result = detect_session_levels(df, session_key)
                current_price = close[-1]
                filters_matched = []
                session_high = float(session_result.get("high", 0.0) or 0.0)
                session_low = float(session_result.get("low", 0.0) or 0.0)
                if session_high > 0 and current_price > session_high:
                    filters_matched.append(f"{session_key}_high_breakout")
                if session_low > 0 and current_price < session_low:
                    filters_matched.append(f"{session_key}_low_breakout")
                score = 80.0 if filters_matched else 0.0
                latest_ts = df.index[-1]
                results.append(
                    ScanResult(
                        symbol=symbol,
                        exchange=self.exchange,
                        timestamp=latest_ts.timestamp()
                        if isinstance(latest_ts, pd.Timestamp)
                        else float(latest_ts),
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
