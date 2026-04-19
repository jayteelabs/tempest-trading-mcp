"""Multi-factor crypto screener."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from tempest_mcp.config import get_config
from tempest_mcp.data.ccxt_adapter import CCXTAdapter
from tempest_mcp.indicators.momentum import calculate_rsi_result
from tempest_mcp.indicators.trend import calculate_ema_result
from tempest_mcp.indicators.volatility import calculate_bollinger_width
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

HIGH_VOLATILITY_BOLLINGER_WIDTH_THRESHOLD = 0.08
LOW_VOLATILITY_BOLLINGER_WIDTH_THRESHOLD = 0.03


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
        if self.filters:
            self.filters = list(dict.fromkeys(self.filters))

    @property
    def adapter(self) -> CCXTAdapter:
        if self._adapter is None:
            self._adapter = CCXTAdapter(exchange_name=self.exchange)
        return self._adapter

    def scan(self, symbols: list[str] | None = None) -> tuple[list[ScanResult], list[ScanFailure]]:
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
        results.sort(key=lambda r: (-r.score, -len(r.filters_matched), r.symbol, r.exchange))

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
                reason="fetch_error",
            )

        close = df["close"].tolist()
        volume = df["volume"].tolist()

        indicator_values: dict[str, float] = {}
        filters_matched: list[str] = []

        def append_match(filter_name: str) -> None:
            if filter_name not in filters_matched:  # noqa: B023
                filters_matched.append(filter_name)  # noqa: B023

        # ── RSI Evaluation ───────────────────────────────────────────────────
        try:
            rsi_result = calculate_rsi_result(close)
            rsi = rsi_result.values.get("rsi", 50.0)
            indicator_values["rsi"] = rsi
            indicator_values["rsi_oversold"] = float(rsi_result.values.get("oversold", False))
            indicator_values["rsi_overbought"] = float(rsi_result.values.get("overbought", False))

            if ScanFilter.RSI_OVERSOLD in filters and indicator_values["rsi_oversold"]:
                append_match("rsi_oversold")
            if ScanFilter.RSI_OVERBOUGHT in filters and indicator_values["rsi_overbought"]:
                append_match("rsi_overbought")
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
                append_match("trend_bullish")
            if ScanFilter.TREND_BEARISH in filters and is_bearish:
                append_match("trend_bearish")
        except Exception as e:
            logger.warning("EMA calculation failed", symbol=symbol, error=str(e))
            indicator_values["ema_7"] = close[-1]
            indicator_values["ema_25"] = close[-1]
            indicator_values["ema_50"] = close[-1]

        # ── Volatility Evaluation ────────────────────────────────────────────
        try:
            bollinger_width = calculate_bollinger_width(pd.Series(close, index=df.index))
            latest_width = float(bollinger_width.iloc[-1]) if not bollinger_width.empty else 0.0
            indicator_values["bollinger_width"] = latest_width

            if (
                ScanFilter.HIGH_VOLATILITY in filters
                and latest_width >= HIGH_VOLATILITY_BOLLINGER_WIDTH_THRESHOLD
            ):
                append_match("high_volatility")
            if (
                ScanFilter.LOW_VOLATILITY in filters
                and latest_width <= LOW_VOLATILITY_BOLLINGER_WIDTH_THRESHOLD
            ):
                append_match("low_volatility")
        except Exception as e:
            logger.warning("Volatility calculation failed", symbol=symbol, error=str(e))
            indicator_values["bollinger_width"] = 0.0

        # ── Volume Evaluation ────────────────────────────────────────────────
        try:
            # Calculate volume spike: current volume vs 20-bar average
            lookback = min(20, len(volume))
            if lookback >= 5:
                avg_volume = (
                    sum(volume[-lookback:-1]) / (lookback - 1) if lookback > 1 else volume[-1]
                )
                current_volume = volume[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                indicator_values["volume_ratio"] = volume_ratio
                # Volume spike threshold: 1.5x average
                if ScanFilter.VOLUME_SPIKE in filters and volume_ratio >= 1.5:
                    append_match("volume_spike")
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
            else:
                indicator_values["momentum_pct"] = 0.0
        except Exception as e:
            logger.warning("Momentum calculation failed", symbol=symbol, error=str(e))
            indicator_values["momentum_pct"] = 0.0

        # ── Score Calculation ────────────────────────────────────────────────
        score = self._calculate_score(filters, filters_matched, indicator_values)  # noqa: B023

        latest_ts = df.index[-1]
        result = ScanResult(
            symbol=symbol,
            exchange=self.exchange,
            timestamp=latest_ts.timestamp()
            if isinstance(latest_ts, pd.Timestamp)
            else float(latest_ts),
            price=close[-1],
            filters_matched=filters_matched,  # noqa: B023
            indicator_values=indicator_values,
            score=score,
        )

        return result, None

    def _calculate_score(
        self,
        filters: list[ScanFilter],
        filters_matched: list[str],  # noqa: B023
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
        unique_filters = list(dict.fromkeys(filters))

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
            matched_filter_names = set(filters_matched)  # noqa: B023
            matched_count = sum(
                1 for f in unique_filters if filter_str_map.get(f) in matched_filter_names
            )
            match_ratio = matched_count / len(unique_filters) if unique_filters else 0
            return min(100, match_ratio * 100)

    def session_breakout_scan(
        self,
        session: SessionType,
        symbols: list[str] | None = None,
        proximity_pct: float = 1.0,
        volume_multiplier: float = 2.0,
    ) -> tuple[list[ScanResult], list[ScanFailure]]:
        """Execute session breakout scan across symbols.

        Evaluates symbols against the requested session (asia, london, ny) using
        detect_session_levels() for session high/low and detect_pdh_pdl() for
        previous-day context. Breakout/proximity flags are computed against both
        session and PDH/PDL levels, with volume confirmation.

        Args:
            session: Session type (asia, london, ny). Accepts 'new_york' alias normalized to 'ny'.
            symbols: List of symbols to scan. Defaults to screener's configured symbol universe.
            proximity_pct: Percentage threshold for near-breakout detection (default 1.0).
                Price within proximity_pct% of session high/low is flagged as near-breakout.
            volume_multiplier: Volume threshold multiplier for confirmation (default 2.0).
                Current volume must be >= volume_multiplier * prior_window_avg.

        Returns:
            Tuple of (results, failures) where:
            - results: List of ScanResult sorted by (-score, -len(filters_matched), symbol, exchange)  # noqa: B023
            - failures: List of ScanFailure for symbols that could not be scanned
        """
        from tempest_mcp.indicators.session_levels import detect_pdh_pdl, detect_session_levels

        symbols_to_scan = symbols or list(self.symbols)
        results: list[ScanResult] = []
        failures: list[ScanFailure] = []

        # Fixed filter name order for deterministic filters_matched ordering  # noqa: B023
        FILTER_ORDER = (
            "session_high_breakout",
            "session_high_near_breakout",
            "session_low_breakout",
            "session_low_near_breakout",
            "pdh_breakout",
            "pdh_near_breakout",
            "pdl_breakout",
            "pdl_near_breakout",
            "volume_confirmation",
        )

        for symbol in symbols_to_scan:
            try:
                df = self.adapter.fetch_ohlcv_live(symbol, timeframe="1h", limit=48)
                if df.empty:
                    failures.append(
                        ScanFailure(
                            symbol=symbol,
                            exchange=self.exchange,
                            reason="empty_ohlcv",
                        )
                    )
                    continue

                close = df["close"].tolist()
                volume = df["volume"].tolist()
                current_price = close[-1]
                session_key = session.value

                # ── Session levels ───────────────────────────────────────────────
                session_result = detect_session_levels(df, session_key)
                session_high = float(session_result.get("high") or 0.0)
                session_low = float(session_result.get("low") or 0.0)
                session_bars = session_result.get("bars", 0)

                if session_bars == 0:
                    failures.append(
                        ScanFailure(
                            symbol=symbol,
                            exchange=self.exchange,
                            reason="insufficient_session_data",
                        )
                    )
                    continue

                # ── PDH/PDL context ──────────────────────────────────────────────
                pdh_pdl_result = detect_pdh_pdl(df)
                pdh = float(pdh_pdl_result.get("previous_day_high") or 0.0)
                pdl = float(pdh_pdl_result.get("previous_day_low") or 0.0)
                pdh_position = pdh_pdl_result.get("position", "insufficient_data")

                if pdh_position == "insufficient_data":
                    failures.append(
                        ScanFailure(
                            symbol=symbol,
                            exchange=self.exchange,
                            reason="insufficient_pdh_pdl_data",
                        )
                    )
                    continue

                # ── Breakout / Proximity checks ───────────────────────────────────
                filters_matched: list[str] = []  # noqa: B023

                def append_match(filter_name: str) -> None:  # noqa: B023
                    """Append filter in fixed order (no duplicates)."""
                    if filter_name not in filters_matched:  # noqa: B023
                        # Insert at correct position to maintain deterministic ordering
                        target_idx = FILTER_ORDER.index(filter_name)
                        inserted = False
                        for i, existing in enumerate(filters_matched):  # noqa: B023
                            if FILTER_ORDER.index(existing) > target_idx:
                                filters_matched.insert(i, filter_name)  # noqa: B023
                                inserted = True
                                break
                        if not inserted:
                            filters_matched.append(filter_name)  # noqa: B023

                # Session high breakout / near-breakout
                if session_high > 0:
                    if current_price > session_high:
                        append_match("session_high_breakout")
                    elif proximity_pct > 0 and current_price >= session_high * (1 - proximity_pct / 100):
                        append_match("session_high_near_breakout")

                # Session low breakout / near-breakout
                if session_low > 0:
                    if current_price < session_low:
                        append_match("session_low_breakout")
                    elif proximity_pct > 0 and current_price <= session_low * (1 + proximity_pct / 100):
                        append_match("session_low_near_breakout")

                # PDH breakout / near-breakout
                if pdh > 0:
                    if current_price > pdh:
                        append_match("pdh_breakout")
                    elif proximity_pct > 0 and current_price >= pdh * (1 - proximity_pct / 100):
                        append_match("pdh_near_breakout")

                # PDL breakout / near-breakout
                if pdl > 0:
                    if current_price < pdl:
                        append_match("pdl_breakout")
                    elif proximity_pct > 0 and current_price <= pdl * (1 + proximity_pct / 100):
                        append_match("pdl_near_breakout")

                # ── Volume confirmation ───────────────────────────────────────────
                lookback = min(20, len(volume))
                volume_confirmed = False
                if lookback >= 5:
                    avg_volume = sum(volume[-lookback:-1]) / (lookback - 1)
                    current_volume = volume[-1]
                    if avg_volume > 0 and current_volume >= avg_volume * volume_multiplier:
                        append_match("volume_confirmation")
                        volume_confirmed = True

                # ── Score calculation ───────────────────────────────────────────
                # Breakout = 30pts each, near-breakout = 15pts each, volume = 10pts
                # Max score = 100, min score = 0
                score = 0.0
                for f in filters_matched:  # noqa: B023
                    if f == "session_high_breakout":
                        score += 30.0
                    elif f == "session_low_breakout":
                        score += 30.0
                    elif f == "pdh_breakout":
                        score += 20.0
                    elif f == "pdl_breakout":
                        score += 20.0
                    elif f == "session_high_near_breakout":
                        score += 15.0
                    elif f == "session_low_near_breakout":
                        score += 15.0
                    elif f == "pdh_near_breakout":
                        score += 10.0
                    elif f == "pdl_near_breakout":
                        score += 10.0
                    elif f == "volume_confirmation":
                        score += 10.0

                score = min(100.0, max(0.0, score))

                latest_ts = df.index[-1]
                results.append(
                    ScanResult(
                        symbol=symbol,
                        exchange=self.exchange,
                        timestamp=latest_ts.timestamp()
                        if isinstance(latest_ts, pd.Timestamp)
                        else float(latest_ts),
                        price=current_price,
                        filters_matched=filters_matched,  # noqa: B023
                        indicator_values={
                            "session_high": session_high,
                            "session_low": session_low,
                            "session_bars": session_bars,
                            "previous_day_high": pdh,
                            "previous_day_low": pdl,
                            "volume_confirmed": float(volume_confirmed),
                            "volume_multiplier": volume_multiplier,
                            "proximity_pct": proximity_pct,
                        },
                        score=score,
                    )
                )

            except Exception as e:
                logger.warning("Session scan failed", symbol=symbol, error=str(e))
                failures.append(
                    ScanFailure(
                        symbol=symbol,
                        exchange=self.exchange,
                        reason="fetch_error",
                    )
                )

        # Deterministic sorting: (-score, -len(filters_matched), symbol, exchange)  # noqa: B023
        results.sort(key=lambda r: (-r.score, -len(r.filters_matched), r.symbol, r.exchange))  # noqa: B023

        # Sort failures deterministically: (symbol, exchange)
        failures.sort(key=lambda f: (f.symbol, f.exchange))

        return results, failures
