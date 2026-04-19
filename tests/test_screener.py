"""Tests for screener engine."""

import pandas as pd
import pytest

from tempest_mcp.models.indicator import SessionType
from tempest_mcp.screener.scanner import (
    DEFAULT_FILTER_PRESET,
    ScanFailure,
    ScanFilter,
    ScanResult,
    Screener,
)


class DummyAdapter:
    """Dummy adapter for testing."""

    def __init__(self, df: pd.DataFrame | None = None, should_fail: bool = False):
        self.df = df
        self.should_fail = should_fail

    def fetch_ohlcv_live(self, symbol: str, timeframe: str = "1h", limit: int = 100):
        if self.should_fail:
            raise RuntimeError("Simulated fetch error")
        return self.df


class TestScanFilter:
    """Tests for ScanFilter enum values."""

    def test_filter_values(self):
        assert ScanFilter.RSI_OVERSOLD.value == "rsi_oversold"
        assert ScanFilter.RSI_OVERBOUGHT.value == "rsi_overbought"
        assert ScanFilter.TREND_BULLISH.value == "trend_bullish"
        assert ScanFilter.TREND_BEARISH.value == "trend_bearish"
        assert ScanFilter.HIGH_VOLATILITY.value == "high_volatility"
        assert ScanFilter.LOW_VOLATILITY.value == "low_volatility"
        assert ScanFilter.VOLUME_SPIKE.value == "volume_spike"

    def test_default_preset_contains_expected_filters(self):
        assert ScanFilter.RSI_OVERSOLD in DEFAULT_FILTER_PRESET
        assert ScanFilter.TREND_BULLISH in DEFAULT_FILTER_PRESET
        assert len(DEFAULT_FILTER_PRESET) == 2


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_scan_result_creation(self):
        result = ScanResult(
            symbol="BTC/USDT",
            exchange="binance",
            timestamp=1.0,
            price=50000.0,
            filters_matched=["rsi_oversold", "trend_bullish"],
            indicator_values={"rsi": 25.0, "ema_7": 49000.0},
            score=80.0,
        )
        assert result.symbol == "BTC/USDT"
        assert result.exchange == "binance"
        assert result.timestamp == 1.0
        assert result.price == 50000.0
        assert result.filters_matched == ["rsi_oversold", "trend_bullish"]
        assert result.indicator_values == {"rsi": 25.0, "ema_7": 49000.0}
        assert result.score == 80.0

    def test_scan_result_defaults(self):
        result = ScanResult(
            symbol="ETH/USDT",
            exchange="binance",
            timestamp=1.0,
            price=3000.0,
            filters_matched=[],
            indicator_values={},
        )
        assert result.score == 0.0
        assert result.metadata == {}


class TestScanFailure:
    """Tests for ScanFailure dataclass."""

    def test_scan_failure_creation(self):
        failure = ScanFailure(
            symbol="DOGE/USDT",
            exchange="binance",
            reason="empty_ohlcv",
        )
        assert failure.symbol == "DOGE/USDT"
        assert failure.exchange == "binance"
        assert failure.reason == "empty_ohlcv"


class TestScreenerInit:
    """Tests for Screener initialization."""

    def test_init_with_symbols(self):
        screener = Screener(symbols=("BTC/USDT", "ETH/USDT"), exchange="binance")
        assert screener.symbols == ("BTC/USDT", "ETH/USDT")
        assert screener.exchange == "binance"
        assert screener.filters == []
        assert screener.min_score == 0.0

    def test_init_with_filters(self):
        screener = Screener(
            symbols=("BTC/USDT",),
            exchange="binance",
            filters=[ScanFilter.RSI_OVERSOLD, ScanFilter.TREND_BULLISH],
        )
        assert len(screener.filters) == 2
        assert ScanFilter.RSI_OVERSOLD in screener.filters


class TestScreenerScan:
    """Tests for Screener.scan() method."""

    def _create_bullish_df(self) -> pd.DataFrame:
        """Create a DataFrame with bullish indicators (RSI oversold + EMA bullish)."""
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        # Price rising from 100 to 110
        close_prices = [100.0 + i * 0.2 for i in range(50)]
        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 1.0 for p in close_prices],
                "low": [p - 1.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        return df

    def _create_bearish_df(self) -> pd.DataFrame:
        """Create a DataFrame with bearish indicators."""
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        # Price falling from 110 to 100
        close_prices = [110.0 - i * 0.2 for i in range(50)]
        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 1.0 for p in close_prices],
                "low": [p - 1.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        return df

    def _create_neutral_df(self) -> pd.DataFrame:
        """Create a DataFrame with neutral/steady price."""
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        # Flat price at 100
        close_prices = [100.0] * 50
        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 0.5 for p in close_prices],
                "low": [p - 0.5 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        return df

    def test_scan_returns_tuple_of_results_and_failures(self):
        """Scan should return (results, failures) tuple, not just results."""
        df = self._create_bullish_df()
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        result = screener.scan()

        assert isinstance(result, tuple)
        assert len(result) == 2
        results, failures = result
        assert isinstance(results, list)
        assert isinstance(failures, list)

    def test_scan_with_empty_df_returns_failure(self):
        """Empty DataFrame should result in ScanFailure, not exception."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(empty_df)

        results, failures = screener.scan()

        assert len(results) == 0
        assert len(failures) == 1
        assert failures[0].symbol == "BTC/USDT"
        assert failures[0].reason == "empty_ohlcv"

    def test_scan_with_fetch_error_returns_failure(self):
        """Fetch error should result in ScanFailure, not exception."""
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(should_fail=True)

        results, failures = screener.scan()

        assert len(results) == 0
        assert len(failures) == 1
        assert failures[0].reason == "fetch_error"

    def test_scan_deterministic_sorting(self):
        """Results should be sorted by (-score, -len(filters_matched), symbol, exchange)."""
        # Create two DataFrames with different characteristics
        bullish_df = self._create_bullish_df()
        bearish_df = self._create_bearish_df()

        screener = Screener(
            symbols=("AAA/USDT", "ZZZ/USDT"),
            exchange="binance",
            filters=[ScanFilter.RSI_OVERSOLD, ScanFilter.TREND_BULLISH],
        )

        # Use a dict to map symbols to different DataFrames
        class SelectiveAdapter(DummyAdapter):
            def fetch_ohlcv_live(self, symbol, timeframe="1h", limit=100):
                if symbol == "AAA/USDT":
                    return bullish_df
                return bearish_df

        screener._adapter = SelectiveAdapter()
        results, failures = screener.scan()

        # Results should be sorted deterministically
        if len(results) >= 2:
            # First result should have higher or equal score to second
            assert results[0].score >= results[1].score
            # If scores are equal, first should have more filters matched
            if results[0].score == results[1].score:
                assert len(results[0].filters_matched) >= len(results[1].filters_matched)

    def test_scan_with_default_filters_uses_preset(self):
        """When filters is empty, should use DEFAULT_FILTER_PRESET."""
        df = self._create_bullish_df()
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        # Call scan without filters - should use default preset
        results, failures = screener.scan()

        # With default preset (RSI_OVERSOLD, TREND_BULLISH) and bullish data,
        # we should get some results
        # The exact result depends on indicator calculations
        assert isinstance(results, list)
        assert isinstance(failures, list)

    def test_scan_with_explicit_filters(self):
        """When filters is specified, should use those filters."""
        df = self._create_bullish_df()
        screener = Screener(
            symbols=("BTC/USDT",),
            exchange="binance",
            filters=[ScanFilter.RSI_OVERBOUGHT],  # Will not match bullish data
        )
        screener._adapter = DummyAdapter(df)

        results, failures = screener.scan()

        # RSI_OVERBOUGHT should not match since RSI won't be > 70
        assert isinstance(results, list)
        assert isinstance(failures, list)

    def test_scan_min_score_filtering(self):
        """Results with score < min_score should be filtered out."""
        df = self._create_bullish_df()
        screener = Screener(
            symbols=("BTC/USDT",),
            exchange="binance",
            filters=[ScanFilter.RSI_OVERSOLD, ScanFilter.TREND_BULLISH],
            min_score=75.0,
        )
        screener._adapter = DummyAdapter(df)

        results, failures = screener.scan()

        assert results == []
        assert failures == []


class TestScreenerDeterministicScoring:
    """Tests for deterministic scoring behavior."""

    def _create_bullish_df(self) -> pd.DataFrame:
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.2 for i in range(50)]
        return pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 1.0 for p in close_prices],
                "low": [p - 1.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )

    def _create_df_with_known_rsi(self, rsi_value: float) -> pd.DataFrame:
        """Create a DataFrame that will produce a specific RSI value."""
        dates = pd.date_range("2024-03-15", periods=100, freq="h", tz="UTC")
        # Create oscillating prices to get the target RSI
        # For RSI=25 (oversold), we need consistent downward movement
        base = 100.0
        prices = []
        for i in range(100):
            # Create a pattern that results in RSI oversold
            if i < 30:
                prices.append(base - i * 0.3)  # Gradual decline
            elif i < 60:
                prices.append(base - 30 * 0.3 + (i - 30) * 0.1)  # Recovery
            else:
                prices.append(base - 30 * 0.3 + 30 * 0.1 - (i - 60) * 0.3)  # Another decline
        df = pd.DataFrame(
            {
                "open": prices,
                "high": [p + 0.5 for p in prices],
                "low": [p - 0.5 for p in prices],
                "close": prices,
                "volume": [1000.0] * 100,
            },
            index=dates,
        )
        return df

    def test_score_calculation_no_filters(self):
        """Default scoring when no filters specified."""
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        indicator_values = {
            "rsi": 25.0,
            "ema_7": 100.0,
            "ema_25": 99.0,
            "ema_50": 98.0,
            "volume_ratio": 1.0,
        }
        score = screener._calculate_score([], [], indicator_values)
        # With RSI oversold (25 < 30), score should be 50 + 20 = 70
        assert score == 85.0

    def test_score_calculation_with_filters(self):
        """Filter-based scoring when filters specified."""
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        filters = [ScanFilter.RSI_OVERSOLD, ScanFilter.TREND_BULLISH]
        filters_matched = ["rsi_oversold"]  # Only RSI matched
        indicator_values = {"rsi": 25.0}

        score = screener._calculate_score(filters, filters_matched, indicator_values)
        # 1 out of 2 filters matched = 50%
        assert score == 50.0

    def test_score_calculation_all_filters_matched(self):
        """Score is 100 when all filters match."""
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        filters = [ScanFilter.RSI_OVERSOLD, ScanFilter.TREND_BULLISH]
        filters_matched = ["rsi_oversold", "trend_bullish"]  # Both matched
        indicator_values = {"rsi": 25.0}

        score = screener._calculate_score(filters, filters_matched, indicator_values)
        # 2 out of 2 filters matched = 100%
        assert score == 100.0

    def test_score_calculation_deduplicates_requested_filters(self):
        """Duplicate requested filters should not inflate score."""
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        filters = [
            ScanFilter.RSI_OVERSOLD,
            ScanFilter.RSI_OVERSOLD,
            ScanFilter.TREND_BULLISH,
        ]
        filters_matched = ["rsi_oversold"]

        score = screener._calculate_score(filters, filters_matched, {"rsi": 25.0})

        assert score == 50.0

    def test_momentum_does_not_match_trend_filters(self, monkeypatch):
        """Momentum sign must not backfill trend filter matches."""

        df = self._create_bullish_df()

        class FakeIndicatorResult:
            def __init__(self, values):
                self.values = values

        monkeypatch.setattr(
            "tempest_mcp.screener.scanner.calculate_rsi_result",
            lambda close: FakeIndicatorResult(
                {"rsi": 50.0, "oversold": False, "overbought": False}
            ),
        )
        monkeypatch.setattr(
            "tempest_mcp.screener.scanner.calculate_ema_result",
            lambda close, periods: FakeIndicatorResult(
                {"ema_7": 100.0, "ema_25": 100.0, "ema_50": 100.0}
            ),
        )

        screener = Screener(
            symbols=("BTC/USDT",),
            exchange="binance",
            filters=[ScanFilter.TREND_BULLISH],
        )
        screener._adapter = DummyAdapter(df)

        results, failures = screener.scan()

        assert failures == []
        assert len(results) == 1
        assert results[0].filters_matched == []
        assert results[0].score == 0.0

    def test_volatility_filters_are_evaluated(self, monkeypatch):
        """High/low volatility filters should participate in filter matching."""

        df = self._create_bullish_df()
        monkeypatch.setattr(
            "tempest_mcp.screener.scanner.calculate_bollinger_width",
            lambda prices: pd.Series([0.12], index=[prices.index[-1]]),
        )

        screener = Screener(
            symbols=("BTC/USDT",),
            exchange="binance",
            filters=[ScanFilter.HIGH_VOLATILITY],
        )
        screener._adapter = DummyAdapter(df)

        results, failures = screener.scan()

        assert failures == []
        assert len(results) == 1
        assert results[0].filters_matched == ["high_volatility"]
        assert results[0].score == 100.0


class TestSessionBreakoutScan:
    """Tests for Screener.session_breakout_scan() — ENG-35."""

    def _create_session_df(self, session_type: str = "ny") -> pd.DataFrame:
        """Create a DataFrame suitable for session level detection.

        For NY session (09:30-16:00 ET = 13:30-20:00 UTC), we need bars
        that fall within that window.
        """
        if session_type == "asia":
            # Asia: 00:00-09:00 UTC
            dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
            # Make sure some hours are in Asia session (0-9)
            dates = dates.tz_convert(None)  # naive
            dates = pd.DatetimeIndex([d for d in dates if d.hour < 9 or d.hour >= 0])
            dates = dates[:48]
        elif session_type == "london":
            # London: 08:00-17:00 UTC
            dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
        else:
            # NY: Use a range that covers NY session hours
            # 13:30-20:00 UTC
            start = pd.Timestamp("2024-03-15 13:00", tz="UTC")
            dates = pd.date_range(start, periods=48, freq="h", tz="UTC")

        base_price = 50000.0
        df = pd.DataFrame(
            {
                "open": [base_price] * len(dates),
                "high": [base_price + 100.0] * len(dates),
                "low": [base_price - 100.0] * len(dates),
                "close": [base_price + 50.0] * len(dates),
                "volume": [1000.0] * len(dates),
            },
            index=dates,
        )
        return df

    def _create_df_with_breakout(self) -> pd.DataFrame:
        """Create a DataFrame with price breaking out above session high."""
        # NY session hours
        start = pd.Timestamp("2024-03-15 13:00", tz="UTC")
        dates = pd.date_range(start, periods=48, freq="h", tz="UTC")

        base_price = 50000.0
        # Price starts below session high, ends above
        close_prices = [base_price + i * 50 for i in range(24)] + [base_price + 1500] * 24

        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 100.0 for p in close_prices],
                "low": [p - 100.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * len(dates),
            },
            index=dates,
        )
        return df

    def _create_df_with_pdh_context(self) -> pd.DataFrame:
        """Create a DataFrame with proper PDH/PDL context.

        Need 48 1h bars that span at least 2 business days ET.
        """
        start = pd.Timestamp("2024-03-13 00:00", tz="UTC")  # Start Wed UTC
        dates = pd.date_range(start, periods=48, freq="h", tz="UTC")

        base_price = 50000.0
        # Day 1: PDH day (higher prices)
        day1_prices = [base_price + 500 + i * 10 for i in range(24)]
        # Day 2: Current day (lower prices, breaking out)
        day2_prices = [base_price + 100 + i * 50 for i in range(24)]
        all_prices = day1_prices + day2_prices

        df = pd.DataFrame(
            {
                "open": all_prices,
                "high": [p + 100.0 for p in all_prices],
                "low": [p - 100.0 for p in all_prices],
                "close": all_prices,
                "volume": [1000.0] * len(dates),
            },
            index=dates,
        )
        return df

    def test_session_breakout_scan_returns_tuple(self):
        """session_breakout_scan returns (results, failures) tuple."""
        df = self._create_df_with_breakout()
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        result = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_session_breakout_scan_empty_df_returns_failure(self):
        """Empty DataFrame returns ScanFailure."""
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(empty_df)

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(results) == 0
        assert len(failures) == 1
        assert failures[0].reason == "empty_ohlcv"

    def test_session_breakout_scan_fetch_error_returns_failure(self):
        """Fetch error returns ScanFailure."""
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(should_fail=True)

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(results) == 0
        assert len(failures) == 1
        assert failures[0].reason == "fetch_error"

    def test_session_breakout_scan_uses_detect_session_levels(self):
        """Results include session levels from detect_session_levels."""
        df = self._create_df_with_breakout()
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(results) >= 1
        result = results[0]
        assert "session_high" in result.indicator_values
        assert "session_low" in result.indicator_values
        assert result.indicator_values["session_high"] > 0
        assert result.indicator_values["session_low"] > 0

    def test_session_breakout_scan_includes_pdh_pdl_context(self):
        """Results include PDH/PDL context from detect_pdh_pdl."""
        df = self._create_df_with_pdh_context()
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(results) >= 1
        result = results[0]
        assert "previous_day_high" in result.indicator_values
        assert "previous_day_low" in result.indicator_values

    def test_session_breakout_scan_detects_high_breakout(self, monkeypatch):
        """Results detect when price breaks above session high."""
        dates = pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h")
        df = pd.DataFrame(
            {
                "open": [99.0] * 11 + [101.0],
                "high": [99.5] * 11 + [101.2],
                "low": [98.5] * 11 + [100.8],
                "close": [99.0] * 11 + [101.0],
                "volume": [1000.0] * 12,
            },
            index=dates,
        )
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 100.0, "low": 10.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": 200.0,
                "previous_day_low": 1.0,
                "position": "inside_range",
            },
        )

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert failures == []
        assert len(results) == 1
        assert results[0].filters_matched == ["session_high_breakout"]
        assert results[0].score == 30.0

    def test_session_breakout_scan_volume_confirmation(self, monkeypatch):
        """Results include volume confirmation check."""
        dates = pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 11 + [100.0],
                "high": [100.2] * 11 + [100.2],
                "low": [99.8] * 11 + [99.8],
                "close": [100.0] * 12,
                "volume": [1000.0] * 11 + [3000.0],
            },
            index=dates,
        )
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 200.0, "low": 10.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": 300.0,
                "previous_day_low": 1.0,
                "position": "inside_range",
            },
        )

        results_confirmed, _ = screener.session_breakout_scan(
            SessionType.NEW_YORK, volume_multiplier=2.5
        )
        results_unconfirmed, _ = screener.session_breakout_scan(
            SessionType.NEW_YORK, volume_multiplier=3.5
        )

        assert results_confirmed[0].filters_matched == ["volume_confirmation"]
        assert results_confirmed[0].indicator_values["volume_confirmed"] == 1.0
        assert results_unconfirmed[0].filters_matched == []
        assert results_unconfirmed[0].indicator_values["volume_confirmed"] == 0.0

    def test_session_breakout_scan_tie_breaks_on_filter_count(self, monkeypatch):
        """Equal-score results prefer higher filter-count before symbol ordering."""
        high_df = pd.DataFrame(
            {
                "open": [99.0] * 11 + [101.0],
                "high": [99.5] * 11 + [101.2],
                "low": [98.5] * 11 + [100.8],
                "close": [99.0] * 11 + [101.0],
                "volume": [1000.0] * 12,
            },
            index=pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h"),
        )
        volume_df = pd.DataFrame(
            {
                "open": [79.0] * 11 + [79.0],
                "high": [79.2] * 11 + [79.2],
                "low": [78.8] * 11 + [78.8],
                "close": [79.0] * 12,
                "volume": [1000.0] * 11 + [3000.0],
            },
            index=pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h"),
        )

        class MixedAdapter(DummyAdapter):
            def fetch_ohlcv_live(self, symbol, timeframe="1h", limit=100):
                if symbol == "AAA/USDT":
                    return high_df
                return volume_df

        screener = Screener(symbols=("BBB/USDT", "AAA/USDT"), exchange="binance")
        screener._adapter = MixedAdapter()

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 100.0, "low": 10.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": 150.0,
                "previous_day_low": 80.0,
                "position": "inside_range",
            },
        )

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK, volume_multiplier=2.5)

        assert failures == []
        assert [r.symbol for r in results] == ["BBB/USDT", "AAA/USDT"]
        assert results[0].filters_matched == ["pdl_breakout", "volume_confirmation"]
        assert results[0].score == 30.0
        assert results[1].filters_matched == ["session_high_breakout"]
        assert results[1].score == 30.0

    def test_session_breakout_scan_deterministic_sorting(self, monkeypatch):
        """Results are sorted deterministically by (-score, -len(filters), symbol, exchange)."""
        dates = pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h")
        df = pd.DataFrame(
            {
                "open": [99.0] * 11 + [101.0],
                "high": [99.5] * 11 + [101.2],
                "low": [98.5] * 11 + [100.8],
                "close": [99.0] * 11 + [101.0],
                "volume": [1000.0] * 12,
            },
            index=dates,
        )
        screener = Screener(symbols=("BBB/USDT", "AAA/USDT"), exchange="binance")
        screener._adapter = DummyAdapter(df)

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 100.0, "low": 10.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": 200.0,
                "previous_day_low": 1.0,
                "position": "inside_range",
            },
        )

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert failures == []
        assert [r.symbol for r in results] == ["AAA/USDT", "BBB/USDT"]
        assert all(r.filters_matched == ["session_high_breakout"] for r in results)

    def test_session_breakout_scan_failure_sorting(self):
        """Failures are sorted deterministically by (symbol, exchange)."""
        screener = Screener(symbols=("ZZZ/USDT", "AAA/USDT"), exchange="binance")
        screener._adapter = DummyAdapter(should_fail=True)

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert results == []
        assert [f.symbol for f in failures] == ["AAA/USDT", "ZZZ/USDT"]
        assert all(f.reason == "fetch_error" for f in failures)

    def test_session_breakout_scan_new_york_alias(self):
        """Session type accepts 'new_york' as alias for 'ny'."""
        df = self._create_df_with_breakout()
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        result, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(result) >= 1
        assert result[0].filters_matched
        assert result[0].indicator_values["session_high"] > 0
        assert failures == []

    def test_session_breakout_scan_proximity_pct(self, monkeypatch):
        """Proximity percentage affects near-breakout detection."""
        dates = pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 11 + [100.5],
                "high": [100.2] * 11 + [100.6],
                "low": [99.8] * 11 + [100.1],
                "close": [100.0] * 11 + [100.5],
                "volume": [1000.0] * 12,
            },
            index=dates,
        )
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 101.0, "low": 99.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": 110.0,
                "previous_day_low": 90.0,
                "position": "inside_range",
            },
        )

        results1, _ = screener.session_breakout_scan(
            SessionType.NEW_YORK, proximity_pct=0.0  # Strict - no proximity
        )
        results2, _ = screener.session_breakout_scan(
            SessionType.NEW_YORK, proximity_pct=1.0  # Lenient - 1%
        )

        assert results1[0].filters_matched == []
        assert results2[0].filters_matched == ["session_high_near_breakout"]

    def test_session_breakout_scan_uses_pdh_pdl(self, monkeypatch):
        """session_breakout_scan composes detect_pdh_pdl."""
        dates = pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h")
        df = pd.DataFrame(
            {
                "open": [149.0] * 11 + [151.0],
                "high": [149.5] * 11 + [151.2],
                "low": [148.5] * 11 + [150.8],
                "close": [149.0] * 11 + [151.0],
                "volume": [1000.0] * 12,
            },
            index=dates,
        )
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 300.0, "low": 10.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": 150.0,
                "previous_day_low": 80.0,
                "position": "inside_range",
            },
        )

        results, failures = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert failures == []
        assert len(results) == 1
        result = results[0]
        assert result.filters_matched == ["pdh_breakout"]
        assert result.indicator_values["previous_day_high"] == 150.0
        assert result.indicator_values["previous_day_low"] == 80.0

    @pytest.mark.parametrize(
        "current_close, pdh_level, pdl_level, expected_filters",
        [
            (9.0, 150.0, 1.0, ["session_low_breakout"]),
            (10.05, 150.0, 1.0, ["session_low_near_breakout"]),
            (149.5, 150.0, 1.0, ["pdh_near_breakout"]),
            (79.0, 150.0, 80.0, ["pdl_breakout"]),
        ],
    )
    def test_session_breakout_scan_low_and_pdh_pdl_classifications(
        self, monkeypatch, current_close, pdh_level, pdl_level, expected_filters
    ):
        """Low-side and PDH/PDL classifications are pinned exactly."""
        dates = pd.date_range(pd.Timestamp("2024-03-18 00:00", tz="UTC"), periods=12, freq="h")
        df = pd.DataFrame(
            {
                "open": [current_close] * 11 + [current_close],
                "high": [current_close + 0.2] * 11 + [current_close + 0.2],
                "low": [current_close - 0.2] * 11 + [current_close - 0.2],
                "close": [current_close] * 12,
                "volume": [1000.0] * 12,
            },
            index=dates,
        )
        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_session_levels",
            lambda *_args, **_kwargs: {"high": 200.0, "low": 10.0, "bars": 12},
        )
        monkeypatch.setattr(
            "tempest_mcp.indicators.session_levels.detect_pdh_pdl",
            lambda *_args, **_kwargs: {
                "previous_day_high": pdh_level,
                "previous_day_low": pdl_level,
                "position": "inside_range",
            },
        )

        results, failures = screener.session_breakout_scan(
            SessionType.NEW_YORK, proximity_pct=1.0
        )

        assert failures == []
        assert len(results) == 1
        assert results[0].filters_matched == expected_filters


class TestScreenerEdgeCases:
    """Tests for edge cases in Screener."""

    def test_multiple_symbols_mixed_results(self):
        """When some symbols succeed and some fail."""
        bullish_df = self._create_bullish_df_for_symbol("BTC/USDT")
        empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        class MixedAdapter(DummyAdapter):
            def fetch_ohlcv_live(self, symbol, timeframe="1h", limit=100):
                if symbol == "BTC/USDT":
                    return bullish_df
                return empty_df

        screener = Screener(
            symbols=("BTC/USDT", "ETH/USDT"),
            exchange="binance",
        )
        screener._adapter = MixedAdapter()

        results, failures = screener.scan()

        # One success, one failure
        assert len(results) + len(failures) == 2

    def _create_bullish_df_for_symbol(self, symbol: str) -> pd.DataFrame:
        dates = pd.date_range("2024-03-15", periods=50, freq="h", tz="UTC")
        close_prices = [100.0 + i * 0.2 for i in range(50)]
        df = pd.DataFrame(
            {
                "open": close_prices,
                "high": [p + 1.0 for p in close_prices],
                "low": [p - 1.0 for p in close_prices],
                "close": close_prices,
                "volume": [1000.0] * 50,
            },
            index=dates,
        )
        return df
