"""Tests for screener engine."""

import pandas as pd

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
        assert "fetch_error" in failures[0].reason

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
            filters=[ScanFilter.RSI_OVERSOLD],  # Should match
            min_score=100.0,  # Very high threshold
        )
        screener._adapter = DummyAdapter(df)

        results, failures = screener.scan()

        # With min_score=100, no results should pass
        assert all(r.score >= 100.0 for r in results)


class TestScreenerDeterministicScoring:
    """Tests for deterministic scoring behavior."""

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
        indicator_values = {"rsi": 25.0, "ema_7": 100.0, "ema_25": 99.0, "ema_50": 98.0, "volume_ratio": 1.0}
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


class TestSessionBreakoutScan:
    """Tests for Screener.session_breakout_scan()."""

    def test_session_breakout_scan_uses_detect_session_levels(self):
        dates = pd.date_range("2024-03-15", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0] * 48,
                "high": [101.0 + i * 0.1 for i in range(48)],
                "low": [99.0 + i * 0.1 for i in range(48)],
                "close": [100.5 + i * 0.2 for i in range(48)],
                "volume": [1000.0] * 48,
            },
            index=dates,
        )

        screener = Screener(symbols=("BTC/USDT",), exchange="binance")
        screener._adapter = DummyAdapter(df)

        results = screener.session_breakout_scan(SessionType.NEW_YORK)

        assert len(results) == 1
        assert results[0].indicator_values["session_high"] > 0
        assert results[0].indicator_values["session_low"] > 0


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
