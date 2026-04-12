"""Integration smoke tests for Phase 1 data layer + indicator engine.

These tests require live network access and are marked with @pytest.mark.integration.
They are skipped by default (run with pytest --run-integration to execute).

No mocking — smoke tests hit real endpoints.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tempest_mcp.data import DataSourceRouter, get_live_adapter
from tempest_mcp.data.yf_adapter import fetch_ohlcv
from tempest_mcp.indicators import (
    calculate_ema_stack,
    calculate_rsi,
    calculate_vwap,
    detect_session_levels,
)


# =============================================================================
# 1. Data Layer — Historical (YFinance adapter)
# =============================================================================


@pytest.mark.integration
class TestYFinanceAdapter:
    """Tests for YFinance historical data adapter."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_yf_btcusd_ohlcv(self):
        """Fetch BTC-USD daily OHLCV (start: 90 days ago)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)

        assert not df.empty, "BTC-USD should return non-empty DataFrame"
        assert isinstance(df, pd.DataFrame)
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Column {col} must be present"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None, "Index must be UTC-aware"

    def test_yf_ethusd_ohlcv(self):
        """Fetch ETH-USD daily OHLCV (start: 90 days ago)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        df = fetch_ohlcv("ETH-USD", interval="1d", start=start, end=end)

        assert not df.empty, "ETH-USD should return non-empty DataFrame"
        assert isinstance(df, pd.DataFrame)
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Column {col} must be present"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None, "Index must be UTC-aware"

    def test_yf_dogeusd_ohlcv(self):
        """Fetch DOGE-USD daily OHLCV (start: 90 days ago)."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        df = fetch_ohlcv("DOGE-USD", interval="1d", start=start, end=end)

        assert not df.empty, "DOGE-USD should return non-empty DataFrame"
        assert isinstance(df, pd.DataFrame)
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Column {col} must be present"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None, "Index must be UTC-aware"

    def test_yf_empty_on_invalid_symbol(self):
        """Invalid symbol returns empty DataFrame (not an exception)."""
        df = fetch_ohlcv("INVALID_SYMBOL_XYZ", interval="1d")

        assert isinstance(df, pd.DataFrame)
        assert df.empty, "Invalid symbol should return empty DataFrame"


# =============================================================================
# 2. Data Layer — Live (CCXT adapter)
# =============================================================================


@pytest.mark.integration
class TestCCXTAdapter:
    """Tests for CCXT live data adapter."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_ccxt_btcusdt_live_price(self):
        """Fetch live price for BTC/USDT. Asserts price > 0, type is float."""
        adapter = get_live_adapter()
        price = adapter.fetch_live_price("BTCUSDT")

        assert isinstance(price, float), "Price must be a float"
        assert price > 0, "Price must be positive"

    def test_ccxt_ethusdt_live_price(self):
        """Fetch live price for ETH/USDT. Asserts price > 0, type is float."""
        adapter = get_live_adapter()
        price = adapter.fetch_live_price("ETHUSDT")

        assert isinstance(price, float), "Price must be a float"
        assert price > 0, "Price must be positive"

    def test_ccxt_ohlcv_fetch(self):
        """Fetch 100 1h candles for BTC/USDT. Asserts non-empty DataFrame, correct columns, UTC-aware index."""
        adapter = get_live_adapter()
        df = adapter.fetch_ohlcv_live("BTCUSDT", timeframe="1h", limit=100)

        assert not df.empty, "Should return non-empty DataFrame"
        assert isinstance(df, pd.DataFrame)
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Column {col} must be present"
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None, "Index must be UTC-aware"


# =============================================================================
# 3. Data Layer — Historical via router
# =============================================================================


@pytest.mark.integration
class TestHistoricalRouter:
    """Tests for DataSourceRouter historical data interface."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_historical_source_crypto(self):
        """DataSourceRouter().route_historical().fetch_ohlcv returns non-empty DataFrame."""
        router = DataSourceRouter()
        hist = router.route_historical()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        df = hist.fetch_ohlcv("BTC/USDT", interval="1d", start=start, end=end)

        assert not df.empty, "Should return non-empty DataFrame for BTC/USDT"
        assert isinstance(df, pd.DataFrame)
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns

    def test_historical_fallback_stock_symbol(self):
        """AAPL (stock symbol) exercises the yfinance fallback path explicitly."""
        router = DataSourceRouter()
        hist = router.route_historical()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        # AAPL is a stock symbol, not crypto — CCXT won't have it,
        # so this explicitly exercises the yfinance fallback
        df = hist.fetch_ohlcv("AAPL", interval="1d", start=start, end=end)

        assert not df.empty, "AAPL should return data via yfinance fallback"
        assert isinstance(df, pd.DataFrame)


# =============================================================================
# 4. Indicator Engine — EMA
# =============================================================================


@pytest.mark.integration
class TestEMATechnical:
    """Tests for EMA indicator engine."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_ema_stack_values_defined(self):
        """Fetch BTC-USD daily OHLCV (90 days). Compute EMA stack. Assert all four EMA series are non-empty and have no NaN in the last 10 values."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=90)

        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)
        assert not df.empty, "BTC-USD should return data"

        close = df["close"]

        # Calculate EMA stack with explicit periods parameter
        stack = calculate_ema_stack(close, [7, 25, 50, 200])

        assert "ema7" in stack
        assert "ema25" in stack
        assert "ema50" in stack
        assert "ema200" in stack

        # All series must be non-empty
        for key in ["ema7", "ema25", "ema50", "ema200"]:
            assert not stack[key].empty, f"{key} must be non-empty"

        # Last 10 values must not be NaN
        for key in ["ema7", "ema25", "ema50", "ema200"]:
            last_10 = stack[key].tail(10)
            assert not last_10.isna().any(), f"{key} should have no NaN in last 10 values"

    def test_ema_stack_insufficient_data_raises(self):
        """Calling calculate_ema_stack with fewer than 200 values raises ValueError."""
        # Create a price series with only 100 values
        prices = pd.Series(range(100, 200))

        with pytest.raises(ValueError, match="Insufficient data"):
            calculate_ema_stack(prices, [7, 25, 50, 200])


# =============================================================================
# 5. Indicator Engine — RSI
# =============================================================================


@pytest.mark.integration
class TestRSITechnical:
    """Tests for RSI indicator engine."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_rsi_btc_known_range(self):
        """Fetch BTC-USD daily OHLCV (180 days). Compute RSI(14). Assert all RSI values in [0, 100]."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=180)

        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)
        assert not df.empty, "BTC-USD should return data"

        close = df["close"]
        rsi = calculate_rsi(close, period=14)

        assert isinstance(rsi, pd.Series)
        # All RSI values must be in [0, 100]
        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI values must be >= 0"
        assert (valid_rsi <= 100).all(), "RSI values must be <= 100"


# =============================================================================
# 6. Indicator Engine — Session Levels
# =============================================================================


@pytest.mark.integration
class TestSessionLevelsTechnical:
    """Tests for session levels indicator engine."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_session_levels_asia_hnl(self):
        """Fetch BTC-USD daily OHLCV (30 days). Assert asia_high >= asia_low."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)
        assert not df.empty, "BTC-USD should return data"

        levels = detect_session_levels(df)

        assert "asia_high" in levels
        assert "asia_low" in levels
        assert levels["asia_high"] >= levels["asia_low"]

    def test_session_levels_london_hnl(self):
        """Fetch BTC-USD daily OHLCV (30 days). Assert london_high >= london_low."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)
        assert not df.empty, "BTC-USD should return data"

        levels = detect_session_levels(df)

        assert "london_high" in levels
        assert "london_low" in levels
        assert levels["london_high"] >= levels["london_low"]

    def test_session_levels_ny_hnl(self):
        """Fetch BTC-USD daily OHLCV (30 days). Assert ny_high >= ny_low."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=30)

        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)
        assert not df.empty, "BTC-USD should return data"

        levels = detect_session_levels(df)

        assert "ny_high" in levels
        assert "ny_low" in levels
        assert levels["ny_high"] >= levels["ny_low"]


# =============================================================================
# 7. Indicator Engine — VWAP
# =============================================================================


@pytest.mark.integration
class TestVWAPTechnical:
    """Tests for VWAP indicator engine."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_vwap_btcusdt(self):
        """Fetch 1h BTC/USDT OHLCV (100 candles via CCXT). Compute VWAP. Assert VWAP is between the HLV tuple lows and highs."""
        adapter = get_live_adapter()
        df = adapter.fetch_ohlcv_live("BTCUSDT", timeframe="1h", limit=100)

        assert not df.empty, "Should return non-empty DataFrame"

        vwap, hlv = calculate_vwap(df)

        assert isinstance(vwap, float), "VWAP must be a float"
        assert hlv is not None, "HLV tuple must be returned"

        low, high, close = hlv
        assert low <= vwap <= high, f"VWAP {vwap} should be between low {low} and high {high}"


# =============================================================================
# 8. Error Handling
# =============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling (D14)."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_empty_df_returns_empty(self):
        """Calling fetch_ohlcv with garbage symbol returns empty DataFrame, does not raise."""
        # Using an invalid symbol that yfinance will reject
        df = fetch_ohlcv("GARBAGE_SYMBOL_XYZ_123")

        assert isinstance(df, pd.DataFrame)
        assert df.empty, "Garbage symbol should return empty DataFrame, not raise"
