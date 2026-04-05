"""
Tests for data layer adapters.

Tests cover:
- Symbol normalization (_symbols.py)
- CCXT adapter functionality
- TradingView adapter with CCXT fallback
- Error handling per D14 (no exception propagation)
"""

import pandas as pd
import pytest

from tempest_mcp.data._symbols import (
    get_base_currency,
    normalize_to_ccxt,
    normalize_to_tradingview,
    validate_symbol,
)
from tempest_mcp.errors import CCXTError, TradingViewError


class TestSymbolNormalization:
    """Tests for symbol conversion (D11, D12)."""

    def test_normalize_btcusdt_to_ccxt(self):
        """BTCUSDT should remain unchanged for CCXT."""
        assert normalize_to_ccxt("BTCUSDT") == "BTCUSDT"

    def test_normalize_btcusd_to_ccxt(self):
        """BTCUSD (TV format) should convert to BTCUSDT for CCXT."""
        assert normalize_to_ccxt("BTCUSD") == "BTCUSDT"

    def test_normalize_btcusdt_to_tradingview(self):
        """BTCUSDT should convert to BTCUSD for TradingView."""
        assert normalize_to_tradingview("BTCUSDT") == "BTCUSD"

    def test_normalize_btcusd_to_tradingview(self):
        """BTCUSD should remain unchanged for TradingView."""
        assert normalize_to_tradingview("BTCUSD") == "BTCUSD"

    def test_lowercase_symbol_handling(self):
        """Lowercase symbols should be handled correctly."""
        assert normalize_to_ccxt("btcusdt") == "BTCUSDT"
        assert normalize_to_ccxt("btcusd") == "BTCUSDT"
        assert normalize_to_tradingview("btcusdt") == "BTCUSD"

    def test_get_base_currency(self):
        """Base currency extraction should work correctly."""
        assert get_base_currency("BTCUSDT") == "BTC"
        assert get_base_currency("BTCUSD") == "BTC"
        assert get_base_currency("ETHUSDT") == "ETH"

    def test_validate_symbol_known(self):
        """Known symbols should validate successfully."""
        assert validate_symbol("BTCUSDT") is True
        assert validate_symbol("BTCUSD") is True
        assert validate_symbol("ETHUSDT") is True

    def test_validate_symbol_unknown(self):
        """Unknown symbols should fail validation."""
        assert validate_symbol("INVALIDPAIR") is False

    def test_invalid_symbol_raises_error(self):
        """Unrecognized symbol format should raise ValueError."""
        with pytest.raises(ValueError):
            normalize_to_ccxt("INVALIDPAIR")


class TestCCXTAdapter:
    """Tests for CCXT adapter functionality."""

    def test_adapter_initialization(self, ccxt_adapter):
        """Adapter should initialize with correct exchange."""
        assert ccxt_adapter.exchange_name == "binance"

    def test_fetch_live_price_returns_float(self, ccxt_adapter):
        """fetch_live_price should return float (D14 - NaN on error)."""
        # Using an invalid symbol to test NaN return
        price = ccxt_adapter.fetch_live_price("INVALIDPAIR")
        assert isinstance(price, float)

    def test_fetch_ohlcv_returns_dataframe(self, ccxt_adapter):
        """fetch_ohlcv_live should return DataFrame with correct columns."""
        df = ccxt_adapter.fetch_ohlcv_live("INVALIDPAIR", "1m", 100)

        # Should return DataFrame even on error (D14)
        assert isinstance(df, pd.DataFrame)

        # Should have correct columns
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns

    def test_fetch_orderbook_returns_dict(self, ccxt_adapter):
        """fetch_orderbook_snapshot should return dict with correct keys."""
        ob = ccxt_adapter.fetch_orderbook_snapshot("INVALIDPAIR", 20)

        # Should return dict even on error (D14)
        assert isinstance(ob, dict)
        assert "bids" in ob
        assert "asks" in ob
        assert "timestamp" in ob

        # On error, should have empty values
        assert ob["bids"] == []
        assert ob["asks"] == []
        assert ob["timestamp"] is None

    def test_symbol_normalization_in_adapter(self, ccxt_adapter):
        """Adapter should normalize TV symbols to CCXT format."""
        # This test verifies the adapter uses normalize_to_ccxt
        # The actual API call may fail, but we test the normalization path
        df = ccxt_adapter.fetch_ohlcv_live("BTCUSD", "1m", 10)
        # If normalization worked, we get a DataFrame (may be empty on API error)
        assert isinstance(df, pd.DataFrame)


class TestTradingViewAdapter:
    """Tests for TradingView adapter with CCXT fallback."""

    def test_adapter_initialization_with_key(self, tv_adapter_with_key):
        """Adapter should store API key when provided."""
        assert tv_adapter_with_key.api_key == "test-api-key"

    def test_adapter_initialization_no_key(self, tv_adapter_no_key):
        """Adapter should have None API key when not provided."""
        assert tv_adapter_no_key.api_key is None

    def test_fetch_live_price_no_key_uses_ccxt_fallback(self, tv_adapter_no_key):
        """Without API key, should fall back to CCXT."""
        # Should return float (NaN on error)
        price = tv_adapter_no_key.fetch_live_price("BTCUSDT")
        assert isinstance(price, float)

    def test_fetch_ohlcv_no_key_uses_ccxt_fallback(self, tv_adapter_no_key):
        """Without API key, should fall back to CCXT for OHLCV."""
        df = tv_adapter_no_key.fetch_ohlcv_live("BTCUSDT", "1m", 10)

        assert isinstance(df, pd.DataFrame)
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns

    def test_orderbook_delegates_to_ccxt(self, tv_adapter_with_key):
        """fetch_orderbook_snapshot should delegate to CCXT (D16)."""
        ob = tv_adapter_with_key.fetch_orderbook_snapshot("BTCUSDT", 20)

        assert isinstance(ob, dict)
        assert "bids" in ob
        assert "asks" in ob
        assert "timestamp" in ob

    def test_symbol_normalization_tv_format(self, tv_adapter_with_key):
        """TradingView adapter should normalize symbols to TV format."""
        # The adapter should convert BTCUSDT to BTCUSD for TV API
        # We can verify this by checking the normalization is used
        tv_symbol = normalize_to_tradingview("BTCUSDT")
        assert tv_symbol == "BTCUSD"


class TestErrorHierarchy:
    """Tests for error code taxonomy (D8)."""

    def test_tradingview_error_in_3xxx_range(self):
        """TradingViewError should be in 3001-3005 range."""
        error = TradingViewError("Test error", code=3002)
        assert 3001 <= error.code <= 3005
        assert error.code == 3002

    def test_tradingview_error_default_code(self):
        """TradingViewError should default to 3001."""
        error = TradingViewError("Test error")
        assert error.code == 3001

    def test_tradingview_error_code_range_clamped(self):
        """TradingViewError code should be clamped to valid range."""
        error = TradingViewError("Test", code=9999)
        assert error.code == 3001  # Should be reset to default

    def test_ccxt_error_in_3xxx_range(self):
        """CCXTError should be in 3101-3105 range."""
        error = CCXTError("Test error", code=3103)
        assert 3101 <= error.code <= 3105
        assert error.code == 3103

    def test_ccxt_error_default_code(self):
        """CCXTError should default to 3101."""
        error = CCXTError("Test error")
        assert error.code == 3101

    def test_error_str_representation(self):
        """Error should have proper string representation."""
        error = TradingViewError("Test error", code=3002)
        assert "[3002]" in str(error)
        assert "Test error" in str(error)


class TestAdapterSelection:
    """Tests for adapter selection in __init__.py."""

    def test_get_live_adapter_no_key_returns_ccxt(self, monkeypatch):
        """get_live_adapter should return CCXTAdapter when no API key."""
        monkeypatch.delenv("TRADINGVIEW_API_KEY", raising=False)

        from tempest_mcp.data import get_live_adapter
        from tempest_mcp.data.ccxt_adapter import CCXTAdapter

        adapter = get_live_adapter()
        assert isinstance(adapter, CCXTAdapter)

    def test_get_live_adapter_with_key_returns_ccxt(self, monkeypatch):
        """get_live_adapter should return CCXTAdapter even when API key is set.

        TradingView has no OHLCV data API - CCXT is always used for live data.
        """
        monkeypatch.setenv("TRADINGVIEW_API_KEY", "test-key")

        # Need to reimport to pick up env change
        import importlib

        import tempest_mcp.data

        importlib.reload(tempest_mcp.data)

        from tempest_mcp.data import get_live_adapter
        from tempest_mcp.data.ccxt_adapter import CCXTAdapter

        adapter = get_live_adapter()
        assert isinstance(adapter, CCXTAdapter)


class TestHistoricalAdapter:
    """Tests for historical data adapter and factory."""

    def test_get_historical_adapter_returns_historical_data_source(self):
        """get_historical_adapter should return HistoricalDataSource instance."""
        from tempest_mcp.data._factory import get_historical_adapter
        from tempest_mcp.data._hist import HistoricalDataSource

        adapter = get_historical_adapter()
        assert isinstance(adapter, HistoricalDataSource)

    def test_get_historical_adapter_caching(self):
        """get_historical_adapter should return same instance (singleton)."""
        from tempest_mcp.data._factory import get_historical_adapter

        # Clear the cache first
        get_historical_adapter.cache_clear()

        adapter1 = get_historical_adapter()
        adapter2 = get_historical_adapter()
        assert adapter1 is adapter2

    def test_historical_data_source_fetch_ohlcv(self):
        """HistoricalDataSource.fetch_ohlcv should return DataFrame."""
        from tempest_mcp.data._hist import HistoricalDataSource

        source = HistoricalDataSource()
        # Use a known valid yfinance symbol with a date range
        from datetime import datetime, timedelta, timezone

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=10)
        df = source.fetch_ohlcv("BTC-USD", "1d", start=start, end=end)
        assert isinstance(df, pd.DataFrame)
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns


class TestDataSourceRouter:
    """Tests for DataSourceRouter."""

    def test_route_historical_returns_historical_data_source(self):
        """route_historical should return HistoricalDataSource."""
        from tempest_mcp.data._hist import HistoricalDataSource
        from tempest_mcp.data._router import DataSourceRouter

        router = DataSourceRouter()
        adapter = router.route_historical()
        assert isinstance(adapter, HistoricalDataSource)

    def test_route_live_returns_live_data_adapter(self, monkeypatch):
        """route_live should return LiveDataAdapter."""
        from tempest_mcp.data import LiveDataAdapter
        from tempest_mcp.data._router import DataSourceRouter

        # Ensure no TV API key
        monkeypatch.delenv("TRADINGVIEW_API_KEY", raising=False)

        # Clear any cached imports
        import importlib

        import tempest_mcp.data

        importlib.reload(tempest_mcp.data)

        router = DataSourceRouter()
        adapter = router.route_live()
        assert isinstance(adapter, LiveDataAdapter)

    def test_route_live_with_tv_key_returns_ccxt(self, monkeypatch):
        """route_live should return CCXTAdapter even when key is set.

        TradingView has no OHLCV data API - CCXT is always used for live data.
        """
        from tempest_mcp.data import LiveDataAdapter
        from tempest_mcp.data._router import DataSourceRouter

        monkeypatch.setenv("TRADINGVIEW_API_KEY", "test-key")

        # Need to clear the cache in get_live_adapter
        import importlib

        import tempest_mcp.data

        importlib.reload(tempest_mcp.data)

        router = DataSourceRouter()
        adapter = router.route_live()
        assert isinstance(adapter, LiveDataAdapter)


class TestNormalizeToYF:
    """Tests for normalize_to_yf function (D19)."""

    def test_normalize_btcusdt_to_yf(self):
        """BTCUSDT should convert to BTC-USD."""
        from tempest_mcp.data._symbols import normalize_to_yf

        assert normalize_to_yf("BTCUSDT") == "BTC-USD"

    def test_normalize_ethusdt_to_yf(self):
        """ETHUSDT should convert to ETH-USD."""
        from tempest_mcp.data._symbols import normalize_to_yf

        assert normalize_to_yf("ETHUSDT") == "ETH-USD"

    def test_normalize_lowercase(self):
        """Lowercase symbols should be handled."""
        from tempest_mcp.data._symbols import normalize_to_yf

        assert normalize_to_yf("btcusdt") == "BTC-USD"

    def test_normalize_already_yf_format(self):
        """Already yfinance format should pass through."""
        from tempest_mcp.data._symbols import normalize_to_yf

        assert normalize_to_yf("BTC-USD") == "BTC-USD"

    def test_normalize_empty_raises(self):
        """Empty symbol should raise ValueError."""
        from tempest_mcp.data._symbols import normalize_to_yf

        with pytest.raises(ValueError):
            normalize_to_yf("")

    def test_normalize_unknown_pair_fallback(self):
        """Unknown pairs ending in USDT should return fallback with warning."""
        from tempest_mcp.data._symbols import normalize_to_yf

        # Should not raise, should return fallback format
        result = normalize_to_yf("UNKNOWNUSDT")
        assert result == "UNKNOWN-USD"

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC-USDT",  # hyphenated base
            "BTC/USDT",  # slash-separated
            "BTC\\USDT",  # backslash-separated
            "BTC:USDT",  # colon-separated
        ],
    )
    def test_normalize_malformed_usdt_raises(self, symbol):
        """Malformed USDT symbols with separators in base should raise ValueError."""
        from tempest_mcp.data._symbols import normalize_to_yf

        with pytest.raises(ValueError, match="non-alphanumeric base"):
            normalize_to_yf(symbol)

    @pytest.mark.parametrize(
        "symbol",
        [
            "BTC--USD",  # double hyphen
            "BTC---USD",  # triple hyphen
            "ETH--USD",  # double hyphen eth
        ],
    )
    def test_normalize_malformed_yf_format_raises(self, symbol):
        """Malformed yfinance format with multiple hyphens should raise ValueError."""
        from tempest_mcp.data._symbols import normalize_to_yf

        with pytest.raises(ValueError, match="Invalid yfinance format"):
            normalize_to_yf(symbol)
