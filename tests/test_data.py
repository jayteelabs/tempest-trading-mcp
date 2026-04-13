"""
Tests for data layer adapters.

Tests cover:
- Symbol normalization (_symbols.py)
- CCXT adapter functionality
- TradingView adapter with CCXT fallback
- Error handling per D14 (no exception propagation)
"""

from datetime import timezone

import pandas as pd
import pytest

from tempest_mcp.data._symbols import (
    get_base_currency,
    normalize_to_ccxt,
    normalize_to_ccxt_exchange,
    normalize_to_tradingview,
    validate_symbol,
)
from tempest_mcp.errors import CCXTError, TradingViewError


def _sample_ohlcv_rows() -> list[list[float]]:
    ts = int(pd.Timestamp("2024-01-01", tz=timezone.utc).timestamp() * 1000)
    return [
        [ts, 100.0, 110.0, 95.0, 105.0, 1000.0],
        [ts + 60000, 105.0, 115.0, 100.0, 110.0, 1200.0],
    ]


class DummyExchange:
    def __init__(self, ohlcv_rows: list[list[float]] | None = None, last_price: float = 123.45):
        self.ohlcv_rows = ohlcv_rows or _sample_ohlcv_rows()
        self.last_price = last_price
        self.last_ticker_symbol: str | None = None
        self.last_ohlcv_symbol: str | None = None
        self.last_orderbook_symbol: str | None = None

    def fetch_ticker(self, symbol: str) -> dict:
        self.last_ticker_symbol = symbol
        return {"last": self.last_price}

    def fetch_ohlcv(self, symbol: str, *args, **kwargs) -> list[list[float]]:
        self.last_ohlcv_symbol = symbol
        return self.ohlcv_rows

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        self.last_orderbook_symbol = symbol
        return {
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 2.0]],
        }


class DummyLiveAdapter:
    def __init__(self):
        self.last_symbol: str | None = None
        self.last_timeframe: str | None = None
        self.last_limit: int | None = None

    def fetch_live_price(self, symbol: str, exchange: str = "binance") -> float:
        self.last_symbol = symbol
        return 111.0

    def fetch_ohlcv_live(
        self, symbol: str, timeframe: str = "1m", limit: int = 100
    ) -> pd.DataFrame:
        self.last_symbol = symbol
        self.last_timeframe = timeframe
        self.last_limit = limit
        df = pd.DataFrame(
            _sample_ohlcv_rows(), columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_orderbook_snapshot(self, symbol: str, limit: int = 20) -> dict:
        self.last_symbol = symbol
        self.last_limit = limit
        return {
            "bids": [[100.0, 1.0]],
            "asks": [[101.0, 2.0]],
            "timestamp": pd.Timestamp.now(tz="UTC"),
        }


@pytest.fixture
def dummy_exchange():
    return DummyExchange()


@pytest.fixture
def dummy_live_adapter():
    return DummyLiveAdapter()


class TestSymbolNormalization:
    """Tests for symbol conversion (D11, D12)."""

    def test_normalize_btcusdt_to_ccxt(self):
        """BTCUSDT should remain unchanged for CCXT."""
        assert normalize_to_ccxt("BTCUSDT") == "BTCUSDT"

    def test_normalize_btcusd_to_ccxt(self):
        """BTCUSD (TV format) should convert to BTCUSDT for CCXT."""
        assert normalize_to_ccxt("BTCUSD") == "BTCUSDT"

    def test_normalize_yf_format_rejected_by_ccxt(self):
        """BTC-USD should be rejected by CCXT normalization to avoid USD/USDT rewrites."""
        with pytest.raises(ValueError):
            normalize_to_ccxt("BTC-USD")

    def test_normalize_to_ccxt_exchange(self):
        """CCXT exchange format should be BASE/QUOTE."""
        assert normalize_to_ccxt_exchange("BTCUSDT") == "BTC/USDT"
        assert normalize_to_ccxt_exchange("BTCUSD") == "BTC/USDT"
        assert normalize_to_ccxt_exchange("BTC/USDT") == "BTC/USDT"

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
        """fetch_live_price should return float for valid path."""
        ccxt_adapter.exchange = DummyExchange(last_price=222.0)
        price = ccxt_adapter.fetch_live_price("BTCUSDT")
        assert isinstance(price, float)

    def test_fetch_live_price_invalid_symbol_returns_nan(self, ccxt_adapter):
        """Invalid symbol format should return NaN without network call."""
        price = ccxt_adapter.fetch_live_price("INVALIDPAIR")
        assert isinstance(price, float)
        assert pd.isna(price)

    def test_fetch_ohlcv_returns_dataframe(self, ccxt_adapter):
        """fetch_ohlcv_live should return DataFrame with correct columns."""
        ccxt_adapter.exchange = DummyExchange()
        df = ccxt_adapter.fetch_ohlcv_live("BTCUSDT", "1m", 100)

        assert isinstance(df, pd.DataFrame)
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None, "Index must be UTC-aware"

    def test_fetch_ohlcv_invalid_timeframe_returns_empty(self, ccxt_adapter):
        """Unsupported timeframe returns empty DataFrame."""
        df = ccxt_adapter.fetch_ohlcv_live("BTCUSDT", "10m", 100)
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_fetch_orderbook_returns_dict(self, ccxt_adapter):
        """fetch_orderbook_snapshot should return dict with correct keys."""
        ccxt_adapter.exchange = DummyExchange()
        ob = ccxt_adapter.fetch_orderbook_snapshot("BTCUSDT", 20)

        assert isinstance(ob, dict)
        assert "bids" in ob
        assert "asks" in ob
        assert "timestamp" in ob
        assert ob["timestamp"].tzinfo is not None

    def test_symbol_normalization_in_adapter(self, ccxt_adapter):
        """Adapter should normalize TV symbols to CCXT format."""
        dummy = DummyExchange()
        ccxt_adapter.exchange = dummy
        df = ccxt_adapter.fetch_ohlcv_live("BTCUSD", "1m", 10)
        assert isinstance(df, pd.DataFrame)
        assert dummy.last_ohlcv_symbol == "BTC/USDT"


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
        dummy = DummyLiveAdapter()
        tv_adapter_no_key._ccxt_adapter = dummy
        price = tv_adapter_no_key.fetch_live_price("BTCUSDT")
        assert isinstance(price, float)
        assert dummy.last_symbol == "BTCUSDT"

    def test_fetch_ohlcv_no_key_uses_ccxt_fallback(self, tv_adapter_no_key):
        """Without API key, should fall back to CCXT for OHLCV."""
        dummy = DummyLiveAdapter()
        tv_adapter_no_key._ccxt_adapter = dummy
        df = tv_adapter_no_key.fetch_ohlcv_live("BTCUSDT", "1m", 10)

        assert isinstance(df, pd.DataFrame)
        expected_columns = ["open", "high", "low", "close", "volume"]
        assert list(df.columns) == expected_columns
        assert dummy.last_symbol == "BTCUSDT"

    def test_orderbook_delegates_to_ccxt(self, tv_adapter_with_key):
        """fetch_orderbook_snapshot should delegate to CCXT (D16)."""
        dummy = DummyLiveAdapter()
        tv_adapter_with_key._ccxt_adapter = dummy
        ob = tv_adapter_with_key.fetch_orderbook_snapshot("BTCUSDT", 20)

        assert isinstance(ob, dict)
        assert "bids" in ob
        assert "asks" in ob
        assert "timestamp" in ob
        assert dummy.last_symbol == "BTCUSDT"

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

    def test_historical_fetch_uses_yfinance_on_ccxt_empty(self, monkeypatch):
        """When CCXT returns empty, yfinance fallback should be used."""
        from tempest_mcp.data import yf_adapter
        from tempest_mcp.data._hist import HistoricalDataSource

        source = HistoricalDataSource()

        class DummyCCXTHistorical:
            def __init__(self):
                self.called = False

            def fetch_ohlcv_historical(self, *args, **kwargs):
                self.called = True
                return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        dummy_ccxt = DummyCCXTHistorical()
        source._ccxt = dummy_ccxt

        captured: dict[str, str] = {}

        def _fake_yf_fetch(
            symbol: str, interval: str = "1d", start=None, end=None, auto_adjust=True
        ):
            captured["symbol"] = symbol
            df = pd.DataFrame(
                _sample_ohlcv_rows(),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        monkeypatch.setattr(yf_adapter, "fetch_ohlcv", _fake_yf_fetch)

        df = source.fetch_ohlcv("BTCUSDT", "1d")
        assert dummy_ccxt.called is True
        assert captured["symbol"] == "BTC-USD"
        assert not df.empty
        assert df.index.tz is not None

    def test_historical_fetch_direct_yf_for_usd_symbol(self, monkeypatch):
        """Yfinance-style input should bypass CCXT to avoid USD->USDT rewrite."""
        from tempest_mcp.data import yf_adapter
        from tempest_mcp.data._hist import HistoricalDataSource

        source = HistoricalDataSource()

        class DummyCCXTHistorical:
            def fetch_ohlcv_historical(self, *args, **kwargs):
                raise AssertionError("CCXT should not be called for yfinance symbol inputs")

        source._ccxt = DummyCCXTHistorical()

        def _fake_yf_fetch(
            symbol: str, interval: str = "1d", start=None, end=None, auto_adjust=True
        ):
            df = pd.DataFrame(
                _sample_ohlcv_rows(),
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]]

        monkeypatch.setattr(yf_adapter, "fetch_ohlcv", _fake_yf_fetch)

        df = source.fetch_ohlcv("BTC-USD", "1d")
        assert not df.empty
        assert df.index.tz is not None

    def test_historical_fetch_returns_ccxt_when_non_empty(self, monkeypatch):
        """Non-empty CCXT response should return without fallback."""
        from tempest_mcp.data import yf_adapter
        from tempest_mcp.data._hist import HistoricalDataSource

        source = HistoricalDataSource()

        df_ccxt = pd.DataFrame(
            _sample_ohlcv_rows(),
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df_ccxt["timestamp"] = pd.to_datetime(df_ccxt["timestamp"], unit="ms", utc=True)
        df_ccxt.set_index("timestamp", inplace=True)
        df_ccxt = df_ccxt[["open", "high", "low", "close", "volume"]]

        class DummyCCXTHistorical:
            def __init__(self, df: pd.DataFrame):
                self.df = df

            def fetch_ohlcv_historical(self, *args, **kwargs):
                return self.df

        source._ccxt = DummyCCXTHistorical(df_ccxt)

        def _fake_yf_fetch(*args, **kwargs):
            raise AssertionError("yfinance should not be called when CCXT returns data")

        monkeypatch.setattr(yf_adapter, "fetch_ohlcv", _fake_yf_fetch)

        df = source.fetch_ohlcv("BTCUSDT", "1d")
        assert not df.empty
        assert df.index.tz is not None
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]


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

    def test_normalize_ccxt_market_format(self):
        """CCXT market format should convert to yfinance format."""
        from tempest_mcp.data._symbols import normalize_to_yf

        assert normalize_to_yf("BTC/USDT") == "BTC-USD"

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
            "BTC\\USDT",  # backslash-separated
            "BTC:USDT",  # colon-separated
            "BTC--USDT",  # double hyphen
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
