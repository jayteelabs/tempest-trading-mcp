"""Tests for data adapters."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from tempest_mcp.data.ccxt_adapter import CCXTAdapter, CCXTError
from tempest_mcp.data.yf_adapter import (
    DataSourceError,
    TempestError,
    YFAdapter,
    YFinanceError,
    _get_empty_ohlcv,
    fetch_ohlcv,
)
from tempest_mcp.models.market import Kline, Ticker


class TestFetchOHLCV:
    """Tests for fetch_ohlcv standalone function."""

    def test_error_hierarchy(self):
        """YFinanceError should inherit from DataSourceError which inherits from TempestError."""
        assert issubclass(YFinanceError, DataSourceError)
        assert issubclass(DataSourceError, TempestError)
        assert issubclass(YFinanceError, TempestError)
        assert issubclass(YFinanceError, Exception)

    def test_yfinance_error_codes_in_3xxx_range(self):
        """YFinanceError codes should be in the 3xxx range per D8."""
        err = YFinanceError("test", 3001)
        assert err.code >= 3000
        assert err.code < 4000

    def test_get_empty_ohlcv_returns_correct_columns(self):
        """Empty DataFrame should have correct OHLCV columns."""
        df = _get_empty_ohlcv()
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.empty

    def test_invalid_symbol_returns_empty_dataframe(self):
        """Invalid symbol should return empty DataFrame with correct columns, no exception."""
        df = fetch_ohlcv("INVALID_SYMBOL_XYZ123")
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_btc_usd_fetch_success(self):
        """BTC-USD should return DataFrame with correct columns."""
        df = fetch_ohlcv("BTC-USD", interval="1d")
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_eth_usd_fetch_success(self):
        """ETH-USD should return DataFrame with correct columns."""
        df = fetch_ohlcv("ETH-USD", interval="1d")
        assert not df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_date_range_filtering(self):
        """Date range parameters should be respected."""
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        df = fetch_ohlcv("BTC-USD", interval="1d", start=start, end=end)
        assert not df.empty
        # Should have ~7-8 rows for 7 days
        assert len(df) <= 10

    def test_utc_index(self):
        """Returned DataFrame index should be UTC-aware."""
        df = fetch_ohlcv("BTC-USD", interval="1d")
        if not df.empty:
            assert df.index.tz is not None
            assert str(df.index.tz) == "UTC"

    def test_auto_adjust_true(self):
        """auto_adjust=True should work."""
        df = fetch_ohlcv("BTC-USD", interval="1d", auto_adjust=True)
        assert not df.empty

    def test_auto_adjust_false(self):
        """auto_adjust=False should work."""
        df = fetch_ohlcv("BTC-USD", interval="1d", auto_adjust=False)
        assert not df.empty

    def test_interval_mapping(self):
        """Different intervals should work."""
        for interval in ["1d", "1wk", "1mo"]:
            df = fetch_ohlcv("BTC-USD", interval=interval)
            # Should either get data or empty DataFrame
            assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    @patch("tempest_mcp.data.yf_adapter.yf.download")
    def test_rate_limit_retry(self, mock_download):
        """Rate limit should trigger retry and eventually return empty DataFrame."""
        import yfinance as yf

        # First two calls raise rate limit, third returns empty
        mock_download.side_effect = [
            yf.exceptions.YFRateLimitError(),
            yf.exceptions.YFRateLimitError(),
            pd.DataFrame(),
        ]
        df = fetch_ohlcv("BTC-USD")
        assert df.empty
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert mock_download.call_count == 3

    @patch("tempest_mcp.data.yf_adapter.yf.download")
    def test_network_error_retry(self, mock_download):
        """Network error should trigger retry and eventually return empty DataFrame."""
        mock_download.side_effect = [
            OSError("Network error"),
            OSError("Network error"),
            pd.DataFrame(),
        ]
        df = fetch_ohlcv("BTC-USD")
        assert df.empty
        assert mock_download.call_count == 3


class TestYFAdapter:
    def test_convert_symbol(self):
        adapter = YFAdapter()
        assert adapter._convert_symbol("BTC/USDT") == "BTC-USD"
        assert adapter._convert_symbol("ETH/USDT") == "ETH-USD"

    def test_convert_timeframe(self):
        adapter = YFAdapter()
        assert adapter._convert_timeframe("1h") == "1h"
        assert adapter._convert_timeframe("1d") == "1d"


class TestCCXTAdapter:
    def test_initialization(self):
        adapter = CCXTAdapter(exchange="binance", timeout=60)
        assert adapter.exchange == "binance"
        assert adapter.timeout == 60

    def test_invalid_exchange(self):
        with pytest.raises(CCXTError):
            CCXTAdapter(exchange="invalid")


class TestModels:
    def test_kline(self):
        k = Kline(timestamp=1.0, open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0)
        assert k.close == 102.0

    def test_ticker(self):
        t = Ticker(symbol="BTC/USDT", exchange="binance", price=50000.0, timestamp=1.0)
        assert t.price == 50000.0
