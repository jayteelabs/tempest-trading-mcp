"""Tests for data adapters."""
import pytest
from tempest_mcp.data.yf_adapter import YFAdapter
from tempest_mcp.data.ccxt_adapter import CCXTAdapter, CCXTError
from tempest_mcp.models.market import Kline, Ticker, OrderBook, OrderBookLevel

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
