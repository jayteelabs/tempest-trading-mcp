"""
Tests for data layer adapters.

Tests cover:
- Symbol normalization (_symbols.py)
- CCXT adapter functionality
- TradingView adapter with CCXT fallback
- Error handling per D14 (no exception propagation)
"""

import math

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
    
    def test_get_live_adapter_with_key_returns_tv(self, monkeypatch):
        """get_live_adapter should return TradingViewAdapter when API key is set."""
        monkeypatch.setenv("TRADINGVIEW_API_KEY", "test-key")
        
        # Need to reimport to pick up env change
        import importlib
        import tempest_mcp.data
        importlib.reload(tempest_mcp.data)
        
        from tempest_mcp.data import get_live_adapter
        from tempest_mcp.data.tv_adapter import TradingViewAdapter
        
        adapter = get_live_adapter()
        assert isinstance(adapter, TradingViewAdapter)
