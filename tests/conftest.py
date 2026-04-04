"""Test configuration."""

import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


@pytest.fixture
def ccxt_adapter():
    """Create a CCXTAdapter instance for testing."""
    from tempest_mcp.data.ccxt_adapter import CCXTAdapter

    return CCXTAdapter()


@pytest.fixture
def tv_adapter_with_key():
    """Create a TradingViewAdapter with a test API key."""
    from tempest_mcp.data.tv_adapter import TradingViewAdapter

    return TradingViewAdapter(api_key="test-api-key")


@pytest.fixture
def tv_adapter_no_key():
    """Create a TradingViewAdapter without an API key (CCXT fallback mode)."""
    from tempest_mcp.data.tv_adapter import TradingViewAdapter

    return TradingViewAdapter()


@pytest.fixture
def sample_klines():
    dates = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(100)]
    return {
        "timestamp": [d.timestamp() for d in dates],
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(100, 110, 100),
        "volume": np.random.uniform(1000, 5000, 100),
    }


@pytest.fixture
def price_data():
    np.random.seed(42)
    base = 100
    returns = np.random.normal(0.0005, 0.02, 200)
    return base * np.exp(np.cumsum(returns))


@pytest.fixture
def ohlcv_data(price_data):
    np.random.seed(42)
    high = price_data * (1 + np.abs(np.random.normal(0, 0.01, len(price_data))))
    low = price_data * (1 - np.abs(np.random.normal(0, 0.01, len(price_data))))
    volume = np.random.uniform(1000, 5000, len(price_data))
    return {"open": price_data, "high": high, "low": low, "close": price_data, "volume": volume}
