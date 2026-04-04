"""
Pytest fixtures and configuration for Tempest MCP tests.
"""

import os
import pytest
import vcr

# Set test environment
os.environ.setdefault("TRADINGVIEW_API_KEY", "")


@pytest.fixture
def vcr_config():
    """VCR configuration for recording/replaying HTTP requests."""
    return {
        "record_mode": "new_episodes",
        "match_on": ["method", "scheme", "host", "port", "path", "query"],
        "filter_headers": ["authorization", "api-key"],
        "cassette_library_dir": "tests/cassettes",
    }


@pytest.fixture
def ccxt_adapter():
    """Create CCXT adapter for testing."""
    from tempest_mcp.data.ccxt_adapter import CCXTAdapter
    return CCXTAdapter()


@pytest.fixture
def tv_adapter_with_key():
    """Create TradingView adapter with mock API key."""
    from tempest_mcp.data.tv_adapter import TradingViewAdapter
    return TradingViewAdapter(api_key="test-api-key")


@pytest.fixture
def tv_adapter_no_key():
    """Create TradingView adapter without API key (tests CCXT fallback)."""
    from tempest_mcp.data.tv_adapter import TradingViewAdapter
    return TradingViewAdapter(api_key=None)
