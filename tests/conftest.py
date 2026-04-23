"""Test configuration."""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))


def _assert_worktree_package_resolution() -> None:
    """Fail fast if pytest resolves tempest_mcp from a sibling checkout."""
    import tempest_mcp

    package_path = Path(tempest_mcp.__file__).resolve()

    try:
        package_path.relative_to(SRC_ROOT)
    except ValueError as exc:
        raise pytest.UsageError(
            "pytest imported tempest_mcp from "
            f"{package_path} instead of this worktree's source tree ({SRC_ROOT}). "
            "Run repo-local validation from this checkout, e.g. `uv run pytest ...`."
        ) from exc


_assert_worktree_package_resolution()


def pytest_addoption(parser):
    """Register custom pytest options for integration test runs."""
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled."""
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


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


@pytest.fixture(scope="session")
def network_available():
    """Check if network is available by attempting connection to DNS server."""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False
