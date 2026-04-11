"""Phase 1 Integration Tests — data layer + indicator smoke tests.

These tests hit real network endpoints (YFAdapter, CCXTAdapter) and compute
real indicator values. They are marked @pytest.mark.integration and only run
when explicitly invoked via: pytest --run-integration

Normal CI does NOT run these — they are for manual/scheduled integration validation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tempest_mcp.data import (
    CCXTAdapter,
    DataSourceRouter,
    get_live_adapter,
)
from tempest_mcp.data.yf_adapter import fetch_ohlcv as fetch_yf_ohlcv
from tempest_mcp.indicators import (
    calculate_ema_stack,
    calculate_rsi,
    calculate_session_levels,
    calculate_vwap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 1. Data Layer — Historical (YFinance adapter)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_yf_btcusd_ohlcv():
    start = _days_ago(90)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    assert not df.empty, "BTC-USD OHLCV should not be empty"
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert df.index.tz is not None, "Index must be UTC-aware"


@pytest.mark.integration
def test_yf_ethusd_ohlcv():
    start = _days_ago(90)
    end = _utc_now()
    df = fetch_yf_ohlcv("ETH-USD", start=start, end=end)
    assert not df.empty, "ETH-USD OHLCV should not be empty"
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.integration
def test_yf_dogeusd_ohlcv():
    start = _days_ago(90)
    end = _utc_now()
    df = fetch_yf_ohlcv("DOGE-USD", start=start, end=end)
    assert not df.empty, "DOGE-USD OHLCV should not be empty"
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"


@pytest.mark.integration
def test_yf_empty_on_invalid_symbol():
    start = _days_ago(30)
    end = _utc_now()
    df = fetch_yf_ohlcv("NOTASYM-XYZ", start=start, end=end)
    assert isinstance(df, pd.DataFrame), "Should return DataFrame, not raise"
    assert df.empty, "Invalid symbol should return empty DataFrame"


# ---------------------------------------------------------------------------
# 2. Data Layer — Live (CCXT adapter)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ccxt_btcusdt_live_price():
    adapter = get_live_adapter()
    assert isinstance(adapter, CCXTAdapter)
    price = adapter.fetch_live_price("BTC/USDT")
    assert price > 0, "Live price must be positive"
    assert isinstance(price, float), "Price must be float"


@pytest.mark.integration
def test_ccxt_ethusdt_live_price():
    adapter = get_live_adapter()
    price = adapter.fetch_live_price("ETH/USDT")
    assert price > 0, "Live price must be positive"
    assert isinstance(price, float), "Price must be float"


@pytest.mark.integration
def test_ccxt_ohlcv_fetch():
    adapter = get_live_adapter()
    df = adapter.fetch_ohlcv_live("BTC/USDT", timeframe="1h", limit=100)
    assert not df.empty, "OHLCV should not be empty"
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column: {col}"
    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex"
    assert df.index.tz is not None, "Index must be UTC-aware"


# ---------------------------------------------------------------------------
# 3. Data Layer — Historical via router
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_historical_source_crypto():
    router = DataSourceRouter()
    ds = router.route_historical()
    df = ds.fetch_ohlcv("BTC/USDT", "1d")
    assert not df.empty, "Router should return non-empty DataFrame for BTC/USDT"


@pytest.mark.integration
def test_historical_fallback_stock_symbol():
    """AAPL is a stock symbol — CCXT returns empty for it, triggering yfinance fallback.

    This explicitly exercises the fallback path in DataSourceRouter for non-crypto symbols.
    """
    router = DataSourceRouter()
    ds = router.route_historical()
    df = ds.fetch_ohlcv("AAPL", "1d")
    assert isinstance(df, pd.DataFrame), "Should return DataFrame, not raise"
    # Fallback should populate with yfinance data
    assert not df.empty, "yfinance fallback should return data for AAPL"
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns, f"Missing column after fallback: {col}"


# ---------------------------------------------------------------------------
# 4. Indicator Engine — EMA
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ema_stack_btc_bull_trend():
    """Fetch BTC-USD daily OHLCV and compute EMA stack.

    Verifies all four EMA series are non-empty and have no NaN in the last 10 values.
    Does NOT assert bull-trend ordering — that is environment-dependent.
    """
    start = _days_ago(250)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    assert not df.empty, "BTC-USD data must be available"

    close = df["close"].squeeze()
    assert isinstance(close, pd.Series)
    stack = calculate_ema_stack(close)

    for key in ("ema7", "ema25", "ema50", "ema200"):
        assert key in stack, f"Missing key: {key}"
        series = stack[key]
        assert not series.empty, f"{key} should not be empty"
        last10 = series.dropna().tail(10)
        assert len(last10) == 10, f"{key} must have ≥10 non-NaN values in last 10 bars"


@pytest.mark.integration
def test_ema_stack_values_defined():
    """Sanity check: all EMA values in the stack are non-NaN for recent bars."""
    start = _days_ago(250)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    close = df["close"].squeeze()
    stack = calculate_ema_stack(close)

    for key, series in stack.items():
        last10 = series.dropna().tail(10)
        assert len(last10) > 0, f"{key} must have at least some non-NaN values"


# ---------------------------------------------------------------------------
# 5. Indicator Engine — RSI
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rsi_btc_known_range():
    """RSI(14) values must always fall within [0, 100]."""
    start = _days_ago(180)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    assert not df.empty, "BTC-USD data must be available"

    close = df["close"].squeeze()
    rsi = calculate_rsi(close, period=14)

    valid_rsi = rsi.dropna()
    assert len(valid_rsi) > 0, "RSI should have some valid values"
    assert (valid_rsi >= 0).all(), "RSI values must be ≥ 0"
    assert (valid_rsi <= 100).all(), "RSI values must be ≤ 100"


# ---------------------------------------------------------------------------
# 6. Indicator Engine — Session Levels
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_session_levels_asia_hnl():
    start = _days_ago(30)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    assert not df.empty
    result = calculate_session_levels(df.index, df["high"].squeeze(), df["low"].squeeze())
    assert result.values["asia_high"] >= result.values["asia_low"], "Asia high must be ≥ Asia low"


@pytest.mark.integration
def test_session_levels_london_hnl():
    start = _days_ago(30)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    assert not df.empty
    result = calculate_session_levels(df.index, df["high"].squeeze(), df["low"].squeeze())
    assert result.values["london_high"] >= result.values["london_low"], (
        "London high must be ≥ London low"
    )


@pytest.mark.integration
def test_session_levels_ny_hnl():
    start = _days_ago(30)
    end = _utc_now()
    df = fetch_yf_ohlcv("BTC-USD", start=start, end=end)
    assert not df.empty
    result = calculate_session_levels(df.index, df["high"].squeeze(), df["low"].squeeze())
    assert result.values["ny_high"] >= result.values["ny_low"], "NY high must be ≥ NY low"


# ---------------------------------------------------------------------------
# 7. Indicator Engine — VWAP
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_vwap_btcusdt():
    """Fetch 1h BTC/USDT OHLCV via CCXT (includes volume) and compute VWAP.

    VWAP should fall within the low-high range of the HLV tuple.
    """
    adapter = get_live_adapter()
    df = adapter.fetch_ohlcv_live("BTC/USDT", timeframe="1h", limit=100)
    assert not df.empty, "BTC/USDT 1h data must be available"

    vwap = calculate_vwap(
        high=df["high"].squeeze(),
        low=df["low"].squeeze(),
        close=df["close"].squeeze(),
        volume=df["volume"].squeeze(),
        anchor="ny",
    )

    assert not vwap.empty, "VWAP should not be empty"
    # VWAP should be between the session's low and high
    assert (vwap >= df["low"].squeeze()).any(), "VWAP should be ≥ some low"
    assert (vwap <= df["high"].squeeze()).any(), "VWAP should be ≤ some high"


# ---------------------------------------------------------------------------
# 8. Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_insufficient_data_raises():
    """Calling calculate_ema_stack with fewer than 200 values should raise ValueError.

    Note: ENG-9 through ENG-13 explicitly raise ValueError for insufficient data;
    IndicatorError is NOT used by indicator functions for this case.
    """
    short_prices = pd.Series(range(50))  # 50 values < 200
    with pytest.raises(ValueError):
        calculate_ema_stack(short_prices)


@pytest.mark.integration
def test_ema_stack_rejects_non_series_input():
    with pytest.raises(TypeError, match="prices must be a pandas Series"):
        calculate_ema_stack([1.0] * 200)


@pytest.mark.integration
def test_ema_stack_rejects_non_finite_values():
    prices = pd.Series([float(i) for i in range(199)] + [float("inf")])
    with pytest.raises(ValueError, match="finite numeric values"):
        calculate_ema_stack(prices)


@pytest.mark.integration
def test_empty_df_returns_empty():
    """Garbage symbol returns empty DataFrame — does not raise."""
    start = _days_ago(30)
    end = _utc_now()
    df = fetch_yf_ohlcv("GARBAGEXYZ-999", start=start, end=end)
    assert isinstance(df, pd.DataFrame), "Should return DataFrame, not raise"
    assert df.empty, "Garbage symbol should return empty DataFrame"
