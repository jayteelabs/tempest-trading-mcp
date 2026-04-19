"""Phase 3 analytical tools integration suite (ENG-38).

This module exercises the ENG-37 analytical tool handlers against real CCXT-fetched
market data and validates the output contracts are sane, stable, and not just
passing synthetic fixtures.

Scope (ENG-38):
- BTCUSDT-centric for reliability/low noise
- Both 1h and 4h timeframe validation
- Bounded retry on transient upstream CCXT instability → skip with loud warning
- Hard fail on contract/sanity violations once data is fetched

Run with:  uv run pytest --run-integration tests/test_phase3_integration.py -v
"""

from __future__ import annotations

import asyncio
import math
import warnings

import ccxt
import numpy as np
import pandas as pd
import pytest

from tempest_mcp.data import get_live_adapter
from tempest_mcp.tools.analytical_tools import (
    calculate_fibonacci,
    calculate_tpo,
    detect_elliot_wave,
    get_market_structure,
)

# =============================================================================
# Constants
# =============================================================================

CANONICAL_SYMBOL = "BTCUSDT"
MAX_RETRIES = 3
TRANSIENT_FETCH_EXCEPTIONS = (ccxt.NetworkError, ConnectionError, TimeoutError)


# =============================================================================
# Helpers — data fetching with bounded retry
# =============================================================================


def _fetch_live_ohlcv_with_retry(
    symbol: str,
    timeframe: str,
    limit: int = 300,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """Fetch live OHLCV with bounded retry for transient upstream failures.

    Treats empty DataFrames as transient (CCXT can return empty df on rate-limit
    or network hiccups without throwing). Retries bounded times, then skips with
    a loud warning if no usable data arrives.
    Skips with a loud diagnostic warning only when failure is clearly transient
    (exchange unavailable, rate-limited, timeout, network instability, empty df).
    Hard-fails on contract/sanity violations once data is fetched.
    """
    adapter = get_live_adapter()
    last_transient: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = adapter.fetch_ohlcv_live(symbol, timeframe=timeframe, limit=limit)
            # Treat empty DataFrame as transient — CCXT can return empty df on
            # rate-limit or network hiccups without throwing an exception.
            if df.empty:
                raise _EmptyDataFrameTranscient("fetch_ohlcv_live returned empty DataFrame")
            _assert_ohlcv_contract(df, symbol, timeframe)
            return df
        except AssertionError:
            raise
        except _EmptyDataFrameTranscient as exc:
            last_transient = str(exc)
            if attempt < max_retries:
                warnings.warn(
                    f"[{symbol} {timeframe}] Attempt {attempt}/{max_retries} failed: "
                    f"empty DataFrame (transient). Retrying...",
                    stacklevel=2,
                )
            continue
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, TRANSIENT_FETCH_EXCEPTIONS):
                raise

            last_transient = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                warnings.warn(
                    f"[{symbol} {timeframe}] Attempt {attempt}/{max_retries} failed: "
                    f"{type(exc).__name__}: {exc}. Retrying...",
                    stacklevel=2,
                )
            continue

    raise pytest.skip.Exception(
        f"[{symbol} {timeframe}] Bounded retries ({max_retries}) exhausted. "
        f"Skipping due to transient upstream CCXT instability: {last_transient or 'unknown'}"
    )


class _EmptyDataFrameTranscient(Exception):
    """Sentinel marking an empty DataFrame from fetch_ohlcv_live as transient."""
    pass


def _assert_ohlcv_contract(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    """Validate fetched OHLCV DataFrame meets the live-data contract.

    Hard-fails on any contract violation.
    """
    assert not df.empty, (
        f"[{symbol} {timeframe}] OHLCV DataFrame must be non-empty."
    )
    assert len(df) >= 10, (
        f"[{symbol} {timeframe}] OHLCV DataFrame must have at least 10 rows for "
        f"analytical processing, got {len(df)}."
    )
    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference(df.columns)
    assert not missing, (
        f"[{symbol} {timeframe}] Missing required columns: {', '.join(sorted(missing))}."
    )

    assert isinstance(df.index, pd.DatetimeIndex), "Index must be DatetimeIndex."
    assert df.index.tz is not None, "DatetimeIndex must be UTC-aware."
    tz_name = getattr(df.index.tz, "key", None) or getattr(df.index.tz, "zone", None) or str(df.index.tz)
    assert tz_name == "UTC", f"DatetimeIndex must be UTC, got {tz_name!r}."
    assert df.index.is_monotonic_increasing, "DatetimeIndex must be monotonic increasing."
    assert not df.index.has_duplicates, "DatetimeIndex must not have duplicates."

    for col in required_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), f"Column '{col}' must be numeric."
        assert df[col].notna().all(), f"Column '{col}' must not contain missing values."
        assert np.isfinite(df[col].to_numpy()).all(), f"Column '{col}' must have only finite values."

    assert (df["high"] >= df["low"]).all(), (
        f"[{symbol} {timeframe}] All rows must satisfy high >= low."
    )
    assert (df["volume"] >= 0).all(), (
        f"[{symbol} {timeframe}] Volume must be non-negative."
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def network_available():
    """Check if network is available."""
    import socket

    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


@pytest.fixture(scope="module")
def btcusdt_1h_ohlcv(network_available):
    """Fetch BTCUSDT 1h OHLCV with bounded retry."""
    if not network_available:
        pytest.skip("Network not available")
    return _fetch_live_ohlcv_with_retry(CANONICAL_SYMBOL, "1h", limit=300)


@pytest.fixture(scope="module")
def btcusdt_4h_ohlcv(network_available):
    """Fetch BTCUSDT 4h OHLCV with bounded retry."""
    if not network_available:
        pytest.skip("Network not available")
    # 4h: 100 bars ~= 16 days, sufficient for analytical tools
    return _fetch_live_ohlcv_with_retry(CANONICAL_SYMBOL, "4h", limit=100)


# =============================================================================
# Helper — run async handler in sync context
# =============================================================================


def _run_async(coro):
    """Run an async coroutine synchronously (for use in sync test methods)."""
    return asyncio.run(coro)


# =============================================================================
# Helper — contract assertions for analytical tool outputs
# =============================================================================


def _assert_fibonacci_contract(result: dict, symbol: str, timeframe: str) -> None:
    """Assert calculate_fibonacci output contract.

    Hard-fails on any contract violation.
    """
    if asyncio.iscoroutine(result):
        result = _run_async(result)

    assert result.get("success") is True, (
        f"[{symbol} {timeframe}] calculate_fibonacci must succeed with live data. "
        f"Error: {result.get('error')}"
    )
    data = result["data"]
    assert data["tool"] == "calculate_fibonacci"
    assert data["symbol"] == symbol
    assert data["timeframe"] == timeframe
    assert "window" in data
    window = data["window"]
    assert "start_at_utc" in window
    assert "end_at_utc" in window
    assert "estimated_bars" in window
    assert "exchange" in window
    assert "fib_levels" in data
    assert "count" in data
    assert data["count"] == len(data["fib_levels"])
    assert isinstance(data["fib_levels"], list)
    for level in data["fib_levels"]:
        assert "level_type" in level
        assert "level_ratio" in level
        assert "price" in level
        assert "swing_high" in level
        assert "swing_low" in level
        assert level["price"] > 0
        assert level["swing_high"] > level["swing_low"]


def _assert_tpo_contract(result: dict, symbol: str, timeframe: str) -> None:
    """Assert calculate_tpo output contract.

    Hard-fails on any contract violation.
    """
    if asyncio.iscoroutine(result):
        result = _run_async(result)

    assert result.get("success") is True, (
        f"[{symbol} {timeframe}] calculate_tpo must succeed with live data. "
        f"Error: {result.get('error')}"
    )
    data = result["data"]
    assert data["tool"] == "calculate_tpo"
    assert data["symbol"] == symbol
    assert data["timeframe"] == timeframe
    assert "window" in data
    assert "session" in data
    session = data["session"]
    assert "row_size" in session
    assert "marker_count" in session
    assert "poc_price" in session
    assert "vah_price" in session
    assert "val_price" in session
    assert session["val_price"] <= session["poc_price"] <= session["vah_price"], (
        f"POC {session['poc_price']} must be between VAL {session['val_price']} "
        f"and VAH {session['vah_price']}"
    )
    assert "tpo_rows" in data
    assert "count" in data
    assert data["count"] == len(data["tpo_rows"])
    assert isinstance(data["tpo_rows"], list)
    if data["tpo_rows"]:
        indices = [row["row_index"] for row in data["tpo_rows"]]
        assert indices == sorted(indices), "TPO rows must be ordered by row_index"


def _assert_elliot_wave_contract(result: dict, symbol: str, timeframe: str) -> None:
    """Assert detect_elliot_wave output contract.

    Hard-fails on any contract violation.
    """
    if asyncio.iscoroutine(result):
        result = _run_async(result)

    assert result.get("success") is True, (
        f"[{symbol} {timeframe}] detect_elliot_wave must succeed with live data. "
        f"Error: {result.get('error')}"
    )
    data = result["data"]
    assert data["tool"] == "detect_elliot_wave"
    assert data["symbol"] == symbol
    assert data["timeframe"] == timeframe
    assert "window" in data
    assert "parameters" in data
    params = data["parameters"]
    assert "swing_window" in params
    assert "min_swing_pct" in params
    assert "wave2_retrace_band" in params
    assert "wave3_extension_min" in params
    assert "wave4_retrace_max" in params
    assert "waveb_retrace_band" in params
    assert "wavec_extension_min" in params
    assert "degree_thresholds" in params
    assert "include_rejected" in params
    assert "wave_sequences" in data
    assert "count" in data
    assert data["count"] == len(data["wave_sequences"])
    assert isinstance(data["wave_sequences"], list)
    for seq in data["wave_sequences"]:
        assert "sequence_id" in seq
        assert "sequence_type" in seq
        assert "wave_label" in seq
        assert "segment_order" in seq
        assert "direction" in seq
        assert "degree" in seq
        assert "is_rule_compliant" in seq
        assert "is_accepted_sequence" in seq


def _assert_market_structure_contract(result: dict, symbol: str, timeframe: str) -> None:
    """Assert get_market_structure output contract.

    Hard-fails on any contract violation.
    """
    if asyncio.iscoroutine(result):
        result = _run_async(result)

    assert result.get("success") is True, (
        f"[{symbol} {timeframe}] get_market_structure must succeed with live data. "
        f"Error: {result.get('error')}"
    )
    data = result["data"]
    assert data["tool"] == "get_market_structure"
    assert data["symbol"] == symbol
    assert data["timeframe"] == timeframe
    assert "window" in data
    assert "insufficient_data" in data
    assert "summary" in data
    if not data["insufficient_data"]:
        summary = data["summary"]
        assert summary is not None
        assert "summary_label" in summary
        assert "structure_classification" in summary
        assert "structure_trend_state" in summary
        assert "adx" in summary
        assert "plus_di" in summary
        assert "minus_di" in summary
        if summary["adx"] is not None:
            assert 0 <= summary["adx"] <= 100, (
                f"ADX {summary['adx']} must be in [0, 100]"
            )
        if summary["di_spread"] is not None:
            # DI spread = plus_di - minus_di; can be negative in bearish markets
            assert math.isfinite(summary["di_spread"]), (
                f"DI spread {summary['di_spread']} must be finite"
            )


# =============================================================================
# Tests — calculate_fibonacci
# =============================================================================


@pytest.mark.integration
class TestCalculateFibonacciIntegration:
    """Live integration tests for calculate_fibonacci."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_fibonacci_retracement_1h(self, btcusdt_1h_ohlcv):
        """calculate_fibonacci retracement mode returns valid contract on 1h BTCUSDT."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()
        swing_high = float(df["high"].max())
        swing_low = float(df["low"].min())

        result = _run_async(calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
            swing_high=swing_high,
            swing_low=swing_low,
        ))

        _assert_fibonacci_contract(result, "BTC/USDT", "1h")
        # retracement levels must be between swing_low and swing_high
        for level in result["data"]["fib_levels"]:
            assert level["swing_low"] <= level["price"] <= level["swing_high"], (
                f"Fib price {level['price']} must be between swing_low {level['swing_low']} "
                f"and swing_high {level['swing_high']}"
            )

    def test_fibonacci_retracement_4h(self, btcusdt_4h_ohlcv):
        """calculate_fibonacci retracement mode returns valid contract on 4h BTCUSDT."""
        df = btcusdt_4h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()
        swing_high = float(df["high"].max())
        swing_low = float(df["low"].min())

        result = _run_async(calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="4h",
            start_at=start_at,
            end_at=end_at,
            swing_high=swing_high,
            swing_low=swing_low,
        ))

        _assert_fibonacci_contract(result, "BTC/USDT", "4h")

    def test_fibonacci_extension_1h(self, btcusdt_1h_ohlcv):
        """calculate_fibonacci extension mode returns valid contract on 1h BTCUSDT."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()
        swing_high = float(df["high"].max())
        swing_low = float(df["low"].min())

        result = _run_async(calculate_fibonacci(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
            swing_high=swing_high,
            swing_low=swing_low,
            output_mode="extension",
            trend_direction="bullish",
        ))

        assert result.get("success") is True, f"Extension mode failed: {result.get('error')}"
        data = result["data"]
        assert data["output_mode"] == "extension"
        assert data["trend_direction"] == "bullish"
        assert "fib_levels" in data
        for level in data["fib_levels"]:
            assert level["level_type"] == "extension"


# =============================================================================
# Tests — calculate_tpo
# =============================================================================


@pytest.mark.integration
class TestCalculateTpoIntegration:
    """Live integration tests for calculate_tpo."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_tpo_single_session_1h(self, btcusdt_1h_ohlcv):
        """calculate_tpo returns valid contract on 1h BTCUSDT single session."""
        df = btcusdt_1h_ohlcv
        # Use first 12 bars to ensure single session (< 4h gap)
        narrow_df = df.iloc[:12]
        start_at = narrow_df.index[0].to_pydatetime().isoformat()
        end_at = narrow_df.index[-1].to_pydatetime().isoformat()
        price_range = float(narrow_df["high"].max() - narrow_df["low"].min())
        row_size = price_range / 20  # 20 rows across the range

        result = _run_async(calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
            row_size=row_size,
        ))

        _assert_tpo_contract(result, "BTC/USDT", "1h")

    def test_tpo_value_area_ordering(self, btcusdt_1h_ohlcv):
        """calculate_tpo VAH >= POC >= VAL (value area ordering)."""
        df = btcusdt_1h_ohlcv
        narrow_df = df.iloc[:12]
        start_at = narrow_df.index[0].to_pydatetime().isoformat()
        end_at = narrow_df.index[-1].to_pydatetime().isoformat()
        price_range = float(narrow_df["high"].max() - narrow_df["low"].min())
        row_size = price_range / 20

        result = _run_async(calculate_tpo(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
            row_size=row_size,
            value_area_pct=0.70,
        ))

        session = result["data"]["session"]
        assert session["val_price"] <= session["poc_price"] <= session["vah_price"], (
            f"Value area ordering violated: VAL={session['val_price']} <= "
            f"POC={session['poc_price']} <= VAH={session['vah_price']}"
        )


# =============================================================================
# Tests — detect_elliot_wave
# =============================================================================


@pytest.mark.integration
class TestDetectElliotWaveIntegration:
    """Live integration tests for detect_elliot_wave."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_elliot_wave_1h(self, btcusdt_1h_ohlcv):
        """detect_elliot_wave returns valid contract on 1h BTCUSDT."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
        ))

        _assert_elliot_wave_contract(result, "BTC/USDT", "1h")

    def test_elliot_wave_4h(self, btcusdt_4h_ohlcv):
        """detect_elliot_wave returns valid contract on 4h BTCUSDT."""
        df = btcusdt_4h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="4h",
            start_at=start_at,
            end_at=end_at,
        ))

        _assert_elliot_wave_contract(result, "BTC/USDT", "4h")

    def test_elliot_wave_wave_directions(self, btcusdt_1h_ohlcv):
        """detect_elliot_wave wave_sequences direction is 'bullish' or 'bearish'."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(detect_elliot_wave(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
        ))

        for seq in result["data"]["wave_sequences"]:
            assert seq["direction"] in ("bullish", "bearish"), (
                f"Wave direction must be 'bullish' or 'bearish', got {seq['direction']}"
            )


# =============================================================================
# Tests — get_market_structure
# =============================================================================


@pytest.mark.integration
class TestGetMarketStructureIntegration:
    """Live integration tests for get_market_structure."""

    @pytest.fixture(autouse=True)
    def check_network(self, network_available):
        if not network_available:
            pytest.skip("Network not available")

    def test_market_structure_1h(self, btcusdt_1h_ohlcv):
        """get_market_structure returns valid contract on 1h BTCUSDT."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
        ))

        _assert_market_structure_contract(result, "BTC/USDT", "1h")

    def test_market_structure_4h(self, btcusdt_4h_ohlcv):
        """get_market_structure returns valid contract on 4h BTCUSDT."""
        df = btcusdt_4h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(get_market_structure(
            symbol="BTC/USDT",
            timeframe="4h",
            start_at=start_at,
            end_at=end_at,
        ))

        _assert_market_structure_contract(result, "BTC/USDT", "4h")

    def test_market_structure_adx_range(self, btcusdt_1h_ohlcv):
        """get_market_structure ADX values are in [0, 100] when present."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
        ))

        summary = result["data"]["summary"]
        if summary and summary["adx"] is not None:
            assert 0 <= summary["adx"] <= 100, (
                f"ADX {summary['adx']} must be in valid range [0, 100]"
            )

    def test_market_structure_di_spread_finite(self, btcusdt_1h_ohlcv):
        """get_market_structure DI spread is finite (can be negative in bearish markets)."""
        df = btcusdt_1h_ohlcv
        start_at = df.index[0].to_pydatetime().isoformat()
        end_at = df.index[-1].to_pydatetime().isoformat()

        result = _run_async(get_market_structure(
            symbol="BTC/USDT",
            timeframe="1h",
            start_at=start_at,
            end_at=end_at,
        ))

        summary = result["data"]["summary"]
        if summary and summary["di_spread"] is not None:
            # DI spread = plus_di - minus_di; can be negative in bearish markets
            assert math.isfinite(summary["di_spread"]), (
                f"DI spread {summary['di_spread']} must be finite"
            )
