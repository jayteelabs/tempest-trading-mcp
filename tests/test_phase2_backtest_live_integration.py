"""Phase 2 backtest live-data integration suite (ENG-64).

This module exercises all Phase 2 backtest strategy entrypoints against
real CCXT-fetched market data and validates the strategy/backtest output
contracts are sane, stable, and not just passing synthetic fixtures.

Scope (ENG-64 design pass v3):
- BTCUSDT-only for reliability/low noise
- Both 1h and 4h timeframe validation (parameterized)
- Bounded retry on transient upstream CCXT instability → skip with loud warning
- Hard fail on contract/sanity violations

Run with:  uv run pytest --run-integration tests/test_phase2_backtest_live_integration.py -v
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from datetime import timedelta, timezone

import ccxt
import numpy as np
import pandas as pd
import pytest

from tempest_mcp import strategies
from tempest_mcp.backtest.engine import (
    BacktestEngine,
    PositionDirection,
    SignalAction,
    Trade,
)
from tempest_mcp.data import get_live_adapter

# =============================================================================
# Constants
# =============================================================================

CANONICAL_SYMBOL = "BTCUSDT"
WARMUP_LIMIT = 200  # minimum bars for EMA stack warmup
MAX_RETRIES = 3
TRANSIENT_FETCH_EXCEPTIONS = (ccxt.NetworkError, ConnectionError, TimeoutError)


# =============================================================================
# Helpers — data fetching with bounded retry
# =============================================================================


def _fetch_live_ohlcv_with_retry(
    symbol: str,
    timeframe: str,
    limit: int = WARMUP_LIMIT + 100,
    max_retries: int = MAX_RETRIES,
) -> pd.DataFrame:
    """Fetch live OHLCV with bounded retry for transient upstream failures.

    Skips with a loud diagnostic warning only when failure is clearly transient
    (exchange unavailable, rate-limited, timeout, network instability).
    Hard-fails on contract/sanity violations once data is fetched.
    """
    adapter = get_live_adapter()
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            df = adapter.fetch_ohlcv_live(symbol, timeframe=timeframe, limit=limit)
            # Validate contract immediately — hard fail on violation
            _assert_ohlcv_contract(df, timeframe)
            return df
        except AssertionError:
            raise
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, TRANSIENT_FETCH_EXCEPTIONS):
                raise

            last_exception = exc
            if attempt < max_retries:
                warnings.warn(
                    f"[{symbol} {timeframe}] Attempt {attempt}/{max_retries} failed: {type(exc).__name__}: {exc}. Retrying...",
                    stacklevel=2,
                )
            continue

    # All retries exhausted — skip with loud diagnostic
    exc_class = type(last_exception).__name__ if last_exception else "Unknown"
    exc_msg = str(last_exception) if last_exception else "no exception details"
    raise pytest.skip.Exception(
        f"[{symbol} {timeframe}] Bounded retries ({max_retries}) exhausted. "
        f"Skipping due to transient upstream CCXT instability: {exc_class}: {exc_msg}"
    )


# =============================================================================
# Helpers — contract assertions
# =============================================================================


def _assert_ohlcv_contract(df: pd.DataFrame, timeframe: str) -> None:
    """Validate fetched OHLCV DataFrame meets the live-data contract.

    Hard-fails on any contract violation.
    """
    assert not df.empty, (
        f"[{timeframe}] OHLCV DataFrame must be non-empty. "
        f"Ensure CCXT returned data for {CANONICAL_SYMBOL} on {timeframe}."
    )

    assert len(df) >= WARMUP_LIMIT, (
        f"[{timeframe}] OHLCV DataFrame must have >= {WARMUP_LIMIT} rows for strategy warmup, "
        f"got {len(df)}. Check CCXT limit parameter."
    )

    required_cols = {"open", "high", "low", "close", "volume"}
    missing = required_cols.difference(df.columns)
    assert not missing, (
        f"[{timeframe}] OHLCV DataFrame missing required columns: {', '.join(sorted(missing))}. "
        f"Columns found: {list(df.columns)}"
    )

    # UTC-aware, monotonic, non-duplicate DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex), (
        f"[{timeframe}] Index must be DatetimeIndex, got {type(df.index).__name__}"
    )
    assert df.index.tz is not None, (
        f"[{timeframe}] DatetimeIndex must be UTC-aware (tz is None). "
        f"Localize or convert the index to UTC."
    )
    tz_name = (
        getattr(df.index.tz, "key", None) or getattr(df.index.tz, "zone", None) or str(df.index.tz)
    )
    assert tz_name == "UTC", (
        f"[{timeframe}] DatetimeIndex must be UTC, got {tz_name!r}. "
        "Convert the index to UTC before running the validation gate."
    )
    assert df.index.is_monotonic_increasing, (
        f"[{timeframe}] DatetimeIndex must be monotonic increasing. "
        f"First 5: {df.index[:5].tolist()}"
    )
    assert not df.index.has_duplicates, (
        f"[{timeframe}] DatetimeIndex must not have duplicates. "
        f"Duplicates found: {df.index[df.index.duplicated()].tolist()}"
    )

    # Numeric sanity: finite values
    for col in required_cols:
        assert pd.api.types.is_numeric_dtype(df[col]), (
            f"[{timeframe}] Column '{col}' must be numeric, got {df[col].dtype}"
        )
        assert df[col].notna().all(), (
            f"[{timeframe}] Column '{col}' must not contain missing values"
        )
        assert np.isfinite(df[col].to_numpy()).all(), (
            f"[{timeframe}] Column '{col}' must have only finite values. Non-finite values found."
        )

    # Price sanity: high >= low, non-negative volume
    assert (df["high"] >= df["low"]).all(), (
        f"[{timeframe}] All rows must satisfy high >= low. "
        f"Violations: {(df['high'] < df['low']).sum()}"
    )
    assert (df["volume"] >= 0).all(), (
        f"[{timeframe}] Volume must be non-negative. "
        f"Negative volume found: {(df['volume'] < 0).sum()} rows"
    )


def _assert_strategy_result_contract(
    signals: pd.Series,
    engine: BacktestEngine,
    expected_index: pd.Index,
    timeframe: str,
) -> None:
    """Validate strategy/backtest result meets the output contract.

    Hard-fails on any contract violation.
    """
    # signals: indexed to input, values in SignalAction domain
    assert isinstance(signals, pd.Series), (
        f"[{timeframe}] signals must be a pd.Series, got {type(signals).__name__}"
    )
    assert len(signals) > 0, f"[{timeframe}] signals must be non-empty"
    assert len(signals) == len(expected_index), (
        f"[{timeframe}] signals length must match OHLCV length. "
        f"Expected {len(expected_index)}, got {len(signals)}"
    )
    assert signals.index.equals(expected_index), (
        f"[{timeframe}] signals index must align exactly to the input OHLCV index"
    )

    # All signal values must be SignalAction members
    invalid_signals = [(i, s) for i, s in enumerate(signals) if not isinstance(s, SignalAction)]
    assert not invalid_signals, (
        f"[{timeframe}] All signal values must be SignalAction enum members. "
        f"Invalid at indices: {invalid_signals[:5]}"
    )

    # engine: populated BacktestEngine with required output structures
    assert isinstance(engine, BacktestEngine), (
        f"[{timeframe}] engine must be BacktestEngine, got {type(engine).__name__}"
    )

    # equity curve exists with expected cardinality (typically len(df)-1)
    equity_curve = engine._equity_curve
    assert len(equity_curve) > 0, (
        f"[{timeframe}] equity_curve must be non-empty, got {len(equity_curve)}"
    )

    # trades structure: each trade has valid fields
    trades = engine._trades
    assert isinstance(trades, list), (
        f"[{timeframe}] trades must be a list, got {type(trades).__name__}"
    )
    for i, trade in enumerate(trades):
        assert trade.entry_price > 0, (
            f"[{timeframe}] Trade {i}: entry_price must be positive, got {trade.entry_price}"
        )
        assert trade.exit_price > 0, (
            f"[{timeframe}] Trade {i}: exit_price must be positive, got {trade.exit_price}"
        )
        assert trade.size > 0, f"[{timeframe}] Trade {i}: size must be positive, got {trade.size}"
        assert math.isfinite(trade.net_pnl), (
            f"[{timeframe}] Trade {i}: net_pnl must be finite, got {trade.net_pnl}"
        )
        assert isinstance(trade.direction, PositionDirection), (
            f"[{timeframe}] Trade {i}: direction must be PositionDirection, got {type(trade.direction).__name__}"
        )
        assert trade.direction in (PositionDirection.LONG, PositionDirection.SHORT), (
            f"[{timeframe}] Trade {i}: direction must be LONG or SHORT"
        )
        assert trade.entry_time < trade.exit_time, (
            f"[{timeframe}] Trade {i}: entry_time must be before exit_time"
        )

    # metrics: required keys present
    metrics = engine._compute_metrics()
    required_metrics = {
        "total_return",
        "win_rate",
        "profit_factor",
        "max_drawdown",
        "expectancy",
        "sharpe_ratio",
        "total_trades",
    }
    missing_metrics = required_metrics.difference(metrics.keys())
    assert not missing_metrics, (
        f"[{timeframe}] metrics missing required keys: {', '.join(sorted(missing_metrics))}"
    )

    # No NaN/invalid numeric states in core metrics (allow inf profit_factor edge case)
    for key in required_metrics - {"profit_factor"}:
        val = metrics[key]
        assert isinstance(val, (int, float)) and not math.isnan(val), (
            f"[{timeframe}] metric '{key}' must be a finite number, got {val}"
        )


def _iter_phase2_strategy_entrypoints() -> list[tuple[str, Callable[..., object]]]:
    """Return list of (name, callable) for all Phase 2 backtest entrypoints.

    Pulls from the exported strategies surface so coverage cannot drift.
    Enforces the minimum required set from ENG-64 design.
    """
    exported_backtest_names = sorted(
        name
        for name in getattr(strategies, "__all__", [])
        if name.startswith("run_") and name.endswith("_backtest")
    )
    entrypoints = [(name, getattr(strategies, name)) for name in exported_backtest_names]

    required_names = {
        "run_pdh_session_backtest",
        "run_ema_stack_backtest",
        "run_vwap_anchored_backtest",
    }
    found_names = {name for name, _ in entrypoints}
    missing = required_names.difference(found_names)
    assert not missing, (
        f"[ENG-64] Required Phase 2 entrypoints missing from strategies export surface: {sorted(missing)}. "
        f"Found: {sorted(found_names)}"
    )

    for name, fn in entrypoints:
        assert callable(fn), f"[ENG-64] Exported entrypoint '{name}' must be callable"

    return entrypoints


def _make_valid_ohlcv_frame(rows: int = WARMUP_LIMIT) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100.0, 100.0 + rows - 1, rows)

    return pd.DataFrame(
        {
            "open": base,
            "high": base + 2.0,
            "low": base - 2.0,
            "close": base + 1.0,
            "volume": np.full(rows, 10.0),
        },
        index=index,
    )


def _make_engine_with_trade(index: pd.DatetimeIndex) -> BacktestEngine:
    engine = BacktestEngine()
    engine._equity_curve = [100_000.0] * max(len(index) - 1, 1)
    engine._trades = [
        Trade(
            entry_time=index[0],
            exit_time=index[1],
            entry_price=100.0,
            exit_price=101.0,
            size=1.0,
            direction=PositionDirection.LONG,
            gross_pnl=1.0,
            net_pnl=0.8,
            commission=0.1,
            slippage_cost=0.1,
            bars_held=1,
        )
    ]
    return engine


# =============================================================================
# Tests — data contract by timeframe
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("timeframe", ["1h", "4h"])
def test_phase2_live_data_contract_by_timeframe(timeframe, network_available):
    """Validate live OHLCV data contract for both 1h and 4h timeframes.

    Fetches real BTCUSDT data via CCXT and asserts:
    - non-empty, warmup-capable (>= 200 rows)
    - required OHLCV columns present
    - UTC-aware, monotonic, non-duplicate DatetimeIndex
    - finite numeric values, high >= low, non-negative volume
    """
    if not network_available:
        pytest.skip("Network not available")

    df = _fetch_live_ohlcv_with_retry(
        symbol=CANONICAL_SYMBOL,
        timeframe=timeframe,
        limit=WARMUP_LIMIT + 100,
    )
    # Contract assertion is embedded in _fetch_live_ohlcv_with_retry
    assert not df.empty


# =============================================================================
# Tests — strategy export surface
# =============================================================================


@pytest.mark.integration
def test_phase2_strategy_export_surface_includes_required_minimum():
    """Verify the strategies module exports all required Phase 2 entrypoints.

    Hard-fails if any required minimum entrypoint is absent.
    """
    entrypoints = _iter_phase2_strategy_entrypoints()
    found_names = [name for name, _ in entrypoints]

    required_names = [
        "run_pdh_session_backtest",
        "run_ema_stack_backtest",
        "run_vwap_anchored_backtest",
    ]
    missing = [n for n in required_names if n not in found_names]
    assert not missing, (
        f"[ENG-64] Required Phase 2 entrypoints missing from strategies __init__.py: {missing}. "
        f"Found: {found_names}"
    )


# =============================================================================
# Tests — all Phase 2 entrypoints on live data by timeframe
# =============================================================================


@pytest.mark.integration
@pytest.mark.parametrize("timeframe", ["1h", "4h"])
def test_phase2_all_exported_backtest_entrypoints_run_on_live_data_by_timeframe(
    timeframe,
    network_available,
):
    """Run all Phase 2 backtest entrypoints end-to-end on live BTCUSDT data.

    For each entrypoint:
    - Fetches real OHLCV via CCXT (bounded retry → skip on transient failure)
    - Executes strategy with live data
    - Validates signal and backtest output contracts
    - Fails if either timeframe path breaks
    """
    if not network_available:
        pytest.skip("Network not available")

    # Fetch live data once per timeframe (retry handled inside helper)
    df = _fetch_live_ohlcv_with_retry(
        symbol=CANONICAL_SYMBOL,
        timeframe=timeframe,
        limit=WARMUP_LIMIT + 100,
    )

    entrypoints = _iter_phase2_strategy_entrypoints()

    for name, run_fn in entrypoints:
        signals, engine = run_fn(ohlcv_df=df)
        _assert_strategy_result_contract(signals, engine, df.index, f"{timeframe}/{name}")


def test_assert_ohlcv_contract_accepts_valid_numeric_series():
    df = _make_valid_ohlcv_frame()

    _assert_ohlcv_contract(df, "1h")


def test_assert_ohlcv_contract_requires_utc_timezone():
    df = _make_valid_ohlcv_frame()
    df.index = df.index.tz_convert(timezone(-timedelta(hours=5)))

    with pytest.raises(AssertionError, match="must be UTC"):
        _assert_ohlcv_contract(df, "1h")


def test_assert_ohlcv_contract_rejects_missing_values():
    df = _make_valid_ohlcv_frame()
    df.loc[df.index[0], "close"] = np.nan

    with pytest.raises(AssertionError, match="must not contain missing values"):
        _assert_ohlcv_contract(df, "1h")


def test_iter_phase2_strategy_entrypoints_matches_export_surface():
    entrypoints = _iter_phase2_strategy_entrypoints()

    assert [name for name, _ in entrypoints] == [
        name
        for name in sorted(strategies.__all__)
        if name.startswith("run_") and name.endswith("_backtest")
    ]
    assert "run_elliot_wave_backtest" in [name for name, _ in entrypoints]


def test_assert_strategy_result_contract_accepts_aligned_signals_and_trade_direction():
    df = _make_valid_ohlcv_frame()
    signals = pd.Series(SignalAction.HOLD, index=df.index, dtype=object)
    engine = _make_engine_with_trade(df.index)

    _assert_strategy_result_contract(signals, engine, df.index, "1h/run_ema_stack_backtest")


def test_assert_strategy_result_contract_rejects_misaligned_signals():
    df = _make_valid_ohlcv_frame()
    shifted_index = df.index.shift(1, freq="h")
    signals = pd.Series(SignalAction.HOLD, index=shifted_index, dtype=object)
    engine = _make_engine_with_trade(df.index)

    with pytest.raises(AssertionError, match="align exactly"):
        _assert_strategy_result_contract(signals, engine, df.index, "1h/run_ema_stack_backtest")


def test_fetch_live_ohlcv_with_retry_re_raises_contract_violations(monkeypatch):
    class DummyAdapter:
        def fetch_ohlcv_live(self, symbol, timeframe, limit):
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    monkeypatch.setattr(
        "tests.test_phase2_backtest_live_integration.get_live_adapter",
        lambda: DummyAdapter(),
    )

    with pytest.raises(AssertionError):
        _fetch_live_ohlcv_with_retry(CANONICAL_SYMBOL, "1h", max_retries=3)


def test_fetch_live_ohlcv_with_retry_re_raises_non_transient_errors(monkeypatch):
    attempts = 0

    class DummyAdapter:
        def fetch_ohlcv_live(self, symbol, timeframe, limit):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("bad auth")

    monkeypatch.setattr(
        "tests.test_phase2_backtest_live_integration.get_live_adapter",
        lambda: DummyAdapter(),
    )

    with pytest.raises(RuntimeError, match="bad auth"):
        _fetch_live_ohlcv_with_retry(CANONICAL_SYMBOL, "1h", max_retries=3)

    assert attempts == 1


def test_fetch_live_ohlcv_with_retry_skips_only_after_transient_retries(monkeypatch):
    attempts = 0

    class DummyAdapter:
        def fetch_ohlcv_live(self, symbol, timeframe, limit):
            nonlocal attempts
            attempts += 1
            raise ccxt.NetworkError("temporary outage")

    monkeypatch.setattr(
        "tests.test_phase2_backtest_live_integration.get_live_adapter",
        lambda: DummyAdapter(),
    )

    with pytest.raises(pytest.skip.Exception, match="temporary outage"):
        _fetch_live_ohlcv_with_retry(CANONICAL_SYMBOL, "1h", max_retries=3)

    assert attempts == 3
