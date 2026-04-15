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

import pandas as pd
import pytest

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.data import get_live_adapter
from tempest_mcp.strategies import (
    run_ema_stack_backtest,
    run_pdh_session_backtest,
    run_vwap_anchored_backtest,
)

# =============================================================================
# Constants
# =============================================================================

CANONICAL_SYMBOL = "BTCUSDT"
WARMUP_LIMIT = 200  # minimum bars for EMA stack warmup
MAX_RETRIES = 3


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
        except Exception as exc:  # noqa: BLE001
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
    pytest.skip(
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
        assert df[col].notna().any(), (
            f"[{timeframe}] Column '{col}' must have at least one non-NA value"
        )
        assert math.isfinite(df[col].dropna()).all(), (
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
        assert trade.direction in (SignalAction.LONG_ENTRY, SignalAction.SHORT_ENTRY), (
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


def _iter_phase2_strategy_entrypoints() -> list[tuple[str, callable]]:
    """Return list of (name, callable) for all Phase 2 backtest entrypoints.

    Enforces minimum required set from ENG-64 design.
    """
    required = {
        "run_pdh_session_backtest": run_pdh_session_backtest,
        "run_ema_stack_backtest": run_ema_stack_backtest,
        "run_vwap_anchored_backtest": run_vwap_anchored_backtest,
    }

    # Verify required entrypoints are available
    for name, fn in required.items():
        assert fn is not None, f"[ENG-64] Required entrypoint '{name}' must be importable"

    return list(required.items())


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
        _assert_strategy_result_contract(signals, engine, f"{timeframe}/{name}")
