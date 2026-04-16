"""Walk-forward evaluation engine for backtest strategies.

ENG-26 spec — backtest-core only. This module does not import from
tempest_mcp.tools.* and does not own fetch/window planning.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from tempest_mcp.backtest.engine import BacktestEngine

RESERVED_STRATEGY_PARAM_KEYS: frozenset[str] = frozenset({"initial_capital"})


# ---------------------------------------------------------------------------
# Config and result contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Configuration for walk-forward split planning.

    All window/step values are expressed in row counts, not wall-clock time.
    """

    train_window: int  # number of rows in the train slice
    test_window: int  # number of rows in the test slice
    step: int  # number of rows to advance between splits


@dataclass(frozen=True)
class WalkForwardWindowResult:
    """Result of a single walk-forward window evaluation."""

    split_index: int
    train_start_at_utc: str
    train_end_at_utc: str
    test_start_at_utc: str
    test_end_at_utc: str
    train_rows: int
    test_rows: int
    strategy_id: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class WalkForwardSummary:
    """Aggregate summary across all walk-forward windows."""

    window_count: int
    mean_out_of_sample_total_return: float | None
    mean_out_of_sample_sharpe_ratio: float | None
    best_window_total_return: float | None
    worst_window_total_return: float | None


@dataclass(frozen=True)
class WalkForwardResult:
    """Complete walk-forward evaluation result."""

    strategy_id: str
    config: WalkForwardConfig
    windows: list[WalkForwardWindowResult]
    summary: WalkForwardSummary


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_config(config: WalkForwardConfig) -> None:
    """Validate walk-forward config parameters.

    Raises
    ------
    ValueError
        If any parameter is invalid.
    """
    if config.train_window <= 0:
        raise ValueError(f"train_window must be positive, got {config.train_window}")
    if config.test_window < 2:
        raise ValueError(
            f"test_window must be at least 2 for meaningful backtest, got {config.test_window}"
        )
    if config.step <= 0:
        raise ValueError(f"step must be positive, got {config.step}")


def _validate_ohlcv(ohlcv_df: pd.DataFrame) -> None:
    """Validate OHLCV DataFrame structure.

    Raises
    ------
    ValueError
        If DataFrame is missing columns or has invalid index.
    """
    required_columns = {"open", "high", "low", "close", "volume"}
    missing = required_columns.difference(ohlcv_df.columns)
    if missing:
        raise ValueError(f"OHLCV DataFrame missing required columns: {', '.join(sorted(missing))}")

    if ohlcv_df.empty:
        raise ValueError("OHLCV DataFrame must not be empty")

    if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
        raise ValueError("OHLCV DataFrame must have a DatetimeIndex")

    if not ohlcv_df.index.is_monotonic_increasing:
        raise ValueError("OHLCV DataFrame index must be monotonically increasing")

    if ohlcv_df.index.has_duplicates:
        raise ValueError("OHLCV DataFrame index must not contain duplicates")

    if ohlcv_df.index.tz is None:
        raise ValueError("OHLCV DataFrame index must be timezone-aware")


def _normalize_ohlcv_to_utc(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a timezone-aware OHLCV index to UTC."""
    return ohlcv_df.tz_convert("UTC")


def _validate_initial_capital(value: Any) -> float:
    """Validate initial capital as a finite positive number."""
    if isinstance(value, bool):
        raise ValueError("initial_capital must be a finite number greater than 0")

    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("initial_capital must be a finite number greater than 0") from exc

    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("initial_capital must be a finite number greater than 0")

    return amount


def _validate_strategy_params(strategy_params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate strategy parameter passthrough kwargs."""
    if strategy_params is None:
        return {}

    params = dict(strategy_params)
    reserved = RESERVED_STRATEGY_PARAM_KEYS.intersection(params)
    if reserved:
        reserved_list = ", ".join(sorted(reserved))
        raise ValueError(f"strategy_params contains reserved keys: {reserved_list}")

    return params


def _validate_strategy_result(result: tuple) -> None:
    """Validate strategy runner return type.

    Raises
    ------
    ValueError
        If the return value does not match the expected (signals, BacktestEngine) contract.
    """
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError(
            f"Strategy runner must return (signals, BacktestEngine), got {type(result).__name__}"
        )
    signals, engine = result
    if not isinstance(signals, pd.Series):
        raise ValueError(f"Strategy runner signals must be pd.Series, got {type(signals).__name__}")
    if not isinstance(engine, BacktestEngine):
        raise ValueError(
            f"Strategy runner engine must be BacktestEngine, got {type(engine).__name__}"
        )


def _validate_signal_alignment(
    signals: pd.Series,
    combined_index: pd.DatetimeIndex,
    test_index: pd.DatetimeIndex,
    *,
    split_index: int,
) -> pd.Series:
    """Require exact signal alignment for combined and test slices."""
    if len(signals) != len(combined_index) or not signals.index.equals(combined_index):
        raise ValueError(
            "Strategy runner signals must align 1:1 with the combined slice index "
            f"for split_index={split_index}"
        )

    try:
        test_slice_signals = signals.loc[test_index]
    except KeyError as exc:
        raise ValueError(
            "Strategy runner signals must align 1:1 with the test slice index "
            f"for split_index={split_index}"
        ) from exc

    if len(test_slice_signals) != len(test_index) or not test_slice_signals.index.equals(
        test_index
    ):
        raise ValueError(
            "Strategy runner signals must align 1:1 with the test slice index "
            f"for split_index={split_index}"
        )

    return test_slice_signals


def _format_timestamp(timestamp: pd.Timestamp) -> str:
    """Serialize timestamps as UTC ISO-8601 strings."""
    return timestamp.tz_convert("UTC").isoformat()


# ---------------------------------------------------------------------------
# Split planning
# ---------------------------------------------------------------------------


def _plan_splits(
    ohlcv_df: pd.DataFrame,
    config: WalkForwardConfig,
) -> list[tuple[int, int, int, int]]:
    """Plan deterministic walk-forward split bounds.

    Each split is represented as a 4-tuple:
        (train_start, train_end, test_start, test_end)
    where indices are inclusive start / exclusive end positions.

    Splits are generated in ascending order from index 0 forward.

    Returns
    -------
    list[tuple[int, int, int, int]]
        Empty list if no valid split can be generated.

    Raises
    ------
    ValueError
        If the configuration plus data cannot produce at least one valid split.
    """
    n_rows = len(ohlcv_df)
    min_required = config.train_window + config.test_window

    if n_rows < min_required:
        raise ValueError(
            f"Insufficient data: need at least {min_required} rows "
            f"(train_window={config.train_window} + test_window={config.test_window}), "
            f"got {n_rows}"
        )

    splits: list[tuple[int, int, int, int]] = []
    train_start = 0

    while True:
        train_end = train_start + config.train_window
        test_start = train_end
        test_end = test_start + config.test_window

        # Check bounds — test_end must not exceed data length
        if test_end > n_rows:
            break

        splits.append((train_start, train_end, test_start, test_end))

        # Advance by step
        train_start += config.step

        # Safety guard against infinite loop
        if train_start >= n_rows:
            break

    if not splits:
        raise ValueError(
            f"Configuration produces no valid splits: "
            f"train_window={config.train_window}, test_window={config.test_window}, "
            f"step={config.step}, data_rows={n_rows}"
        )

    return splits


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _summarize_windows(windows: list[WalkForwardWindowResult]) -> WalkForwardSummary:
    """Compute deterministic aggregate summary from ordered window results.

    Non-finite values (nan/inf) are ignored for mean/best/worst reducers.
    If all values are invalid, the corresponding summary field returns None.
    """
    returns = []
    sharpes = []

    for w in windows:
        tr = w.metrics.get("total_return")
        if tr is not None and math.isfinite(tr):
            returns.append(tr)
        sr = w.metrics.get("sharpe_ratio")
        if sr is not None and math.isfinite(sr):
            sharpes.append(sr)

    mean_return = float(sum(returns) / len(returns)) if returns else None
    mean_sharpe = float(sum(sharpes) / len(sharpes)) if sharpes else None
    best_return = max(returns) if returns else None
    worst_return = min(returns) if returns else None

    return WalkForwardSummary(
        window_count=len(windows),
        mean_out_of_sample_total_return=mean_return,
        mean_out_of_sample_sharpe_ratio=mean_sharpe,
        best_window_total_return=best_return,
        worst_window_total_return=worst_return,
    )


# ---------------------------------------------------------------------------
# Per-split runner
# ---------------------------------------------------------------------------


def _run_split(
    ohlcv_df: pd.DataFrame,
    strategy_runner: Callable[..., tuple[pd.Series, BacktestEngine]],
    split_index: int,
    train_start: int,
    train_end: int,
    test_start: int,
    test_end: int,
    train_window: int,
    strategy_id: str,
    initial_capital: float,
    strategy_params: Mapping[str, Any] | None,
) -> WalkForwardWindowResult:
    """Run a single walk-forward split.

    Train slice provides context for signal generation; test slice is used for
    out-of-sample metric scoring. Position state is reset at the test boundary
    so OOS scoring is isolated and deterministic per split.
    """
    # Combined slice for signal generation (train + test)
    combined_start = train_start
    combined_end = test_end
    combined_df = ohlcv_df.iloc[combined_start:combined_end]

    # Test slice OHLCV data
    test_df = ohlcv_df.iloc[test_start:test_end]

    # Generate signals using the strategy runner on combined data
    try:
        result = strategy_runner(
            combined_df,
            initial_capital=initial_capital,
            **(strategy_params or {}),
        )
    except Exception as exc:
        raise ValueError(f"Strategy runner failed for split_index={split_index}: {exc}") from exc

    try:
        _validate_strategy_result(result)
    except ValueError as exc:
        raise ValueError(f"Invalid strategy result for split_index={split_index}: {exc}") from exc

    signals, _ = result
    test_slice_signals = _validate_signal_alignment(
        signals,
        combined_df.index,
        test_df.index,
        split_index=split_index,
    )

    # Run backtest engine on test slice only (out-of-sample)
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.run(test_df, test_slice_signals)

    # Extract timestamps
    train_ts_start = _format_timestamp(ohlcv_df.index[train_start])
    train_ts_end = _format_timestamp(ohlcv_df.index[train_end - 1])
    test_ts_start = _format_timestamp(ohlcv_df.index[test_start])
    test_ts_end = _format_timestamp(ohlcv_df.index[test_end - 1])

    return WalkForwardWindowResult(
        split_index=split_index,
        train_start_at_utc=train_ts_start,
        train_end_at_utc=train_ts_end,
        test_start_at_utc=test_ts_start,
        test_end_at_utc=test_ts_end,
        train_rows=train_window,
        test_rows=test_end - test_start,
        strategy_id=strategy_id,
        metrics=engine.metrics,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_walk_forward(
    ohlcv_df: pd.DataFrame,
    strategy_runner: Callable[..., tuple[pd.Series, BacktestEngine]],
    *,
    strategy_id: str,
    config: WalkForwardConfig,
    initial_capital: float = 100_000.0,
    strategy_params: Mapping[str, Any] | None = None,
) -> WalkForwardResult:
    """Run walk-forward evaluation over rolling train/test splits.

    Parameters
    ----------
    ohlcv_df:
        Pre-resolved OHLCV DataFrame with UTC-aware DatetimeIndex and columns
        [open, high, low, close, volume]. The DataFrame must be monotonically
        increasing without duplicate timestamps.
    strategy_runner:
        Callable that accepts an OHLCV DataFrame and initial_capital keyword argument,
        returning ``(signals, BacktestEngine)`` where signals is a pd.Series and
        engine is a configured BacktestEngine instance. The strategy function is
        responsible for its own parameterisation via ``strategy_params``.
    strategy_id:
        Identifier for the strategy being evaluated (used in results).
    config:
        WalkForwardConfig specifying train_window, test_window, and step in rows.
    initial_capital:
        Starting capital for each walk-forward backtest run (default 100_000.0).
    strategy_params:
        Optional mapping of additional keyword arguments to pass to the strategy runner.

    Returns
    -------
    WalkForwardResult
        Contains the strategy_id, config, ordered list of per-window results,
        and aggregate summary.

    Raises
    ------
    ValueError
        For invalid configuration, insufficient data, malformed OHLCV DataFrame,
        or strategy runner contract violations.

    Notes
    -----
    - "Train" means the historical slice used to anchor each rolling evaluation
      window; it does not imply model fitting or parameter search.
    - Position state is reset at the test boundary so out-of-sample scoring
      is deterministic and isolated per split.
    - Split ordering is stable and reproducible for the same input DataFrame
      and configuration.
    """
    # Validate inputs
    _validate_config(config)
    _validate_ohlcv(ohlcv_df)
    initial_capital = _validate_initial_capital(initial_capital)
    strategy_params = _validate_strategy_params(strategy_params)
    ohlcv_df = _normalize_ohlcv_to_utc(ohlcv_df)

    # Plan splits
    splits = _plan_splits(ohlcv_df, config)

    # Run each split
    windows: list[WalkForwardWindowResult] = []
    for idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        window_result = _run_split(
            ohlcv_df=ohlcv_df,
            strategy_runner=strategy_runner,
            split_index=idx,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_window=config.train_window,
            strategy_id=strategy_id,
            initial_capital=initial_capital,
            strategy_params=strategy_params,
        )
        windows.append(window_result)

    # Compute aggregate summary
    summary = _summarize_windows(windows)

    return WalkForwardResult(
        strategy_id=strategy_id,
        config=config,
        windows=windows,
        summary=summary,
    )
