"""Tests for walk-forward evaluation engine (ENG-26)."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tempest_mcp.backtest.engine import BacktestEngine, SignalAction
from tempest_mcp.backtest.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardSummary,
    WalkForwardWindowResult,
    _plan_splits,
    _summarize_windows,
    _validate_config,
    _validate_ohlcv,
    run_walk_forward,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n: int,
    start_price: float = 100.0,
    step: float = 0.5,
    start_time: datetime | None = None,
) -> pd.DataFrame:
    """Helper to create OHLCV DataFrame."""
    if start_time is None:
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start_time + timedelta(hours=i) for i in range(n)]
    data = {
        "open": [start_price + i * step for i in range(n)],
        "high": [start_price + i * step + 1 for i in range(n)],
        "low": [start_price + i * step - 1 for i in range(n)],
        "close": [start_price + i * step for i in range(n)],
        "volume": [1000.0] * n,
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex(times))


def _make_simple_strategy(rising: bool = True) -> callable:
    """Make a simple strategy runner for testing.

    Returns a callable that generates simple LONG_ENTRY at bar 0 and LONG_EXIT
    at the last bar, compatible with the (signals, BacktestEngine) contract.
    """

    def strategy(
        ohlcv_df: pd.DataFrame,
        *,
        initial_capital: float = 100_000.0,
    ) -> tuple[pd.Series, BacktestEngine]:
        signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)
        # Entry at first bar
        signals.iloc[0] = SignalAction.LONG_ENTRY
        # Exit at last bar
        signals.iloc[-1] = SignalAction.LONG_EXIT
        engine = BacktestEngine(initial_capital=initial_capital)
        return signals, engine

    return strategy


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestWalkForwardConfigValidation:
    """Tests for WalkForwardConfig validation."""

    def test_valid_config(self):
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        _validate_config(config)  # Should not raise

    def test_train_window_zero_raises(self):
        config = WalkForwardConfig(train_window=0, test_window=5, step=5)
        with pytest.raises(ValueError, match="train_window must be positive"):
            _validate_config(config)

    def test_train_window_negative_raises(self):
        config = WalkForwardConfig(train_window=-1, test_window=5, step=5)
        with pytest.raises(ValueError, match="train_window must be positive"):
            _validate_config(config)

    def test_test_window_less_than_two_raises(self):
        config = WalkForwardConfig(train_window=10, test_window=1, step=5)
        with pytest.raises(ValueError, match="test_window must be at least 2"):
            _validate_config(config)

    def test_step_zero_raises(self):
        config = WalkForwardConfig(train_window=10, test_window=5, step=0)
        with pytest.raises(ValueError, match="step must be positive"):
            _validate_config(config)

    def test_step_negative_raises(self):
        config = WalkForwardConfig(train_window=10, test_window=5, step=-1)
        with pytest.raises(ValueError, match="step must be positive"):
            _validate_config(config)


# ---------------------------------------------------------------------------
# OHLCV validation tests
# ---------------------------------------------------------------------------


class TestOHLCVValidation:
    """Tests for OHLCV DataFrame validation."""

    def test_valid_ohlcv(self):
        df = _make_ohlcv(20)
        _validate_ohlcv(df)  # Should not raise

    def test_missing_columns_raises(self):
        df = pd.DataFrame(
            {"open": [100.0, 101.0], "close": [100.5, 101.5]},
            index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        )
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_ohlcv(df)

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([]),
        )
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_ohlcv(df)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1000.0],
            },
            index=pd.Index([0, 1]),
        )
        with pytest.raises(ValueError, match="DatetimeIndex"):
            _validate_ohlcv(df)

    def test_non_monotonic_index_raises(self):
        df = _make_ohlcv(10)
        df = df.iloc[::-1]  # Reverse to make non-monotonic
        with pytest.raises(ValueError, match="monotonically increasing"):
            _validate_ohlcv(df)

    def test_duplicate_index_raises(self):
        times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(5)]
        times[2] = times[1]  # Create duplicate
        data = {
            "open": [100.0 + i for i in range(5)],
            "high": [101.0 + i for i in range(5)],
            "low": [99.0 + i for i in range(5)],
            "close": [100.5 + i for i in range(5)],
            "volume": [1000.0] * 5,
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex(times))
        with pytest.raises(ValueError, match="must not contain duplicates"):
            _validate_ohlcv(df)

    def test_timezone_naive_index_raises(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1000.0],
            },
            index=pd.DatetimeIndex([datetime(2024, 1, 1), datetime(2024, 1, 1, 1)]),
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            _validate_ohlcv(df)


# ---------------------------------------------------------------------------
# Split planning tests
# ---------------------------------------------------------------------------


class TestSplitPlanning:
    """Tests for deterministic split planning."""

    def test_single_split(self):
        """A single valid split is produced when data is exactly train+test."""
        df = _make_ohlcv(15)  # 10 train + 5 test = 15
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        splits = _plan_splits(df, config)
        assert len(splits) == 1
        assert splits[0] == (0, 10, 10, 15)

    def test_multiple_splits(self):
        """Multiple splits are produced with correct step advancement."""
        df = _make_ohlcv(25)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        splits = _plan_splits(df, config)
        # Split 1: 0-10 (train), 10-15 (test)
        # Split 2: 5-15 (train), 15-20 (test)  <- advance by step=5
        # Split 3: 10-20 (train), 20-25 (test)
        assert len(splits) == 3
        assert splits[0] == (0, 10, 10, 15)
        assert splits[1] == (5, 15, 15, 20)
        assert splits[2] == (10, 20, 20, 25)

    def test_insufficient_data_raises(self):
        """Insufficient data for even one split raises ValueError."""
        df = _make_ohlcv(10)  # Less than 10 + 5 = 15
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        with pytest.raises(ValueError, match="Insufficient data"):
            _plan_splits(df, config)

    def test_no_valid_split_produced(self):
        """Step larger than window produces no splits."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=10)
        # First split would be (0, 10, 10, 15), then (10, 20, 20, 25) but step=10 means next train_start=20 which equals n_rows
        splits = _plan_splits(df, config)
        assert len(splits) == 1  # Only the first split fits

    def test_split_order_deterministic(self):
        """Split order is always ascending and deterministic."""
        df = _make_ohlcv(30)
        config = WalkForwardConfig(train_window=5, test_window=3, step=2)
        splits1 = _plan_splits(df, config)
        splits2 = _plan_splits(df, config)
        assert splits1 == splits2
        # Verify ascending order
        for i in range(len(splits1) - 1):
            assert splits1[i][0] < splits1[i + 1][0]

    def test_large_step_produces_single_split(self):
        """Step larger than train+test still produces one split with n == train+test."""
        # With n=15, train=10, test=5, step=16: the first split fits exactly (test_end=15),
        # but the second iteration has train_start=16 >= n=15, so loop ends.
        df = _make_ohlcv(15)
        config = WalkForwardConfig(train_window=10, test_window=5, step=16)
        splits = _plan_splits(df, config)
        # Exactly 1 split since n == train + test means the first split barely fits
        assert len(splits) == 1


# ---------------------------------------------------------------------------
# Summary computation tests
# ---------------------------------------------------------------------------


class TestSummaryComputation:
    """Tests for aggregate summary computation."""

    def test_mean_computation(self):
        windows = [
            WalkForwardWindowResult(
                split_index=0,
                train_start_at_utc="2024-01-01",
                train_end_at_utc="2024-01-02",
                test_start_at_utc="2024-01-02",
                test_end_at_utc="2024-01-03",
                train_rows=10,
                test_rows=5,
                strategy_id="test",
                metrics={"total_return": 0.1, "sharpe_ratio": 1.5},
            ),
            WalkForwardWindowResult(
                split_index=1,
                train_start_at_utc="2024-01-03",
                train_end_at_utc="2024-01-04",
                test_start_at_utc="2024-01-04",
                test_end_at_utc="2024-01-05",
                train_rows=10,
                test_rows=5,
                strategy_id="test",
                metrics={"total_return": 0.2, "sharpe_ratio": 2.0},
            ),
        ]
        summary = _summarize_windows(windows)
        assert summary.window_count == 2
        assert summary.mean_out_of_sample_total_return == pytest.approx(0.15)
        assert summary.mean_out_of_sample_sharpe_ratio == pytest.approx(1.75)
        assert summary.best_window_total_return == 0.2
        assert summary.worst_window_total_return == 0.1

    def test_nan_values_ignored(self):
        """Non-finite values are excluded from mean/best/worst computation."""

        windows = [
            WalkForwardWindowResult(
                split_index=0,
                train_start_at_utc="2024-01-01",
                train_end_at_utc="2024-01-02",
                test_start_at_utc="2024-01-02",
                test_end_at_utc="2024-01-03",
                train_rows=10,
                test_rows=5,
                strategy_id="test",
                metrics={"total_return": 0.1, "sharpe_ratio": float("nan")},
            ),
            WalkForwardWindowResult(
                split_index=1,
                train_start_at_utc="2024-01-03",
                train_end_at_utc="2024-01-04",
                test_start_at_utc="2024-01-04",
                test_end_at_utc="2024-01-05",
                train_rows=10,
                test_rows=5,
                strategy_id="test",
                metrics={"total_return": float("inf"), "sharpe_ratio": 2.0},
            ),
            WalkForwardWindowResult(
                split_index=2,
                train_start_at_utc="2024-01-05",
                train_end_at_utc="2024-01-06",
                test_start_at_utc="2024-01-06",
                test_end_at_utc="2024-01-07",
                train_rows=10,
                test_rows=5,
                strategy_id="test",
                metrics={"total_return": 0.3, "sharpe_ratio": 1.0},
            ),
        ]
        summary = _summarize_windows(windows)
        assert summary.window_count == 3
        # Filtering is per-metric: inf return is filtered, nan sharpe is filtered
        # returns = [0.1, 0.3] -> mean = 0.2
        # sharpes = [2.0, 1.0] -> mean = 1.5
        assert summary.mean_out_of_sample_total_return == pytest.approx(0.2)
        assert summary.mean_out_of_sample_sharpe_ratio == pytest.approx(1.5)
        assert summary.best_window_total_return == 0.3
        assert summary.worst_window_total_return == 0.1

    def test_empty_windows_returns_none_fields(self):
        """Empty window list returns None for aggregate fields."""
        summary = _summarize_windows([])
        assert summary.window_count == 0
        assert summary.mean_out_of_sample_total_return is None
        assert summary.mean_out_of_sample_sharpe_ratio is None
        assert summary.best_window_total_return is None
        assert summary.worst_window_total_return is None


# ---------------------------------------------------------------------------
# Walk-forward integration tests
# ---------------------------------------------------------------------------


class TestRunWalkForward:
    """Integration tests for run_walk_forward."""

    def test_single_split_full_cycle(self):
        """Run walk-forward with a single split and verify result structure."""
        df = _make_ohlcv(20)  # 10 train + 5 test + 5 buffer
        config = WalkForwardConfig(train_window=10, test_window=5, step=10)
        strategy = _make_simple_strategy()

        result = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="test_strategy",
            config=config,
            initial_capital=100_000.0,
        )

        assert isinstance(result, WalkForwardResult)
        assert result.strategy_id == "test_strategy"
        assert result.config == config
        assert len(result.windows) == 1

        window = result.windows[0]
        assert isinstance(window, WalkForwardWindowResult)
        assert window.split_index == 0
        assert window.train_rows == 10
        assert window.test_rows == 5
        assert window.strategy_id == "test_strategy"
        assert "total_return" in window.metrics
        assert "sharpe_ratio" in window.metrics

        summary = result.summary
        assert isinstance(summary, WalkForwardSummary)
        assert summary.window_count == 1

    def test_multiple_splits(self):
        """Run walk-forward with multiple splits."""
        df = _make_ohlcv(30)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        strategy = _make_simple_strategy()

        result = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="multi_split",
            config=config,
        )

        # With train=10, test=5, step=5, n=30:
        # Split 1: train [0:10], test [10:15]
        # Split 2: train [5:15], test [15:20]
        # Split 3: train [10:20], test [20:25]
        # Split 4: train [15:25], test [25:30]
        assert len(result.windows) == 4
        # Verify split indices are ordered
        for i, window in enumerate(result.windows):
            assert window.split_index == i

    def test_strategy_params_passed_through(self):
        """Strategy_params are passed to the strategy runner."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=10)

        received_params = {}

        def capturing_strategy(
            ohlcv_df: pd.DataFrame,
            *,
            initial_capital: float,
            **kwargs,
        ) -> tuple[pd.Series, BacktestEngine]:
            received_params.update(kwargs)
            signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)
            engine = BacktestEngine(initial_capital=initial_capital)
            return signals, engine

        run_walk_forward(
            ohlcv_df=df,
            strategy_runner=capturing_strategy,
            strategy_id="param_test",
            config=config,
            strategy_params={"foo": "bar", "baz": 42},
        )

        assert received_params.get("foo") == "bar"
        assert received_params.get("baz") == 42

    def test_timestamp_boundaries_correct(self):
        """Train/test timestamp boundaries are correctly extracted."""
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        df = _make_ohlcv(20, start_time=start)
        config = WalkForwardConfig(train_window=5, test_window=3, step=5)
        strategy = _make_simple_strategy()

        result = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="ts_test",
            config=config,
        )

        window = result.windows[0]
        # Train: index 0 to 4 (5 rows), so end is index 4
        assert window.train_start_at_utc == "2024-01-01T00:00:00+00:00"
        assert window.train_end_at_utc == "2024-01-01T04:00:00+00:00"
        # Test: index 5 to 7 (3 rows), so end is index 7
        assert window.test_start_at_utc == "2024-01-01T05:00:00+00:00"
        assert window.test_end_at_utc == "2024-01-01T07:00:00+00:00"

    def test_non_utc_timezone_is_normalized_in_output(self):
        start = datetime(2024, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
        df = _make_ohlcv(20, start_time=start)
        config = WalkForwardConfig(train_window=5, test_window=3, step=5)
        strategy = _make_simple_strategy()

        result = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="tz_normalized",
            config=config,
        )

        window = result.windows[0]
        assert window.train_start_at_utc == "2024-01-01T05:00:00+00:00"
        assert window.test_start_at_utc == "2024-01-01T10:00:00+00:00"

    def test_insufficient_data_raises(self):
        """Insufficient data raises ValueError."""
        df = _make_ohlcv(5)  # Too few rows
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        strategy = _make_simple_strategy()

        with pytest.raises(ValueError, match="Insufficient data"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=strategy,
                strategy_id="insufficient",
                config=config,
            )

    def test_invalid_config_raises(self):
        """Invalid config parameters raise ValueError."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=-1, test_window=5, step=5)
        strategy = _make_simple_strategy()

        with pytest.raises(ValueError, match="train_window must be positive"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=strategy,
                strategy_id="bad_config",
                config=config,
            )

    def test_missing_ohlcv_columns_raises(self):
        """Missing OHLCV columns raise ValueError."""
        df = pd.DataFrame(
            {"open": [100.0, 101.0], "close": [100.5, 101.5]},
            index=pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        )
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        strategy = _make_simple_strategy()

        with pytest.raises(ValueError, match="missing required columns"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=strategy,
                strategy_id="bad_columns",
                config=config,
            )

    def test_duplicate_index_raises(self):
        """Duplicate datetime index raises ValueError."""
        times = [datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i) for i in range(10)]
        times[5] = times[4]  # Duplicate
        data = {
            "open": [100.0 + i for i in range(10)],
            "high": [101.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [1000.0] * 10,
        }
        df = pd.DataFrame(data, index=pd.DatetimeIndex(times))
        config = WalkForwardConfig(train_window=5, test_window=3, step=3)
        strategy = _make_simple_strategy()

        with pytest.raises(ValueError, match="must not contain duplicates"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=strategy,
                strategy_id="dup_index",
                config=config,
            )

    def test_invalid_initial_capital_raises(self):
        """Initial capital must be finite and positive."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        strategy = _make_simple_strategy()

        with pytest.raises(
            ValueError, match="initial_capital must be a finite number greater than 0"
        ):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=strategy,
                strategy_id="bad_capital",
                config=config,
                initial_capital=0,
            )

    def test_reserved_strategy_param_raises(self):
        """Reserved strategy params are rejected before runner invocation."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        strategy = _make_simple_strategy()

        with pytest.raises(ValueError, match="reserved keys: initial_capital"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=strategy,
                strategy_id="reserved_param",
                config=config,
                strategy_params={"initial_capital": 1.0},
            )

    def test_invalid_strategy_result_is_wrapped(self):
        """Invalid strategy runner contracts raise ValueError with split context."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)

        def bad_strategy(
            ohlcv_df: pd.DataFrame,
            *,
            initial_capital: float,
        ) -> list[object]:
            return [
                pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object),
                BacktestEngine(initial_capital=initial_capital),
            ]

        with pytest.raises(ValueError, match="Invalid strategy result for split_index=0"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=bad_strategy,
                strategy_id="bad_result",
                config=config,
            )

    def test_runner_exception_is_wrapped(self):
        """Runner exceptions are surfaced as ValueError with split context."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)

        def exploding_strategy(
            ohlcv_df: pd.DataFrame,
            *,
            initial_capital: float,
        ) -> tuple[pd.Series, BacktestEngine]:
            raise RuntimeError("boom")

        with pytest.raises(ValueError, match="Strategy runner failed for split_index=0: boom"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=exploding_strategy,
                strategy_id="runner_error",
                config=config,
            )

    def test_misaligned_signals_raise(self):
        """Signals must align exactly to the combined/test slice index."""
        df = _make_ohlcv(20)
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)

        def shifted_index_strategy(
            ohlcv_df: pd.DataFrame,
            *,
            initial_capital: float,
        ) -> tuple[pd.Series, BacktestEngine]:
            shifted_index = ohlcv_df.index + pd.Timedelta(hours=1)
            signals = pd.Series(SignalAction.HOLD, index=shifted_index, dtype=object)
            engine = BacktestEngine(initial_capital=initial_capital)
            return signals, engine

        with pytest.raises(ValueError, match="combined slice index for split_index=0"):
            run_walk_forward(
                ohlcv_df=df,
                strategy_runner=shifted_index_strategy,
                strategy_id="misaligned_signals",
                config=config,
            )

    def test_deterministic_results(self):
        """Same input produces identical results (determinism)."""
        df = _make_ohlcv(25)
        config = WalkForwardConfig(train_window=5, test_window=3, step=3)
        strategy = _make_simple_strategy()

        result1 = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="det",
            config=config,
        )
        result2 = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="det",
            config=config,
        )

        assert result1.strategy_id == result2.strategy_id
        assert len(result1.windows) == len(result2.windows)
        for w1, w2 in zip(result1.windows, result2.windows, strict=True):
            assert w1.split_index == w2.split_index
            assert w1.metrics == w2.metrics

    def test_summary_aggregates_all_windows(self):
        """Summary correctly aggregates metrics across all windows."""
        df = _make_ohlcv(30)
        config = WalkForwardConfig(train_window=5, test_window=3, step=5)
        strategy = _make_simple_strategy()

        result = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=strategy,
            strategy_id="agg_test",
            config=config,
        )

        # All windows should have same count in summary
        assert result.summary.window_count == len(result.windows)

        # Mean should be computable from windows
        returns = [w.metrics["total_return"] for w in result.windows]
        if returns:
            expected_mean = sum(returns) / len(returns)
            assert result.summary.mean_out_of_sample_total_return == pytest.approx(expected_mean)
            assert result.summary.best_window_total_return == max(returns)
            assert result.summary.worst_window_total_return == min(returns)


# ---------------------------------------------------------------------------
# WalkForwardResult dataclass tests
# ---------------------------------------------------------------------------


class TestWalkForwardDataclasses:
    """Tests for walk-forward result dataclasses."""

    def test_walk_forward_config_frozen(self):
        """WalkForwardConfig is immutable (frozen=True)."""
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        with pytest.raises(AttributeError):
            config.train_window = 20

    def test_walk_forward_window_result_frozen(self):
        """WalkForwardWindowResult is immutable."""
        window = WalkForwardWindowResult(
            split_index=0,
            train_start_at_utc="2024-01-01",
            train_end_at_utc="2024-01-02",
            test_start_at_utc="2024-01-02",
            test_end_at_utc="2024-01-03",
            train_rows=10,
            test_rows=5,
            strategy_id="test",
            metrics={"total_return": 0.1},
        )
        with pytest.raises(AttributeError):
            window.split_index = 1

    def test_walk_forward_summary_frozen(self):
        """WalkForwardSummary is immutable."""
        summary = WalkForwardSummary(
            window_count=1,
            mean_out_of_sample_total_return=0.1,
            mean_out_of_sample_sharpe_ratio=1.0,
            best_window_total_return=0.2,
            worst_window_total_return=0.0,
        )
        with pytest.raises(AttributeError):
            summary.window_count = 2

    def test_walk_forward_result_contains_all_parts(self):
        """WalkForwardResult contains config, windows, and summary."""
        config = WalkForwardConfig(train_window=10, test_window=5, step=5)
        window = WalkForwardWindowResult(
            split_index=0,
            train_start_at_utc="2024-01-01",
            train_end_at_utc="2024-01-02",
            test_start_at_utc="2024-01-02",
            test_end_at_utc="2024-01-03",
            train_rows=10,
            test_rows=5,
            strategy_id="test",
            metrics={"total_return": 0.1},
        )
        summary = WalkForwardSummary(
            window_count=1,
            mean_out_of_sample_total_return=0.1,
            mean_out_of_sample_sharpe_ratio=1.0,
            best_window_total_return=0.1,
            worst_window_total_return=0.1,
        )
        result = WalkForwardResult(
            strategy_id="test",
            config=config,
            windows=[window],
            summary=summary,
        )
        assert result.strategy_id == "test"
        assert result.config == config
        assert len(result.windows) == 1
        assert result.summary == summary


# ---------------------------------------------------------------------------
# Existing strategy runner compatibility
# ---------------------------------------------------------------------------


class TestStrategyRunnerCompatibility:
    """Tests for compatibility with existing strategy runner surfaces."""

    def test_compatible_with_direct_runner_contract(self):
        """Walk-forward works with strategy runners returning (signals, BacktestEngine)."""
        # n=24 with train=10, test=5, step=10 gives exactly 1 split:
        # Split: (0, 10, 10, 15) fits, train_start=10 -> (10, 20, 20, 25) doesn't fit (25 > 24)
        df = _make_ohlcv(24)
        config = WalkForwardConfig(train_window=10, test_window=5, step=10)

        # This matches the contract used by existing strategies like run_vwap_anchored_backtest
        def direct_runner(
            ohlcv_df: pd.DataFrame,
            *,
            initial_capital: float,
        ) -> tuple[pd.Series, BacktestEngine]:
            signals = pd.Series(SignalAction.HOLD, index=ohlcv_df.index, dtype=object)
            signals.iloc[0] = SignalAction.LONG_ENTRY
            signals.iloc[-1] = SignalAction.LONG_EXIT
            engine = BacktestEngine(initial_capital=initial_capital)
            return signals, engine

        # Should not raise
        result = run_walk_forward(
            ohlcv_df=df,
            strategy_runner=direct_runner,
            strategy_id="direct_runner_test",
            config=config,
            initial_capital=50_000.0,
        )
        assert len(result.windows) == 1
