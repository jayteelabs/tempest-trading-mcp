"""Tests for backtest engine (ENG-16)."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.backtest.commission import (
    apply_slippage,
    calculate_commission,
    calculate_net_pnl,
)
from tempest_mcp.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    PositionDirection,
    SignalAction,
    Trade,
)


class TestCommissionFunctions:
    """Tests for standalone commission/slippage functions."""

    def test_calculate_commission_positive(self):
        result = calculate_commission(50000.0, 0.001)
        assert result == 50.0

    def test_calculate_commission_default_rate(self):
        result = calculate_commission(100000.0)
        assert result == 100.0  # 0.1% of 100000

    def test_calculate_commission_zero_value(self):
        result = calculate_commission(0.0)
        assert result == 0.0

    def test_calculate_commission_negative_value(self):
        result = calculate_commission(-1000.0)
        assert result == 0.0

    def test_apply_slippage_buy(self):
        # 5 bps = 0.0005
        result = apply_slippage(50000.0, 1.0, 1, 5.0)
        assert result == 50000.0 * (1 + 5 / 10000)

    def test_apply_slippage_sell(self):
        result = apply_slippage(50000.0, 1.0, -1, 5.0)
        assert result == 50000.0 * (1 - 5 / 10000)

    def test_apply_slippage_zero_price(self):
        result = apply_slippage(0.0, 1.0, 1, 5.0)
        assert result == 0.0

    def test_apply_slippage_negative_price(self):
        result = apply_slippage(-100.0, 1.0, 1, 5.0)
        assert result == -100.0

    def test_apply_slippage_zero_size(self):
        result = apply_slippage(50000.0, 0.0, 1, 5.0)
        assert result == 50000.0

    def test_apply_slippage_default_bps(self):
        result = apply_slippage(10000.0, 1.0, 1)
        assert result == 10000.0 * (1 + 5 / 10000)

    def test_calculate_net_pnl_standard(self):
        # entry=100, exit=110, size=100
        # gross_pnl = (110-100) * 100 = 1000
        # entry_comm = 100 * 100 * 0.001 = 10
        # exit_comm = 110 * 100 * 0.001 = 11
        # avg_price = 105
        # slippage_cost = 100 * 0.0005 * 105 = 5.25
        # net_pnl = 1000 - 10 - 11 - 5.25 = 973.75
        result = calculate_net_pnl(100.0, 110.0, 100.0, 0.001, 5.0)
        expected = 973.75
        assert abs(result - expected) < 0.01

    def test_calculate_net_pnl_losing_trade(self):
        # entry=100, exit=90, size=100
        # gross_pnl = (90-100) * 100 = -1000
        result = calculate_net_pnl(100.0, 90.0, 100.0, 0.001, 5.0)
        assert result < 0

    def test_calculate_net_pnl_zero_size(self):
        result = calculate_net_pnl(100.0, 110.0, 0.0, 0.001, 5.0)
        assert result == 0.0


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def _make_ohlcv(self, n: int, start_price: float = 100.0, step: float = 0.5) -> pd.DataFrame:
        """Helper to create OHLCV DataFrame."""
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        data = {
            "open": [start_price + i * step for i in range(n)],
            "high": [start_price + i * step + 1 for i in range(n)],
            "low": [start_price + i * step - 1 for i in range(n)],
            "close": [start_price + i * step for i in range(n)],
            "volume": [1000.0] * n,
        }
        return pd.DataFrame(data, index=pd.DatetimeIndex(times))

    def _make_signals(self, n: int, entries: list[int], exits: list[int]) -> pd.Series:
        """Helper to create signals series with matching DatetimeIndex."""
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        signals = pd.Series(0, index=pd.DatetimeIndex(times))
        for e in entries:
            if 0 <= e < n:
                signals.iloc[e] = 1
        for x in exits:
            if 0 <= x < n:
                signals.iloc[x] = -1
        return signals

    def test_insufficient_data_raises(self):
        df = self._make_ohlcv(1)
        signals = pd.Series([0])
        engine = BacktestEngine()
        with pytest.raises(ValueError):
            engine.run(df, signals)

    def test_single_trade_entry_then_exit(self):
        # Bar 0: price=100, signal=1 (entry)
        # Bar 1: price=110, no signal
        # Bar 2: price=120, signal=-1 (exit)
        # Entry executes at bar 1 open=105 (slippage), exit at bar 3 open=125 (slippage)
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=5.0)
        signals = self._make_signals(n, entries=[0], exits=[2])
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.001, slippage_bps=5.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.entry_price > 100.0  # slippage applied
        assert trade.exit_price > 110.0  # slippage applied
        assert trade.net_pnl > 0
        assert result.open_position is False

    def test_losing_trade(self):
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=-5.0)  # falling prices
        signals = self._make_signals(n, entries=[0], exits=[2])
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.net_pnl < 0

    def test_multiple_trades_sequence(self):
        # Three complete round trips
        n = 8
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0, 2, 4], exits=[1, 3, 5])
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 3
        for trade in result.trades:
            assert trade.net_pnl is not None
            assert trade.commission > 0

    def test_flat_position_hold_at_end(self):
        # Entry at bar 0, never exit
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0], exits=[])  # no exit
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 0  # position not closed
        assert result.open_position is True

    def test_consecutive_entry_signals_no_op(self):
        # Two consecutive entry signals, only first should open position
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0, 1], exits=[3])  # second entry is no-op
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1

    def test_consecutive_exit_signals_no_op(self):
        # Two consecutive exit signals, only first should close
        n = 6
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0], exits=[2, 3])  # second exit is no-op
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1

    def test_flat_to_position_to_flat(self):
        # Complete round trip
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0], exits=[2])
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        assert result.open_position is False
        assert result.final_equity != result.initial_capital

    def test_position_sizing_round_to_8_decimals(self):
        # Entry at bar 1 open = 100.0 (start_price=99, step=1), exit at bar 2 open = 101.0
        # With cash=100000 and entry_price=100, size = round(100000/100, 8) = 1000
        n = 3
        ohlcv = self._make_ohlcv(n, start_price=99.0, step=1.0)  # bar 1 open = 100.0
        signals = self._make_signals(n, entries=[0], exits=[1])
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        # entry at bar 1 open = 100.0 (start_price=99), size = round(100000/100, 8) = 1000
        assert trade.size == 1000.0
        # gross_pnl = (exit_price - entry_price) * size = (101 - 100) * 1000 = 1000
        assert abs(trade.gross_pnl - (101.0 - 100.0) * 1000) < 0.01

    def test_metrics_computed(self):
        n = 6
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=2.0)
        signals = self._make_signals(n, entries=[0], exits=[3])
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        metrics = result.metrics
        assert "total_return" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "max_drawdown" in metrics
        assert "expectancy" in metrics
        assert "sharpe_ratio" in metrics
        assert "total_trades" in metrics

    def test_sharpe_ratio_zero_std(self):
        # Flat equity curve should give sharpe = 0.0
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=0.0)  # no price change
        signals = self._make_signals(n, entries=[0], exits=[2])
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert result.metrics["sharpe_ratio"] == 0.0

    def test_equity_curve_index_matches_timestamps(self):
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0], exits=[2])
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        # Equity curve starts from bar 1 (index 1 onwards)
        assert len(result.equity_curve) == n - 1
        # First equity should be after first trade execution
        assert result.equity_curve.iloc[0] is not None

    def test_no_trades_no_position(self):
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = pd.Series([0] * n)  # no signals at all
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 0
        assert result.open_position is False
        assert result.final_equity == result.initial_capital


class TestTradeDataclass:
    """Tests for Trade dataclass fields."""

    def test_trade_fields_present(self):
        trade = Trade(
            entry_time=pd.Timestamp("2024-01-01"),
            exit_time=pd.Timestamp("2024-01-02"),
            entry_price=100.0,
            exit_price=110.0,
            size=100.0,
            direction=PositionDirection.LONG,
            gross_pnl=1000.0,
            net_pnl=973.75,
            commission=26.0,
            slippage_cost=5.25,
            bars_held=5,
        )
        assert trade.entry_time == pd.Timestamp("2024-01-01")
        assert trade.exit_time == pd.Timestamp("2024-01-02")
        assert trade.entry_price == 100.0
        assert trade.exit_price == 110.0
        assert trade.size == 100.0
        assert trade.direction == PositionDirection.LONG
        assert trade.gross_pnl == 1000.0
        assert trade.net_pnl == 973.75
        assert trade.commission == 26.0
        assert trade.slippage_cost == 5.25
        assert trade.bars_held == 5


class TestBacktestResultDataclass:
    """Tests for BacktestResult dataclass."""

    def test_backtest_result_fields(self):
        result = BacktestResult(
            trades=[],
            equity_curve=pd.Series([100000.0]),
            metrics={"total_return": 0.0},
            open_position=False,
            initial_capital=100000.0,
            final_equity=100000.0,
        )
        assert len(result.trades) == 0
        assert result.equity_curve.iloc[0] == 100000.0
        assert result.open_position is False
        assert result.initial_capital == 100000.0
        assert result.final_equity == 100000.0


class TestSignalActionEnum:
    """Tests for SignalAction enum (ENG-58)."""

    def test_signal_action_values(self):
        assert SignalAction.LONG_ENTRY.value == "long_entry"
        assert SignalAction.LONG_EXIT.value == "long_exit"
        assert SignalAction.SHORT_ENTRY.value == "short_entry"
        assert SignalAction.SHORT_EXIT.value == "short_exit"
        assert SignalAction.HOLD.value == "hold"

    def test_signal_action_from_int_1(self):
        assert SignalAction.from_int(1) == SignalAction.LONG_ENTRY

    def test_signal_action_from_int_minus_1(self):
        assert SignalAction.from_int(-1) == SignalAction.LONG_EXIT

    def test_signal_action_from_int_0(self):
        assert SignalAction.from_int(0) == SignalAction.HOLD

    def test_signal_action_from_int_invalid(self):
        with pytest.raises(ValueError, match="Invalid signal int value"):
            SignalAction.from_int(99)

    def test_signal_action_from_int_bool_rejected(self):
        with pytest.raises(TypeError, match="Boolean signals are not supported"):
            SignalAction.from_int(True)


class TestPositionDirectionEnum:
    """Tests for PositionDirection enum (ENG-58)."""

    def test_position_direction_values(self):
        assert PositionDirection.FLAT.value == "flat"
        assert PositionDirection.LONG.value == "long"
        assert PositionDirection.SHORT.value == "short"


class TestBidirectionalEngine:
    """Tests for bidirectional backtest engine (ENG-58)."""

    def _make_ohlcv(self, n: int, start_price: float = 100.0, step: float = 0.5) -> pd.DataFrame:
        """Helper to create OHLCV DataFrame."""
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        data = {
            "open": [start_price + i * step for i in range(n)],
            "high": [start_price + i * step + 1 for i in range(n)],
            "low": [start_price + i * step - 1 for i in range(n)],
            "close": [start_price + i * step for i in range(n)],
            "volume": [1000.0] * n,
        }
        return pd.DataFrame(data, index=pd.DatetimeIndex(times))

    def _make_signal_series(self, n: int, mapping: dict[int, SignalAction]) -> pd.Series:
        """Helper to create SignalAction series with matching DatetimeIndex."""
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        signals = pd.Series(SignalAction.HOLD, index=pd.DatetimeIndex(times))
        for idx, action in mapping.items():
            if 0 <= idx < n:
                signals.iloc[idx] = action
        return signals

    def test_short_trade_lifecycle(self):
        """Short trade: price falls, profit when short."""
        # Bar 0: price=100, SHORT_ENTRY
        # Bar 1: price=95 (falls), no signal
        # Bar 2: price=90, SHORT_EXIT
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=-5.0)  # falling prices
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY, 2: SignalAction.SHORT_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.001, slippage_bps=5.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.direction == PositionDirection.SHORT
        # For short: entry_price > exit_price (short at high, cover at low for profit)
        assert trade.entry_price > trade.exit_price
        assert trade.net_pnl > 0  # profit on short
        assert result.open_position is False

    def test_flat_to_short_to_flat(self):
        """Complete short round trip: FLAT -> SHORT -> FLAT."""
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=-5.0)
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY, 2: SignalAction.SHORT_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        assert result.trades[0].direction == PositionDirection.SHORT
        assert result.open_position is False

    def test_short_loss_when_price_rises(self):
        """Verify short PnL formula: loss when exit_price > entry_price."""
        n = 4
        # entry at bar 1 open = 100 (start_price=99, step=1)
        ohlcv = self._make_ohlcv(n, start_price=99.0, step=1.0)  # bar 1 open = 100, bar 2 open = 101
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY, 1: SignalAction.SHORT_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        trade = result.trades[0]
        # Short: entry=100, exit=101, size=1000
        # gross_pnl = (entry - exit) * size = (100 - 101) * 1000 = -1000 (loss)
        assert trade.gross_pnl < 0

    def test_short_profit_when_price_falls(self):
        """Verify short PnL formula: profit when entry_price > exit_price."""
        n = 4
        # price falls: start=99, step=-1 so bar 1 open = 98 (entry), bar 2 open = 97
        ohlcv = self._make_ohlcv(n, start_price=99.0, step=-1.0)  # bar 1 open = 98, bar 2 open = 97
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY, 1: SignalAction.SHORT_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        trade = result.trades[0]
        # Short: entry=98, exit=97, size=~1020
        # gross_pnl = (entry - exit) * size = (98 - 97) * size > 0 (profit)
        assert trade.gross_pnl > 0

    def test_long_short_sequence_through_flat(self):
        """LONG -> FLAT -> SHORT -> FLAT (valid state transitions)."""
        n = 8
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        # Bar 0: LONG_ENTRY, Bar 2: LONG_EXIT (go flat), Bar 4: SHORT_ENTRY, Bar 6: SHORT_EXIT
        signals = self._make_signal_series(
            n,
            {
                0: SignalAction.LONG_ENTRY,
                2: SignalAction.LONG_EXIT,
                4: SignalAction.SHORT_ENTRY,
                6: SignalAction.SHORT_EXIT,
            },
        )
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 2
        assert result.trades[0].direction == PositionDirection.LONG
        assert result.trades[1].direction == PositionDirection.SHORT
        assert result.open_position is False

    def test_invalid_long_to_short_direct_flip_raises(self):
        """LONG -> SHORT without FLAT intermediate should raise ValueError."""
        n = 6
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        # Bar 0: LONG_ENTRY, Bar 2: SHORT_ENTRY (direct flip - invalid)
        signals = self._make_signal_series(
            n,
            {
                0: SignalAction.LONG_ENTRY,
                2: SignalAction.SHORT_ENTRY,
            },
        )
        engine = BacktestEngine(initial_capital=100000.0)
        with pytest.raises(ValueError, match="Invalid transition.*cannot SHORT_ENTRY"):
            engine.run(ohlcv, signals)

    def test_invalid_short_to_long_direct_flip_raises(self):
        """SHORT -> LONG without FLAT intermediate should raise ValueError."""
        n = 6
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        # Bar 0: SHORT_ENTRY, Bar 2: LONG_ENTRY (direct flip - invalid)
        signals = self._make_signal_series(
            n,
            {
                0: SignalAction.SHORT_ENTRY,
                2: SignalAction.LONG_ENTRY,
            },
        )
        engine = BacktestEngine(initial_capital=100000.0)
        with pytest.raises(ValueError, match="Invalid transition.*cannot LONG_ENTRY"):
            engine.run(ohlcv, signals)

    def test_short_exit_without_position_is_noop(self):
        """SHORT_EXIT when flat is a no-op (no position to close)."""
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_EXIT})  # no entry first
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)  # should not raise, just no-op
        assert len(result.trades) == 0
        assert result.open_position is False

    def test_long_exit_without_position_is_noop(self):
        """LONG_EXIT when flat is a no-op (no position to close)."""
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signal_series(n, {0: SignalAction.LONG_EXIT})  # no entry first
        engine = BacktestEngine(initial_capital=100000.0)
        result = engine.run(ohlcv, signals)  # should not raise, just no-op
        assert len(result.trades) == 0
        assert result.open_position is False

    def test_short_trade_with_commission_and_slippage(self):
        """Short trade correctly accounts for commission and slippage."""
        n = 4
        # start_price=100, step=-5 so bar 1 open=95 (entry), bar 3 open=85 (exit)
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=-5.0)
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY, 2: SignalAction.SHORT_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.001, slippage_bps=5.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.commission > 0
        assert trade.slippage_cost > 0
        # net_pnl should be less than gross_pnl due to costs
        assert abs(trade.net_pnl) < abs(trade.gross_pnl)

    def test_long_trade_still_works_correctly(self):
        """Verify existing long-only behavior is preserved."""
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=5.0)  # rising prices
        signals = self._make_signal_series(n, {0: SignalAction.LONG_ENTRY, 2: SignalAction.LONG_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.direction == PositionDirection.LONG
        assert trade.entry_price < trade.exit_price  # entry < exit for long profit
        assert trade.net_pnl > 0

    def test_short_position_sizing(self):
        """Short position size is based on cash, same as long."""
        n = 3
        # bar 1 open = 100.0 (start_price=99, step=1)
        ohlcv = self._make_ohlcv(n, start_price=99.0, step=1.0)
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY, 1: SignalAction.SHORT_EXIT})
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        trade = result.trades[0]
        # size = round(100000 / 100, 8) = 1000
        assert trade.size == 1000.0

    def test_equity_tracks_unrealized_short_pnl(self):
        """Equity correctly reflects unrealized PnL for open short position."""
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=-5.0)  # falling prices
        signals = self._make_signal_series(n, {0: SignalAction.SHORT_ENTRY})  # no exit
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        # Position still open
        assert result.open_position is True
        assert len(result.trades) == 0
        # Equity should increase as price falls (short profits)
        assert result.final_equity > result.initial_capital

    def test_equity_tracks_unrealized_long_pnl(self):
        """Equity correctly reflects unrealized PnL for open long position."""
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=5.0)  # rising prices
        signals = self._make_signal_series(n, {0: SignalAction.LONG_ENTRY})  # no exit
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        # Position still open
        assert result.open_position is True
        assert len(result.trades) == 0
        # Equity should increase as price rises (long profits)
        assert result.final_equity > result.initial_capital

    def test_multiple_short_trades_sequence(self):
        """Multiple short round trips in sequence."""
        n = 8
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=-1.0)  # falling
        signals = self._make_signal_series(
            n,
            {
                0: SignalAction.SHORT_ENTRY,
                2: SignalAction.SHORT_EXIT,
                4: SignalAction.SHORT_ENTRY,
                6: SignalAction.SHORT_EXIT,
            },
        )
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 2
        for trade in result.trades:
            assert trade.direction == PositionDirection.SHORT

    def test_signal_action_enum_series_input(self):
        """Engine accepts SignalAction enum values directly in signals Series."""
        n = 4
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=5.0)
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        signals = pd.Series(
            [SignalAction.LONG_ENTRY, SignalAction.HOLD, SignalAction.LONG_EXIT, SignalAction.HOLD],
            index=pd.DatetimeIndex(times),
        )
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        assert result.trades[0].direction == PositionDirection.LONG

    def test_float_signal_series_with_nan_is_normalized(self):
        """Float signals containing NaN should normalize to HOLD instead of crashing."""
        n = 5
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        signals = pd.Series([1.0, np.nan, -1.0], index=pd.DatetimeIndex(times[:3]), dtype=float)
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        assert result.trades[0].direction == PositionDirection.LONG
        assert result.open_position is False

    def test_bool_signal_series_is_rejected(self):
        """Boolean legacy signals should be rejected explicitly instead of coercing True/False."""
        n = 4
        ohlcv = self._make_ohlcv(n)
        times = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(n)]
        signals = pd.Series([True, False], index=pd.DatetimeIndex(times[:2]), dtype=bool)
        engine = BacktestEngine(initial_capital=100000.0)

        with pytest.raises(TypeError, match="Boolean signals are not supported"):
            engine.run(ohlcv, signals)

    def test_missing_required_ohlcv_columns_raise(self):
        """Missing OHLCV columns should fail fast with ValueError."""
        ohlcv = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1000.0],
            },
            index=pd.date_range("2024-01-01", periods=2, freq="h"),
        )
        signals = pd.Series([SignalAction.HOLD, SignalAction.HOLD], index=ohlcv.index)
        engine = BacktestEngine(initial_capital=100000.0)

        with pytest.raises(ValueError, match="missing required columns: low"):
            engine.run(ohlcv, signals)
