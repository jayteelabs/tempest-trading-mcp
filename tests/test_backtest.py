"""Tests for backtest engine (ENG-16)."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from tempest_mcp.backtest.commission import (
    calculate_commission,
    apply_slippage,
    calculate_net_pnl,
)
from tempest_mcp.backtest.engine import BacktestEngine, Trade, BacktestResult


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
        """Helper to create signals series."""
        signals = [0] * n
        for e in entries:
            if 0 <= e < n:
                signals[e] = 1
        for x in exits:
            if 0 <= x < n:
                signals[x] = -1
        return pd.Series(signals, index=pd.RangeIndex(n))

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
        # With price=100 and cash=100000, size should be 1000 (100000/100)
        n = 3
        ohlcv = self._make_ohlcv(n, start_price=100.0, step=1.0)
        signals = self._make_signals(n, entries=[0], exits=[1])
        engine = BacktestEngine(initial_capital=100000.0, commission_pct=0.0, slippage_bps=0.0)
        result = engine.run(ohlcv, signals)

        assert len(result.trades) == 1
        trade = result.trades[0]
        # size = round(100000 / 100, 8) = 1000
        assert trade.size == 1000.0
        # gross_pnl should be (exit - entry) * size
        assert abs(trade.gross_pnl - (ohlcv["close"].iloc[1] - ohlcv["open"].iloc[1]) * 1000) < 0.01

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
            direction=1,
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
        assert trade.direction == 1
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
