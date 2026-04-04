"""Tests for backtest engine."""
from datetime import datetime

import pytest

from tempest_mcp.backtest.commission import CommissionModel
from tempest_mcp.backtest.engine import BacktestEngine
from tempest_mcp.models.backtest import OrderSide
from tempest_mcp.models.market import Kline


class TestCommission:
    def test_commission(self):
        model = CommissionModel()
        commission = model.calculate_commission(50000.0, 0.1)
        assert abs(commission - 5.0) < 0.01

    def test_slippage(self):
        model = CommissionModel(slippage_rate=0.001)
        assert model.apply_slippage(50000.0, OrderSide.BUY) == 50000.0 * 1.001

class TestEngine:
    @pytest.fixture
    def sample_klines(self):
        return [Kline(timestamp=datetime(2024, 1, 1).timestamp() + i * 3600, open=100 + i * 0.5, high=105 + i * 0.5, low=95 + i * 0.5, close=100 + i * 0.5, volume=1000.0) for i in range(100)]

    def test_init(self):
        engine = BacktestEngine(initial_capital=10000.0)
        assert engine.initial_capital == 10000.0

    def test_run(self, sample_klines):
        engine = BacktestEngine(initial_capital=10000.0)
        result = engine.run(klines=sample_klines, strategy_func=lambda ctx: 1 if ctx["position"] is None else 0, strategy_id="test", symbol="BTC/USDT", timeframe="1h")
        assert result.total_trades >= 1
