"""Commission and slippage model."""
from dataclasses import dataclass
from enum import Enum
from tempest_mcp.models.backtest import OrderSide

class SlippageModel(Enum):
    FIXED = "fixed"
    VOLUME_BASED = "volume_based"
    VOLATILITY_BASED = "volatility_based"

@dataclass
class CommissionModel:
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    min_commission: float = 0.0
    slippage_model: SlippageModel = SlippageModel.FIXED

    def calculate_commission(self, price: float, volume: float) -> float:
        commission = price * volume * self.commission_rate
        return max(commission, self.min_commission)

    def apply_slippage(self, price: float, side: OrderSide, volume: float | None = None, atr: float | None = None) -> float:
        slippage = self.slippage_rate
        if side == OrderSide.BUY:
            return price * (1 + slippage)
        return price * (1 - slippage)

    def calculate_total_cost(self, price: float, volume: float, side: OrderSide, atr: float | None = None):
        executed_price = self.apply_slippage(price, side, volume, atr)
        commission = self.calculate_commission(executed_price, volume)
        slippage_cost = abs(executed_price - price) * volume
        return executed_price, commission, slippage_cost

def create_binance_model() -> CommissionModel:
    return CommissionModel(commission_rate=0.001, slippage_rate=0.0005)

def create_bybit_model() -> CommissionModel:
    return CommissionModel(commission_rate=0.001, slippage_rate=0.0005)
