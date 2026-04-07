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


# ---------------------------------------------------------------------------
# Standalone commission/slippage functions (per ENG-16 spec)
# ---------------------------------------------------------------------------


def calculate_commission(trade_value: float, commission_pct: float = 0.001) -> float:
    """Fixed percentage commission on trade notional.

    Args:
        trade_value: Notional value of the trade (price × size).
        commission_pct: Commission rate (default 0.001 = 0.1% per side, Binance Futures taker).

    Returns:
        Commission amount. Returns 0.0 if trade_value <= 0.
    """
    if trade_value <= 0:
        return 0.0
    return trade_value * commission_pct


def apply_slippage(price: float, size: float, direction: int, slippage_bps: float = 5.0) -> float:
    """Asymmetric slippage: buy pays more, sell receives less.

    Args:
        price: Base price before slippage.
        size: Position size (used only for validation).
        direction: 1 for buy, -1 for sell.
        slippage_bps: Slippage in basis points (default 5.0 bps).

    Returns:
        Executed price after slippage. Returns price unchanged if price <= 0 or size <= 0.
    """
    if price <= 0 or size <= 0:
        return price
    if direction == 1:  # buy
        return price * (1 + slippage_bps / 10000)
    else:  # sell (direction == -1)
        return price * (1 - slippage_bps / 10000)


def calculate_net_pnl(
    entry_price: float,
    exit_price: float,
    size: float,
    commission_pct: float = 0.001,
    slippage_bps: float = 5.0,
) -> float:
    """Standalone analytical PnL utility (NOT called by BacktestEngine internally).

    Args:
        entry_price: Entry execution price.
        exit_price: Exit execution price.
        size: Position size.
        commission_pct: Commission rate per side (default 0.001 = 0.1%).
        slippage_bps: Slippage in basis points (default 5.0 bps).

    Returns:
        Net PnL after commission and slippage.
    """
    entry_value = entry_price * size
    exit_value = exit_price * size
    entry_commission = entry_value * commission_pct
    exit_commission = exit_value * commission_pct
    avg_price = (entry_price + exit_price) / 2
    slippage_cost = size * (slippage_bps / 10000) * avg_price
    gross_pnl = (exit_price - entry_price) * size
    return gross_pnl - entry_commission - exit_commission - slippage_cost
