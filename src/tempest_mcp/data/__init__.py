"""
Data layer for Tempest MCP.

This module provides adapters for fetching market data from various sources:
- Yahoo Finance: Historical data for backtesting
- TradingView: Real-time data (primary when API key is set)
- CCXT: Real-time data fallback / orderbook data

Adapter Selection:
- If TRADINGVIEW_API_KEY is set → TradingViewAdapter (primary)
- Otherwise → CCXTAdapter (fallback)

Design Decisions:
- D3: Yahoo Finance + TradingView + CCXT data adapters
- D11: Symbol conversion via _symbols.py
- D16: TradingViewAdapter delegates orderbook to CCXT
"""

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import structlog

if TYPE_CHECKING:
    import pandas as pd

logger = structlog.get_logger()


@runtime_checkable
class LiveDataAdapter(Protocol):
    """Protocol defining the interface for live data adapters.
    
    All adapters must implement these three methods:
    - fetch_live_price: Get latest trade price
    - fetch_ohlcv_live: Get OHLCV candlestick data
    - fetch_orderbook_snapshot: Get order book depth
    """
    
    def fetch_live_price(
        self,
        symbol: str,
        exchange: str = "binance",
    ) -> float:
        """Fetch latest trade price for a symbol.
        
        Args:
            symbol: Symbol in any supported format
            exchange: Target exchange (default: "binance")
        
        Returns:
            Latest trade price as float, or float('nan') on error
        """
        ...
    
    def fetch_ohlcv_live(
        self,
        symbol: str,
        timeframe: str = "1m",
        limit: int = 100,
    ) -> "pd.DataFrame":
        """Fetch OHLCV candlestick data.
        
        Args:
            symbol: Symbol in any supported format
            timeframe: Timeframe string (e.g., "1m", "5m", "1h")
            limit: Number of candles to fetch
        
        Returns:
            DataFrame with OHLCV columns and UTC-aware index,
            or empty DataFrame on error
        """
        ...
    
    def fetch_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 20,
    ) -> dict:
        """Fetch order book snapshot.
        
        Args:
            symbol: Symbol in any supported format
            limit: Depth of orderbook to fetch
        
        Returns:
            Dict with bids, asks, and timestamp,
            or empty structure on error
        """
        ...


def get_live_adapter() -> LiveDataAdapter:
    """Get the appropriate live data adapter based on environment.
    
    Selection logic:
    1. If TRADINGVIEW_API_KEY is set → TradingViewAdapter (primary)
    2. Otherwise → CCXTAdapter (fallback)
    
    Returns:
        LiveDataAdapter instance (TradingViewAdapter or CCXTAdapter)
    
    Example:
        >>> adapter = get_live_adapter()
        >>> price = adapter.fetch_live_price("BTCUSDT")
    """
    api_key = os.environ.get("TRADINGVIEW_API_KEY")
    
    if api_key:
        from tempest_mcp.data.tv_adapter import TradingViewAdapter
        
        logger.info(
            "adapter_selected",
            adapter="TradingViewAdapter",
            reason="TRADINGVIEW_API_KEY is set",
        )
        return TradingViewAdapter(api_key=api_key)
    else:
        from tempest_mcp.data.ccxt_adapter import CCXTAdapter
        
        logger.warning(
            "adapter_selected",
            adapter="CCXTAdapter",
            reason="TRADINGVIEW_API_KEY not set - using CCXT fallback",
        )
        return CCXTAdapter()


# Re-export key components
from tempest_mcp.data._symbols import (
    get_base_currency,
    normalize_to_ccxt,
    normalize_to_tradingview,
    validate_symbol,
)

__all__ = [
    "LiveDataAdapter",
    "get_live_adapter",
    "normalize_to_ccxt",
    "normalize_to_tradingview",
    "get_base_currency",
    "validate_symbol",
]
