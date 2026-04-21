"""MCP tool stubs — Phase 2 backtest tools. ENG-17."""

from tempest_mcp.tools.analysis_tools import calculate_volume_profile, detect_order_blocks
from tempest_mcp.tools.analytical_tools import (
    calculate_fibonacci,
    calculate_tpo,
    detect_elliot_wave,
    get_market_structure,
)
from tempest_mcp.tools.backtest_tools import (
    BACKTEST_TOOLS,
    backtest_strategy,
)
from tempest_mcp.tools.indicator_tools import indicator_rsi
from tempest_mcp.tools.market_tools import fetch_klines, fetch_orderbook, fetch_ticker
from tempest_mcp.tools.screener_tools import (
    order_block_screener_scan,
    screener_scan,
    session_breakout_scan,
)
from tempest_mcp.tools.sentiment_tools import (
    get_combined_sentiment_dashboard,
)

__all__ = [
    # Market tools
    "fetch_ticker",
    "fetch_klines",
    "fetch_orderbook",
    # Indicator tools
    "indicator_rsi",
    # Backtest tools (Phase 2 — ENG-17) — access via BACKTEST_TOOLS dict
    "backtest_strategy",  # legacy deprecated stub
    # Screener tools
    "screener_scan",
    "session_breakout_scan",  # ENG-35
    "order_block_screener_scan",  # ENG-36
    # Backtest tool registry for server registration
    "BACKTEST_TOOLS",
    # Analysis tools (ENG-28)
    "calculate_volume_profile",
    "detect_order_blocks",
    # Analytical tools (ENG-37)
    "calculate_fibonacci",
    "calculate_tpo",
    "detect_elliot_wave",
    "get_market_structure",
    # Sentiment tools (ENG-41)
    "get_combined_sentiment_dashboard",
]
