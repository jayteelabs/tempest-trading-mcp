"""Internal MCP tool handler registry."""

from collections.abc import Callable
from typing import Any

from tempest_mcp.tools import (
    BACKTEST_TOOLS,
    backtest_strategy,
    calculate_volume_profile,
    detect_order_blocks,
    fetch_klines,
    fetch_orderbook,
    fetch_ticker,
    get_combined_sentiment_dashboard,
    indicator_rsi,
    order_block_screener_scan,
    screener_scan,
    session_breakout_scan,
)
from tempest_mcp.tools.analytical_tools import (
    calculate_fibonacci,
    calculate_tpo,
    detect_elliot_wave,
    get_market_structure,
)

ToolHandler = Callable[..., Any]

# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOLS: dict[str, ToolHandler] = {
    "fetch_ticker": fetch_ticker,
    "fetch_klines": fetch_klines,
    "fetch_orderbook": fetch_orderbook,
    "indicator_rsi": indicator_rsi,
    "screener_scan": screener_scan,
    "session_breakout_scan": session_breakout_scan,
    "order_block_screener_scan": order_block_screener_scan,
    # Legacy backtest_strategy (deprecated — deterministic error response)
    "backtest_strategy": backtest_strategy,
}
# Phase 2 dedicated backtest tools (ENG-17) — populate from BACKTEST_TOOLS registry
TOOLS.update(BACKTEST_TOOLS)
# Phase 2 analysis tools (ENG-28)
TOOLS["calculate_volume_profile"] = calculate_volume_profile
TOOLS["detect_order_blocks"] = detect_order_blocks
# ENG-37 analytical tools
TOOLS["calculate_fibonacci"] = calculate_fibonacci
TOOLS["calculate_tpo"] = calculate_tpo
TOOLS["detect_elliot_wave"] = detect_elliot_wave
TOOLS["get_market_structure"] = get_market_structure
# ENG-41 sentiment tools
TOOLS["get_combined_sentiment_dashboard"] = get_combined_sentiment_dashboard



def lookup_handler(name: str) -> ToolHandler | None:
    """Return the registered handler for a tool name, or None when unknown."""
    return TOOLS.get(name)
