"""MCP Server entry point."""
import asyncio
import json
from typing import Any
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from tempest_mcp.config import ErrorCodes, get_config
from tempest_mcp.logging_config import get_logger, setup_logging
from tempest_mcp.tools import (
    fetch_ticker, fetch_klines, fetch_orderbook,
    indicator_ema, indicator_vwap, indicator_rsi, indicator_macd, indicator_atr, indicator_supertrend,
    indicator_session_levels, indicator_adx, indicator_stochastic, indicator_cci, indicator_williams_r,
    indicator_roc, indicator_bollinger_width, indicator_obv, indicator_mfi, indicator_historical_volatility,
    backtest_strategy, compare_strategies, screener_scan, session_breakout_scan,
)

logger = get_logger(__name__)

TOOLS = {
    "fetch_ticker": fetch_ticker, "fetch_klines": fetch_klines, "fetch_orderbook": fetch_orderbook,
    "indicator_ema": indicator_ema, "indicator_vwap": indicator_vwap, "indicator_rsi": indicator_rsi,
    "indicator_macd": indicator_macd, "indicator_atr": indicator_atr, "indicator_supertrend": indicator_supertrend,
    "indicator_session_levels": indicator_session_levels, "indicator_adx": indicator_adx,
    "indicator_stochastic": indicator_stochastic, "indicator_bollinger_width": indicator_bollinger_width,
    "indicator_obv": indicator_obv, "indicator_mfi": indicator_mfi, "indicator_historical_volatility": indicator_historical_volatility,
    "backtest_strategy": backtest_strategy, "compare_strategies": compare_strategies,
    "screener_scan": screener_scan, "session_breakout_scan": session_breakout_scan,
}

TOOL_SCHEMAS = [
    # Market Data (3)
    Tool(name="fetch_ticker", description="Fetch real-time ticker", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="fetch_klines", description="Fetch OHLCV klines", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}, "timeframe": {"type": "string", "default": "1h"}}, "required": ["symbol"]}),
    Tool(name="fetch_orderbook", description="Fetch orderbook depth", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    # Indicators (13)
    Tool(name="indicator_ema", description="Calculate EMA (Exponential Moving Average)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_vwap", description="Calculate VWAP (Volume Weighted Average Price)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_rsi", description="Calculate RSI (Relative Strength Index)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_macd", description="Calculate MACD", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_atr", description="Calculate ATR (Average True Range)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_supertrend", description="Calculate Supertrend indicator", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_session_levels", description="Calculate session levels", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_adx", description="Calculate ADX (Average Directional Index)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_stochastic", description="Calculate Stochastic oscillator", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_bollinger_width", description="Calculate Bollinger Band Width", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_obv", description="Calculate OBV (On-Balance Volume)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_mfi", description="Calculate MFI (Money Flow Index)", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="indicator_historical_volatility", description="Calculate historical volatility", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    # Backtest (2)
    Tool(name="backtest_strategy", description="Run backtest", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}, "strategy_id": {"type": "string"}}, "required": ["symbol"]}),
    Tool(name="compare_strategies", description="Compare multiple strategies", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
    # Screener (2)
    Tool(name="screener_scan", description="Scan for opportunities", inputSchema={"type": "object", "properties": {}}),
    Tool(name="session_breakout_scan", description="Detect session breakouts", inputSchema={"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}),
]

async def run_server():
    config = get_config()
    setup_logging()
    logger.info("Starting MCP server", name=config.mcp_server_name, version=config.mcp_server_version)
    server = Server(config.mcp_server_name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOL_SCHEMAS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        logger.info("Tool called", name=name)
        if name not in TOOLS:
            return [TextContent(type="text", text=json.dumps({"success": False, "error": {"code": ErrorCodes.INVALID_PARAMETER, "message": f"Unknown tool: {name}"}}))]
        try:
            result = await TOOLS[name](**arguments)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"success": False, "error": {"code": ErrorCodes.INTERNAL_ERROR, "message": str(e)}}))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

def main():
    asyncio.run(run_server())

if __name__ == "__main__":
    main()
