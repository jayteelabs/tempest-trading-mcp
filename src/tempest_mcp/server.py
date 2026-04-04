"""MCP Server entry point — stdio transport."""
import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tempest_mcp.config import ErrorCodes, get_config
from tempest_mcp.logging_config import get_logger, setup_logging
from tempest_mcp.tools import (
    fetch_ticker,
    fetch_klines,
    fetch_orderbook,
    indicator_rsi,
    backtest_strategy,
    screener_scan,
)

logger = get_logger(__name__)

# ── Tool Registry ─────────────────────────────────────────────────────────────
TOOLS: dict[str, Any] = {
    "fetch_ticker": fetch_ticker,
    "fetch_klines": fetch_klines,
    "fetch_orderbook": fetch_orderbook,
    "indicator_rsi": indicator_rsi,
    "backtest_strategy": backtest_strategy,
    "screener_scan": screener_scan,
}

# ── Tool Schemas (MCP protocol surface) ──────────────────────────────────────
TOOL_SCHEMAS: list[Tool] = [
    Tool(
        name="fetch_ticker",
        description="Fetch real-time ticker price + 24h volume for a crypto symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "exchange": {"type": "string", "default": "binance"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="fetch_klines",
        description="Fetch OHLCV klines for a symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": {"type": "string", "default": "1h"},
                "since": {"type": "string", "nullable": True},
                "limit": {"type": "integer", "default": 100},
                "exchange": {"type": "string", "default": "binance"},
                "source": {"type": "string", "default": "ccxt"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="fetch_orderbook",
        description="Fetch order book depth for a symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
                "exchange": {"type": "string", "default": "binance"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="indicator_rsi",
        description="Calculate RSI. Oscillator 0-100: <30 oversold, >70 overbought.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "period": {"type": "integer", "default": 14},
                "timeframe": {"type": "string", "default": "1h"},
                "limit": {"type": "integer", "default": 100},
                "exchange": {"type": "string", "default": "binance"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="backtest_strategy",
        description="Run a backtest for a single strategy on a symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "strategy_id": {"type": "string", "default": "rsi_mean_reversion"},
                "timeframe": {"type": "string", "default": "1h"},
                "period": {"type": "string", "default": "1y"},
                "initial_capital": {"type": "number", "default": 10000.0},
                "exchange": {"type": "string", "default": "binance"},
                "source": {"type": "string", "default": "yf"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="screener_scan",
        description="Multi-factor crypto screener.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {"type": "array", "items": {"type": "string"}, "nullable": True},
                "filters": {"type": "array", "items": {"type": "string"}, "nullable": True},
                "min_score": {"type": "number", "default": 0.0},
                "exchange": {"type": "string", "default": "binance"},
            },
        },
    ),
]


# ── Server ────────────────────────────────────────────────────────────────────
async def run_server() -> None:
    config = get_config()
    setup_logging()
    logger.info(
        "Starting MCP server",
        name=config.mcp_server_name,
        version=config.mcp_server_version,
    )
    server = Server(config.mcp_server_name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOL_SCHEMAS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        logger.info("Tool called", name=name)
        if name not in TOOLS:
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": {
                            "code": ErrorCodes.INVALID_PARAMETER,
                            "message": f"Unknown tool: {name}"
                        }
                    })
                )
            ]
        try:
            result = await TOOLS[name](**arguments)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            logger.error("Tool raised exception", name=name, error=str(e))
            return [
                TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": {
                            "code": ErrorCodes.INTERNAL_ERROR,
                            "message": str(e)
                        }
                    })
                )
            ]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
