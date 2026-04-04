"""MCP Server entry point — SSE/HTTP transport."""

import json
import re
import time
from typing import Any

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.routing import Route

from tempest_mcp.config import ErrorCodes, get_config
from tempest_mcp.logging_config import get_logger, setup_logging
from tempest_mcp.tools import (
    backtest_strategy,
    fetch_klines,
    fetch_orderbook,
    fetch_ticker,
    indicator_rsi,
    screener_scan,
)

# Server listens on port 9001 for HTTP/SSE transport
# Binds to 127.0.0.1 — portmapped externally via Docker
SERVER_PORT = 9001
SERVER_HOST = "127.0.0.1"

# Rate limiting: max requests per client per window
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60

# Symbol format: uppercase alphanumeric, 2-20 chars
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{2,20}$")

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


# ── Rate Limiting ──────────────────────────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter per client IP."""

    def __init__(self, app: Any, max_requests: int, window_seconds: int) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}

    def _clean_old(self, client_ip: str, now: float) -> None:
        if client_ip in self._requests:
            self._requests[client_ip] = [
                t for t in self._requests[client_ip] if now - t < self.window_seconds
            ]

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Only rate-limit /messages (not /sse which is streaming)
        if request.url.path != "/messages":
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        self._clean_old(client_ip, now)

        if client_ip not in self._requests:
            self._requests[client_ip] = []

        if len(self._requests[client_ip]) >= self.max_requests:
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            from starlette.responses import JSONResponse

            return JSONResponse(
                {"success": False, "error": {"code": 429, "message": "Rate limit exceeded"}},
                status_code=429,
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


# ── Input Validation ──────────────────────────────────────────────────────────
def validate_symbol(symbol: str, field_name: str = "symbol") -> str | None:
    """Validate symbol format. Returns None if valid, error message if invalid."""
    if not isinstance(symbol, str):
        return f"{field_name} must be a string"
    if not SYMBOL_PATTERN.match(symbol):
        return f"Invalid {field_name} format: {symbol!r} — must be 2-20 uppercase alphanumeric characters"
    return None


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> str | None:
    """Validate tool arguments. Returns error message or None if valid."""
    if name == "fetch_ticker":
        return validate_symbol(arguments.get("symbol", ""))
    if name == "fetch_klines":
        return validate_symbol(arguments.get("symbol", ""))
    if name == "fetch_orderbook":
        return validate_symbol(arguments.get("symbol", ""))
    if name == "indicator_rsi":
        return validate_symbol(arguments.get("symbol", ""))
    if name == "backtest_strategy":
        return validate_symbol(arguments.get("symbol", ""))
    return None


# ── Server ────────────────────────────────────────────────────────────────────
def create_app() -> Starlette:
    """Create the Starlette ASGI app with SSE transport."""
    config = get_config()
    setup_logging()
    logger.info(
        "Starting MCP server",
        name=config.mcp_server_name,
        version=config.mcp_server_version,
        transport="sse",
        port=SERVER_PORT,
        host=SERVER_HOST,
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
                    text=json.dumps(
                        {
                            "success": False,
                            "error": {
                                "code": ErrorCodes.INVALID_PARAMETER,
                                "message": f"Unknown tool: {name}",
                            },
                        }
                    ),
                )
            ]
        # Validate inputs
        if validation_error := validate_tool_arguments(name, arguments):
            logger.warning("Validation failed", name=name, error=validation_error)
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": {
                                "code": ErrorCodes.INVALID_PARAMETER,
                                "message": validation_error,
                            },
                        }
                    ),
                )
            ]
        try:
            result = await TOOLS[name](**arguments)
            return [TextContent(type="text", text=json.dumps(result))]
        except Exception as e:
            # Log full error server-side; return generic message to client
            logger.error("Tool raised exception", name=name, error=str(e))
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "success": False,
                            "error": {
                                "code": ErrorCodes.INTERNAL_ERROR,
                                "message": "An internal error occurred",
                            },
                        }
                    ),
                )
            ]

    transport = SseServerTransport("/messages")

    async def sse_handler(scope: dict, receive: callable, send: callable) -> None:
        # TODO: Add authentication before allowing SSE connection.
        # Currently the service binds to 127.0.0.1 and is accessible only via
        # Docker port mapping + VPS firewall + Tailscale. Production deployment
        # may require API key, Bearer token, or mTLS. Investigate MCP HTTP/SSE
        # auth patterns before exposing beyond trusted networks.
        async with transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    async def message_handler(scope: dict, receive: callable, send: callable) -> None:
        await transport.handle_post_message(scope, receive, send)

    return Starlette(
        middleware=[
            (
                RateLimitMiddleware,
                {"max_requests": RATE_LIMIT_REQUESTS, "window_seconds": RATE_LIMIT_WINDOW_SECONDS},
            ),
        ],
        routes=[
            Route("/sse", sse_handler, methods=["GET"]),
            Route("/messages", message_handler, methods=["POST"]),
        ],
    )


def main() -> None:
    logger.info(f"Starting tempest-tradingview-mcp on port {SERVER_PORT}")
    uvicorn.run(create_app(), host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
