"""MCP Server entry point — SSE/HTTP transport."""

import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
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

# SSE connection limiting: max concurrent SSE connections per IP
SSE_MAX_CONNECTIONS_PER_IP = 10

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
# Module-level registry for cleanup task — accessible from lifespan handler
_rate_limit_cleanup_task: asyncio.Task | None = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per client IP — thread-safe for asyncio.

    Handles both:
    - /messages: request-count rate limiting (100 req/min per IP)
    - /sse: concurrent connection limiting (10 per IP)
    """

    def __init__(self, app: Any, max_requests: int, window_seconds: int) -> None:
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = {}  # IP -> list of request timestamps
        self._sse_connections: dict[str, int] = {}  # IP -> active SSE connection count
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    def _get_lock(self, client_ip: str) -> asyncio.Lock:
        """Get or create a lock for a specific client IP."""
        if client_ip not in self._locks:
            self._locks[client_ip] = asyncio.Lock()
        return self._locks[client_ip]

    def _clean_stale_requests(self, now: float) -> None:
        """Remove all IPs with no recent requests to prevent memory leak."""
        stale = []
        for ip, times in list(self._requests.items()):
            try:
                # Guard against race: times could become empty between check and max()
                if times:
                    if now - max(times) >= self.window_seconds * 2:
                        stale.append(ip)
                else:
                    stale.append(ip)
            except ValueError:
                # max() arg empty — treat as stale
                stale.append(ip)
        for ip in stale:
            self._requests.pop(ip, None)
            self._locks.pop(ip, None)

    def _clean_stale_sse(self, now: float) -> None:
        """Clean SSE connections dict of entries with zero count."""
        stale = [ip for ip, count in self._sse_connections.items() if count <= 0]
        for ip in stale:
            self._sse_connections.pop(ip, None)

    async def _periodic_request_cleanup(self) -> None:
        """Background task: clean stale entries every window period."""
        global _rate_limit_cleanup_task
        try:
            while True:
                await asyncio.sleep(self.window_seconds * 2)
                async with self._global_lock:
                    self._clean_stale_requests(time.time())
                    self._clean_stale_sse(time.time())
        except asyncio.CancelledError:
            pass
        finally:
            if _rate_limit_cleanup_task is self._cleanup_task:
                _rate_limit_cleanup_task = None

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        global _rate_limit_cleanup_task
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Start cleanup task on first request (singleton — cancel previous if exists)
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._periodic_request_cleanup())
            _rate_limit_cleanup_task = self._cleanup_task
        elif self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = asyncio.create_task(self._periodic_request_cleanup())
            _rate_limit_cleanup_task = self._cleanup_task

        if request.url.path == "/messages":
            # Request-count rate limiting
            lock = self._get_lock(client_ip)
            async with lock:
                async with self._global_lock:
                    self._clean_stale_requests(now)
                    if client_ip not in self._requests:
                        self._requests[client_ip] = []
                    # Remove expired entries for this IP
                    self._requests[client_ip] = [
                        t for t in self._requests[client_ip] if now - t < self.window_seconds
                    ]
                    if len(self._requests[client_ip]) >= self.max_requests:
                        logger.warning("Rate limit exceeded", client_ip=client_ip)
                        return JSONResponse(
                            {
                                "success": False,
                                "error": {"code": 429, "message": "Rate limit exceeded"},
                            },
                            status_code=429,
                        )
                    self._requests[client_ip].append(now)

        elif request.url.path == "/sse":
            # Concurrent SSE connection limiting
            async with self._global_lock:
                current = self._sse_connections.get(client_ip, 0)
                if current >= SSE_MAX_CONNECTIONS_PER_IP:
                    logger.warning("SSE connection limit exceeded", client_ip=client_ip)
                    return JSONResponse(
                        {
                            "success": False,
                            "error": {"code": 429, "message": "Too many SSE connections"},
                        },
                        status_code=429,
                    )
                self._sse_connections[client_ip] = current + 1

            # Decrement on connection close
            async def decrement_sse():
                try:
                    await call_next(request)
                finally:
                    async with self._global_lock:
                        self._sse_connections[client_ip] = max(
                            0, self._sse_connections.get(client_ip, 1) - 1
                        )

            return await decrement_sse()

        return await call_next(request)


def cancel_rate_limit_cleanup() -> None:
    """Cancel the rate limit background cleanup task. Call from lifespan shutdown."""
    global _rate_limit_cleanup_task
    if _rate_limit_cleanup_task is not None:
        _rate_limit_cleanup_task.cancel()
        _rate_limit_cleanup_task = None


# ── Input Validation ──────────────────────────────────────────────────────────
def validate_symbol(symbol: str, field_name: str = "symbol") -> str | None:
    """Validate symbol format. Returns None if valid, error message if invalid."""
    if not isinstance(symbol, str):
        return f"{field_name} must be a string"
    if not symbol:
        return f"{field_name} cannot be empty"
    if not SYMBOL_PATTERN.match(symbol):
        return f"Invalid {field_name} format: {symbol!r} — must be 2-20 uppercase alphanumeric characters"
    return None


def validate_tool_arguments(name: str, arguments: dict[str, Any]) -> str | None:
    """Validate tool arguments. Returns error message or None if valid."""
    if name == "fetch_ticker":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    if name == "fetch_klines":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    if name == "fetch_orderbook":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    if name == "indicator_rsi":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    if name == "backtest_strategy":
        return validate_symbol(arguments.get("symbol", ""), "symbol")
    if name == "screener_scan":
        symbols = arguments.get("symbols")
        if symbols is None:
            return None
        if not isinstance(symbols, list):
            return "symbols must be an array of strings"
        for i, sym in enumerate(symbols):
            if err := validate_symbol(sym, f"symbols[{i}]"):
                return err
        return None
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

    @asynccontextmanager
    async def lifespan(app: Starlette):
        """Gracefully cancel background tasks on shutdown."""
        yield
        cancel_rate_limit_cleanup()
        logger.info("Rate limit cleanup task cancelled on shutdown")

    # TODO: In-memory rate limiting is per-process. If uvicorn is run with multiple
    # workers (--workers N), each worker has independent state — a client could
    # bypass rate limits by hitting different workers. For multi-worker deployments,
    # replace with Redis-backed rate limiting (e.g., Redis + lua script for atomic
    # check-and-increment). This is not a concern for single-worker deployments.

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
        lifespan=lifespan,
    )


def main() -> None:
    logger.info(f"Starting tempest-tradingview-mcp on port {SERVER_PORT}")
    uvicorn.run(create_app(), host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
