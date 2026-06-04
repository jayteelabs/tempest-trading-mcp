"""MCP Server entry point — SSE/HTTP transport."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tempest_mcp import catalogue
from tempest_mcp.config import get_config
from tempest_mcp.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# Backward-compatible internal imports for existing characterization tests.
TOOL_SCHEMAS = catalogue.TOOL_SCHEMAS
TOOLS = catalogue.TOOLS
validate_tool_arguments = catalogue.validate_tool_arguments

# Server listens on port 9001 for HTTP/SSE transport
# Binds to 0.0.0.0 so Docker port publishing can reach the app.
SERVER_PORT = 9001
SERVER_HOST = "0.0.0.0"

# Rate limiting: max requests per client per window
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW_SECONDS = 60

# SSE connection limiting: max concurrent SSE connections per IP
SSE_MAX_CONNECTIONS_PER_IP = 10

# ── Rate Limiting ──────────────────────────────────────────────────────────────
# Module-level registry for cleanup task — accessible from lifespan handler
_rate_limit_cleanup_task: asyncio.Task | None = None

# ── Streamable HTTP Session Manager ───────────────────────────────────────────
# Module-level reference to the StreamableHTTPSessionManager — set in create_app()
# so cancel can fire during shutdown if lifespan exits unexpectedly.
_http_session_manager: "StreamableHTTPSessionManager | None" = None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory rate limiter per client IP — thread-safe for asyncio.

    Handles both:
    - /messages: request-count rate limiting (100 req/min per IP)
    - /mcp: request-count rate limiting (100 req/min per IP)
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

        if request.url.path in {"/messages", "/messages/", "/mcp"}:
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
                    return await call_next(request)
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


# ── Server ────────────────────────────────────────────────────────────────────
def create_app() -> Starlette:
    """Create the Starlette ASGI app with SSE and streamable-HTTP transport."""
    config = get_config()
    setup_logging()
    logger.info(
        "Starting MCP server",
        name=config.mcp_server_name,
        version=config.mcp_server_version,
        transport="sse+streamable-http",
        port=SERVER_PORT,
        host=SERVER_HOST,
    )
    server = Server(config.mcp_server_name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return catalogue.list_public_tools()

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        logger.info("Tool called", name=name)
        return await catalogue.dispatch_tool_call(name, arguments)

    transport = SseServerTransport("/messages")

    class SseApp:
        def __init__(self, server: Server, transport: SseServerTransport):
            self._server = server
            self._transport = transport

        async def __call__(self, scope: dict, receive: callable, send: callable) -> None:
            # TODO: Add authentication before allowing SSE connection.
            # The server binds to 0.0.0.0 in-container, and the repo-owned Compose
            # config publishes 127.0.0.1:9001 on the host by default. If this is
            # exposed beyond trusted networks, add compensating controls such as a
            # reverse proxy, Tailscale, firewall rules, and/or API auth (Bearer
            # token, API key, mTLS) before widening exposure.
            async with self._transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )

    class MessageApp:
        def __init__(self, transport: SseServerTransport):
            self._transport = transport

        async def __call__(self, scope: dict, receive: callable, send: callable) -> None:
            await self._transport.handle_post_message(scope, receive, send)

    # Streamable HTTP session manager — uses real MCP SDK session lifecycle (ENG-123)
    # Stateless=True gives each POST request its own server run + transport instance,
    # which cleanly parallels how SseApp creates a per-connection transport.
    http_session_manager = StreamableHTTPSessionManager(
        app=server,
        stateless=True,
    )

    class StreamableHTTPApp:
        """ASGI wrapper around StreamableHTTPSessionManager — handles POST /mcp."""

        def __init__(self, mgr: StreamableHTTPSessionManager):
            self._mgr = mgr

        async def __call__(self, scope: dict, receive: callable, send: callable) -> None:
            await self._mgr.handle_request(scope, receive, send)

    sse_handler = SseApp(server, transport)
    message_handler = MessageApp(transport)
    streamable_http_handler = StreamableHTTPApp(http_session_manager)

    # Module-level reference so cancel can fire during shutdown
    global _http_session_manager
    _http_session_manager = http_session_manager

    @asynccontextmanager
    async def lifespan(app: Starlette):
        """Manage StreamableHTTPSessionManager lifecycle; cancel cleanup tasks on shutdown."""
        async with http_session_manager.run():
            yield
        global _http_session_manager
        _http_session_manager = None
        cancel_rate_limit_cleanup()
        logger.info("Rate limit cleanup task cancelled on shutdown")

    # TODO: In-memory rate limiting is per-process. If uvicorn is run with multiple
    # workers (--workers N), each worker has independent state — a client could
    # bypass rate limits by hitting different workers. For multi-worker deployments,
    # replace with Redis-backed rate limiting (e.g., Redis + lua script for atomic
    # check-and-increment). This is not a concern for single-worker deployments.

    return Starlette(
        middleware=[
            Middleware(
                RateLimitMiddleware,
                max_requests=RATE_LIMIT_REQUESTS,
                window_seconds=RATE_LIMIT_WINDOW_SECONDS,
            ),
        ],
        routes=[
            Route("/sse", sse_handler, methods=["GET"]),
            Route("/messages", message_handler, methods=["POST"]),
            Route("/messages/", message_handler, methods=["POST"]),
            Route("/mcp", streamable_http_handler, methods=["POST"]),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    logger.info(f"Starting tempest-trading-mcp on port {SERVER_PORT}")
    uvicorn.run(create_app(), host=SERVER_HOST, port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
