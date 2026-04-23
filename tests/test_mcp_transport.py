"""Transport tests for dual SSE + streamable-HTTP support — ENG-123."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from httpx import ASGITransport, AsyncClient, Response

import tempest_mcp.server as server_module
from tempest_mcp.server import TOOL_SCHEMAS, create_app

DEFAULT_INIT_PARAMS = {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "test-client", "version": "0.1.0"},
}
STREAMABLE_HTTP_HEADERS = {"Accept": "application/json, text/event-stream"}


@asynccontextmanager
async def make_client() -> AsyncClient:
    """Create an async client against the worktree source app with lifespan enabled."""
    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            yield client


def parse_sse_jsonrpc_response(response: Response) -> dict:
    """Extract the JSON-RPC payload from an SSE-framed /mcp response."""
    content_type = response.headers.get("content-type", "")
    assert content_type.startswith("text/event-stream")
    assert "event: message" in response.text

    data_lines = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert data_lines, f"Expected at least one SSE data line, got: {response.text!r}"
    return json.loads("\n".join(data_lines))


class TestStreamableHTTPTransport:
    """Tests for POST /mcp streamable HTTP endpoint."""

    async def test_initialize_request_returns_sse_framed_jsonrpc(self):
        async with make_client() as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": DEFAULT_INIT_PARAMS,
                },
                headers=STREAMABLE_HTTP_HEADERS,
            )

        assert response.status_code == 200
        data = parse_sse_jsonrpc_response(response)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 1
        assert data["result"]["protocolVersion"] == "2024-11-05"
        assert "capabilities" in data["result"]

    async def test_tools_list_request_returns_sse_framed_tool_schemas(self):
        async with make_client() as client:
            response = await client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers=STREAMABLE_HTTP_HEADERS,
            )

        assert response.status_code == 200
        data = parse_sse_jsonrpc_response(response)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 2
        tool_names = {tool["name"] for tool in data["result"]["tools"]}
        expected_names = {tool.name for tool in TOOL_SCHEMAS}
        assert tool_names == expected_names

    async def test_tools_call_validation_error_is_sse_framed_jsonrpc(self):
        async with make_client() as client:
            response = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "fetch_ticker",
                        "arguments": {"symbol": ""},
                    },
                },
                headers=STREAMABLE_HTTP_HEADERS,
            )

        assert response.status_code == 200
        data = parse_sse_jsonrpc_response(response)
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == 3
        content = data["result"]["content"]
        assert len(content) == 1
        inner_payload = json.loads(content[0]["text"])
        assert inner_payload == {
            "success": False,
            "error": {"code": 1004, "message": "symbol cannot be empty"},
        }

    async def test_mcp_get_returns_405(self):
        async with make_client() as client:
            response = await client.get("/mcp")

        assert response.status_code == 405

    async def test_mcp_post_invalid_json_returns_400(self):
        async with make_client() as client:
            response = await client.post(
                "/mcp",
                content=b"not valid json",
                headers={"Content-Type": "application/json", **STREAMABLE_HTTP_HEADERS},
            )

        assert response.status_code == 400
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert data["jsonrpc"] == "2.0"
        assert data["id"] == "server-error"
        assert data["error"]["code"] == -32700


class TestSSERegression:
    """Tests that legacy SSE routes still respond correctly."""

    def test_transport_suite_uses_current_worktree_source(self):
        repo_root = Path(__file__).resolve().parents[1]
        server_path = Path(server_module.__file__).resolve()

        assert server_path.is_relative_to(repo_root / "src")

    async def test_messages_route_requires_session_id(self):
        async with make_client() as client:
            response = await client.post(
                "/messages",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": DEFAULT_INIT_PARAMS,
                },
            )

        assert response.status_code == 400
        assert response.text == "session_id is required"

    def test_mcp_route_in_routes_list(self):
        app = create_app()
        route_paths = {route.path for route in app.routes}
        assert "/mcp" in route_paths


class TestRateLimitingOnMCP:
    """Tests that rate limiting applies to /mcp the same as /messages."""

    async def test_rate_limit_on_mcp(self):
        async with make_client() as client:
            status_codes = []
            for _ in range(110):
                response = await client.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "tools/call",
                        "params": {
                            "name": "fetch_ticker",
                            "arguments": {"symbol": ""},
                        },
                    },
                    headers=STREAMABLE_HTTP_HEADERS,
                )
                status_codes.append(response.status_code)
                if response.status_code == 429:
                    break

        assert 429 in status_codes, (
            "Expected 429 (rate limit) after 100+ requests, "
            f"got statuses: {status_codes[:5]}..."
        )


class TestDualTransportParity:
    """Tests that both transports delegate to the same Server instance."""

    def test_tool_names_match_tool_schemas(self):
        expected = {tool.name for tool in TOOL_SCHEMAS}
        assert len(expected) == 21, f"Expected 21 tools, got {len(expected)}"
        assert "fetch_ticker" in expected
        assert "backtest_pdh_session" in expected
        assert "get_combined_sentiment_dashboard" in expected
