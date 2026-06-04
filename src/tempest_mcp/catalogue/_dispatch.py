"""MCP tool-call dispatch and response-envelope compatibility."""

import json
from typing import Any

from mcp.types import TextContent

from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger

from ._registry import lookup_handler
from ._validation import validate_tool_arguments

logger = get_logger(__name__)


def _error_content(code: int, message: str) -> list[TextContent]:
    return [
        TextContent(
            type="text",
            text=json.dumps(
                {
                    "success": False,
                    "error": {"code": code, "message": message},
                }
            ),
        )
    ]


async def dispatch_tool_call(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Lookup, validate, execute, and envelope a tool call exactly as the server did."""
    handler = lookup_handler(name)
    if handler is None:
        return _error_content(ErrorCodes.INVALID_PARAMETER, f"Unknown tool: {name}")

    if validation_error := validate_tool_arguments(name, arguments):
        logger.warning("Validation failed", name=name, error=validation_error)
        return _error_content(ErrorCodes.INVALID_PARAMETER, validation_error)

    try:
        result = await handler(**arguments)
        return [TextContent(type="text", text=json.dumps(result))]
    except Exception as e:
        logger.error("Tool raised exception", name=name, error=str(e))
        return _error_content(ErrorCodes.INTERNAL_ERROR, "An internal error occurred")
