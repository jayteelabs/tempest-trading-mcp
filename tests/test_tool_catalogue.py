"""Focused MCP catalogue seam characterization tests — ENG-180."""

import json

from mcp.types import Tool

from tempest_mcp import catalogue
from tempest_mcp.config import ErrorCodes

INTERNAL_TOOL_NAMES = [
    "fetch_ticker",
    "fetch_klines",
    "fetch_orderbook",
    "indicator_rsi",
    "screener_scan",
    "session_breakout_scan",
    "order_block_screener_scan",
    "backtest_strategy",
    "backtest_pdh_session",
    "backtest_rsi",
    "backtest_vwap",
    "backtest_ema_stack",
    "backtest_order_blocks",
    "backtest_elliot_wave",
    "compare_strategies",
    "calculate_volume_profile",
    "detect_order_blocks",
    "calculate_fibonacci",
    "calculate_tpo",
    "detect_elliot_wave",
    "get_market_structure",
    "get_combined_sentiment_dashboard",
]

PUBLIC_TOOL_NAMES = [
    "fetch_ticker",
    "fetch_klines",
    "fetch_orderbook",
    "indicator_rsi",
    "backtest_pdh_session",
    "backtest_rsi",
    "backtest_vwap",
    "backtest_ema_stack",
    "backtest_order_blocks",
    "backtest_elliot_wave",
    "compare_strategies",
    "screener_scan",
    "session_breakout_scan",
    "order_block_screener_scan",
    "calculate_volume_profile",
    "detect_order_blocks",
    "calculate_fibonacci",
    "calculate_tpo",
    "detect_elliot_wave",
    "get_market_structure",
    "get_combined_sentiment_dashboard",
]


def _payload(contents):
    assert len(contents) == 1
    return json.loads(contents[0].text)


def test_internal_registry_names_and_hidden_legacy_policy():
    assert list(catalogue.TOOLS) == INTERNAL_TOOL_NAMES
    assert len(catalogue.TOOLS) == 22
    assert "backtest_strategy" in catalogue.TOOLS


def test_public_schema_names_order_and_metadata_are_exposed():
    tools = catalogue.list_public_tools()
    assert tools is catalogue.TOOL_SCHEMAS
    assert [tool.name for tool in tools] == PUBLIC_TOOL_NAMES
    assert len(tools) == 21
    assert all(isinstance(tool, Tool) for tool in tools)
    assert "backtest_strategy" not in {tool.name for tool in tools}


def test_representative_schema_content_is_preserved():
    by_name = {tool.name: tool for tool in catalogue.TOOL_SCHEMAS}
    assert by_name["fetch_klines"].inputSchema["properties"]["source"] == {
        "type": "string",
        "default": "ccxt",
        "description": "Must be 'ccxt' (historical routing is CCXT+yfinance fallback)",
    }
    vwap_props = by_name["backtest_vwap"].inputSchema["properties"]
    assert vwap_props["timeframe"]["description"].startswith("Supported OHLCV timeframe")
    assert "America/New_York" in vwap_props["start_at"]["description"]
    assert vwap_props["vwap_anchor"]["enum"] == ["asia", "london", "ny", "daily"]
    assert by_name["get_combined_sentiment_dashboard"].inputSchema["required"] == [
        "symbol",
        "price_bias",
    ]


def test_lookup_handler_contract():
    assert callable(catalogue.lookup_handler("fetch_ticker"))
    assert callable(catalogue.lookup_handler("backtest_strategy"))
    assert catalogue.lookup_handler("unknown_tool") is None


def test_validation_routing_examples_are_preserved():
    cases = [
        ("fetch_ticker", {"symbol": ""}, "symbol cannot be empty"),
        (
            "fetch_klines",
            {"symbol": "BTC/USDT", "source": "yf"},
            'source must be "ccxt" (historical routing is CCXT+yfinance fallback)',
        ),
        ("fetch_orderbook", {"symbol": "BTC/USDT", "limit": 101}, "limit must be between 1 and 100"),
        ("backtest_strategy", {"symbol": ""}, "symbol cannot be empty"),
        ("screener_scan", {"symbols": []}, "symbols must contain at least 1 entry"),
        ("session_breakout_scan", {}, "session is required"),
        ("order_block_screener_scan", {"atr_period": 1}, "atr_period must be between 2 and 200"),
        ("get_combined_sentiment_dashboard", {"symbol": "BTC/USDT"}, "price_bias is required"),
        (
            "get_combined_sentiment_dashboard",
            {"symbol": "BTC/USDT", "price_bias": "mixed"},
            "price_bias must be one of: bullish, bearish, neutral",
        ),
    ]
    for name, arguments, expected in cases:
        assert catalogue.validate_tool_arguments(name, arguments) == expected


def test_limit_validation_rejects_boolean_values():
    assert (
        catalogue.validate_tool_arguments("fetch_klines", {"symbol": "BTC/USDT", "limit": True})
        == "limit must be an integer"
    )


async def test_dispatch_unknown_tool_envelope():
    assert _payload(await catalogue.dispatch_tool_call("missing", {})) == {
        "success": False,
        "error": {"code": ErrorCodes.INVALID_PARAMETER, "message": "Unknown tool: missing"},
    }


async def test_dispatch_validation_error_envelope():
    assert _payload(await catalogue.dispatch_tool_call("fetch_ticker", {"symbol": ""})) == {
        "success": False,
        "error": {"code": ErrorCodes.INVALID_PARAMETER, "message": "symbol cannot be empty"},
    }


async def test_dispatch_hidden_legacy_backtest_strategy_remains_callable():
    payload = _payload(await catalogue.dispatch_tool_call("backtest_strategy", {"symbol": "BTC/USDT"}))
    assert payload["success"] is False
    assert payload["error"]["code"] == ErrorCodes.VALIDATION_ERROR
    assert "backtest_strategy is deprecated" in payload["error"]["message"]


async def test_dispatch_internal_error_envelope(monkeypatch):
    async def broken_handler(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setitem(catalogue.TOOLS, "fetch_ticker", broken_handler)

    assert _payload(await catalogue.dispatch_tool_call("fetch_ticker", {"symbol": "BTC/USDT"})) == {
        "success": False,
        "error": {"code": ErrorCodes.INTERNAL_ERROR, "message": "An internal error occurred"},
    }
