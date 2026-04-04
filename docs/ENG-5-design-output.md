# ENG-5 Design Output — MCP Server Skeleton + 6 Tool Stubs

**Ticket:** [ENG-5](https://linear.app/jayteelabs/issue/ENG-5/tvmcp-002-mcp-server-skeleton-serverpy-6-tool-stubs) — TVMCP-002  
**Author:** Shuna (Design Architect)  
**Date:** 2026-04-04  
**Status:** Draft — Pending Josh Design Review (HitL Gate #2)  
**Branch:** `raphael/eng-5-tvmcp-002-mcp-server-skeleton-serverpy-6-tool-stubs`

---

## 1. Overview

This design covers the `server.py` skeleton and 6 tool stubs for the tempest-tradingview-mcp MCP server. All stubs follow the design architecture doc exactly: **pure data/analytics server, NO trading/execution tools**. Stubs return valid JSON responses enabling MCP protocol validation without crashing.

> **Critical Note — C1/C2 Discrepancy:** The existing repo at `/home/tempest/apps/tempest-tradingview-mcp/` contains a **trading bot** implementation (order execution via `connectors/binance.py`, mandatory `TEMPEST_BINANCE_API_KEY/SECRET`). This directly contradicts the design doc which states: *"What this project is NOT: A trading bot. No order execution, no position management."* ENG-5 stubs follow the **design doc**, NOT the existing repo implementations. The full implementations in the repo are ahead-of-scope for this skeleton ticket.

---

## 2. The 6 Tool Stubs

### 2.1 `fetch_ticker` — market_tools.py

**Function Signature:**
```python
async def fetch_ticker(symbol: str, exchange: str = "binance") -> dict[str, Any]:
```

**Stub Body:**
```python
async def fetch_ticker(symbol: str, exchange: str = "binance") -> dict[str, Any]:
    """Fetch real-time ticker price + volume for a symbol."""
    logger.info("Tool invoked: fetch_ticker", symbol=symbol, exchange=exchange)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "fetch_ticker",
            "symbol": symbol,
            "exchange": exchange,
            "note": "Implementation pending — data layer (ENG-4 successor tickets)"
        }
    }
```

**Tool Schema:**
```json
{
  "name": "fetch_ticker",
  "description": "Fetch real-time ticker price + 24h volume for a crypto symbol. Returns price, bid/ask, volume, and change %.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string", "description": "Trading symbol, e.g. BTC/USDT"},
      "exchange": {"type": "string", "default": "binance", "description": "Exchange name"}
    },
    "required": ["symbol"]
  }
}
```

---

### 2.2 `fetch_klines` — market_tools.py

**Function Signature:**
```python
async def fetch_klines(
    symbol: str,
    timeframe: str = "1h",
    since: str | None = None,
    limit: int = 100,
    exchange: str = "binance",
    source: str = "ccxt"
) -> dict[str, Any]:
```

**Stub Body:**
```python
async def fetch_klines(
    symbol: str,
    timeframe: str = "1h",
    since: str | None = None,
    limit: int = 100,
    exchange: str = "binance",
    source: str = "ccxt"
) -> dict[str, Any]:
    """Fetch OHLCV klines (candlestick data) for a symbol."""
    logger.info("Tool invoked: fetch_klines", symbol=symbol, timeframe=timeframe, limit=limit)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "fetch_klines",
            "symbol": symbol,
            "timeframe": timeframe,
            "limit": limit,
            "note": "Implementation pending — data layer (ENG-4 successor tickets)"
        }
    }
```

**Tool Schema:**
```json
{
  "name": "fetch_klines",
  "description": "Fetch OHLCV (candlestick) klines for a symbol. Returns array of {timestamp, open, high, low, close, volume}.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string", "description": "Trading symbol, e.g. BTC/USDT"},
      "timeframe": {"type": "string", "default": "1h", "description": "Kline timeframe: 1m, 5m, 15m, 1h, 4h, 1d"},
      "since": {"type": "string", "nullable": true, "description": "ISO8601 start time or unix timestamp"},
      "limit": {"type": "integer", "default": 100, "description": "Max number of klines"},
      "exchange": {"type": "string", "default": "binance"},
      "source": {"type": "string", "default": "ccxt", "description": "Data source: ccxt (real-time) or yf (yahoo finance, historical)"}
    },
    "required": ["symbol"]
  }
}
```

---

### 2.3 `fetch_orderbook` — market_tools.py

**Function Signature:**
```python
async def fetch_orderbook(symbol: str, limit: int = 20, exchange: str = "binance") -> dict[str, Any]:
```

**Stub Body:**
```python
async def fetch_orderbook(symbol: str, limit: int = 20, exchange: str = "binance") -> dict[str, Any]:
    """Fetch order book (bid/ask depth) for a symbol."""
    logger.info("Tool invoked: fetch_orderbook", symbol=symbol, limit=limit)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "fetch_orderbook",
            "symbol": symbol,
            "exchange": exchange,
            "limit": limit,
            "note": "Implementation pending — data layer (ENG-4 successor tickets)"
        }
    }
```

**Tool Schema:**
```json
{
  "name": "fetch_orderbook",
  "description": "Fetch order book depth (bids/asks) for a symbol.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string", "description": "Trading symbol, e.g. BTC/USDT"},
      "limit": {"type": "integer", "default": 20, "description": "Depth level (number of bids/asks)"},
      "exchange": {"type": "string", "default": "binance"}
    },
    "required": ["symbol"]
  }
}
```

---

### 2.4 `indicator_rsi` — indicator_tools.py

**Function Signature:**
```python
async def indicator_rsi(
    symbol: str,
    period: int = 14,
    timeframe: str = "1h",
    limit: int = 100,
    exchange: str = "binance"
) -> dict[str, Any]:
```

**Stub Body:**
```python
async def indicator_rsi(
    symbol: str,
    period: int = 14,
    timeframe: str = "1h",
    limit: int = 100,
    exchange: str = "binance"
) -> dict[str, Any]:
    """Calculate Relative Strength Index (RSI) for a symbol."""
    logger.info("Tool invoked: indicator_rsi", symbol=symbol, period=period)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "indicator_rsi",
            "symbol": symbol,
            "period": period,
            "timeframe": timeframe,
            "values": [],
            "note": "Implementation pending — indicator engine (ENG-4 successor tickets)"
        }
    }
```

**Tool Schema:**
```json
{
  "name": "indicator_rsi",
  "description": "Calculate Relative Strength Index (RSI). Oscillator 0-100: <30 oversold, >70 overbought.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string", "description": "Trading symbol"},
      "period": {"type": "integer", "default": 14, "description": "RSI period"},
      "timeframe": {"type": "string", "default": "1h"},
      "limit": {"type": "integer", "default": 100},
      "exchange": {"type": "string", "default": "binance"}
    },
    "required": ["symbol"]
  }
}
```

---

### 2.5 `backtest_strategy` — backtest_tools.py

**Function Signature:**
```python
async def backtest_strategy(
    symbol: str,
    strategy_id: str = "rsi_mean_reversion",
    timeframe: str = "1h",
    period: str = "1y",
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    source: str = "yf"
) -> dict[str, Any]:
```

**Stub Body:**
```python
async def backtest_strategy(
    symbol: str,
    strategy_id: str = "rsi_mean_reversion",
    timeframe: str = "1h",
    period: str = "1y",
    initial_capital: float = 10000.0,
    exchange: str = "binance",
    source: str = "yf"
) -> dict[str, Any]:
    """Run a backtest for a single strategy on a symbol."""
    logger.info("Tool invoked: backtest_strategy", symbol=symbol, strategy=strategy_id)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "backtest_strategy",
            "symbol": symbol,
            "strategy_id": strategy_id,
            "timeframe": timeframe,
            "period": period,
            "note": "Implementation pending — backtest engine (ENG-4 successor tickets)"
        }
    }
```

**Tool Schema:**
```json
{
  "name": "backtest_strategy",
  "description": "Run a backtest for a single strategy on a symbol using historical data.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbol": {"type": "string"},
      "strategy_id": {"type": "string", "default": "rsi_mean_reversion", "description": "Strategy ID"},
      "timeframe": {"type": "string", "default": "1h"},
      "period": {"type": "string", "default": "1y", "description": "Backtest period, e.g. 1y, 6mo"},
      "initial_capital": {"type": "number", "default": 10000.0},
      "exchange": {"type": "string", "default": "binance"},
      "source": {"type": "string", "default": "yf", "description": "Data source: yf (yahoo finance)"}
    },
    "required": ["symbol"]
  }
}
```

---

### 2.6 `screener_scan` — screener_tools.py

**Function Signature:**
```python
async def screener_scan(
    symbols: list[str] | None = None,
    filters: list[str] | None = None,
    min_score: float = 0.0,
    exchange: str = "binance"
) -> dict[str, Any]:
```

**Stub Body:**
```python
async def screener_scan(
    symbols: list[str] | None = None,
    filters: list[str] | None = None,
    min_score: float = 0.0,
    exchange: str = "binance"
) -> dict[str, Any]:
    """Multi-factor crypto screener — scan symbols against technical filters."""
    logger.info("Tool invoked: screener_scan", symbols=symbols, filters=filters)
    # STUB — full implementation in later phase ticket
    return {
        "success": True,
        "data": {
            "stub": True,
            "tool": "screener_scan",
            "filters": filters,
            "min_score": min_score,
            "results": [],
            "note": "Implementation pending — screener engine (ENG-4 successor tickets)"
        }
    }
```

**Tool Schema:**
```json
{
  "name": "screener_scan",
  "description": "Multi-factor crypto screener. Scan symbols against technical indicator filters (RSI, trend, volatility, volume).",
  "inputSchema": {
    "type": "object",
    "properties": {
      "symbols": {"type": "array", "items": {"type": "string"}, "nullable": true, "description": "List of symbols to scan. Null = default watchlist."},
      "filters": {"type": "array", "items": {"type": "string"}, "nullable": true, "description": "Filter names: rsi_oversold, rsi_overbought, trend_bullish, trend_bearish, high_volatility, low_volatility, volume_spike"},
      "min_score": {"type": "number", "default": 0.0},
      "exchange": {"type": "string", "default": "binance"}
    }
  }
}
```

---

## 3. server.py Skeleton

```python
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
```

---

## 4. tools/__init__.py — Stub Exports

```python
"""MCP tool stubs — ENG-5 skeleton. Implementations in later phase tickets."""

from tempest_mcp.tools.market_tools import fetch_ticker, fetch_klines, fetch_orderbook
from tempest_mcp.tools.indicator_tools import indicator_rsi
from tempest_mcp.tools.backtest_tools import backtest_strategy
from tempest_mcp.tools.screener_tools import screener_scan

__all__ = [
    "fetch_ticker",
    "fetch_klines",
    "fetch_orderbook",
    "indicator_rsi",
    "backtest_strategy",
    "screener_scan",
]
```

---

## 5. HitL Gate #2 — Design Review

**Next step:** Raphael will post this design doc to Linear on ENG-5 and set the `hitl-pending` label. JACE will notify Josh on Discord for Design Review approval. Josh must approve before the ticket can advance to Development.

---

## 6. Stub Body Design Decision

**Decision:** Stub bodies return `{"success": True, "data": {"stub": True, ...}}` — NOT `NotImplementedError`.

**Rationale:**
- MCP protocol requires `call_tool` to return `TextContent` — an exception would crash the transport
- Returning a valid JSON success envelope allows MCP protocol handshake validation
- Kurisu/tempest-mesh can confirm the tool is registered and callable even if the body is a stub
- Later phase tickets replace the stub body with real implementation — the call path (server -> tool function) stays identical

---

**Status:** Ready for Josh Design Review (HitL Gate #2)
