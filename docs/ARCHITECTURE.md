# Architecture Deep-Dive

This document explains the internal structure of `tempest-tradingview-mcp` and how contributors extend it. It is grounded on the live `origin/main` codebase.

> **TradingView note:** Despite the repository name, TradingView is not an active market-data provider in the current architecture. Legacy TradingView compatibility code (`data/tv_adapter.py`) is retained only to avoid breaking older imports. CCXT is the primary live/historical crypto path; yfinance is the fallback/non-crypto path.

---

## Transport Boundary

**Runtime:** HTTP/SSE on port `9001`, bound to `0.0.0.0`

```
MCP Client ──HTTP/SSE──▶ /sse (GET)   ← SSE connection endpoint
                          /messages (POST) ← Tool invocation endpoint
```

- **`/sse`** — Server-Sent Events connection. MCP clients establish an SSE stream first, then POST tool calls to `/messages`.
- **`/messages`** — JSON-RPC over HTTP POST. Each request carries a JSON-RPC payload describing the tool name and arguments.
- Rate limiting: 100 requests/minute per IP on `/messages`; max 10 concurrent SSE connections per IP.
- The server is a Starlette ASGI app wrapping an `mcp.server.Server` instance (`server.py`).

---

## MCP Tool Registration Surface

**File:** `src/tempest_mcp/server.py`

All 21 public MCP tools are registered in two places:

1. **`TOOLS` dict** (lines 76–98) — maps tool name strings to handler callables.
2. **`TOOL_SCHEMAS` list** (lines 101–663) — declares each tool's name, description, and JSON input schema to the MCP protocol.

```python
# Example: adding a new tool
TOOLS["my_new_tool"] = my_new_tool_handler  # handler callable

TOOL_SCHEMAS.append(Tool(
    name="my_new_tool",
    description="...",
    inputSchema={...},
))
```

Tool argument validation is centralized in `validate_tool_arguments()` (lines 832–993). Each tool name branches to its own validation logic.

**Tool categories:**

| Category | Tools |
|----------|-------|
| Market data | `fetch_ticker`, `fetch_klines`, `fetch_orderbook` |
| Indicators | `indicator_rsi` |
| Backtesting | `backtest_pdh_session`, `backtest_rsi`, `backtest_vwap`, `backtest_ema_stack`, `backtest_order_blocks`, `backtest_elliot_wave`, `compare_strategies` |
| Screening | `screener_scan`, `session_breakout_scan`, `order_block_screener_scan` |
| Analysis | `calculate_volume_profile`, `detect_order_blocks`, `calculate_fibonacci`, `calculate_tpo`, `detect_elliot_wave`, `get_market_structure` |
| Sentiment | `get_combined_sentiment_dashboard` |

---

## Data Layer Architecture

**Files:** `src/tempest_mcp/data/`

### Primary Data Path: CCXT

CCXT is the primary adapter for all live and historical crypto market data (`ccxt_adapter.py`).

```python
CCXTAdapter  # live + historical OHLCV, ticker, orderbook
```

- Supports Binance, Bybit, Kraken, Coinbase public REST APIs — no API keys required.
- `fetch_live_price()` — latest trade price (float, NaN on error)
- `fetch_ohlcv_live()` — recent OHLCV candles (DataFrame)
- `fetch_ohlcv_historical()` — historical OHLCV via `since`/`until` params, supports auto-pagination
- `fetch_orderbook_snapshot()` — bids/asks depth

All CCXT methods return empty/safe values on error (no exception propagation — per D14 design contract).

### Fallback Data Path: yfinance

yfinance is the fallback for stocks and crypto data gaps CCXT does not cover (`yf_adapter.py`). The module exposes `fetch_ohlcv(...)` and a `YFAdapter` compatibility wrapper for historical OHLCV retrieval.

### Data Source Router

**File:** `src/tempest_mcp/data/_router.py`

```python
DataSourceRouter
  ├── route_live()   → LiveDataAdapter (CCXT only)
  └── route_historical() → HistoricalDataSource (CCXT primary, yfinance fallback)
```

### Historical Data Source

**Files:** `src/tempest_mcp/data/_hist.py`, `src/tempest_mcp/data/_factory.py`

`HistoricalDataSource` lives in `_hist.py` and wraps CCXT-backed historical retrieval with yfinance fallback behind a single interface. The cached `get_historical_adapter()` singleton helper lives in `_factory.py` and returns one `HistoricalDataSource` instance via `lru_cache(maxsize=1)`.

### Legacy TradingView Adapter

**File:** `src/tempest_mcp/data/tv_adapter.py`

Retained as legacy compatibility only. **Not the active primary data architecture.**

### Data Contracts

**File:** `src/tempest_mcp/data/_contracts.py`

Canonical OHLCV schema: `["open", "high", "low", "close", "volume"]` with UTC-aware DatetimeIndex. All data adapters normalize to this contract.

---

## Indicators Engine

**Directory:** `src/tempest_mcp/indicators/`

Internal engine (30+ indicators). Not all are exposed as standalone MCP tools — some are used internally by strategy and analysis tools.

| Subdirectory | Contents |
|---------------|----------|
| `momentum/` | RSI (`rsi.py`), MACD/ADX/Stochastic (`macd_adx_stoch.py`), secondary momentum |
| `trend/` | EMA (`ema.py`) |
| `volatility/` | ATR (`atr.py`) |
| `volume/` | VWAP (`vwap.py`), TPO (`tpo.py`), Volume Profile (`volume_profile.py`) |
| `structure.py` | Fibonacci, pivots, HH/HL detection |
| `session_levels.py` | Asia/London/NY PDH/PDL session levels |
| `ta_wrapper.py` | ta-lib C extension wrapper (internal) |

Adding a new indicator: create a new module under the appropriate subdirectory and expose it via the relevant tool handler in `src/tempest_mcp/tools/`.

---

## Backtest Engine

**Files:** `src/tempest_mcp/backtest/`

### Core Engine — `engine.py`

`BacktestEngine` — bidirectional backtest engine with commission and slippage modeling.

- **Signal model:** `SignalAction` enum — `LONG_ENTRY`, `LONG_EXIT`, `SHORT_ENTRY`, `SHORT_EXIT`, `HOLD`
- **Position model:** `PositionDirection` enum — `FLAT`, `LONG`, `SHORT`
- Directional flips must transition through `FLAT` (no direct LONG→SHORT).
- Execution: signal at bar `i` triggers order at bar `i+1` open (no lookahead).
- Commission: symmetric percentage on both entry and exit.
- Slippage: configurable basis points, direction-aware.
- Metrics: total_return, sharpe_ratio, max_drawdown, win_rate, profit_factor, expectancy.

### Walk-Forward Evaluation — `walk_forward.py`

Internal backtest-core module (not a separate MCP tool surface).

- `run_walk_forward()` — rolling train/test splits for out-of-sample strategy validation.
- Train slice anchors signal generation; test slice scores OOS performance.
- Position state resets at each test boundary.
- Split planning is deterministic and reproducible.

---

## Strategies

**Directory:** `src/tempest_mcp/strategies/`

Strategy modules owned by the backtest team. The surface is mixed: some modules return `(signals, BacktestEngine)` directly, while RSI and Order Blocks expose signal generators that are wrapped by thin adapters in `src/tempest_mcp/tools/backtest_tools.py`.

| File | Strategy |
|------|----------|
| `backtest_pdh_session.py` | PDH/PDL + Session Levels |
| `backtest_rsi.py` | RSI Mean Reversion |
| `backtest_vwap.py` | VWAP Anchored |
| `backtest_ema_stack.py` | Multi-EMA Stack |
| `backtest_order_blocks.py` | Order Blocks |
| `backtest_elliot_wave.py` | Elliott Wave |

- Direct runners: `run_pdh_session_backtest`, `run_vwap_anchored_backtest`, `run_ema_stack_backtest`, and `run_elliot_wave_backtest` return `(signals, BacktestEngine)` directly.
- Signal-generator modules: `backtest_rsi.py` exposes `generate_rsi_signals(...)` and `backtest_order_blocks.py` exposes `generate_order_block_signals(...)`; `tools/backtest_tools.py` wraps those outputs with the shared `BacktestEngine`.

---

## Screening

**Directory:** `src/tempest_mcp/screener/`

`scanner.py` — multi-factor crypto screener engine. Handles `screener_scan`, `session_breakout_scan`, and `order_block_screener_scan`.

---

## Sentiment Analysis

**Directory:** `src/tempest_mcp/sentiment/`

| File | Source |
|------|--------|
| `reddit.py` | Reddit post/search sentiment |
| `rss.py` | RSS feed sentiment |
| `combined_sentiment.py` | Weighted combination (40% Reddit / 60% RSS) + cross-signal detection |

`get_combined_sentiment_dashboard` is the single MCP tool surfacing this module.

---

## Discord Formatting

**File:** `src/tempest_mcp/formatters/discord.py`

`DiscordFormatter` — converts MCP result envelopes into Discord embed dicts. Pure utility, no I/O, no external API calls, no `discord.py` dependency.

Dispatches by tool name to per-category formatters: `format_backtest`, `format_screener`, `format_sentiment`, `format_analytical`, etc.

---

## Models

**Directory:** `src/tempest_mcp/models/`

Dataclass models for typed data contracts:

| File | Contents |
|------|----------|
| `backtest.py` | Backtest results |
| `indicator.py` | Indicator results |
| `market.py` | Ticker, kline, orderbook |

---

## MCP Tool Handlers

**Directory:** `src/tempest_mcp/tools/`

Thin handlers that bridge MCP tool calls to internal engines:

| File | Tool handlers |
|------|---------------|
| `market_tools.py` | `fetch_ticker`, `fetch_klines`, `fetch_orderbook` |
| `indicator_tools.py` | `indicator_rsi` |
| `backtest_tools.py` | `backtest_*` (Phase 2 backtest tools) |
| `backtest_window.py` | Timeframe constants for backtest tools |
| `screener_tools.py` | `screener_scan`, `session_breakout_scan`, `order_block_screener_scan` |
| `analysis_tools.py` | `calculate_volume_profile`, `detect_order_blocks` |
| `analytical_tools.py` | `calculate_fibonacci`, `calculate_tpo`, `detect_elliot_wave`, `get_market_structure` |
| `sentiment_tools.py` | `get_combined_sentiment_dashboard` |

---

## Extension Points

### Adding a new MCP tool

1. Implement the handler function (e.g., in `tools/`).
2. Add to `TOOLS` dict in `server.py`: `TOOLS["my_tool"] = my_handler`
3. Add `Tool(...)` schema entry to `TOOL_SCHEMAS` list in `server.py`.
4. Add argument validation in `validate_tool_arguments()` if needed.

### Adding a new data source

1. Create a new adapter in `src/tempest_mcp/data/` (e.g., `my_adapter.py`).
2. If historical: update `HistoricalDataSource` in `_hist.py` to route to it.
3. If the historical singleton construction changes, update `get_historical_adapter()` in `_factory.py`.
4. If live: update `get_live_adapter()` / `DataSourceRouter.route_live()` as needed.
5. No changes to MCP tool schemas required — data sources are internal.

### Adding a new indicator

1. Create a new module in `src/tempest_mcp/indicators/<category>/`.
2. If it should be exposed as a standalone MCP tool: add handler in `tools/` and register in `server.py`.
3. If it is internal only: import it directly from the relevant strategy or analysis module.

### Adding a new backtest strategy

1. Create a new module in `src/tempest_mcp/strategies/`.
2. Choose the live surface: either expose a direct runner that returns `(signals, BacktestEngine)` or expose a signal generator and add a thin adapter in `tools/backtest_tools.py`.
3. Register the strategy in `tools/backtest_tools.py` (`_STRATEGY_SPECS` plus `_run_direct_runner()` / `_run_adapter()` as appropriate).
4. If it should participate in `compare_strategies`, update `ALLOWED_STRATEGY_IDS` in `tools/backtest_tools.py` and the `compare_strategies` schema text in `server.py`.

---

## Error Envelope Contract

All MCP tool responses follow a standardized envelope format:

```json
{
  "success": true,
  "data": { ... }
}
```

On failure:

```json
{
  "success": false,
  "error": {
    "code": <integer>,
    "message": <string>
  }
}
```

Error code ranges:

| Range | Category |
|-------|----------|
| 1xxx | Validation errors |
| 3xxx | Data source errors (CCXT, yfinance) |
| 5xxx | Indicator/calculation errors |
| 9xxx | Internal/unexpected errors |

See `docs/error-envelope-contract.md` for full reference.

---

## Directory Structure

```
src/tempest_mcp/
├── server.py              # MCP server entry point (HTTP/SSE :9001)
├── config.py              # Environment config loader (TEMPEST_* vars)
├── errors.py              # Error code definitions
├── logging_config.py      # Structured logging setup
├── time_utils.py          # Time utilities
├── data/                  # Data layer
│   ├── _contracts.py     # Canonical OHLCV/orderbook schemas
│   ├── _factory.py       # HistoricalDataSource singleton
│   ├── _hist.py          # HistoricalDataSource (CCXT primary, yfinance fallback)
│   ├── _router.py        # DataSourceRouter
│   ├── _symbols.py       # Symbol normalization
│   ├── ccxt_adapter.py   # CCXT adapter (live primary)
│   ├── tv_adapter.py     # Legacy TradingView compatibility (NOT active)
│   └── yf_adapter.py    # yfinance adapter (fallback)
├── indicators/            # Technical indicator engine (30+ indicators)
│   ├── ta_wrapper.py     # ta-lib C extension wrapper
│   ├── session_levels.py # Asia/London/NY PDH/PDL
│   ├── structure.py      # Fibonacci, pivots, HH/HL
│   ├── momentum/         # RSI, MACD, ADX, Stochastic
│   ├── trend/            # EMA
│   ├── volatility/       # ATR
│   └── volume/           # VWAP, TPO, Volume Profile
├── backtest/             # Backtesting engine
│   ├── engine.py         # Bidirectional BacktestEngine
│   ├── walk_forward.py   # Walk-forward evaluation (internal, not MCP surface)
│   └── commission.py     # Commission/slippage modeling
├── strategies/            # Backtest strategy definitions
│   ├── backtest_pdh_session.py
│   ├── backtest_rsi.py
│   ├── backtest_vwap.py
│   ├── backtest_ema_stack.py
│   ├── backtest_order_blocks.py
│   └── backtest_elliot_wave.py
├── screener/             # Multi-factor crypto screener
│   └── scanner.py
├── sentiment/            # Sentiment analysis
│   ├── reddit.py
│   ├── rss.py
│   └── combined_sentiment.py
├── formatters/            # Output formatters
│   └── discord.py        # Discord embed formatter
├── charting/             # mplfinance (debug only)
│   └── mpl_chart.py
├── models/               # Dataclass models
│   ├── backtest.py
│   ├── indicator.py
│   └── market.py
└── tools/                # MCP tool handlers
    ├── __init__.py
    ├── analysis_tools.py
    ├── analytical_tools.py
    ├── backtest_tools.py
    ├── backtest_window.py
    ├── indicator_tools.py
    ├── market_tools.py
    ├── screener_tools.py
    └── sentiment_tools.py
```
