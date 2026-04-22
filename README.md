# tempest-tradingview-mcp

**Market data & analytics MCP server** — provides technical indicators, backtesting, and multi-factor screening for crypto markets via the Model Context Protocol.

**NOT a trading bot.** No order execution, no position management, no trade placement.

## Features

- **Market Data** — CCXT is the primary source for live crypto market data; Yahoo Finance is historical fallback for coverage gaps
- **Technical Indicators** — 30+ indicators via ta-lib C extension: EMA, RSI, MACD, ATR, VWAP, Supertrend, Bollinger, Stochastic, and more
- **Backtesting** — Commission/slippage model with 6 built-in strategies
- **Screening** — Multi-factor crypto screener with session breakout detection
- **Structured Logging** — JSON output via structlog

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/jayteelabs/tempest-tradingview-mcp.git
cd tempest-tradingview-mcp

# 2. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. Configure environment (optional — defaults work out of the box)
cp .env.example .env

# 5. Run the MCP server (stdio transport)
uv run python -m tempest_mcp.server
```

## Configuration

Configuration is loaded from environment variables with the `TEMPEST_` prefix:

| Variable | Description | Default |
|----------|-------------|---------|
| `TEMPEST_YF_CACHE_TTL` | Yahoo Finance cache TTL (seconds) | `3600` |
| `TEMPEST_CCXT_DEFAULT_EXCHANGE` | Default exchange for real-time quotes | `binance` |
| `TEMPEST_LOG_LEVEL` | Logging level | `INFO` |
| `TEMPEST_ENABLE_SSE` | Enable SSE transport (deferred to Phase 2) | `false` |

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

**No API keys required** — the active data sources use public endpoints.

> Note: despite the repository name, TradingView is not an active market-data
> provider in the current architecture. Legacy TradingView compatibility code is
> retained only to avoid breaking older imports.

## Deployment

### Docker

```bash
# Build the image
sg docker -c "docker build -t tempest-tradingview-mcp ."

# Run the container with the MCP port published
sg docker -c "docker run --rm -p 9001:9001 tempest-tradingview-mcp"
```

Notes:
- The MCP server must bind to `0.0.0.0` inside the container for Docker port publishing to work.
- Binding to `127.0.0.1` inside the container makes `/sse` reachable only from inside the container and breaks host-side MCP clients.
- For live verification, use a real SSE/MCP handshake instead of only checking whether port `9001` is open.

### Docker Compose

```bash
# Build and start the service
docker compose up -d --build

# View logs
docker compose logs -f

# Stop the service
docker compose down
```

The compose stack starts the MCP server on port `9001` with the HTTP/SSE transport (`/sse` for SSE connections, `/messages` for POST requests).

**Health check:** The service includes a Docker healthcheck that verifies the SSE endpoint responds on `http://localhost:9001/sse`. Use `docker compose ps` to see the health status.

**Note:** The MCP server must bind to `0.0.0.0` inside the container for Docker port publishing to work. The Dockerfile is pre-configured to run the server on `0.0.0.0:9001`.

## Architecture

```
Kurisu → tempest-mesh → MCP Server → [CCXT primary / Yahoo Finance fallback]
```

### Directory Structure

```
tempest-tradingview-mcp/
├── src/tempest_mcp/          # Main package
│   ├── __init__.py
│   ├── server.py             # MCP server entry point (stdio)
│   ├── config.py             # Environment config loader
│   ├── logging_config.py     # Structured logging setup
│   ├── models/               # Dataclass models (D6)
│   │   ├── market.py         # Ticker, kline, orderbook
│   │   ├── indicator.py      # Indicator results
│   │   └── backtest.py       # Backtest results
│   ├── data/                 # Data layer
│   │   ├── yf_adapter.py     # Yahoo Finance adapter
│   │   └── ccxt_adapter.py   # CCXT adapter (public, no keys)
│   ├── indicators/           # Technical indicator engine
│   │   ├── ta_wrapper.py     # ta-lib C extension wrapper
│   │   ├── session_levels.py # Asia/London/NY PDH/PDL
│   │   ├── trend.py          # EMA, VWAP, Supertrend, ADX
│   │   ├── momentum.py       # RSI, MACD, Stochastic, CCI
│   │   ├── volatility.py     # ATR, HV, Bollinger Width
│   │   ├── volume.py         # OBV, MFI, VWAP
│   │   └── structure.py      # Fibonacci, Pivots, HH/HL
│   ├── backtest/             # Backtesting engine
│   ├── screener/             # Multi-factor crypto screener
│   ├── sentiment/            # Sentiment analysis (Phase 4)
│   ├── charting/             # mplfinance (debug only)
│   └── tools/                # MCP tool handlers
├── tests/                    # Test suite
├── pyproject.toml
├── Dockerfile
├── .env.example
├── .gitignore
└── README.md
```

## Development

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Linting
uv run ruff check src/

# Format
uv run ruff format src/
```

### Live-Data Integration Tests

```bash
# Run Phase 2 backtest live-data integration suite (ENG-64)
# This fetches real BTCUSDT OHLCV via CCXT and validates strategy contracts
# across both 1h and 4h timeframes
uv run pytest --run-integration tests/test_phase2_backtest_live_integration.py -v
```

### Error Codes

| Range | Category |
|-------|----------|
| 1xxx | Validation errors |
| 2xxx | Authentication/authorization errors |
| 3xxx | Data source errors (Yahoo Finance, CCXT) |
| 5xxx | Indicator calculation errors |
| 9xxx | Internal/unexpected errors |

## License

MIT License
