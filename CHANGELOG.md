# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-23

### Added

#### Data Layer
- **CCXT adapter** (`data/ccxt_adapter.py`) — Primary live and historical crypto data adapter supporting Binance, Bybit, Kraken, and Coinbase public REST APIs. No API keys required.
- **yfinance adapter** (`data/yf_adapter.py`) — Fallback data source for stocks and crypto pairs not covered by CCXT.
- **Data source router** (`data/_router.py`) — Routes requests to the appropriate adapter based on data type (live vs. historical).
- **Data contracts** (`data/_contracts.py`) — Canonical OHLCV schema (`["open", "high", "low", "close", "volume"]`) with UTC-aware DatetimeIndex normalization.

#### Market Data Tools (3 tools)
- `fetch_ticker` — Latest trade price for a symbol
- `fetch_klines` — OHLCV candle data (historical and live)
- `fetch_orderbook` — Order book depth snapshot

#### Indicator Engine (30+ indicators)
- **Momentum**: RSI, MACD, ADX, Stochastic, CCI, Williams %R, ROC
- **Trend**: EMA (7, 25, 50, 200), VWAP
- **Volatility**: ATR, Bollinger Bands, historical volatility
- **Volume**: VWAP, TPO (Time Price Opportunity), Volume Profile

#### Backtesting Engine
- **Bidirectional backtest engine** (`backtest/engine.py`) — Supports LONG and SHORT positions with through-flat transitions
- **Commission and slippage modeling** (`backtest/commission.py`) — Symmetric percentage commission, configurable basis-point slippage
- **Walk-forward evaluation** (`backtest/walk_forward.py`) — Rolling train/test splits for out-of-sample strategy validation
- **Metrics**: total return, Sharpe ratio, max drawdown, win rate, profit factor, expectancy

#### Backtest Strategies (6 strategies)
- `backtest_pdh_session` — PDH/PDL + Asia/London/NY session levels
- `backtest_rsi` — RSI mean reversion
- `backtest_vwap` — VWAP anchored
- `backtest_ema_stack` — Multi-EMA stack trend following
- `backtest_order_blocks` — Order block detection + reaction
- `backtest_elliot_wave` — Elliott Wave pattern recognition

#### Screening Tools (3 tools)
- `screener_scan` — Multi-factor crypto screener
- `session_breakout_scan` — Session breakout detection
- `order_block_screener_scan` — Order block screening across symbols

#### Analysis Tools (6 tools)
- `calculate_volume_profile` — Volume profile analysis
- `detect_order_blocks` — Order block detection
- `calculate_fibonacci` — Fibonacci retracement/extension levels
- `calculate_tpo` — TPO (Time Price Opportunity) analysis
- `detect_elliot_wave` — Elliott Wave pattern detection
- `get_market_structure` — Market structure summary (HH/HL/LH/LL detection)

#### Sentiment Analysis
- **Reddit sentiment** (`sentiment/reddit.py`) — Reddit post/search sentiment
- **RSS feed sentiment** (`sentiment/rss.py`) — RSS feed monitoring and sentiment
- **Combined sentiment dashboard** (`sentiment/combined_sentiment.py`) — Weighted 40% Reddit / 60% RSS combination with cross-signal detection

#### Discord Integration
- **Discord formatter** (`formatters/discord.py`) — Converts MCP result envelopes into Discord embed dicts for notification workflows

### Documentation
- `docs/ARCHITECTURE.md` — Internal structure and extension points
- `docs/error-envelope-contract.md` — Error code reference
- `EXAMPLES.md` — Usage examples for all 21 public MCP tools
- `CONTRIBUTING.md` — Contribution guidelines and PR process
- `CODE_OF_CONDUCT.md` — Community guidelines

### Infrastructure
- **CI workflow** (`.github/workflows/ci.yml`) — Ruff linting, pytest suite with VCR cassette recording
- **Review workflow** (`.github/workflows/review.yml`) — Automated PR review comment tracking
- **Docker support** (`Dockerfile`, `docker-compose.yml`) — Containerized deployment on port 9001
- **SSE transport** — HTTP/SSE on port 9001 with `/sse` (GET) and `/messages` (POST) endpoints

### MCP Tool Surface

| Category | Tools |
|----------|-------|
| Market data | `fetch_ticker`, `fetch_klines`, `fetch_orderbook` |
| Indicators | `indicator_rsi` |
| Backtesting | `backtest_pdh_session`, `backtest_rsi`, `backtest_vwap`, `backtest_ema_stack`, `backtest_order_blocks`, `backtest_elliot_wave`, `compare_strategies` |
| Screening | `screener_scan`, `session_breakout_scan`, `order_block_screener_scan` |
| Analysis | `calculate_volume_profile`, `detect_order_blocks`, `calculate_fibonacci`, `calculate_tpo`, `detect_elliot_wave`, `get_market_structure` |
| Sentiment | `get_combined_sentiment_dashboard` |

**Total: 21 public MCP tools**

### Changed
- Package development status promoted from Alpha (3) to Beta (4)
- Repository initialized with MCP server skeleton and 6 initial tool stubs
- Data source architecture migrated to CCXT primary with yfinance fallback

### Fixed
- TA-Lib NumPy 2 compatibility via wheel-backed installation
- Workflow permissions and self-hosted runner configuration hardened
- Coverage reporting isolated from PR execution context
