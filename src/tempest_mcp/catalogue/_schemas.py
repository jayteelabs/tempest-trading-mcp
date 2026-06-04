"""Ordered public MCP tool schemas."""

from mcp.types import Tool

from tempest_mcp.tools.backtest_window import SUPPORTED_TIMEFRAMES

BACKTEST_TIMEFRAME_PROPERTY = {
    "type": "string",
    "enum": list(SUPPORTED_TIMEFRAMES),
    "description": "Supported OHLCV timeframe. Must be one of the explicitly supported intervals.",
}

BACKTEST_DATETIME_DESCRIPTION = (
    "ISO 8601 datetime; required when trade_style=custom. "
    "If timezone is omitted, the value is interpreted in America/New_York before conversion to UTC."
)

# ── Tool Schemas (MCP protocol surface) ──────────────────────────────────────
TOOL_SCHEMAS: list[Tool] = [
    Tool(
        name="fetch_ticker",
        description="Fetch real-time ticker price + metadata for a crypto symbol. Returns price (required), bid/ask/change_pct_24h/volume_24h (nullable).",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol (e.g., BTCUSDT, ETHUSD, BTC/USDT)"},
                "exchange": {
                    "type": "string",
                    "default": "binance",
                    "enum": ["binance", "bybit", "coinbase", "kraken"],
                    "description": "Exchange name",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="fetch_klines",
        description="Fetch OHLCV klines for a symbol. Routes through historical abstraction (CCXT primary, yfinance fallback). source must be 'ccxt'.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol (e.g., BTCUSDT, ETHUSD, BTC/USDT)"},
                "timeframe": {
                    "type": "string",
                    "default": "1h",
                    "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"],
                    "description": "OHLCV interval",
                },
                "since": {"type": "string", "nullable": True, "description": "ISO-8601 start time (naive interpreted as America/New_York)"},
                "limit": {"type": "integer", "default": 100, "minimum": 1, "maximum": 1000, "description": "Max candles to return"},
                "exchange": {
                    "type": "string",
                    "default": "binance",
                    "enum": ["binance", "bybit", "coinbase", "kraken"],
                    "description": "Exchange name",
                },
                "source": {"type": "string", "default": "ccxt", "description": "Must be 'ccxt' (historical routing is CCXT+yfinance fallback)"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="fetch_orderbook",
        description="Fetch order book (bid/ask depth) for a symbol. One-sided snapshots are allowed; both sides empty are treated as a data-source error.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol (e.g., BTCUSDT, ETHUSD, BTC/USDT)"},
                "limit": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100, "description": "Orderbook depth"},
                "exchange": {
                    "type": "string",
                    "default": "binance",
                    "enum": ["binance", "bybit", "coinbase", "kraken"],
                    "description": "Exchange name",
                },
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
    # ── Phase 2 dedicated backtest tools (ENG-17) ───────────────────────────────
    Tool(
        name="backtest_pdh_session",
        description="Backtest PDH/PDL + Session Levels strategy. Enters long when close > PDH, short when close < PDL, within eligible session windows.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {
                    "type": "integer",
                    "description": "Safety cap on estimated candle count.",
                },
                "atr_period": {"type": "integer", "default": 14},
                "atr_multiplier": {"type": "number", "default": 1.5},
                "session_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Eligible sessions: asia, london, ny",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="backtest_rsi",
        description="Backtest RSI Mean Reversion strategy. LONG at oversold, SHORT at overbought, with optional divergence confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {"type": "integer"},
                "rsi_period": {"type": "integer", "default": 14},
                "confirmation_enabled": {"type": "boolean", "default": False},
                "oversold_threshold": {"type": "number", "default": 30.0},
                "overbought_threshold": {"type": "number", "default": 70.0},
                "risk_reward_ratio": {"type": "number", "default": 2.0},
                "atr_stop_multiplier": {"type": "number", "default": 1.5},
                "divergence_window": {"type": "integer", "default": 20},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="backtest_vwap",
        description="Backtest VWAP Anchored strategy. Trend-following using anchored VWAP with fast/slow EMA confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {"type": "integer"},
                "vwap_anchor": {
                    "type": "string",
                    "enum": ["asia", "london", "ny", "daily"],
                    "default": "ny",
                },
                "trend_fast_period": {"type": "integer", "default": 7},
                "trend_slow_period": {"type": "integer", "default": 25},
                "volume_lookback": {"type": "integer", "default": 20},
                "volume_multiplier": {"type": "number", "default": 1.2},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="backtest_ema_stack",
        description="Backtest EMA Stack strategy. Multi-EMA trend-following with risk management.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {"type": "integer"},
                "ema_periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [7, 25, 50, 200],
                    "description": "List of EMA periods, e.g. [7,25,50,200]",
                },
                "rr_multiple": {"type": "number", "default": 2.0},
                "trend_confirmation_bars": {"type": "integer", "default": 1},
                "stop_buffer_pct": {"type": "number", "default": 0.0},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="backtest_order_blocks",
        description="Backtest Order Blocks strategy. Institutional order block detection with retest confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {"type": "integer"},
                "confirmation_enabled": {"type": "boolean", "default": True},
                "atr_period": {"type": "integer", "default": 14},
                "impulse_atr_mult": {"type": "number", "default": 1.0},
                "retest_atr_tolerance": {"type": "number", "default": 0.5},
                "min_bars_before_entry": {"type": "integer", "default": 2},
                "max_zone_age_bars": {"type": "integer", "default": 20},
                "risk_reward_ratio": {"type": "number", "default": 2.0},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="backtest_elliot_wave",
        description="Backtest Elliott Wave strategy. Wave counting with trend confirmation.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol"],
        },
    ),
    # ── Compare strategies tool (ENG-25) ─────────────────────────────────────────
    Tool(
        name="compare_strategies",
        description="Compare multiple backtest strategies using a single OHLCV dataset. Strategies are ranked by total_return descending, then sharpe_ratio descending, then strategy_id ascending.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "strategy_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Array of 2+ strategy IDs to compare. Allowed: pdh_session, rsi, vwap, ema_stack, order_blocks, elliot_wave",
                },
                "trade_style": {
                    "type": "string",
                    "enum": ["day_trade", "swing_trade", "custom"],
                    "default": "day_trade",
                },
                "start_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "end_at": {
                    "type": "string",
                    "description": BACKTEST_DATETIME_DESCRIPTION,
                },
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "exchange": {"type": "string", "default": "binance"},
                "initial_capital": {"type": "number", "default": 100000.0},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "strategy_ids"],
        },
    ),
    # ── Legacy deprecated tool (handled separately in call_tool) ──
    Tool(
        name="screener_scan",
        description="Multi-factor crypto screener.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "nullable": True,
                },
                "filters": {"type": "array", "items": {"type": "string"}, "nullable": True},
                "min_score": {"type": "number", "default": 0.0},
                "exchange": {"type": "string", "default": "binance"},
            },
        },
    ),
    # ── Session breakout screener (ENG-35) ─────────────────────────────────────────
    Tool(
        name="session_breakout_scan",
        description="Session breakout screener. Evaluates symbols for session breakout/proximity patterns against session high/low and previous-day high/low.",
        inputSchema={
            "type": "object",
            "properties": {
                "session": {
                    "type": "string",
                    "description": "Session type: asia, london, ny (new_york accepted as alias for ny)",
                },
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "nullable": True,
                },
                "exchange": {"type": "string", "default": "binance"},
                "proximity_pct": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Near-breakout threshold as percentage",
                },
                "volume_multiplier": {
                    "type": "number",
                    "default": 2.0,
                    "description": "Volume confirmation multiplier",
                },
            },
            "required": ["session"],
        },
    ),
    # ── Order-block screener (ENG-36) ────────────────────────────────────────────────
    Tool(
        name="order_block_screener_scan",
        description="Order-block screener. Scans symbols across fixed horizons (1h/1d and 4h/7d) for active order-block zones. Returns one best candidate per (symbol, horizon) job.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "nullable": True,
                },
                "exchange": {"type": "string", "default": "binance"},
                "atr_period": {
                    "type": "integer",
                    "default": 14,
                    "description": "ATR period for order-block detection (range 2-200)",
                },
                "impulse_atr_mult": {
                    "type": "number",
                    "default": 1.0,
                    "description": "Body size must be >= impulse_atr_mult * ATR (range >0 to 10)",
                },
                "max_zone_age_bars": {
                    "type": "integer",
                    "default": 20,
                    "description": "Max zone age in bars (range 1-500)",
                },
            },
        },
    ),
    # ── Analysis tools (ENG-28) ──────────────────────────────────────────────────
    Tool(
        name="calculate_volume_profile",
        description="Calculate volume profile for a symbol over a time window. Returns profile rows, POC, VAH, VAL, and shape classification.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "start_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "end_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "exchange": {"type": "string", "default": "binance"},
                "bin_count": {"type": "integer", "default": 100, "minimum": 1, "maximum": 500},
                "profile_type": {
                    "type": "string",
                    "enum": ["fixed", "dynamic"],
                    "default": "fixed",
                },
                "dynamic_mode": {"type": "string", "enum": ["atr", "pct"]},
                "atr_period": {"type": "integer", "default": 14},
                "atr_mult": {"type": "number", "default": 1.0},
                "range_pct": {"type": "number"},
                "value_area_pct": {"type": "number", "default": 0.70},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "timeframe", "start_at", "end_at"],
        },
    ),
    Tool(
        name="detect_order_blocks",
        description="Detect active order-block zones as of the end of the requested window. Read-only analytical output: candidate detection + invalidation + age filtering only. No retest, entry, or PnL output.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "start_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "end_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "exchange": {"type": "string", "default": "binance"},
                "atr_period": {"type": "integer", "default": 14},
                "impulse_atr_mult": {"type": "number", "default": 1.0},
                "max_zone_age_bars": {"type": "integer", "default": 20},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "timeframe", "start_at", "end_at"],
        },
    ),
    # ── ENG-37 analytical tools ─────────────────────────────────────────────────
    Tool(
        name="calculate_fibonacci",
        description="Calculate Fibonacci retracement or extension levels using deterministic anchors.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "start_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "end_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "swing_high": {"type": "number"},
                "swing_low": {"type": "number"},
                "exchange": {"type": "string", "default": "binance"},
                "output_mode": {
                    "type": "string",
                    "enum": ["retracement", "extension"],
                    "default": "retracement",
                },
                "trend_direction": {"type": "string", "enum": ["bullish", "bearish"]},
                "levels": {"type": "array", "items": {"type": "number"}},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "timeframe", "start_at", "end_at", "swing_high", "swing_low"],
        },
    ),
    Tool(
        name="calculate_tpo",
        description="Calculate TPO (Time-Price Opportunity) chart for a single session.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "start_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "end_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "row_size": {"type": "number", "description": "Required price increment for row buckets. Must be positive."},
                "exchange": {"type": "string", "default": "binance"},
                "value_area_pct": {"type": "number", "default": 0.70},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "timeframe", "start_at", "end_at", "row_size"],
        },
    ),
    Tool(
        name="detect_elliot_wave",
        description="Detect Elliott Wave patterns from OHLCV data.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "start_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "end_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "exchange": {"type": "string", "default": "binance"},
                "swing_window": {"type": "integer", "default": 2},
                "min_swing_pct": {"type": "number", "default": 0.05},
                "wave2_retrace_band": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Acceptable retracement range for wave 2 as [min, max]",
                },
                "wave3_extension_min": {"type": "number", "default": 1.0},
                "wave4_retrace_max": {"type": "number", "default": 0.618},
                "waveb_retrace_band": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Acceptable retracement range for wave B as [min, max]",
                },
                "wavec_extension_min": {"type": "number", "default": 1.0},
                "degree_thresholds": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Thresholds for degree classification as [micro_max, minor_max]",
                },
                "include_rejected": {"type": "boolean", "default": True},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "timeframe", "start_at", "end_at"],
        },
    ),
    Tool(
        name="get_market_structure",
        description="Get deterministic market structure summary for a time window.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "timeframe": BACKTEST_TIMEFRAME_PROPERTY,
                "start_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "end_at": {"type": "string", "description": BACKTEST_DATETIME_DESCRIPTION},
                "exchange": {"type": "string", "default": "binance"},
                "swing_window": {"type": "integer", "default": 2},
                "min_swing_pct": {"type": "number", "default": 0.02},
                "range_lookback": {"type": "integer", "default": 20},
                "max_range_pct": {"type": "number", "default": 0.03},
                "breakout_confirm_bars": {"type": "integer", "default": 1},
                "adx_period": {"type": "integer", "default": 14},
                "adx_trend_threshold": {"type": "number", "default": 25.0},
                "adx_range_ceiling": {"type": "number", "default": 20.0},
                "di_spread_min": {"type": "number", "default": 2.0},
                "breakout_recency_bars": {"type": "integer", "default": 3},
                "max_bars": {"type": "integer"},
            },
            "required": ["symbol", "timeframe", "start_at", "end_at"],
        },
    ),
    # ENG-41 Combined Sentiment Dashboard tool
    Tool(
        name="get_combined_sentiment_dashboard",
        description="Combined Reddit + RSS sentiment dashboard. Returns a weighted sentiment_index (40% Reddit / 60% RSS) when both sources are available, or falls back to the single usable source. Includes per-source diagnostics and cross-signal detection against caller-supplied price_bias.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "price_bias": {
                    "type": "string",
                    "enum": ["bullish", "bearish", "neutral"],
                    "description": "Caller-supplied directional bias used for cross-signal detection",
                },
            },
            "required": ["symbol", "price_bias"],
        },
    ),
]




def list_public_tools() -> list[Tool]:
    """Return the canonical ordered public MCP tool schema list."""
    return TOOL_SCHEMAS
