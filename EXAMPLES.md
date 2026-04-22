# EXAMPLES.md — Usage Examples for the Public MCP Surface

This document provides practical usage examples for each of the 21 public MCP tools exposed by `tempest-tradingview-mcp`. Examples are grounded in the live tool contracts defined in `src/tempest_mcp/server.py`.

For high-level tool descriptions and architecture, see [README.md](README.md).

---

## Table of Contents

1. [Market Data](#1-market-data)
2. [Technical Indicators](#2-technical-indicators)
3. [Backtesting](#3-backtesting)
4. [Screening](#4-screening)
5. [Analysis](#5-analysis)
6. [Sentiment](#6-sentiment)

---

## 1. Market Data

### `fetch_ticker` — Real-time ticker price + 24h volume

```json
{
  "name": "fetch_ticker",
  "arguments": {
    "symbol": "BTCUSDT",
    "exchange": "binance"
  }
}
```

**Notes:**
- Default exchange is `binance`.
- Symbol must be alphanumeric with an optional single `/` or `-` separator.

---

### `fetch_klines` — OHLCV klines

```json
{
  "name": "fetch_klines",
  "arguments": {
    "symbol": "ETHUSDT",
    "timeframe": "1h",
    "since": "2026-04-01T00:00:00",
    "limit": 100,
    "exchange": "binance"
  }
}
```

**Notes:**
- `timeframe` defaults to `"1h"`. Supported timeframes are defined in `src/tempest_mcp/tools/backtest_window.py`.
- `since` is an ISO 8601 datetime string; if timezone is omitted, interpreted in `America/New_York` before conversion to UTC.
- `limit` defaults to 100 candles.
- `source` defaults to `"ccxt"` (live primary).

---

### `fetch_orderbook` — Order book depth

```json
{
  "name": "fetch_orderbook",
  "arguments": {
    "symbol": "BTCUSDT",
    "limit": 20,
    "exchange": "binance"
  }
}
```

**Notes:**
- `limit` defaults to 20 price levels per side (bids/asks).

---

## 2. Technical Indicators

### `indicator_rsi` — RSI oscillator

```json
{
  "name": "indicator_rsi",
  "arguments": {
    "symbol": "BTCUSDT",
    "period": 14,
    "timeframe": "1h",
    "limit": 100,
    "exchange": "binance"
  }
}
```

**Notes:**
- RSI ranges 0–100: values below 30 indicate oversold conditions, above 70 indicate overbought.
- `period` defaults to 14.
- `timeframe` defaults to `"1h"`.
- `limit` defaults to 100 data points.

---

## 3. Backtesting

All backtest tools accept `trade_style` (`"day_trade"`, `"swing_trade"`, `"custom"`), `start_at`, `end_at`, `timeframe`, `exchange`, `initial_capital`, and `max_bars`. The `custom` trade style requires explicit `start_at` and `end_at` in ISO 8601 format. Timezone handling on `start_at`/`end_at` matches `fetch_klines`.

### `backtest_pdh_session` — PDH/PDL + Session Levels

```json
{
  "name": "backtest_pdh_session",
  "arguments": {
    "symbol": "BTCUSDT",
    "trade_style": "day_trade",
    "timeframe": "1h",
    "exchange": "binance",
    "initial_capital": 100000.0,
    "atr_period": 14,
    "atr_multiplier": 1.5,
    "session_types": ["asia", "london", "ny"]
  }
}
```

**Notes:**
- Long entry when close > PDH; short entry when close < PDL, within eligible session windows.
- `session_types` filters which sessions are eligible (`asia`, `london`, `ny`).

---

### `backtest_rsi` — RSI Mean Reversion

```json
{
  "name": "backtest_rsi",
  "arguments": {
    "symbol": "BTCUSDT",
    "trade_style": "swing_trade",
    "timeframe": "4h",
    "exchange": "binance",
    "rsi_period": 14,
    "confirmation_enabled": false,
    "oversold_threshold": 30.0,
    "overbought_threshold": 70.0,
    "risk_reward_ratio": 2.0,
    "atr_stop_multiplier": 1.5
  }
}
```

**Notes:**
- Long at oversold (< 30), short at overbought (> 70) by default.
- Optional divergence confirmation via `confirmation_enabled`.

---

### `backtest_vwap` — VWAP Anchored trend-following

```json
{
  "name": "backtest_vwap",
  "arguments": {
    "symbol": "BTCUSDT",
    "trade_style": "day_trade",
    "timeframe": "15m",
    "exchange": "binance",
    "vwap_anchor": "ny",
    "trend_fast_period": 7,
    "trend_slow_period": 25,
    "volume_lookback": 20,
    "volume_multiplier": 1.2
  }
}
```

**Notes:**
- `vwap_anchor` determines the VWAP reset point: `"asia"`, `"london"`, `"ny"`, or `"daily"`.
- Fast/slow EMA confirmation uses `trend_fast_period` and `trend_slow_period`.

---

### `backtest_ema_stack` — Multi-EMA trend-following

```json
{
  "name": "backtest_ema_stack",
  "arguments": {
    "symbol": "ETHUSDT",
    "trade_style": "swing_trade",
    "timeframe": "1d",
    "exchange": "binance",
    "ema_periods": [7, 25, 50, 200],
    "rr_multiple": 2.0,
    "trend_confirmation_bars": 1,
    "stop_buffer_pct": 0.0
  }
}
```

**Notes:**
- Default EMA periods are `[7, 25, 50, 200]`.
- `rr_multiple` sets the risk/reward ratio for stop-loss and take-profit.

---

### `backtest_order_blocks` — Institutional order blocks

```json
{
  "name": "backtest_order_blocks",
  "arguments": {
    "symbol": "BTCUSDT",
    "trade_style": "day_trade",
    "timeframe": "1h",
    "exchange": "binance",
    "confirmation_enabled": true,
    "atr_period": 14,
    "impulse_atr_mult": 1.0,
    "retest_atr_tolerance": 0.5,
    "min_bars_before_entry": 2,
    "max_zone_age_bars": 20,
    "risk_reward_ratio": 2.0
  }
}
```

**Notes:**
- `confirmation_enabled` defaults to `true` — waits for retest confirmation before entry.
- `impulse_atr_mult` sets the minimum body size as a multiple of ATR.
- `max_zone_age_bars` limits how old a zone can be before it is ignored.

---

### `backtest_elliot_wave` — Elliott Wave counting

```json
{
  "name": "backtest_elliot_wave",
  "arguments": {
    "symbol": "BTCUSDT",
    "trade_style": "day_trade",
    "timeframe": "1h",
    "exchange": "binance"
  }
}
```

**Notes:**
- Wave counting with trend confirmation. Accepts optional advanced parameters for wave identification constraints.

---

### `compare_strategies` — Compare 2+ strategies on one dataset

```json
{
  "name": "compare_strategies",
  "arguments": {
    "symbol": "BTCUSDT",
    "strategy_ids": ["pdh_session", "rsi", "vwap", "ema_stack", "order_blocks", "elliot_wave"],
    "trade_style": "day_trade",
    "start_at": "2026-01-01T00:00:00",
    "end_at": "2026-04-01T00:00:00",
    "timeframe": "1h",
    "exchange": "binance",
    "initial_capital": 100000.0
  }
}
```

**Notes:**
- Requires at least 2 strategy IDs.
- Allowed strategy IDs: `pdh_session`, `rsi`, `vwap`, `ema_stack`, `order_blocks`, `elliot_wave`.
- Results are ranked by `total_return` descending, then `sharpe_ratio` descending, then `strategy_id` ascending.

---

## 4. Screening

### `screener_scan` — Multi-factor crypto screener

```json
{
  "name": "screener_scan",
  "arguments": {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "filters": null,
    "min_score": 0.0,
    "exchange": "binance"
  }
}
```

**Notes:**
- `symbols` is optional; when omitted the screener scans across the full universe.
- `min_score` ranges 0–100 (default 0.0).
- Maximum symbols per scan is defined in `src/tempest_mcp/tools/screener_tools.py` as `MAX_SCAN_SYMBOLS`.

---

### `session_breakout_scan` — Session breakout screener

```json
{
  "name": "session_breakout_scan",
  "arguments": {
    "session": "ny",
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "exchange": "binance",
    "proximity_pct": 1.0,
    "volume_multiplier": 2.0
  }
}
```

**Notes:**
- `session` accepts `"asia"`, `"london"`, `"ny"` (`"new_york"` is accepted as an alias for `"ny"`).
- `proximity_pct` is the near-breakout threshold as a percentage.
- `volume_multiplier` is the volume confirmation threshold (default 2.0x).

---

### `order_block_screener_scan` — Order-block zone screener

```json
{
  "name": "order_block_screener_scan",
  "arguments": {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "exchange": "binance",
    "atr_period": 14,
    "impulse_atr_mult": 1.0,
    "max_zone_age_bars": 20
  }
}
```

**Notes:**
- Scans across fixed horizons (1h/1d and 4h/7d) for active order-block zones.
- Returns one best candidate per `(symbol, horizon)` job.
- `atr_period` range: 2–200 (default 14).
- `impulse_atr_mult` range: >0 to 10 (default 1.0).
- `max_zone_age_bars` range: 1–500 (default 20).

---

## 5. Analysis

All analysis tools require `symbol`, `timeframe`, `start_at`, and `end_at`. The `start_at`/`end_at` parameter notes from `fetch_klines` apply here as well.

### `calculate_volume_profile` — Volume profile for a time window

```json
{
  "name": "calculate_volume_profile",
  "arguments": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_at": "2026-04-01T00:00:00",
    "end_at": "2026-04-22T00:00:00",
    "exchange": "binance",
    "bin_count": 100,
    "profile_type": "fixed",
    "value_area_pct": 0.70
  }
}
```

**Notes:**
- Returns profile rows, POC (Point of Control), VAH (Value Area High), VAL (Value Area Low), and shape classification.
- `bin_count` range: 1–500 (default 100).
- `profile_type` is `"fixed"` or `"dynamic"`.
- `value_area_pct` defaults to 0.70 (70% of volume).

---

### `detect_order_blocks` — Active order-block zone detection

```json
{
  "name": "detect_order_blocks",
  "arguments": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_at": "2026-04-01T00:00:00",
    "end_at": "2026-04-22T00:00:00",
    "exchange": "binance",
    "atr_period": 14,
    "impulse_atr_mult": 1.0,
    "max_zone_age_bars": 20
  }
}
```

**Notes:**
- Read-only analytical output: candidate detection, invalidation, and age filtering only.
- No retest, entry, or PnL output.

---

### `calculate_fibonacci` — Fibonacci retracement or extension

```json
{
  "name": "calculate_fibonacci",
  "arguments": {
    "symbol": "BTCUSDT",
    "timeframe": "1d",
    "start_at": "2026-03-01T00:00:00",
    "end_at": "2026-04-22T00:00:00",
    "swing_high": 74000,
    "swing_low": 58000,
    "exchange": "binance",
    "output_mode": "retracement",
    "trend_direction": "bullish",
    "levels": [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
  }
}
```

**Notes:**
- `swing_high` and `swing_low` are required and must be explicit price values.
- `output_mode` is `"retracement"` (default) or `"extension"`.
- `trend_direction` is `"bullish"` or `"bearish"`.
- `levels` defaults to standard Fibonacci levels if not provided.

---

### `calculate_tpo` — TPO chart for a single session

```json
{
  "name": "calculate_tpo",
  "arguments": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_at": "2026-04-21T00:00:00",
    "end_at": "2026-04-22T00:00:00",
    "row_size": 100,
    "exchange": "binance",
    "value_area_pct": 0.70
  }
}
```

**Notes:**
- `row_size` is the required price increment for row buckets (must be positive).
- Returns TPO rows, POC, VAH, VAL for the session window.

---

### `detect_elliot_wave` — Elliott Wave pattern detection

```json
{
  "name": "detect_elliot_wave",
  "arguments": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_at": "2026-03-01T00:00:00",
    "end_at": "2026-04-22T00:00:00",
    "exchange": "binance",
    "swing_window": 2,
    "min_swing_pct": 0.05,
    "wave2_retrace_band": [0.5, 0.618],
    "wave3_extension_min": 1.0,
    "wave4_retrace_max": 0.618,
    "include_rejected": true
  }
}
```

**Notes:**
- `wave2_retrace_band` and `waveb_retrace_band` are `[min, max]` tuples.
- `degree_thresholds` is `[micro_max, minor_max]` for degree classification.
- `include_rejected` defaults to `true`.

---

### `get_market_structure` — Deterministic market structure summary

```json
{
  "name": "get_market_structure",
  "arguments": {
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "start_at": "2026-04-01T00:00:00",
    "end_at": "2026-04-22T00:00:00",
    "exchange": "binance",
    "swing_window": 2,
    "min_swing_pct": 0.02,
    "range_lookback": 20,
    "max_range_pct": 0.03,
    "breakout_confirm_bars": 1,
    "adx_period": 14,
    "adx_trend_threshold": 25.0,
    "adx_range_ceiling": 20.0,
    "di_spread_min": 2.0,
    "breakout_recency_bars": 3
  }
}
```

**Notes:**
- Returns HH/HL and LH/LL structure with trend vs. range regime classification.
- `adx_trend_threshold` (default 25.0) and `adx_range_ceiling` (default 20.0) control regime determination.

---

## 6. Sentiment

### `get_combined_sentiment_dashboard` — Combined Reddit + RSS sentiment

```json
{
  "name": "get_combined_sentiment_dashboard",
  "arguments": {
    "symbol": "BTCUSDT",
    "price_bias": "bullish"
  }
}
```

**Notes:**
- `price_bias` is required and must be `"bullish"`, `"bearish"`, or `"neutral"`.
- Returns a weighted `sentiment_index` (40% Reddit / 60% RSS) when both sources are available; falls back to the single usable source if one is unavailable.
- Includes per-source diagnostics and cross-signal detection against the caller-supplied `price_bias`.

---

## Shared Parameter Notes

| Parameter | Notes |
|-----------|-------|
| `symbol` | Alphanumeric with optional single `/` or `-` separator. |
| `exchange` | Defaults to `"binance"`. Other exchanges supported per `src/tempest_mcp/tools/screener_tools.py`. |
| `timeframe` | Must be one of the supported OHLCV intervals from `src/tempest_mcp/tools/backtest_window.py`. |
| `start_at` / `end_at` | ISO 8601 datetime. If timezone is omitted, interpreted in `America/New_York` before UTC conversion. Required for `custom` trade style. |
| `max_bars` | Safety cap on estimated candle count; applied when the window would otherwise generate excessive bars. |
