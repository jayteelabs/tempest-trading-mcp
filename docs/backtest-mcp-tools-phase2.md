# Backtest MCP Tools — Phase 2 Reference

**ENG-17 Phase 2 Contract** — This document explains exactly how each MCP backtest tool should be used, including required args, optional args, examples, defaults, and expected validation/deprecation failures.

---

## Overview

Phase 2 replaces the legacy single `backtest_strategy` stub with six dedicated per-strategy MCP tools:

| Tool | Strategy | Runner Mode |
|------|----------|-------------|
| `backtest_pdh_session` | PDH/PDL + Session Levels | Direct |
| `backtest_rsi` | RSI Mean Reversion | Adapter |
| `backtest_vwap` | VWAP Anchored | Direct |
| `backtest_ema_stack` | EMA Stack | Direct |
| `backtest_order_blocks` | Order Blocks | Adapter |
| `backtest_elliot_wave` | Elliott Wave | Direct |

**Adapter** = signal generator wrapped with BacktestEngine  
**Direct** = existing backtest runner

---

## Shared Parameters (All 6 Tools)

### Required

| Parameter | Type | Description |
|-----------|------|-------------|
| `symbol` | string | Trading symbol, e.g. `BTC/USDT`, `ETH/USDT` |

### Trade Style Presets

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trade_style` | enum | `day_trade` | `day_trade`, `swing_trade`, or `custom` |

**Preset defaults:**

| trade_style | Timeframe | Duration |
|-------------|-----------|----------|
| `day_trade` | `1h` | 24 hours |
| `swing_trade` | `4h` | 7 days |
| `custom` | caller-supplied or `1h` | caller-supplied |

### Custom Window (Required for `trade_style=custom`)

| Parameter | Type | Description |
|-----------|------|-------------|
| `start_at` | ISO datetime string | Start of window. If timezone is omitted, it is interpreted in `America/New_York` before conversion to UTC. Required when `trade_style=custom`. |
| `end_at` | ISO datetime string | End of window. If timezone is omitted, it is interpreted in `America/New_York` before conversion to UTC. Required when `trade_style=custom`. |

> **Strict reject behavior:** For `day_trade` and `swing_trade`, supplying `start_at` or `end_at` will result in a deterministic **validation error**. Use `trade_style=custom` with explicit timestamps.

### Optional

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeframe` | string | preset default | OHLCV interval: `1m`, `5m`, `15m`, `30m`, `1h`, `4h`, `1d`, `1wk`, `1mo` |
| `exchange` | string | `binance` | Exchange identifier |
| `initial_capital` | number | `100000.0` | Starting capital |
| `max_bars` | integer | none | Safety cap on estimated candle count |

---

## Tool-Specific Parameters

### `backtest_pdh_session`

PDH/PDL + Session Levels — enters long when close > PDH, short when close < PDL, within eligible session windows.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atr_period` | integer | `14` | ATR period (Wilder's smoothing) |
| `atr_multiplier` | number | `1.5` | Stop distance = atr_multiplier × ATR |
| `session_types` | array[string] | `["london", "ny"]` | Eligible sessions: `asia`, `london`, `ny` |

**Example:**
```json
{
  "symbol": "BTC/USDT",
  "trade_style": "day_trade",
  "atr_period": 14,
  "atr_multiplier": 1.5
}
```

---

### `backtest_rsi`

RSI Mean Reversion — LONG at oversold, SHORT at overbought, with optional divergence confirmation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rsi_period` | integer | `14` | RSI lookback period |
| `confirmation_enabled` | boolean | `false` | Require divergence confirmation |
| `oversold_threshold` | number | `30.0` | Long entry threshold |
| `overbought_threshold` | number | `70.0` | Short entry threshold |
| `risk_reward_ratio` | number | `2.0` | Stop/target ratio |
| `atr_stop_multiplier` | number | `1.5` | ATR-based stop distance |
| `divergence_window` | integer | `20` | Lookback for local extrema in divergence detection |

**Example:**
```json
{
  "symbol": "ETH/USDT",
  "trade_style": "swing_trade",
  "rsi_period": 14,
  "oversold_threshold": 30.0,
  "overbought_threshold": 70.0,
  "divergence_window": 20
}
```

---

### `backtest_vwap`

VWAP Anchored — trend-following using anchored VWAP with fast/slow EMA confirmation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vwap_anchor` | enum | `ny` | VWAP anchor point: `asia`, `london`, `ny`, `daily` |
| `trend_fast_period` | integer | `7` | Fast EMA period |
| `trend_slow_period` | integer | `25` | Slow EMA period |
| `volume_lookback` | integer | `20` | Volume lookback period |
| `volume_multiplier` | number | `1.2` | Volume threshold multiplier |

**Example:**
```json
{
  "symbol": "BTC/USDT",
  "trade_style": "day_trade",
  "vwap_anchor": "ny",
  "trend_fast_period": 7,
  "trend_slow_period": 25
}
```

---

### `backtest_ema_stack`

EMA Stack — multi-EMA trend-following with risk management.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ema_periods` | array[integer] | `[7, 25, 50, 200]` | List of EMA periods |
| `rr_multiple` | number | `2.0` | Risk/reward multiple for take profit |
| `trend_confirmation_bars` | integer | `1` | Bars required for trend confirmation |
| `stop_buffer_pct` | number | `0.0` | Stop buffer as fraction of price |

**Example:**
```json
{
  "symbol": "ETH/USDT",
  "trade_style": "swing_trade",
  "ema_periods": [7, 25, 50, 200],
  "rr_multiple": 2.0
}
```

---

### `backtest_order_blocks`

Order Blocks — institutional order block detection with retest confirmation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confirmation_enabled` | boolean | `true` | Require retest confirmation |
| `atr_period` | integer | `14` | ATR period for zone sizing |
| `impulse_atr_mult` | number | `1.0` | Impulse move ATR multiplier |
| `retest_atr_tolerance` | number | `0.5` | Retest tolerance (ATR fraction) |
| `min_bars_before_entry` | integer | `2` | Minimum bars before entry |
| `max_zone_age_bars` | integer | `20` | Maximum zone age in bars |
| `risk_reward_ratio` | number | `2.0` | Risk/reward ratio |

**Example:**
```json
{
  "symbol": "BTC/USDT",
  "trade_style": "day_trade",
  "confirmation_enabled": true,
  "atr_period": 14,
  "risk_reward_ratio": 2.0
}
```

---

### `backtest_elliot_wave`

Elliott Wave — wave counting with trend confirmation.

No additional parameters beyond shared window parameters. The tool exposes only the contract supported by `run_elliot_wave_backtest(...)`.

**Example:**
```json
{
  "symbol": "BTC/USDT",
  "trade_style": "swing_trade"
}
```

---

## Validation Failures

### `start_at >= end_at`

```json
{
  "success": false,
  "error": {
    "code": 1004,
    "message": "Invalid window: start_at (2024-01-10 00:00:00+00:00) must be before end_at (2024-01-01 00:00:00+00:00)"
  }
}
```

### `trade_style` mismatch with timestamps

```json
{
  "success": false,
  "error": {
    "code": 1004,
    "message": "trade_style='day_trade' does not support start_at or end_at. Use trade_style='custom' with explicit start_at and end_at."
  }
}
```

### `custom` without timestamps

```json
{
  "success": false,
  "error": {
    "code": 1004,
    "message": "trade_style='custom' requires both start_at and end_at to be specified"
  }
}
```

### Window too large (exceeds hard cap)

```json
{
  "success": false,
  "error": {
    "code": 1004,
    "message": "Window too large: estimated 17520 bars exceeds hard safety cap of 1000 bars. Use a shorter date range or increase timeframe."
  }
}
```

### Window exceeds caller max_bars

```json
{
  "success": false,
  "error": {
    "code": 1004,
    "message": "Window too large: estimated 216 bars exceeds caller-supplied max_bars=10. Reduce date range or increase max_bars."
  }
}
```

---

## Legacy `backtest_strategy` Deprecation

The legacy `backtest_strategy` tool is **deprecated**. It remains in the registry for one transition cycle but returns a deterministic deprecation error:

```json
{
  "success": false,
  "error": {
    "code": 1000,
    "message": "backtest_strategy is deprecated. Use one of the dedicated Phase 2 backtest tools: backtest_pdh_session, backtest_rsi, backtest_vwap, backtest_ema_stack, backtest_order_blocks, backtest_elliot_wave. Legacy inputs (period, source) are no longer supported."
  }
}
```

**Migration path:** Replace calls to `backtest_strategy` with the appropriate dedicated tool from the table above based on your strategy.

---

## Success Response Shape

```json
{
  "success": true,
  "data": {
    "tool": "backtest_pdh_session",
    "strategy_id": "pdh_session",
    "symbol": "BTC/USDT",
    "window": {
      "trade_style": "day_trade",
      "timeframe": "1h",
      "start_at_utc": "2024-01-01T00:00:00+00:00",
      "end_at_utc": "2024-01-02T00:00:00+00:00",
      "estimated_bars": 24,
      "exchange": "binance"
    },
    "metrics": {
      "total_return": 0.0523,
      "sharpe_ratio": 1.45,
      "max_drawdown": -0.0312,
      "win_rate": 0.62
    },
    "trade_count": 8,
    "open_position": false,
    "initial_capital": 100000.0,
    "final_equity": 105230.0
  }
}
```

---

## Excluded Legacy Parameters

The following legacy parameters are **not supported** on any Phase 2 backtest tool:

| Excluded Parameter | Reason |
|-------------------|--------|
| `period` | Replaced by `trade_style` presets + explicit window |
| `source` | Data source is always CCXT primary, yfinance fallback |

These inputs were part of the original `backtest_strategy` stub but are intentionally excluded from Phase 2 schemas to enforce the new contract.
