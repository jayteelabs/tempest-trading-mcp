# Error Envelope Contract — Data Layer

**Document:** Error Envelope Contract  
**Date:** 2026-04-04  
**Owner:** eru  
**Status:** Draft for Haga's Security Review  
**Design Decision:** D14 (Empty DataFrame on error, no exception propagation)

---

## Purpose

This document defines the contract between `data/` adapters and `market_tools.py` (and other downstream consumers) for error handling. Per D14, data adapters **NEVER propagate exceptions** to callers — instead, they return error envelopes that downstream tools must interpret.

---

## Adapter Return Contracts

### `fetch_live_price(symbol, exchange) -> float`

| Condition | Return Value | Log Level |
|-----------|--------------|-----------|
| Success | `float` (actual price) | INFO |
| Invalid symbol | `float('nan')` | ERROR |
| Network error | `float('nan')` | ERROR |
| Rate limit exceeded | `float('nan')` | ERROR |
| API unavailable | `float('nan')` | ERROR |

**Downstream Handling:**
```python
price = adapter.fetch_live_price("BTCUSDT")
if math.isnan(price):
    # Handle error case - no price available
    return {"success": False, "error": {"code": 3000, "message": "Price unavailable"}}
```

---

### `fetch_ohlcv_live(symbol, timeframe, limit) -> pd.DataFrame`

| Condition | Return Value | Log Level |
|-----------|--------------|-----------|
| Success | DataFrame with OHLCV columns, UTC index | INFO |
| Invalid symbol | Empty DataFrame with correct columns | ERROR |
| Network error | Empty DataFrame with correct columns | ERROR |
| API unavailable | Empty DataFrame with correct columns | ERROR |

**DataFrame Columns (always present):**
- `open`: float
- `high`: float
- `low`: float
- `close`: float
- `volume`: float
- Index: `pd.DatetimeIndex` (UTC-aware)

**Downstream Handling:**
```python
df = adapter.fetch_ohlcv_live("BTCUSDT", "1m", 100)
if df.empty:
    # Handle error case - no OHLCV data available
    return {"success": False, "error": {"code": 3000, "message": "OHLCV data unavailable"}}

# DataFrame is valid - proceed with analysis
```

---

### `fetch_orderbook_snapshot(symbol, limit) -> dict`

| Condition | Return Value | Log Level |
|-----------|--------------|-----------|
| Success | `{"bids": [...], "asks": [...], "timestamp": Timestamp}` | INFO |
| Invalid symbol | `{"bids": [], "asks": [], "timestamp": None}` | ERROR |
| Network error | `{"bids": [], "asks": [], "timestamp": None}` | ERROR |
| API unavailable | `{"bids": [], "asks": [], "timestamp": None}` | ERROR |

**Structure:**
```python
{
    "bids": [[price: float, amount: float], ...],  # Sorted by price desc
    "asks": [[price: float, amount: float], ...],  # Sorted by price asc
    "timestamp": pd.Timestamp(tz="UTC") | None,
}
```

**Downstream Handling:**
```python
ob = adapter.fetch_orderbook_snapshot("BTCUSDT", 20)
if ob["timestamp"] is None or len(ob["bids"]) == 0:
    # Handle error case - no orderbook data
    return {"success": False, "error": {"code": 3000, "message": "Orderbook unavailable"}}

# Orderbook is valid - proceed with analysis
```

---

## Error Code Taxonomy

Per D8, error codes are organized in ranges:

| Range | Category | Source |
|-------|----------|--------|
| 1xxx | Validation errors | Input validation |
| 3xxx | Data source errors | Active adapter/runtime failures |
| 5xxx | Indicator errors | Calculation failures |
| 9xxx | Internal errors | Unexpected failures |

The current public MCP error envelope follows `src/tempest_mcp/config.py::ErrorCodes`.
No 2xxx auth codes are currently assigned, and the older TradingView/CCXT/YFinance split ranges are legacy/stale design notes.

### Implemented Data Source Error Codes (3000-3005)

| Code | Name | Description |
|------|------|-------------|
| 3000 | DATA_SOURCE_ERROR | Generic upstream data-source failure |
| 3001 | YFINANCE_ERROR | Yahoo Finance adapter error |
| 3002 | CCXT_ERROR | CCXT adapter error |
| 3003 | DATA_NOT_FOUND | Requested data was not found |
| 3004 | RATE_LIMIT_ERROR | Upstream rate limit exceeded |
| 3005 | NETWORK_ERROR | Network/timeout failure |

---

## Logging Requirements

### Log Levels by Outcome

| Outcome | Level | Keys Required |
|---------|-------|---------------|
| Success | INFO | `source`, `symbol`, `timeframe` (if applicable) |
| Fallback activation | INFO | `symbol`, `reason` |
| Failure | ERROR | `error`, `symbol`, `source` |

### Example Log Entries

**Success:**
```json
{
  "level": "info",
  "message": "fetch_live_price_success",
  "source": "ccxt",
  "exchange": "binance",
  "symbol": "BTCUSDT",
  "price": 67234.50
}
```

**Fallback:**
```json
{
  "level": "info",
  "message": "historical_fetch_fallback_yfinance",
  "symbol": "BTC-USD",
  "reason": "CCXT returned empty"
}
```

**Error:**
```json
{
  "level": "error",
  "message": "fetch_ohlcv_network_error",
  "source": "ccxt",
  "symbol": "BTCUSDT",
  "timeframe": "1m",
  "error": "Connection refused"
}
```

---

## market_tools.py Integration

The `market_tools.py` module should:

1. **Never catch exceptions from adapters** — they don't raise any
2. **Check return values for error indicators:**
   - `math.isnan(price)` for `fetch_live_price`
   - `df.empty` for `fetch_ohlcv_live`
   - `ob["timestamp"] is None` for `fetch_orderbook_snapshot`
3. **Map to appropriate MCP response envelope:**

```python
# Example market_tools.py pattern
def fetch_ticker(symbol: str) -> dict:
    """MCP tool wrapper for ticker data."""
    adapter = get_live_adapter()
    price = adapter.fetch_live_price(symbol)
    
    if math.isnan(price):
        return {
            "success": False,
            "error": {
                "code": 3000,
                "message": f"Unable to fetch price for {symbol}",
            }
        }
    
    return {
        "success": True,
        "data": {
            "symbol": symbol,
            "price": price,
            "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        }
    }
```

---

## Testing Requirements

Tests must verify:

1. **Error conditions return correct envelopes:**
   - Invalid symbol -> `float('nan')` / empty DataFrame / empty dict
   - Network errors -> same as above
   
2. **No exceptions propagate:**
   ```python
   # This should NEVER raise
   price = adapter.fetch_live_price("DEFINITELY_NOT_A_REAL_SYMBOL")
   assert math.isnan(price)
   ```

3. **Structured logging occurs:**
   - INFO on success
   - WARNING on fallback
   - ERROR on failure

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-04 | eru | Initial draft for security review |
