# Error Envelope Contract — Data Layer

**Document:** Error Envelope Contract  
**Date:** 2026-04-04  
**Owner:** eru  
**Status:** Draft for Haga's Security Review  
**Design Decision:** D14 (sentinel returns for covered adapter entry points)

---

## Purpose

This document defines the contract between `data/` adapters and downstream MCP consumers for error handling. For the D14-covered adapter entry points used by current MCP flows — `CCXTAdapter.fetch_live_price`, `CCXTAdapter.fetch_ohlcv_live`, `CCXTAdapter.fetch_orderbook_snapshot`, `HistoricalDataSource.fetch_ohlcv`, and module-level `yf_adapter.fetch_ohlcv` — failures are returned as sentinel values (`float('nan')`, empty OHLCV frames, empty order book payloads) instead of being propagated as exceptions. This is **not** universal across every adapter/compatibility API: legacy `YFAdapter` object methods such as `fetch_ticker`, `fetch_klines`, and `get_historical_prices` still raise `YFinanceError`.

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
    # Handle error case - NaN is a generic failure sentinel
    return {"success": False, "error": {"code": 3000, "message": "Unable to fetch price data"}}
```

`math.isnan(price)` is a lossy sentinel: it can mean invalid symbol, upstream outage,
rate limiting, or other adapter failures. Downstream callers should validate inputs
before invoking the adapter if they need 1xxx validation codes; once inputs are known
valid, a raw NaN sentinel should map to `3000 DATA_SOURCE_ERROR` unless separate
metadata proves a true "not found" condition.

---

### `fetch_ohlcv_live(symbol, timeframe, limit) -> pd.DataFrame`

| Condition | Return Value | Log Level |
|-----------|--------------|-----------|
| Success | DataFrame with OHLCV columns, UTC-aware `DatetimeIndex` | INFO |
| Invalid symbol | Empty DataFrame with correct columns | ERROR |
| Network error | Empty DataFrame with correct columns | ERROR |
| API unavailable | Empty DataFrame with correct columns | ERROR |

**DataFrame Contract:**
- `open`: float
- `high`: float
- `low`: float
- `close`: float
- `volume`: float
- Success path index: `pd.DatetimeIndex` (UTC-aware)
- Empty error sentinels only guarantee the canonical OHLCV columns; implementations
  often return `pd.DataFrame(columns=OHLCV_COLUMNS)`, which uses the default
  `RangeIndex`

**Downstream Handling:**
```python
df = adapter.fetch_ohlcv_live("BTCUSDT", "1m", 100)
if df.empty:
    # Handle error case - empty frame is a generic failure sentinel
    return {"success": False, "error": {"code": 3000, "message": "Unable to fetch OHLCV data"}}

# DataFrame is valid - proceed with analysis
```

`df.empty` is also lossy at this boundary: it may represent invalid input, network
failure, rate limiting, or a genuine no-data result. Downstream callers should validate
inputs before invoking the adapter if they need 1xxx validation codes. Reserve
`3003 DATA_NOT_FOUND` for cases where the caller has an explicit upstream
classification instead of only the empty-frame sentinel.

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

## Downstream MCP Integration

Public MCP layers (`server.py`, current tool handlers, and any future `market_tools.py`
wrappers) should:

1. **Validate request arguments before adapter calls** when the caller needs to emit
   1xxx validation errors instead of collapsing everything into a data-source failure
2. **Prefer sentinel checks for D14-covered entry points, but keep downstream wrappers defensive:** legacy compatibility APIs (notably `YFAdapter` methods) can still raise typed adapter exceptions, and future regressions should still be converted into MCP envelopes instead of leaking raw exceptions
3. **Check return values for error indicators:**
    - `math.isnan(price)` for `fetch_live_price`
    - `df.empty` for `fetch_ohlcv_live`
    - `ob["timestamp"] is None` for `fetch_orderbook_snapshot`
4. **Map raw post-validation sentinel-only failures to `3000 DATA_SOURCE_ERROR`:**
   - use `3003 DATA_NOT_FOUND` only when upstream classification explicitly says
     the requested data does not exist

```python
# Example market_tools.py pattern
def fetch_ticker(symbol: str) -> dict:
    """MCP tool wrapper for ticker data."""
    adapter = get_live_adapter()
    try:
        price = adapter.fetch_live_price(symbol)
    except Exception:
        return {
            "success": False,
            "error": {
                "code": 3000,
                "message": f"Unable to fetch price for {symbol}",
            }
        }
    
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
   
2. **D14-covered entry points do not propagate exceptions:**
    ```python
    # This should not raise for the sentinel-based adapter entry points
    price = adapter.fetch_live_price("DEFINITELY_NOT_A_REAL_SYMBOL")
    assert math.isnan(price)
    ```

   Legacy compatibility methods on `YFAdapter` are a separate contract and should be
   tested for typed `YFinanceError` failures where they remain supported.

3. **Structured logging occurs:**
   - INFO on success
   - WARNING on fallback
   - ERROR on failure

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-04-04 | eru | Initial draft for security review |
