"""Tests for DiscordFormatter — TVMCP-039 (ENG-42)."""

from __future__ import annotations

from copy import deepcopy

import pytest

from tempest_mcp.formatters import DiscordFormatter
from tempest_mcp.formatters import discord as discord_module

# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def formatter():
    return DiscordFormatter()


# ── Helper: Deep-copy invariant check ─────────────────────────────────────────

def assert_non_mutated(formatter: DiscordFormatter, original: dict, result: dict):
    """Verify the input dict is not mutated by format()."""
    assert result is not None
    assert original == deepcopy(original)


# ── Dispatch Conformance ────────────────────────────────────────────────────────

DISPATCH_TEST_CASES = [
    # (tool_name, expected_method_name)
    ("backtest_pdh_session", "format_backtest"),
    ("backtest_rsi", "format_backtest"),
    ("backtest_vwap", "format_backtest"),
    ("backtest_ema_stack", "format_backtest"),
    ("backtest_order_blocks", "format_backtest"),
    ("backtest_elliot_wave", "format_backtest"),
    ("compare_strategies", "format_compare"),
    ("screener_scan", "format_screener"),
    ("session_breakout_scan", "format_screener"),
    ("order_block_screener_scan", "format_screener"),
    ("get_combined_sentiment_dashboard", "format_sentiment"),
    ("fetch_ticker", "format_market"),
    ("fetch_klines", "format_market"),
    ("fetch_orderbook", "format_market"),
    ("calculate_volume_profile", "format_analytical"),
    ("detect_order_blocks", "format_analytical"),
    ("calculate_fibonacci", "format_analytical"),
    ("calculate_tpo", "format_analytical"),
    ("detect_elliot_wave", "format_analytical"),
    ("get_market_structure", "format_analytical"),
    ("indicator_rsi", "format_indicator"),
]


@pytest.mark.parametrize("tool_name,expected_method", DISPATCH_TEST_CASES)
def test_dispatch_routes_to_correct_formatter(formatter, tool_name, expected_method):
    """format() must route each known tool to the correct formatter method."""
    result = {"success": True, "data": {"tool": tool_name}}
    formatted = formatter.format(result)
    assert formatted is not None
    assert isinstance(formatted, dict)
    assert "title" in formatted
    assert "color" in formatted
    assert "fields" in formatted


def test_dispatch_error_envelope(formatter):
    """success=False must route to format_error."""
    result = {"success": False, "error": {"code": 500, "message": "Server error"}}
    formatted = formatter.format(result)
    assert formatted["color"] == 0xE74C3C
    assert formatted["title"] == "❌ Error"


def test_dispatch_unknown_tool_routes_to_generic(formatter):
    """Unknown tool must route to format_generic."""
    result = {"success": True, "data": {"tool": "unknown_tool_xyz"}}
    formatted = formatter.format(result)
    assert formatted["color"] == 0x3498DB
    assert "fields" in formatted


def test_dispatch_missing_tool_routes_to_generic(formatter):
    """success=True but missing tool must route to format_generic."""
    result = {"success": True, "data": {}}
    formatted = formatter.format(result)
    assert formatted["color"] == 0x3498DB


def test_dispatch_non_dict_routes_to_error_embed(formatter):
    """Non-dict input must return an error embed."""
    formatted = formatter.format("not a dict")
    assert formatted["color"] == 0xE74C3C
    assert formatted["title"] == "❌ Invalid Payload"


@pytest.mark.parametrize("bad_data", [None, [], "boom"])
def test_dispatch_malformed_success_data_routes_to_error_embed(formatter, bad_data):
    """Malformed success envelopes should return an invalid-payload embed."""
    formatted = formatter.format({"success": True, "data": bad_data})
    assert formatted["color"] == 0xE74C3C
    assert formatted["title"] == "❌ Invalid Payload"


# ── format_backtest ─────────────────────────────────────────────────────────────

BACKTEST_FIXTURES = [
    # Profitable (green)
    {
        "success": True,
        "data": {
            "tool": "backtest_pdh_session",
            "strategy_id": "pdh_session",
            "symbol": "BTC/USDT",
            "trade_count": 42,
            "initial_capital": 100000.0,
            "final_equity": 115000.0,
            "metrics": {
                "total_return": 15.0,
                "sharpe_ratio": 1.8,
                "max_drawdown": -8.5,
                "win_rate": 55.0,
                "profit_factor": 1.6,
                "avg_win": 500.0,
                "avg_loss": -300.0,
            },
        },
    },
    # Loss (red)
    {
        "success": True,
        "data": {
            "tool": "backtest_rsi",
            "strategy_id": "rsi",
            "symbol": "ETH/USDT",
            "trade_count": 30,
            "initial_capital": 100000.0,
            "final_equity": 85000.0,
            "metrics": {
                "total_return": -15.0,
                "sharpe_ratio": -0.5,
                "max_drawdown": -25.0,
                "win_rate": 40.0,
                "profit_factor": 0.8,
                "avg_win": 400.0,
                "avg_loss": -500.0,
            },
        },
    },
]


@pytest.mark.parametrize("fixture", BACKTEST_FIXTURES)
def test_format_backtest_happy_path(formatter, fixture):
    """format_backtest renders expected fields and correct color."""
    formatted = formatter.format_backtest(fixture)
    assert "title" in formatted
    assert "fields" in formatted
    # Should have 12 fields
    assert len(formatted["fields"]) == 12
    # Correct color
    if fixture["data"]["final_equity"] >= fixture["data"]["initial_capital"]:
        assert formatted["color"] == 0x2ECC71
    else:
        assert formatted["color"] == 0xE74C3C
    assert_non_mutated(formatter, fixture, formatted)


def test_format_backtest_missing_fields(formatter):
    """Missing fields must render as N/A without crashing."""
    result = {"success": True, "data": {"tool": "backtest_vwap", "strategy_id": None}}
    formatted = formatter.format_backtest(result)
    assert formatted["color"] in (0x2ECC71, 0xE74C3C, 0xF1C40F)
    field_values = [f["value"] for f in formatted["fields"]]
    assert any("N/A" in v for v in field_values)


def test_format_backtest_non_finite_metrics(formatter):
    """Non-finite floats (inf, nan) must render as N/A."""
    result = {
        "success": True,
        "data": {
            "tool": "backtest_ema_stack",
            "strategy_id": "ema_stack",
            "symbol": "BTC/USDT",
            "trade_count": 10,
            "initial_capital": float("inf"),
            "final_equity": float("nan"),
            "metrics": {
                "total_return": float("nan"),
                "sharpe_ratio": float("inf"),
                "max_drawdown": -5.0,
                "win_rate": 50.0,
                "profit_factor": 1.2,
                "avg_win": 100.0,
                "avg_loss": -90.0,
            },
        },
    }
    formatted = formatter.format_backtest(result)
    field_values = [f["value"] for f in formatted["fields"]]
    # inf and nan should appear as N/A
    na_count = sum(1 for v in field_values if v == "N/A")
    assert na_count >= 2  # initial_capital and final_equity


# ── format_compare ──────────────────────────────────────────────────────────────

def test_format_compare_top_n_plus_count(formatter):
    """Locked decision: format_compare must use top-N + count metadata."""
    # Create 15 results — only top 10 should be displayed
    results = [
        {
            "strategy_id": f"strategy_{i}",
            "rank": i + 1,
            "total_return": 10.0 - i * 0.5,
            "sharpe_ratio": 2.0 - i * 0.1,
            "trade_count": 50 - i * 2,
            "open_position": False,
        }
        for i in range(15)
    ]
    result = {
        "success": True,
        "data": {
            "tool": "compare_strategies",
            "best_strategy_id": "strategy_1",
            "ranking_metric": "total_return",
            "results": results,
        },
    }
    formatted = formatter.format_compare(result)
    # Should have 10 strategy rows + 1 metadata field = 11 fields
    assert len(formatted["fields"]) == 11
    # Check metadata field
    meta_field = formatted["fields"][-1]
    assert "Showing top 10 of 15 strategies" in meta_field["value"]
    assert "truncated" in meta_field["value"].lower()
    assert_non_mutated(formatter, result, formatted)


def test_format_compare_empty_results(formatter):
    """Empty results should still render with metadata."""
    result = {
        "success": True,
        "data": {
            "tool": "compare_strategies",
            "best_strategy_id": "N/A",
            "results": [],
        },
    }
    formatted = formatter.format_compare(result)
    # Only metadata field (no strategy rows)
    assert len(formatted["fields"]) == 1


# ── format_screener ────────────────────────────────────────────────────────────

def test_format_screener_results_shape(formatter):
    """screener_scan / session_breakout_scan use 'results' shape."""
    result = {
        "success": True,
        "data": {
            "tool": "screener_scan",
            "results": [
                {"symbol": "BTC/USDT", "exchange": "binance", "score": 0.85, "filters_matched": 3, "price": 50000.0},
                {"symbol": "ETH/USDT", "exchange": "binance", "score": 0.55, "filters_matched": 2, "price": 3000.0},
                {"symbol": "SOL/USDT", "exchange": "binance", "score": 0.30, "filters_matched": 1, "price": 100.0},
            ],
        },
    }
    formatted = formatter.format_screener(result)
    assert formatted["color"] == 0x3498DB
    # 3 rows: BTC (0.85 -> 🔒), ETH (0.55 -> ⚠️), SOL (0.30 -> 🔴)
    field_names = [f["name"] for f in formatted["fields"]]
    assert "🔒 BTC/USDT (binance)" in field_names[0]
    assert "⚠️ ETH/USDT (binance)" in field_names[1]
    assert "🔴 SOL/USDT (binance)" in field_names[2]
    assert_non_mutated(formatter, result, formatted)


def test_format_screener_candidates_shape(formatter):
    """order_block_screener_scan uses 'candidates' shape with zone_type."""
    result = {
        "success": True,
        "data": {
            "tool": "order_block_screener_scan",
            "candidates": [
                {"symbol": "BTC/USDT", "exchange": "binance", "score": 0.9, "zone_type": "bullish_ob", "price": 50000.0},
            ],
        },
    }
    formatted = formatter.format_screener(result)
    assert "🔒 BTC/USDT (binance)" in formatted["fields"][0]["name"]
    assert "bullish_ob" in formatted["fields"][0]["value"]


def test_format_screener_prefers_results_shape_when_candidates_empty(formatter):
    """Empty candidates must not force candidate row formatting when results are used."""
    result = {
        "success": True,
        "data": {
            "tool": "order_block_screener_scan",
            "candidates": [],
            "results": [
                {
                    "symbol": "BTC/USDT",
                    "exchange": "binance",
                    "score": 0.55,
                    "filters_matched": 3,
                    "price": 50000.0,
                }
            ],
        },
    }
    formatted = formatter.format_screener(result)
    assert "Filters: 3" in formatted["fields"][0]["value"]


def test_format_screener_failures(formatter):
    """Failures block is rendered when present."""
    result = {
        "success": True,
        "data": {
            "tool": "screener_scan",
            "results": [],
            "failures": [{"symbol": "DOGE/USDT", "reason": "No data"}],
        },
    }
    formatted = formatter.format_screener(result)
    field_names = [f["name"] for f in formatted["fields"]]
    assert "⚠️ Failures" in field_names


def test_format_screener_score_normalization_0_1_scale(formatter):
    """Score 0-1 scale normalized correctly."""
    result = {
        "success": True,
        "data": {
            "tool": "screener_scan",
            "results": [
                {"symbol": "A", "exchange": "binance", "score": 0.75, "filters_matched": 2, "price": 100.0},
                {"symbol": "B", "exchange": "binance", "score": 0.65, "filters_matched": 1, "price": 100.0},
                {"symbol": "C", "exchange": "binance", "score": 0.35, "filters_matched": 1, "price": 100.0},
            ],
        },
    }
    formatted = formatter.format_screener(result)
    names = [f["name"] for f in formatted["fields"]]
    assert names[0].startswith("🔒")  # 75% >= 70%
    assert names[1].startswith("⚠️")  # 65% >= 40%
    assert names[2].startswith("🔴")  # 35% < 40%


def test_format_screener_score_normalization_0_100_scale(formatter):
    """Score 0-100 scale normalized correctly."""
    result = {
        "success": True,
        "data": {
            "tool": "screener_scan",
            "results": [
                {"symbol": "A", "exchange": "binance", "score": 85.0, "filters_matched": 3, "price": 100.0},
                {"symbol": "B", "exchange": "binance", "score": 55.0, "filters_matched": 2, "price": 100.0},
                {"symbol": "C", "exchange": "binance", "score": 25.0, "filters_matched": 1, "price": 100.0},
            ],
        },
    }
    formatted = formatter.format_screener(result)
    names = [f["name"] for f in formatted["fields"]]
    assert names[0].startswith("🔒")  # 85 >= 70
    assert names[1].startswith("⚠️")  # 55 >= 40
    assert names[2].startswith("🔴")  # 25 < 40


# ── format_sentiment ───────────────────────────────────────────────────────────

SENTIMENT_FIXTURES = [
    ("bullish", 0x2ECC71, "😀"),
    ("bearish", 0xE74C3C, "😠"),
    ("neutral", 0xF1C40F, "😐"),
]


@pytest.mark.parametrize("polarity,expected_color,expected_mood", SENTIMENT_FIXTURES)
def test_format_sentiment_polarity(formatter, polarity, expected_color, expected_mood):
    """Sentiment polarity determines color and mood emoji."""
    result = {
        "success": True,
        "data": {
            "tool": "get_combined_sentiment_dashboard",
            "sentiment_polarity": polarity,
            "sentiment_index": 0.65,
            "combination_mode": "weighted",
        },
    }
    formatted = formatter.format_sentiment(result)
    assert formatted["color"] == expected_color
    assert expected_mood in formatted["fields"][0]["name"]


@pytest.mark.parametrize(
    "polarity,expected_color,expected_mood",
    [
        ("Bullish", 0x2ECC71, "😀"),
        ("Bearish", 0xE74C3C, "😠"),
        ("Neutral", 0xF1C40F, "😐"),
    ],
)
def test_format_sentiment_polarity_is_case_insensitive(formatter, polarity, expected_color, expected_mood):
    """Mood and color should stay aligned for cased polarity values."""
    result = {
        "success": True,
        "data": {
            "tool": "get_combined_sentiment_dashboard",
            "sentiment_polarity": polarity,
            "sentiment_index": 0.65,
            "combination_mode": "weighted",
        },
    }
    formatted = formatter.format_sentiment(result)
    assert formatted["color"] == expected_color
    assert expected_mood in formatted["fields"][0]["name"]


def test_format_sentiment_cross_signal_flags(formatter):
    """Cross-signal flags are rendered when present."""
    result = {
        "success": True,
        "data": {
            "tool": "get_combined_sentiment_dashboard",
            "sentiment_polarity": "neutral",
            "sentiment_index": 0.5,
            "combination_mode": "weighted",
            "cross_signal_flags": ["bearish_divergence", "oversold_rsi"],
        },
    }
    formatted = formatter.format_sentiment(result)
    field_names = [f["name"] for f in formatted["fields"]]
    assert any("Cross-Signal" in n for n in field_names)


def test_format_sentiment_missing_fields(formatter):
    """Missing fields render as N/A without crashing."""
    result = {
        "success": True,
        "data": {
            "tool": "get_combined_sentiment_dashboard",
            "sentiment_polarity": None,
            "sentiment_index": None,
        },
    }
    formatted = formatter.format_sentiment(result)
    assert formatted["color"] == 0xF1C40F
    field_values = [f["value"] for f in formatted["fields"]]
    assert any("N/A" in v for v in field_values)


# ── format_market ───────────────────────────────────────────────────────────────

def test_format_market_all_fields(formatter):
    """format_market renders all available fields."""
    result = {
        "success": True,
        "data": {
            "tool": "fetch_klines",
            "symbol": "BTC/USDT",
            "exchange": "binance",
            "timeframe": "1h",
            "limit": 100,
            "note": "Live data",
        },
    }
    formatted = formatter.format_market(result)
    assert formatted["color"] == 0x3498DB
    field_names = [f["name"] for f in formatted["fields"]]
    assert "Symbol" in field_names
    assert "Exchange" in field_names
    assert "Timeframe" in field_names
    assert "Limit" in field_names
    assert "Note" in field_names
    assert_non_mutated(formatter, result, formatted)


def test_format_market_minimal_fields(formatter):
    """format_market renders minimal fields without crashing."""
    result = {"success": True, "data": {"tool": "fetch_ticker", "symbol": "ETH/USDT"}}
    formatted = formatter.format_market(result)
    assert formatted["color"] == 0x3498DB


# ── format_analytical ───────────────────────────────────────────────────────────

def test_format_analytical_market_structure_scalar(formatter):
    """get_market_structure renders scalar summary with all fields."""
    result = {
        "success": True,
        "data": {
            "tool": "get_market_structure",
            "symbol": "BTC/USDT",
            "summary": {
                "trend": "bullish",
                "adx": 35.5,
                "rsi": 58.2,
                "support": 49000.0,
                "resistance": 52000.0,
            },
        },
    }
    formatted = formatter.format_analytical(result)
    assert formatted["color"] == 0x3498DB
    field_names = [f["name"] for f in formatted["fields"]]
    assert "Trend" in field_names
    assert "Adx" in field_names
    assert_non_mutated(formatter, result, formatted)


def test_format_analytical_list_rows_summarized(formatter):
    """List-row analytical tools render count + top 5 rows in one field."""
    result = {
        "success": True,
        "data": {
            "tool": "detect_order_blocks",
            "symbol": "BTC/USDT",
            "count": 20,
            "rows": [
                {"zone_type": "bullish_ob", "zone_high": 50000.0, "zone_low": 49000.0},
                {"zone_type": "bearish_ob", "zone_high": 51000.0, "zone_low": 50000.0},
                {"zone_type": "bullish_ob", "zone_high": 49500.0, "zone_low": 48500.0},
            ],
        },
    }
    formatted = formatter.format_analytical(result)
    assert formatted["color"] == 0x3498DB
    results_field = formatted["fields"][0]
    assert "Count: 20" in results_field["value"]
    assert "bullish_ob" in results_field["value"]
    assert_non_mutated(formatter, result, formatted)


def test_format_analytical_row_heavy_truncated(formatter):
    """Row-heavy payload is summarized with top 5 + count metadata."""
    rows = [
        {"zone_type": f"type_{i}", "zone_high": 50000.0 + i, "zone_low": 49000.0 + i}
        for i in range(20)
    ]
    result = {
        "success": True,
        "data": {
            "tool": "detect_order_blocks",
            "symbol": "BTC/USDT",
            "count": 20,
            "rows": rows,
        },
    }
    formatted = formatter.format_analytical(result)
    # 1 results field + 1 metadata field
    assert len(formatted["fields"]) == 2
    assert "Showing top 5 of 20 rows" in formatted["fields"][1]["value"]


def test_format_analytical_fields_never_exceed_25(formatter):
    """Embed fields must never exceed Discord's 25-field limit."""
    rows = [{"zone_type": f"type_{i}", "zone_high": 50000.0 + i, "zone_low": 49000.0 + i} for i in range(30)]
    result = {
        "success": True,
        "data": {
            "tool": "detect_order_blocks",
            "symbol": "BTC/USDT",
            "count": 30,
            "rows": rows,
        },
    }
    formatted = formatter.format_analytical(result)
    assert len(formatted["fields"]) <= 25


# ── format_indicator ────────────────────────────────────────────────────────────

def test_format_indicator_happy_path(formatter):
    """format_indicator renders symbol, period, timeframe, values count."""
    result = {
        "success": True,
        "data": {
            "tool": "indicator_rsi",
            "symbol": "BTC/USDT",
            "period": 14,
            "timeframe": "1h",
            "values": [30.5, 35.2, 40.1, 45.0, 50.2],
        },
    }
    formatted = formatter.format_indicator(result)
    assert formatted["color"] == 0x3498DB
    field_names = [f["name"] for f in formatted["fields"]]
    assert "Symbol" in field_names
    assert "Period" in field_names
    assert "Timeframe" in field_names
    assert "Values Count" in field_names
    assert_non_mutated(formatter, result, formatted)


# ── format_alert ────────────────────────────────────────────────────────────────

ALERT_FIXTURES = [
    ({"symbol": "BTC/USDT", "signal": "bullish", "message": "Breakout"}, 0x2ECC71),
    ({"symbol": "ETH/USDT", "signal": "bearish", "message": "Drop"}, 0xE74C3C),
    ({"symbol": "SOL/USDT", "signal": "neutral", "message": "Wait"}, 0xF1C40F),
    ({"symbol": "DOGE/USDT", "signal": "long", "message": "Long entry"}, 0x2ECC71),
    ({"symbol": "DOGE/USDT", "signal": "short", "message": "Short entry"}, 0xE74C3C),
]


@pytest.mark.parametrize("alert,expected_color", ALERT_FIXTURES)
def test_format_alert_signal_color(formatter, alert, expected_color):
    """Signal type determines embed color."""
    formatted = formatter.format_alert(alert)
    assert formatted["color"] == expected_color


def test_format_alert_with_confidence(formatter):
    """Confidence is formatted as percentage."""
    alert = {
        "symbol": "BTC/USDT",
        "signal": "bullish",
        "confidence": 0.85,
        "message": "High confidence",
    }
    formatted = formatter.format_alert(alert)
    field_names = [f["name"] for f in formatted["fields"]]
    conf_idx = field_names.index("Confidence")
    assert "85.00%" in formatted["fields"][conf_idx]["value"]


def test_format_alert_timestamp_passed_through(formatter):
    """Timestamp is passed to embed if provided."""
    alert = {
        "symbol": "BTC/USDT",
        "signal": "bullish",
        "timestamp": "2026-04-21T10:00:00Z",
    }
    formatted = formatter.format_alert(alert)
    assert "timestamp" in formatted


@pytest.mark.parametrize("timestamp", [None, "N/A", "not-a-timestamp", "2026-04-21"])
def test_format_alert_omits_missing_or_invalid_timestamp(formatter, timestamp):
    """Missing/invalid timestamps must be omitted from the embed."""
    alert = {
        "symbol": "BTC/USDT",
        "signal": "bullish",
        "timestamp": timestamp,
    }
    formatted = formatter.format_alert(alert)
    assert "timestamp" not in formatted


# ── format_error ───────────────────────────────────────────────────────────────

def test_format_error_happy_path(formatter):
    """format_error renders code and message in red embed."""
    result = {"success": False, "error": {"code": 404, "message": "Not found"}}
    formatted = formatter.format_error(result)
    assert formatted["color"] == 0xE74C3C
    assert formatted["title"] == "❌ Error"
    field_names = [f["name"] for f in formatted["fields"]]
    assert "Code" in field_names
    assert "Error" in field_names
    assert_non_mutated(formatter, result, formatted)


def test_format_error_with_diagnostics(formatter):
    """Diagnostic data is rendered as a compact Diagnostics field."""
    result = {
        "success": False,
        "error": {"code": 500, "message": "Internal error"},
        "data": {"stack": "file.py:42", "request_id": "abc123"},
    }
    formatted = formatter.format_error(result)
    field_names = [f["name"] for f in formatted["fields"]]
    assert "🔧 Diagnostics" in field_names


def test_format_error_missing_code_and_message(formatter):
    """Missing code/message renders as N/A without crashing."""
    result = {"success": False, "error": {}}
    formatted = formatter.format_error(result)
    assert formatted["color"] == 0xE74C3C


@pytest.mark.parametrize("bad_error", ["boom", [], None])
def test_format_error_malformed_error_routes_to_invalid_payload(formatter, bad_error):
    """Malformed error payloads should return an invalid-payload embed."""
    formatted = formatter.format_error({"success": False, "error": bad_error})
    assert formatted["color"] == 0xE74C3C
    assert formatted["title"] == "❌ Invalid Payload"


# ── format_generic ─────────────────────────────────────────────────────────────

def test_format_generic_happy_path(formatter):
    """format_generic renders data as indented JSON code block."""
    result = {"success": True, "data": {"tool": "unknown_tool", "foo": "bar", "num": 42}}
    formatted = formatter.format_generic(result)
    assert formatted["color"] == 0x3498DB
    raw_field = formatted["fields"][0]
    assert "```" in raw_field["value"]
    assert "foo" in raw_field["value"]
    assert "bar" in raw_field["value"]
    assert_non_mutated(formatter, result, formatted)


def test_format_generic_truncation(formatter):
    """Large JSON is truncated to _GENERIC_TRUNCATE_LIMIT chars."""
    original_limit = discord_module._GENERIC_TRUNCATE_LIMIT
    discord_module._GENERIC_TRUNCATE_LIMIT = 120
    result = {
        "success": True,
        "data": {
            "tool": "unknown_tool",
            "large_field": "x" * 500,
        },
    }
    try:
        formatted = formatter.format_generic(result)
    finally:
        discord_module._GENERIC_TRUNCATE_LIMIT = original_limit

    raw_field = formatted["fields"][0]
    assert "..." in raw_field["value"]
    assert "/tmp/" not in raw_field["value"]
    assert len(raw_field["value"]) <= 1024


def test_format_generic_file_write_on_oversized(tmp_path, monkeypatch):
    """Oversized payload after hard truncation is securely written to temp storage."""
    f = DiscordFormatter()
    monkeypatch.setattr(discord_module, "_TMP_DIR", str(tmp_path))
    original_limit = discord_module._GENERIC_TRUNCATE_LIMIT
    discord_module._GENERIC_TRUNCATE_LIMIT = 1100
    large_data = {"tool": "unknown", "arr": list(range(10000))}
    result = {"success": True, "data": large_data}
    try:
        formatted = f.format_generic(result)
    finally:
        discord_module._GENERIC_TRUNCATE_LIMIT = original_limit

    raw_field = formatted["fields"][0]
    value = raw_field["value"]
    assert str(tmp_path) not in value
    assert "local temp storage" in value.lower()
    assert "path intentionally omitted" in value.lower()

    written_files = list(tmp_path.iterdir())
    assert len(written_files) == 1
    payload_path = written_files[0]
    assert payload_path.exists()
    assert oct(payload_path.stat().st_mode & 0o777) == "0o600"
    payload_text = payload_path.read_text(encoding="utf-8")
    assert len(payload_text) <= 1100
    assert payload_text.endswith("...")
    assert "TODO" in value.upper() or "cloud" in value.lower()


def test_embed_content_escapes_mentions_and_caps_lengths(formatter):
    """Untrusted embed content is mention-safe and capped to Discord field limits."""
    long_symbol = "BTC" * 120
    result = {
        "success": True,
        "data": {
            "tool": "screener_scan",
            "results": [
                {
                    "symbol": f"@everyone {long_symbol}",
                    "exchange": "binance",
                    "score": 0.9,
                    "filters_matched": f"@here {'x' * 1500}",
                    "price": 50000.0,
                }
            ],
        },
    }

    formatted = formatter.format_screener(result)
    field = formatted["fields"][0]

    assert "@\u200beveryone" in field["name"]
    assert "@\u200bhere" in field["value"]
    assert len(field["name"]) <= 256
    assert len(field["value"]) <= 1024


# ── Number Formatting ──────────────────────────────────────────────────────────

def test_safe_value_non_finite(formatter):
    """_safe_value returns N/A for None and non-finite floats."""
    assert formatter._safe_value(None) == "N/A"
    assert formatter._safe_value(float("nan")) == "N/A"
    assert formatter._safe_value(float("inf")) == "N/A"
    assert formatter._safe_value(float("-inf")) == "N/A"
    assert formatter._safe_value(42) == "42"
    assert formatter._safe_value("hello") == "hello"


def test_fmt_price(formatter):
    """_fmt_price formats to 4 decimal places."""
    assert formatter._fmt_price(123.12345678) == "123.1235"
    assert formatter._fmt_price(123.0) == "123.0000"
    assert formatter._fmt_price("N/A") == "N/A"
    assert formatter._fmt_price("invalid") == "invalid"


def test_fmt_percent(formatter):
    """_fmt_percent formats to 2 decimal places with % sign."""
    assert formatter._fmt_percent(15.5) == "15.50%"
    assert formatter._fmt_percent(-8.25) == "-8.25%"
    assert formatter._fmt_percent("N/A") == "N/A"


def test_fmt_number(formatter):
    """_fmt_number formats to specified decimal places."""
    assert formatter._fmt_number(3.14159, decimals=2) == "3.14"
    assert formatter._fmt_number(3.14159, decimals=3) == "3.142"
    assert formatter._fmt_number("N/A") == "N/A"


# ── Score Normalization ────────────────────────────────────────────────────────

def test_normalize_score_0_1_scale(formatter):
    """Scores <= 1.0 are multiplied by 100."""
    assert formatter._normalize_score(0.75) == 75.0
    assert formatter._normalize_score(0.4) == 40.0
    assert formatter._normalize_score(0.35) == 35.0


def test_normalize_score_0_100_scale(formatter):
    """Scores > 1.0 are treated as 0-100 and clamped."""
    assert formatter._normalize_score(75.0) == 75.0
    assert formatter._normalize_score(150.0) == 100.0  # clamped
    assert formatter._normalize_score(-10.0) == 0.0  # clamped


def test_normalize_score_none_and_invalid(formatter):
    """None and non-numeric return None."""
    assert formatter._normalize_score(None) is None
    assert formatter._normalize_score("invalid") is None


# ── Score Emoji ────────────────────────────────────────────────────────────────

def test_score_emoji_thresholds(formatter):
    """Emoji thresholds: >=70 🔒, 40-69 ⚠️, <40 🔴."""
    assert formatter._score_emoji(80.0) == "🔒"
    assert formatter._score_emoji(70.0) == "🔒"
    assert formatter._score_emoji(55.0) == "⚠️"
    assert formatter._score_emoji(40.0) == "⚠️"
    assert formatter._score_emoji(30.0) == "🔴"
    assert formatter._score_emoji(None) == ""


# ── Mood Emoji ─────────────────────────────────────────────────────────────────

def test_mood_emoji(formatter):
    """Mood emoji maps bullish/positive/buy -> 😀, bearish/negative/sell -> 😠, else -> 😐."""
    assert formatter._mood_emoji("bullish") == "😀"
    assert formatter._mood_emoji("positive") == "😀"
    assert formatter._mood_emoji("buy") == "😀"
    assert formatter._mood_emoji("bearish") == "😠"
    assert formatter._mood_emoji("negative") == "😠"
    assert formatter._mood_emoji("sell") == "😠"
    assert formatter._mood_emoji("neutral") == "😐"
    assert formatter._mood_emoji(None) == "😐"


# ── Field Capping ─────────────────────────────────────────────────────────────

def test_cap_fields_hard_limit(formatter):
    """_cap_fields must hard-cap at DISCORD_TOTAL_FIELDS_LIMIT (25)."""
    fields = [{"name": f"f{i}", "value": f"v{i}", "inline": True} for i in range(30)]
    capped = formatter._cap_fields(fields)
    assert len(capped) == 25


# ── Non-Mutation ──────────────────────────────────────────────────────────────

def test_format_backtest_does_not_mutate_input(formatter):
    """format_backtest must not mutate the input result dict."""
    original = {
        "success": True,
        "data": {
            "tool": "backtest_pdh_session",
            "strategy_id": "pdh_session",
            "symbol": "BTC/USDT",
            "trade_count": 42,
            "initial_capital": 100000.0,
            "final_equity": 115000.0,
            "metrics": {
                "total_return": 15.0,
                "sharpe_ratio": 1.8,
                "max_drawdown": -8.5,
                "win_rate": 55.0,
                "profit_factor": 1.6,
                "avg_win": 500.0,
                "avg_loss": -300.0,
            },
        },
    }
    original_copy = deepcopy(original)
    formatter.format_backtest(original)
    assert original == original_copy


def test_format_generic_does_not_mutate_input(formatter):
    """format_generic must not mutate the input result dict."""
    original = {"success": True, "data": {"tool": "unknown", "key": "value"}}
    original_copy = deepcopy(original)
    formatter.format_generic(original)
    assert original == original_copy


def test_format_does_not_mutate_input_overall(formatter):
    """format() must not mutate input across all paths."""
    test_cases = [
        {"success": True, "data": {"tool": "backtest_rsi", "strategy_id": "rsi", "symbol": "ETH", "trade_count": 10, "initial_capital": 10000.0, "final_equity": 12000.0, "metrics": {"total_return": 20.0, "sharpe_ratio": 1.5, "max_drawdown": -5.0, "win_rate": 60.0, "profit_factor": 1.8, "avg_win": 100.0, "avg_loss": -80.0}}},
        {"success": False, "error": {"code": 500, "message": "fail"}},
        {"success": True, "data": {}},
    ]
    for original in test_cases:
        original_copy = deepcopy(original)
        formatter.format(original)
        assert original == original_copy


# ── Edge Cases ────────────────────────────────────────────────────────────────

def test_format_backtest_no_metrics_key(formatter):
    """Missing metrics key must not crash."""
    result = {
        "success": True,
        "data": {
            "tool": "backtest_pdh_session",
            "strategy_id": "pdh_session",
            "symbol": "BTC/USDT",
        },
    }
    formatted = formatter.format_backtest(result)
    assert formatted is not None


def test_format_screener_empty_results(formatter):
    """Empty results renders explicit 'No rows' metadata."""
    result = {
        "success": True,
        "data": {
            "tool": "screener_scan",
            "results": [],
        },
    }
    formatted = formatter.format_screener(result)
    # Should only have metadata field
    assert len(formatted["fields"]) == 1
    assert "Showing top 5 of 0 results" in formatted["fields"][0]["value"]


def test_format_compare_ranking_metric_included(formatter):
    """ranking_metric is included in metadata when present."""
    result = {
        "success": True,
        "data": {
            "tool": "compare_strategies",
            "best_strategy_id": "strategy_1",
            "ranking_metric": "sharpe_ratio",
            "results": [
                {"strategy_id": "s1", "rank": 1, "total_return": 10.0, "sharpe_ratio": 2.0, "trade_count": 50, "open_position": False},
            ],
        },
    }
    formatted = formatter.format_compare(result)
    meta_field = formatted["fields"][-1]
    assert "sharpe_ratio" in meta_field["value"]


def test_empty_error_envelope(formatter):
    """Empty error envelope renders without crashing."""
    result = {"success": False, "error": {}}
    formatted = formatter.format_error(result)
    assert formatted["color"] == 0xE74C3C
    assert "N/A" in [f["value"] for f in formatted["fields"]]


def test_sentiment_diagnostics_field(formatter):
    """Diagnostics from sentiment are rendered."""
    result = {
        "success": True,
        "data": {
            "tool": "get_combined_sentiment_dashboard",
            "sentiment_polarity": "neutral",
            "sentiment_index": 0.5,
            "combination_mode": "weighted",
            "diagnostics": {"reddit_score": 0.6, "rss_score": 0.4},
        },
    }
    formatted = formatter.format_sentiment(result)
    field_names = [f["name"] for f in formatted["fields"]]
    assert "🔧 Diagnostics" in field_names
