"""MCP tool handler for combined sentiment dashboard — ENG-41."""

from __future__ import annotations

from typing import Any

from tempest_mcp.config import ErrorCodes
from tempest_mcp.logging_config import get_logger
from tempest_mcp.sentiment.combined_sentiment import CombinedSentimentDashboard

logger = get_logger(__name__)

# Singleton dashboard instance (lazy initialization)
_dashboard: CombinedSentimentDashboard | None = None


def _get_dashboard() -> CombinedSentimentDashboard:
    """Get or create the singleton CombinedSentimentDashboard instance."""
    global _dashboard
    if _dashboard is None:
        _dashboard = CombinedSentimentDashboard()
    return _dashboard


async def get_combined_sentiment_dashboard(
    symbol: str, price_bias: str, **kwargs: Any
) -> dict[str, Any]:
    """MCP tool handler for get_combined_sentiment_dashboard.

    Calls the CombinedSentimentDashboard and maps the result to the
    public MCP envelope. The dashboard itself is synchronous, so we
    call it directly (not await).

    Parameters
    ----------
    symbol : str
        Trading symbol (e.g. "BTCUSDT", "ETH/USD")
    price_bias : str
        Caller-supplied directional bias: "bullish" | "bearish" | "neutral"

    Returns
    -------
    dict
        Public MCP envelope conforming to ENG-41 schema:
        {
            "success": true,
            "data": {
                "tool": "get_combined_sentiment_dashboard",
                "symbol": str,
                "fetched_at": str,
                "price_bias": str,
                "sentiment_index": float | null,
                "sentiment_polarity": str,
                "combination_mode": str,
                "diagnostics": {...},
                "cross_signal_flags": list[str]
            }
        }

        On failure (neither source usable):
        {
            "success": false,
            "error": { "code": 3000, "message": "No usable sentiment sources" },
            "data": { ... diagnostics payload ... }
        }
    """
    try:
        dashboard = _get_dashboard()
        result = dashboard.analyze(symbol=symbol, price_bias=price_bias)

        # Determine success based on combination_mode
        if result["combination_mode"] == "unavailable":
            return {
                "success": False,
                "error": {
                    "code": ErrorCodes.DATA_SOURCE_ERROR,
                    "message": "No usable sentiment sources",
                },
                "data": {
                    "tool": "get_combined_sentiment_dashboard",
                    **result,
                },
            }

        return {
            "success": True,
            "data": {
                "tool": "get_combined_sentiment_dashboard",
                **result,
            },
        }

    except Exception as exc:
        logger.error(
            "sentiment_dashboard_error",
            symbol=symbol,
            price_bias=price_bias,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return {
            "success": False,
            "error": {
                "code": ErrorCodes.INTERNAL_ERROR,
                "message": "An internal error occurred",
            },
            "data": {
                "tool": "get_combined_sentiment_dashboard",
                "symbol": symbol,
                "price_bias": price_bias,
                "sentiment_index": None,
                "sentiment_polarity": "neutral",
                "combination_mode": "unavailable",
                "diagnostics": {
                    "sources": {
                        "reddit": {
                            "status": "error",
                            "usable": False,
                            "avg_sentiment": None,
                            "sample_count": 0,
                        },
                        "rss": {
                            "status": "error",
                            "usable": False,
                            "avg_sentiment": None,
                            "sample_count": 0,
                        },
                    },
                    "weights": {
                        "configured": {"reddit": 0.4, "rss": 0.6},
                        "applied": {"reddit": 0.0, "rss": 0.0},
                    },
                    "fallback_reason": "internal error during analysis",
                },
                "cross_signal_flags": [],
            },
        }
