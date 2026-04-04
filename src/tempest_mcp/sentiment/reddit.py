"""Reddit sentiment analyzer (Phase 2+ stub)."""
from dataclasses import dataclass
from typing import Any

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class RedditSentimentAnalyzer:
    subreddits: tuple[str, ...] = ("CryptoCurrency", "Bitcoin")
    def analyze(self, symbol: str) -> dict[str, Any]:
        logger.warning("Reddit sentiment not implemented (Phase 2+)")
        return {"symbol": symbol, "status": "not_implemented"}
