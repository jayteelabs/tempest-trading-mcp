"""mplfinance chart generator (Phase 2+ stub)."""
from dataclasses import dataclass
from typing import Any
from tempest_mcp.logging_config import get_logger
logger = get_logger(__name__)

@dataclass
class ChartGenerator:
    def generate_candlestick(self, symbol: str, klines: list[dict[str, Any]], indicators: list[str] | None = None) -> bytes | None:
        logger.warning("Chart generation not implemented (Phase 2+)")
        return None
