"""RSS sentiment analyzer for financial news feeds."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

from tempest_mcp.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Fixed RSS feed sources
# ---------------------------------------------------------------------------
_DEFAULT_RSS_FEEDS: tuple[str, ...] = (
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cryptoslate.com/feed/",
    "https://www.tradingview.com/ideas/feed/",
)

# ---------------------------------------------------------------------------
# Symbol aliases for common tokens
# ---------------------------------------------------------------------------
_QUOTE_SUFFIXES: tuple[str, ...] = (
    "USDT",
    "USDC",
    "FDUSD",
    "TUSD",
    "USD",
    "BTC",
    "ETH",
    "BNB",
    "EUR",
)

_SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin",),
    "ETH": ("ethereum",),
    "DOGE": ("dogecoin",),
    "SOL": ("solana",),
    "ADA": ("cardano",),
    "XRP": ("ripple",),
    "LTC": ("litecoin",),
    "BNB": ("binance coin",),
}

# ---------------------------------------------------------------------------
# Keyword boost tables (mirrors reddit.py pattern)
# ---------------------------------------------------------------------------
_ALL_BOOST_TERMS: list[tuple[str, float]] = [
    ("to the moon", 0.15),
    ("alts are up", 0.15),
    ("bullish", 0.15),
    ("moon", 0.15),
    ("pump", 0.15),
    ("rugpull", -0.15),
    ("bearish", -0.15),
    ("dump", -0.15),
    ("crash", -0.15),
    ("rug", -0.15),
    ("bear", -0.15),
]

# Deduplicated boost map: term -> boost (first occurrence wins for determinism)
_BOOST_MAP: dict[str, float] = dict(_ALL_BOOST_TERMS)
# Ordered list for deterministic iteration
_BOOST_TERMS: list[tuple[str, float]] = list(_BOOST_MAP.items())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_base_token(symbol: str) -> str:
    """Extract a plausible base token from a trading symbol.

    Handles common formats:
      - USDT pairs  (BTCUSDT -> BTC)
      - USD pairs   (BTCUSD  -> BTC)
      - BTC pairs   (ETHBTC  -> ETH)
      - Slash pairs (BTC/USD -> BTC)
      - Dash pairs  (BTC-USDT -> BTC)
      - Plain symbols (BTC -> BTC, DOGE -> DOGE)

    Returns the input unchanged if no recognized quote suffix is found.
    """
    s = symbol.upper()
    if "/" in s:
        return s.split("/")[0]
    if "-" in s:
        return s.split("-")[0]
    for sep in _QUOTE_SUFFIXES:
        if s.endswith(sep) and len(s) > len(sep):
            return s[: -len(sep)]
    return s


def _coerce_text(value: Any) -> str:
    """Return a deterministic string for RSS text fields."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _validate_symbol_input(symbol: Any) -> tuple[str | None, str]:
    """Validate and normalize a symbol input for analysis."""
    if not isinstance(symbol, str):
        return None, _coerce_text(symbol) or "<invalid>"

    normalized = symbol.strip()
    if not normalized:
        return None, "<empty>"

    return normalized, normalized


def _symbol_matches_text(symbol: str, base_token: str, text: str) -> bool:
    """Return True if text mentions the raw symbol, base token, or a known alias.

    Uses word-boundary matching to prevent false positives.
    """
    lower_text = _coerce_text(text).lower()
    symbol_l = symbol.lower()
    base_l = base_token.lower()

    def matches_term(term: str) -> bool:
        if not term:
            return False
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        return pattern.search(lower_text) is not None

    if matches_term(symbol_l) or matches_term(base_l):
        return True

    for alias in _SYMBOL_ALIASES.get(base_token.upper(), ()):
        if matches_term(alias):
            return True

    return False


def _compute_keyword_boost(text: str) -> float:
    """Compute deterministic keyword modifier for RSS entry text.

    Each known term is applied at most once. Boosts are summed and clamped
    to [-1.0, 1.0]. Matching is case-insensitive and overlap-aware.
    """
    lower_text = _coerce_text(text).lower()
    occupied_spans: list[tuple[int, int]] = []
    boost = 0.0

    def overlaps(span: tuple[int, int]) -> bool:
        start, end = span
        for other_start, other_end in occupied_spans:
            if start < other_end and end > other_start:
                return True
        return False

    for term, value in _BOOST_TERMS:
        pattern = (
            re.compile(rf"\b{re.escape(term)}\b") if term.isalpha() else re.compile(re.escape(term))
        )
        match = pattern.search(lower_text)
        if match is None:
            continue
        span = match.span()
        if overlaps(span):
            continue
        occupied_spans.append(span)
        boost += value

    return max(-1.0, min(1.0, boost))


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# RSSSentimentAnalyzer
# ---------------------------------------------------------------------------


@dataclass
class RSSSentimentAnalyzer:
    """Deterministic RSS-based financial news sentiment analyzer.

    Fetches from fixed free RSS sources (CoinDesk, CryptoSlate, TradingView Ideas),
    filters entries by symbol/base token and aliases, scores with VADER + keyword
    boost, and returns a structured result envelope.
    """

    feeds: tuple[str, ...] = _DEFAULT_RSS_FEEDS
    _http_client: httpx.Client = field(default=None, init=False, repr=False)
    _vader: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # HTTP client with timeout and user-agent
        self._http_client = httpx.Client(
            timeout=15.0,
            headers={"User-Agent": "tempest-tradingview-mcp/1.0"},
        )
        # VADER scorer instantiated once per analyzer instance
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._vader = SentimentIntensityAnalyzer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying HTTP client resources."""
        if self._http_client is None:
            return
        self._http_client.close()
        self._http_client = None

    def __enter__(self) -> RSSSentimentAnalyzer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def analyze(self, symbol: str) -> dict[str, Any]:
        """Analyze RSS feed sentiment for a trading symbol.

        Fetches from configured RSS feeds, filters entries whose title or
        description mentions the symbol or its base token, scores each
        title with VADER + keyword boost, and returns a structured result
        envelope. Individual feed failures are logged and skipped so later
        feeds can still contribute results; an error envelope is returned
        only if every configured feed fails.

        Returns
        -------
        dict
            A dict conforming to the schema defined in ENG-40:
            {
                "symbol": str,
                "fetched_at": str,          # ISO-8601 UTC
                "feeds": list[str],
                "items": list[dict],        # empty when status != "ok"
                "summary": dict,
                "status": "ok" | "no_results" | "error"
            }
        """
        fetched_at = datetime.now(timezone.utc).isoformat()
        validated_symbol, result_symbol = _validate_symbol_input(symbol)
        if validated_symbol is None:
            return self._error_result(
                result_symbol,
                fetched_at,
                ValueError("symbol must be a non-empty string"),
            )

        normalized_symbol = validated_symbol
        base_token = _extract_base_token(normalized_symbol)
        all_items: list[dict[str, Any]] = []
        failed_feeds: list[Exception] = []
        successful_feeds = 0

        for feed_url in self.feeds:
            try:
                entries = self._fetch_and_parse_feed(feed_url)
            except httpx.HTTPError as exc:
                logger.warning(
                    "rss_fetch_failed",
                    feed=feed_url,
                    symbol=normalized_symbol,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                failed_feeds.append(exc)
                continue
            except ET.ParseError as exc:
                logger.warning(
                    "rss_parse_failed",
                    feed=feed_url,
                    symbol=normalized_symbol,
                    error=str(exc),
                )
                failed_feeds.append(exc)
                continue

            successful_feeds += 1

            for entry in entries:
                title = _coerce_text(entry.get("title"))
                description = _coerce_text(entry.get("description"))
                # Match against title and description
                if not _symbol_matches_text(normalized_symbol, base_token, title) and not _symbol_matches_text(
                    normalized_symbol, base_token, description
                ):
                    continue
                item_record = self._build_item_record(feed_url, entry, title)
                all_items.append(item_record)

        if failed_feeds and successful_feeds == 0:
            return self._error_result(normalized_symbol, fetched_at, failed_feeds[0])

        if not all_items:
            return self._no_results_result(normalized_symbol, fetched_at)

        summary = self._summarize_items(all_items)
        return {
            "symbol": normalized_symbol,
            "fetched_at": fetched_at,
            "feeds": list(self.feeds),
            "items": all_items,
            "summary": summary,
            "status": "ok",
        }

    # ------------------------------------------------------------------
    # Private seams (mockable in tests)
    # ------------------------------------------------------------------

    def _fetch_feed_xml(self, feed_url: str) -> str:
        """Fetch raw XML text from an RSS feed."""
        response = self._http_client.get(feed_url)
        response.raise_for_status()
        return response.text

    def _parse_feed_entries(self, xml_text: str) -> list[dict[str, Any]]:
        """Parse RSS XML text into a list of entry dicts.

        Handles both RSS 2.0 and Atom formats. Normalizes fields to:
        - title (str)
        - link (str)
        - description (str)
        - published (str, may be empty)
        """
        entries: list[dict[str, Any]] = []
        root = ET.fromstring(xml_text)

        # Determine if this is an Atom feed by checking for namespace
        ATOM_NS = "http://www.w3.org/2005/Atom"
        is_atom = root.tag == f"{{{ATOM_NS}}}feed"

        if is_atom:
            # Atom: <feed> -> <entry>
            entries_container = root.findall(f"{{{ATOM_NS}}}entry")
        else:
            # RSS 2.0: <rss> -> <channel> -> <item>
            channel = root.find("channel")
            entries_container = channel.findall("item") if channel is not None else []

        for entry in entries_container:
            title = ""
            link = ""
            description = ""
            published = ""

            if is_atom:
                # Atom namespace-aware lookups
                ns = {"atom": ATOM_NS}
                title_elem = entry.find("atom:title", ns)
                link_elem = entry.find("atom:link", ns)
                content_elem = entry.find("atom:content", ns)
                pub_elem = entry.find("atom:published", ns)
                if pub_elem is None:
                    pub_elem = entry.find("atom:updated", ns)
            else:
                # RSS 2.0 - no namespace
                title_elem = entry.find("title")
                link_elem = entry.find("link")
                content_elem = entry.find("content")
                if content_elem is None:
                    content_elem = entry.find("description")
                pub_elem = entry.find("pubDate")

            # Title
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

            # Link: RSS has text content, Atom has href attribute
            if link_elem is not None:
                if is_atom:
                    link = link_elem.get("href", "").strip()
                else:
                    link = (link_elem.text or "").strip()

            # Description: <content> (Atom) or <description> (RSS)
            if content_elem is not None and content_elem.text:
                description = content_elem.text.strip()

            # Published date
            if pub_elem is not None and pub_elem.text:
                published = pub_elem.text.strip()

            entries.append(
                {
                    "title": title,
                    "link": link,
                    "description": description,
                    "published": published,
                }
            )

        return entries

    def _fetch_and_parse_feed(self, feed_url: str) -> list[dict[str, Any]]:
        """Fetch and parse an RSS feed, returning entry dicts."""
        xml_text = self._fetch_feed_xml(feed_url)
        return self._parse_feed_entries(xml_text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_item_record(
        self, feed_url: str, entry: dict[str, Any], title: str
    ) -> dict[str, Any]:
        """Normalize an RSS entry into the required schema shape."""
        vader_scores = self._vader.polarity_scores(title)
        keyword_boost = _compute_keyword_boost(title)
        final_score = _clamp(vader_scores["compound"] + keyword_boost)

        return {
            "feed": feed_url,
            "title": title,
            "link": entry.get("link", ""),
            "published_at": entry.get("published", ""),
            "sentiment": {
                "vader_pos": round(vader_scores["pos"], 4),
                "vader_neu": round(vader_scores["neu"], 4),
                "vader_neg": round(vader_scores["neg"], 4),
                "vader_compound": round(vader_scores["compound"], 4),
                "keyword_boost": round(keyword_boost, 4),
                "final_score": round(final_score, 4),
            },
        }

    def _summarize_items(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        """Compute aggregate statistics from scored items."""
        total = len(items)
        final_scores = [item["sentiment"]["final_score"] for item in items]
        avg = round(sum(final_scores) / total, 4) if total else 0.0
        positive = sum(1 for s in final_scores if s > 0.05)
        negative = sum(1 for s in final_scores if s < -0.05)
        neutral = total - positive - negative
        return {
            "total_items": total,
            "avg_sentiment": avg,
            "positive_count": positive,
            "negative_count": negative,
            "neutral_count": neutral,
        }

    def _no_results_result(self, symbol: str, fetched_at: str) -> dict[str, Any]:
        """Deterministic envelope returned when no matching entries were found."""
        return {
            "symbol": symbol,
            "fetched_at": fetched_at,
            "feeds": list(self.feeds),
            "items": [],
            "summary": {
                "total_items": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "status": "no_results",
        }

    def _error_result(
        self, symbol: str, fetched_at: str, reason: Exception | str
    ) -> dict[str, Any]:
        """Deterministic envelope returned on any fetch or parse failure."""
        reason_text = str(reason)
        log_kwargs = {"symbol": symbol, "reason": reason_text}
        if isinstance(reason, Exception):
            log_kwargs["error_type"] = type(reason).__name__
        logger.warning("rss_analyze_error", **log_kwargs)
        return {
            "symbol": symbol,
            "fetched_at": fetched_at,
            "feeds": list(self.feeds),
            "items": [],
            "summary": {
                "total_items": 0,
                "avg_sentiment": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            },
            "status": "error",
        }
