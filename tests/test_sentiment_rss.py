"""Unit tests for RSSSentimentAnalyzer."""

import sys
import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from tempest_mcp.sentiment.rss import (
    RSSSentimentAnalyzer,
    _clamp,
    _compute_keyword_boost,
    _extract_base_token,
    _symbol_matches_text,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_rss_entry(
    title: str = "BTC update",
    link: str = "https://example.com/btc",
    description: str = "",
    published: str = "Mon, 01 Jan 2024 12:00:00 GMT",
) -> dict:
    return {
        "title": title,
        "link": link,
        "description": description,
        "published": published,
    }


RSS_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Crypto News</title>
    <item>
      <title>Bitcoin surges to new highs</title>
      <link>https://example.com/btc-surges</link>
      <description>Bitcoin BTC is up 5% today</description>
      <pubDate>Mon, 01 Jan 2024 12:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Ethereum update</title>
      <link>https://example.com/eth-update</link>
      <description>ETH price analysis</description>
      <pubDate>Tue, 02 Jan 2024 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Crypto Feed</title>
  <entry>
    <title>Bitcoin halving approaching</title>
    <link href="https://example.com/btc-halving"/>
    <content>BTC mining reward will be cut in half</content>
    <published>2024-01-01T12:00:00Z</published>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestExtractBaseToken:
    def test_usdt_pair(self):
        assert _extract_base_token("BTCUSDT") == "BTC"
        assert _extract_base_token("ETHUSDT") == "ETH"

    def test_usd_pair(self):
        assert _extract_base_token("BTCUSD") == "BTC"
        assert _extract_base_token("ETHUSD") == "ETH"

    def test_extended_quote_pairs(self):
        assert _extract_base_token("SOLUSDC") == "SOL"
        assert _extract_base_token("XRPFDUSD") == "XRP"
        assert _extract_base_token("ADATUSD") == "ADA"

    def test_btc_pair(self):
        assert _extract_base_token("ETHBTC") == "ETH"

    def test_slash_pair(self):
        assert _extract_base_token("BTC/USD") == "BTC"

    def test_dash_pair(self):
        assert _extract_base_token("BTC-USDT") == "BTC"

    def test_plain_symbol(self):
        assert _extract_base_token("BTC") == "BTC"
        assert _extract_base_token("DOGE") == "DOGE"

    def test_lowercase_normalized(self):
        assert _extract_base_token("btcusdt") == "BTC"


class TestSymbolMatchesText:
    def test_raw_symbol_match(self):
        assert _symbol_matches_text("BTC", "BTC", "BTC is pumping!")

    def test_base_token_match(self):
        assert _symbol_matches_text("BTCUSDT", "BTC", "Bitcoin BTC is up today")

    def test_alias_match_bitcoin(self):
        assert _symbol_matches_text("BTC", "BTC", "Bitcoin price analysis")

    def test_alias_match_ethereum(self):
        assert _symbol_matches_text("ETH", "ETH", "Ethereum network upgrade")

    def test_no_match(self):
        assert not _symbol_matches_text("BTC", "BTC", "ETH is mooning")

    def test_case_insensitive(self):
        assert _symbol_matches_text("btc", "BTC", "btc to the moon")

    def test_word_boundaries_prevent_false_positive(self):
        assert not _symbol_matches_text("ETH", "ETH", "the market is choppy today")

    def test_doge_alias_match(self):
        assert _symbol_matches_text("DOGE", "DOGE", "Dogecoin community grows")

    def test_solana_alias_match(self):
        assert _symbol_matches_text("SOL", "SOL", "Solana ecosystem expands")

    def test_cardano_alias_match(self):
        assert _symbol_matches_text("ADA", "ADA", "Cardano roadmap update")

    def test_ripple_alias_match(self):
        assert _symbol_matches_text("XRP", "XRP", "Ripple sees renewed attention")

    def test_description_matching(self):
        assert _symbol_matches_text("BTC", "BTC", "Bitcoin is moving")

    def test_empty_terms_do_not_match_everything(self):
        assert not _symbol_matches_text("", "", "BTC is pumping!")


class TestClamp:
    def test_within_bounds(self):
        assert _clamp(0.5) == 0.5
        assert _clamp(-0.5) == -0.5

    def test_above_upper(self):
        assert _clamp(2.0) == 1.0

    def test_below_lower(self):
        assert _clamp(-2.0) == -1.0

    def test_custom_bounds(self):
        assert _clamp(5.0, lo=0.0, hi=1.0) == 1.0
        assert _clamp(-5.0, lo=0.0, hi=1.0) == 0.0


# ---------------------------------------------------------------------------
# Keyword boost tests
# ---------------------------------------------------------------------------


class TestKeywordBoost:
    def test_bullish_boost_15(self):
        assert _compute_keyword_boost("BTC looks really bullish") > 0

    def test_bearish_boost_15(self):
        assert _compute_keyword_boost("Market looks bearish") < 0

    def test_moon_boost(self):
        boost = _compute_keyword_boost("DOGE to the moon!")
        assert boost > 0

    def test_dump_boost(self):
        boost = _compute_keyword_boost("Coin dump incoming")
        assert boost < 0

    def test_combined_boosts(self):
        boost = _compute_keyword_boost("BTC is bullish and going to the moon!")
        assert boost == pytest.approx(0.30, rel=1e-3)

    def test_combined_bearish(self):
        boost = _compute_keyword_boost("Bearish outlook, dump incoming")
        assert boost == pytest.approx(-0.30, rel=1e-3)

    def test_boost_capped_at_1(self):
        boost = _compute_keyword_boost("BTC is bullish, moon, pump, to the moon, alts are up!")
        assert boost <= 1.0

    def test_boost_capped_at_minus_1(self):
        boost = _compute_keyword_boost("Bearish market, dump, crash, rug, rugpull, bear!")
        assert boost >= -1.0

    def test_term_applied_once_per_post(self):
        boost = _compute_keyword_boost("BTC is bullish and super bullish!")
        assert boost == pytest.approx(0.15, rel=1e-3)

    def test_no_boost(self):
        boost = _compute_keyword_boost("Bitcoin price update today")
        assert boost == 0.0

    def test_case_insensitive(self):
        assert _compute_keyword_boost("BITCOIN IS BULLISH") == _compute_keyword_boost(
            "bitcoin is bullish"
        )


# ---------------------------------------------------------------------------
# RSSSentimentAnalyzer tests
# ---------------------------------------------------------------------------


class TestRSSSentimentAnalyzerHappyPath:
    """Tests for the happy-path scenario with matching entries."""

    def test_returns_ok_status_with_matching_entries(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [
            make_rss_entry("BTC is going to the moon!"),
            make_rss_entry("Bitcoin dump incoming"),
        ]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert result["symbol"] == "BTC"
        assert len(result["items"]) == 2
        assert result["feeds"] == ["https://example.com/rss"]
        assert "fetched_at" in result

    def test_schema_fields_present_on_every_item(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC update for holders")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        item = result["items"][0]
        required_keys = {
            "feed",
            "title",
            "link",
            "published_at",
            "sentiment",
        }
        assert required_keys <= item.keys()
        sentiment_keys = {
            "vader_pos",
            "vader_neu",
            "vader_neg",
            "vader_compound",
            "keyword_boost",
            "final_score",
        }
        assert sentiment_keys <= item["sentiment"].keys()

    def test_base_token_matching(self):
        """Symbol like BTCUSDT should match entries mentioning BTC."""
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC holding strong at support")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTCUSDT")

        assert result["status"] == "ok"
        assert len(result["items"]) == 1

    def test_description_matching_included(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [
            make_rss_entry("News update", description="Bitcoin price analysis"),
        ]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert len(result["items"]) == 1

    def test_summary_counts(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [
            make_rss_entry("BTC to the moon!"),  # positive
            make_rss_entry("BTC dump incoming"),  # negative
            make_rss_entry("Bitcoin price analysis"),  # neutral
        ]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        assert result["summary"]["total_items"] == 3
        assert (
            result["summary"]["positive_count"]
            + result["summary"]["negative_count"]
            + result["summary"]["neutral_count"]
            == 3
        )

    def test_fetched_at_is_iso_format(self):
        import re

        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC update")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        iso_pat = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pat, result["fetched_at"])

    def test_items_order_preserved(self):
        """Items should be in feed iteration order, then entry order in feed."""
        analyzer = RSSSentimentAnalyzer(feeds=("https://feed1.com/rss", "https://feed2.com/rss"))

        def fake_fetch(feed_url):
            if feed_url == "https://feed1.com/rss":
                return [
                    make_rss_entry("BTC post A"),
                    make_rss_entry("BTC post B"),
                ]
            return [make_rss_entry("BTC post C")]

        with patch.object(analyzer, "_fetch_and_parse_feed", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["items"][0]["title"] == "BTC post A"
        assert result["items"][1]["title"] == "BTC post B"
        assert result["items"][2]["title"] == "BTC post C"


class TestRSSSentimentAnalyzerNoResults:
    """Tests for the no-results envelope."""

    def test_no_matching_entries_returns_no_results(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [
            make_rss_entry("ETH update"),
            make_rss_entry("Solana news"),
        ]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        assert result["status"] == "no_results"
        assert result["items"] == []
        assert result["summary"]["total_items"] == 0
        assert result["summary"]["avg_sentiment"] == 0.0
        assert result["summary"]["positive_count"] == 0
        assert result["summary"]["negative_count"] == 0
        assert result["summary"]["neutral_count"] == 0

    def test_empty_feed_response_returns_no_results(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=[]):
            result = analyzer.analyze("BTC")

        assert result["status"] == "no_results"
        assert result["items"] == []


class TestRSSSentimentAnalyzerError:
    """Tests for the error envelope on provider failure."""

    def test_http_error_returns_error(self):
        import httpx

        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))

        with patch.object(
            analyzer,
            "_fetch_and_parse_feed",
            side_effect=httpx.HTTPStatusError(
                "429",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            ),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["items"] == []
        assert result["summary"]["total_items"] == 0

    def test_network_timeout_returns_error(self):
        import httpx

        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))

        with patch.object(
            analyzer,
            "_fetch_and_parse_feed",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["items"] == []

    def test_error_first_feed_uses_remaining_feeds(self):
        """A failed feed should not block later feeds from contributing results."""
        import httpx

        analyzer = RSSSentimentAnalyzer(feeds=("https://feed1.com/rss", "https://feed2.com/rss"))

        def fake_fetch(feed_url):
            if feed_url == "https://feed1.com/rss":
                raise httpx.HTTPStatusError(
                    "429",
                    request=MagicMock(),
                    response=MagicMock(status_code=429),
                )
            return [make_rss_entry("BTC post")]

        with patch.object(analyzer, "_fetch_and_parse_feed", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert [item["title"] for item in result["items"]] == ["BTC post"]

    def test_all_feeds_failing_returns_error(self):
        import httpx

        analyzer = RSSSentimentAnalyzer(feeds=("https://feed1.com/rss", "https://feed2.com/rss"))

        def fake_fetch(_feed_url):
            raise httpx.TimeoutException("timeout")

        with patch.object(analyzer, "_fetch_and_parse_feed", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["items"] == []

    def test_partial_failure_without_matches_returns_no_results(self):
        import httpx

        analyzer = RSSSentimentAnalyzer(feeds=("https://feed1.com/rss", "https://feed2.com/rss"))

        def fake_fetch(feed_url):
            if feed_url == "https://feed1.com/rss":
                raise httpx.TimeoutException("timeout")
            return [make_rss_entry("ETH post")]

        with patch.object(analyzer, "_fetch_and_parse_feed", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "no_results"
        assert result["items"] == []

    def test_later_feed_failure_preserves_earlier_matches(self):
        import httpx

        analyzer = RSSSentimentAnalyzer(feeds=("https://feed1.com/rss", "https://feed2.com/rss"))

        def fake_fetch(feed_url):
            if feed_url == "https://feed1.com/rss":
                return [make_rss_entry("BTC strength returns")]
            raise httpx.TimeoutException("timeout")

        with patch.object(analyzer, "_fetch_and_parse_feed", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert [item["title"] for item in result["items"]] == ["BTC strength returns"]

    def test_http_errors_return_error_but_programming_errors_surface(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))

        with patch.object(
            analyzer,
            "_fetch_and_parse_feed",
            side_effect=TypeError("bug"),
        ):
            with pytest.raises(TypeError, match="bug"):
                analyzer.analyze("BTC")

    def test_malformed_xml_returns_error(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))

        with patch.object(
            analyzer,
            "_fetch_and_parse_feed",
            side_effect=ET.ParseError("invalid xml"),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["items"] == []

    def test_parse_error_uses_remaining_feeds(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://feed1.com/rss", "https://feed2.com/rss"))

        def fake_fetch(feed_url):
            if feed_url == "https://feed1.com/rss":
                raise ET.ParseError("invalid xml")
            return [make_rss_entry("BTC recovery story")]

        with patch.object(analyzer, "_fetch_and_parse_feed", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert [item["title"] for item in result["items"]] == ["BTC recovery story"]

    @pytest.mark.parametrize("symbol", [None, 123, "", "   "])
    def test_invalid_symbol_returns_error(self, symbol):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))

        result = analyzer.analyze(symbol)

        assert result["status"] == "error"
        assert result["items"] == []
        assert result["summary"]["total_items"] == 0

    def test_symbol_is_trimmed_before_analysis(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC momentum returns")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("  BTC  ")

        assert result["status"] == "ok"
        assert result["symbol"] == "BTC"
        assert [item["title"] for item in result["items"]] == ["BTC momentum returns"]

    def test_altcoin_aliases_match_common_quote_pairs(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("Solana ecosystem momentum builds")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("SOLUSDC")

        assert result["status"] == "ok"
        assert [item["title"] for item in result["items"]] == ["Solana ecosystem momentum builds"]


class TestRSSSentimentAnalyzerBoostBehavior:
    """Tests for keyword boost application and capping."""

    def test_bullish_keyword_boost_applied(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC looks really bullish right now")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        item = result["items"][0]
        assert item["sentiment"]["keyword_boost"] > 0
        assert item["sentiment"]["final_score"] != item["sentiment"]["vader_compound"]

    def test_bearish_keyword_boost_applied(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("Bitcoin dump incoming, bearish signals")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        item = result["items"][0]
        assert item["sentiment"]["keyword_boost"] < 0

    def test_final_score_capped_at_1(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC is super bullish, moon, pump, to the moon, alts are up!!")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        item = result["items"][0]
        assert item["sentiment"]["final_score"] <= 1.0

    def test_final_score_capped_at_minus_1(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [make_rss_entry("BTC Bearish market, dump, crash, rug, rugpull, bear!")]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        item = result["items"][0]
        assert item["sentiment"]["final_score"] >= -1.0


class TestRSSSentimentAnalyzerSchemaDefaults:
    """Tests for missing/null RSS entry field defaults."""

    def test_missing_fields_use_defaults(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [{"title": "BTC post"}]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        item = result["items"][0]
        assert item["link"] == ""
        assert item["published_at"] == ""

    def test_null_fields_coerced_to_defaults(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [
            {
                "title": None,
                "link": None,
                "description": None,
                "published": None,
            }
        ]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        assert result["status"] == "no_results"
        assert result["items"] == []
        assert result["summary"]["total_items"] == 0

    def test_non_string_title_is_coerced_without_crashing(self):
        analyzer = RSSSentimentAnalyzer(feeds=("https://example.com/rss",))
        entries = [{"title": 123, "link": "", "description": "BTC discussion", "published": ""}]

        with patch.object(analyzer, "_fetch_and_parse_feed", return_value=entries):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert result["items"][0]["title"] == "123"


class TestRSSFeedParsing:
    """Tests for RSS/Atom XML parsing."""

    def test_parse_rss_2_format(self):
        analyzer = RSSSentimentAnalyzer()
        entries = analyzer._parse_feed_entries(RSS_SAMPLE)

        assert len(entries) == 2
        assert entries[0]["title"] == "Bitcoin surges to new highs"
        assert entries[0]["link"] == "https://example.com/btc-surges"
        assert entries[0]["description"] == "Bitcoin BTC is up 5% today"
        assert "Mon, 01 Jan 2024" in entries[0]["published"]

    def test_parse_atom_format(self):
        analyzer = RSSSentimentAnalyzer()
        entries = analyzer._parse_feed_entries(ATOM_SAMPLE)

        assert len(entries) == 1
        assert entries[0]["title"] == "Bitcoin halving approaching"
        assert entries[0]["link"] == "https://example.com/btc-halving"
        assert "BTC mining reward" in entries[0]["description"]
        assert "2024-01-01" in entries[0]["published"]

    def test_parse_malformed_xml_returns_error(self):
        analyzer = RSSSentimentAnalyzer()

        with pytest.raises(ET.ParseError):
            analyzer._parse_feed_entries("not valid xml <>")


class TestPackageImportStability:
    """Verify the package export surface remains stable."""

    def test_can_import_from_sentiment_package(self):
        from tempest_mcp.sentiment import RSSSentimentAnalyzer as Imported

        assert Imported is RSSSentimentAnalyzer

    def test_can_instantiate(self):
        analyzer = RSSSentimentAnalyzer()
        assert hasattr(analyzer, "analyze")
        assert hasattr(analyzer, "feeds")

    def test_default_feeds(self):
        assert RSSSentimentAnalyzer.feeds == (
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cryptoslate.com/feed/",
            "https://www.tradingview.com/ideas/feed/",
        )

    def test_context_manager_closes_http_client(self):
        with RSSSentimentAnalyzer(feeds=("https://example.com/rss",)) as analyzer:
            assert analyzer._http_client is not None

        assert analyzer._http_client is None


class TestDefaultRSSFeeds:
    """Verify the default RSS feed configuration."""

    def test_coindesk_feed_configured(self):
        assert "https://www.coindesk.com/arc/outboundfeeds/rss/" in RSSSentimentAnalyzer.feeds

    def test_cryptoslate_feed_configured(self):
        assert "https://cryptoslate.com/feed/" in RSSSentimentAnalyzer.feeds

    def test_tradingview_feed_configured(self):
        assert "https://www.tradingview.com/ideas/feed/" in RSSSentimentAnalyzer.feeds
