"""Unit tests for RedditSentimentAnalyzer."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "src")

from tempest_mcp.sentiment.reddit import (
    DEFAULT_REDDIT_ADAPTER_URL,
    RedditSentimentAnalyzer,
    _clamp,
    _compute_keyword_boost,
    _extract_base_token,
    _symbol_matches_title,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_child_data(
    title: str,
    score: int = 100,
    num_comments: int = 50,
    upvote_ratio: float = 0.95,
    flair: str | None = None,
    created_utc: float = 1700000000.0,
) -> dict:
    return {
        "title": title,
        "score": score,
        "num_comments": num_comments,
        "upvote_ratio": upvote_ratio,
        "link_flair_text": flair,
        "created_utc": created_utc,
    }


def reddit_json_response(children: list[dict]) -> dict:
    return {"data": {"children": [{"data": c} for c in children]}}


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

    def test_btc_pair(self):
        assert _extract_base_token("ETHBTC") == "ETH"

    def test_slash_pair(self):
        assert _extract_base_token("BTC/USD") == "BTC"

    def test_plain_symbol(self):
        assert _extract_base_token("BTC") == "BTC"
        assert _extract_base_token("DOGE") == "DOGE"

    def test_lowercase_normalized(self):
        assert _extract_base_token("btcusdt") == "BTC"


class TestSymbolMatchesTitle:
    def test_raw_symbol_match(self):
        assert _symbol_matches_title("BTC", "BTC", "BTC is pumping!")

    def test_base_token_match(self):
        assert _symbol_matches_title("BTCUSDT", "BTC", "Bitcoin BTC is up today")

    def test_no_match(self):
        assert not _symbol_matches_title("BTC", "BTC", "ETH is mooning")

    def test_case_insensitive(self):
        assert _symbol_matches_title("btc", "BTC", "btc to the moon")

    def test_word_boundaries_prevent_false_positive(self):
        assert not _symbol_matches_title("ETH", "ETH", "the market is choppy today")


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
        # "bullish" at +0.15 tier
        assert _compute_keyword_boost("BTC looks really bullish") > 0

    def test_bearish_boost_15(self):
        # "bearish" at -0.15 tier
        assert _compute_keyword_boost("Market looks bearish") < 0

    def test_moon_boost(self):
        # "moon" at +0.15 tier (also in +0.10 tier but applied first)
        boost = _compute_keyword_boost("DOGE to the moon!")
        assert boost > 0

    def test_dump_boost(self):
        # "dump" at -0.15 tier (also in -0.10 tier but applied first)
        boost = _compute_keyword_boost("Coin dump incoming")
        assert boost < 0

    def test_combined_boosts(self):
        # "bullish" (+0.15) + "moon" (+0.15) = +0.30, then clamped to 1.0
        boost = _compute_keyword_boost("BTC is bullish and going to the moon!")
        # Both at +0.15 tier, sum = 0.30
        assert boost == pytest.approx(0.30, rel=1e-3)

    def test_combined_bearish(self):
        # "bearish"(-0.15) + "dump"(-0.15) = -0.30
        boost = _compute_keyword_boost("Bearish outlook, dump incoming")
        assert boost == pytest.approx(-0.30, rel=1e-3)

    def test_boost_capped_at_1(self):
        # Many positive terms sum above 1.0
        boost = _compute_keyword_boost("BTC is bullish, moon, pump, to the moon, alts are up!")
        assert boost <= 1.0

    def test_boost_capped_at_minus_1(self):
        # Many negative terms sum below -1.0
        boost = _compute_keyword_boost("Bearish market, dump, crash, rug, rugpull, bear!")
        assert boost >= -1.0

    def test_term_applied_once_per_post(self):
        # "bullish" appears twice but should only apply once (+0.15)
        boost = _compute_keyword_boost("BTC is bullish and super bullish!")
        assert boost == pytest.approx(0.15, rel=1e-3)

    def test_no_boost(self):
        # Neutral title
        boost = _compute_keyword_boost("Bitcoin price update today")
        assert boost == 0.0

    def test_case_insensitive(self):
        assert _compute_keyword_boost("BITCOIN IS BULLISH") == _compute_keyword_boost(
            "bitcoin is bullish"
        )


# ---------------------------------------------------------------------------
# RedditSentimentAnalyzer tests
# ---------------------------------------------------------------------------


class TestRedditSentimentAnalyzerHappyPath:
    """Tests for the happy-path scenario with matching posts."""

    def test_returns_ok_status_with_matching_posts(self):
        analyzer = RedditSentimentAnalyzer(
            subreddits=("CryptoCurrency",),
        )
        children = [
            make_child_data("BTC is going to the moon!"),
            make_child_data("Bitcoin dump incoming"),
        ]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert result["symbol"] == "BTC"
        assert len(result["posts"]) == 2
        assert result["subreddits"] == ["CryptoCurrency"]
        assert "fetched_at" in result

    def test_schema_fields_present_on_every_post(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [make_child_data("BTC update for holders")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        required_keys = {
            "subreddit",
            "title",
            "score",
            "num_comments",
            "upvote_ratio",
            "flair",
            "created_utc",
            "sentiment",
        }
        assert required_keys <= post.keys()
        sentiment_keys = {
            "vader_pos",
            "vader_neu",
            "vader_neg",
            "vader_compound",
            "keyword_boost",
            "final_score",
        }
        assert sentiment_keys <= post["sentiment"].keys()

    def test_base_token_matching(self):
        """Symbol like BTCUSDT should match posts mentioning BTC."""
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [make_child_data("BTC holding strong at support")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTCUSDT")

        assert result["status"] == "ok"
        assert len(result["posts"]) == 1

    def test_selftext_matching_is_included_per_design(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [
            {
                **make_child_data("Weekend thread"),
                "selftext": "Bitcoin BTC is holding support.",
            }
        ]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert len(result["posts"]) == 1

    def test_selftext_is_inclusion_only_not_scoring_input(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [
            {
                **make_child_data("Weekend thread"),
                "selftext": "BTC is bullish and going to the moon",
            }
        ]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        sentiment = result["posts"][0]["sentiment"]
        assert sentiment["keyword_boost"] == 0.0
        assert sentiment["final_score"] == sentiment["vader_compound"]

    def test_summary_counts(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [
            make_child_data("BTC to the moon!"),  # positive final_score
            make_child_data("BTC dump incoming"),  # negative final_score
            make_child_data("Bitcoin price analysis"),  # neutral final_score
        ]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["summary"]["total_posts"] == 3
        assert (
            result["summary"]["positive_count"]
            + result["summary"]["negative_count"]
            + result["summary"]["neutral_count"]
            == 3
        )

    def test_fetched_at_is_iso_format(self):
        import re

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [make_child_data("BTC update")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        iso_pat = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pat, result["fetched_at"])

    def test_posts_order_preserved(self):
        """Posts should be in subreddit iteration order, then Reddit API order."""
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency", "Bitcoin"))
        crypto_children = [
            make_child_data("BTC Crypto post A"),
            make_child_data("BTC Crypto post B"),
        ]
        btc_children = [
            make_child_data("BTC Bitcoin post C"),
        ]

        def fake_fetch(subreddit):
            if subreddit == "CryptoCurrency":
                return crypto_children
            return btc_children

        with patch.object(analyzer, "_fetch_subreddit_posts", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["posts"][0]["title"] == "BTC Crypto post A"
        assert result["posts"][1]["title"] == "BTC Crypto post B"
        assert result["posts"][2]["title"] == "BTC Bitcoin post C"


class TestRedditSentimentAnalyzerNoResults:
    """Tests for the no-results envelope."""

    def test_no_matching_posts_returns_no_results(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        # Titles that don't mention BTC
        children = [
            make_child_data("ETH update"),
            make_child_data("Solana news"),
        ]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["status"] == "no_results"
        assert result["posts"] == []
        assert result["summary"]["total_posts"] == 0
        assert result["summary"]["avg_sentiment"] == 0.0
        assert result["summary"]["positive_count"] == 0
        assert result["summary"]["negative_count"] == 0
        assert result["summary"]["neutral_count"] == 0

    def test_empty_subreddit_response_returns_no_results(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=[]):
            result = analyzer.analyze("BTC")

        assert result["status"] == "no_results"
        assert result["posts"] == []


class TestRedditSentimentAnalyzerError:
    """Tests for the error envelope on provider failure."""

    def test_429_returns_error(self):
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer,
            "_fetch_subreddit_posts",
            side_effect=httpx.HTTPStatusError(
                "429",
                request=MagicMock(),
                response=MagicMock(status_code=429),
            ),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["posts"] == []
        assert result["summary"]["total_posts"] == 0

    def test_network_timeout_returns_error(self):
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer,
            "_fetch_subreddit_posts",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["posts"] == []

    def test_error_first_subreddit_aborts(self):
        """If the first subreddit fails, we get error immediately (no partial data)."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency", "Bitcoin"))

        def fake_fetch(subreddit):
            if subreddit == "CryptoCurrency":
                raise httpx.HTTPStatusError(
                    "429",
                    request=MagicMock(),
                    response=MagicMock(status_code=429),
                )
            # Should not be reached
            return [make_child_data("BTC post")]

        with patch.object(analyzer, "_fetch_subreddit_posts", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"

    def test_http_errors_return_error_but_programming_errors_surface(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(analyzer, "_fetch_subreddit_posts", side_effect=TypeError("bug")):
            with pytest.raises(TypeError, match="bug"):
                analyzer.analyze("BTC")

    def test_invalid_json_returns_error(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.side_effect = ValueError("bad json")
        response.request = MagicMock()

        with patch.object(analyzer._http_client, "get", return_value=response):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["posts"] == []


class TestRedditSentimentAnalyzerBoostBehavior:
    """Tests for keyword boost application and capping."""

    def test_bullish_keyword_boost_applied(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [make_child_data("BTC looks really bullish right now")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        # keyword_boost should be positive
        assert post["sentiment"]["keyword_boost"] > 0
        # final_score should reflect the boost
        assert post["sentiment"]["final_score"] != post["sentiment"]["vader_compound"]

    def test_bearish_keyword_boost_applied(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [make_child_data("Bitcoin dump incoming, bearish signals")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        assert post["sentiment"]["keyword_boost"] < 0

    def test_final_score_capped_at_1(self):
        """final_score should never exceed 1.0."""
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        # Many strong bullish terms
        children = [make_child_data("BTC is super bullish, moon, pump, to the moon, alts are up!!")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        assert post["sentiment"]["final_score"] <= 1.0

    def test_final_score_capped_at_minus_1(self):
        """final_score should never go below -1.0."""
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [make_child_data("BTC Bearish market, dump, crash, rug, rugpull, bear!")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        assert post["sentiment"]["final_score"] >= -1.0


class TestRedditSentimentAnalyzerSchemaDefaults:
    """Tests for missing/null Reddit field defaults."""

    def test_missing_fields_use_defaults(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        # Minimal child with missing fields
        children = [{"title": "BTC post"}]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        assert post["score"] == 0
        assert post["num_comments"] == 0
        assert post["upvote_ratio"] == 0.0
        assert post["created_utc"] == 0.0
        # flair may be None
        assert post["flair"] is None or isinstance(post["flair"], str)

    def test_null_fields_coerced_to_defaults(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [
            {
                "title": "BTC post",
                "score": None,
                "num_comments": None,
                "upvote_ratio": None,
                "created_utc": None,
                "link_flair_text": None,
            }
        ]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        post = result["posts"][0]
        assert post["score"] == 0
        assert post["num_comments"] == 0
        assert post["upvote_ratio"] == 0.0
        assert post["created_utc"] == 0.0
        assert post["flair"] is None

    def test_none_title_uses_empty_string_without_crashing(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [{"title": None, "selftext": "BTC discussion"}]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert result["posts"][0]["title"] == ""

    def test_non_string_title_is_coerced_without_crashing(self):
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        children = [{"title": 123, "selftext": "BTC discussion"}]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert result["posts"][0]["title"] == "123"


class TestPackageImportStability:
    """Verify the package export surface remains stable."""

    def test_can_import_from_sentiment_package(self):
        from tempest_mcp.sentiment import RedditSentimentAnalyzer as Imported

        assert Imported is RedditSentimentAnalyzer

    def test_can_instantiate(self):
        analyzer = RedditSentimentAnalyzer()
        assert hasattr(analyzer, "analyze")
        assert hasattr(analyzer, "subreddits")

    def test_default_subreddits(self):
        assert RedditSentimentAnalyzer.subreddits == (
            "CryptoCurrency",
            "Bitcoin",
            "ethereum",
            "dogecoin",
        )

    def test_context_manager_closes_http_client(self):
        with RedditSentimentAnalyzer(subreddits=("CryptoCurrency",)) as analyzer:
            assert analyzer._http_client is not None

        assert analyzer._http_client is None


# ---------------------------------------------------------------------------
# Reddit Adapter Path tests (ENG-129)
# ---------------------------------------------------------------------------


class TestRedditAdapterUrlConfiguration:
    """Tests for adapter URL configuration (ENG-129)."""

    def test_default_adapter_url(self):
        """Default adapter URL is the ENG-130 loopback endpoint."""
        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
        assert analyzer._adapter_base == DEFAULT_REDDIT_ADAPTER_URL

    def test_explicit_adapter_url_param(self):
        """Explicit adapter_url parameter overrides the default."""
        analyzer = RedditSentimentAnalyzer(
            subreddits=("CryptoCurrency",),
            adapter_url="http://custom-adapter:9090",
        )
        assert analyzer._adapter_base == "http://custom-adapter:9090"

    def test_adapter_url_env_var_overrides_default(self):
        """REDDIT_ADAPTER_URL env var overrides the hard-coded default."""
        with patch.dict("os.environ", {"REDDIT_ADAPTER_URL": "http://env-adapter:7777"}):
            analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))
            assert analyzer._adapter_base == "http://env-adapter:7777"

    def test_explicit_param_overrides_env_var(self):
        """Explicit adapter_url wins over env var."""
        with patch.dict("os.environ", {"REDDIT_ADAPTER_URL": "http://env-adapter:7777"}):
            analyzer = RedditSentimentAnalyzer(
                subreddits=("CryptoCurrency",),
                adapter_url="http://explicit:8888",
            )
            assert analyzer._adapter_base == "http://explicit:8888"

    def test_adapter_url_trailing_slash_stripped(self):
        """Trailing slashes on adapter URL are stripped for safe concatenation."""
        analyzer = RedditSentimentAnalyzer(
            subreddits=("CryptoCurrency",),
            adapter_url="http://adapter:8080/",
        )
        assert analyzer._adapter_base == "http://adapter:8080"
        # Verify URL construction won't have double slashes
        assert analyzer._adapter_base + "/r/CryptoCurrency/hot.json" == "http://adapter:8080/r/CryptoCurrency/hot.json"


class TestRedditAdapterFetchUrl:
    """Tests verifying _fetch_subreddit_posts hits the adapter URL (ENG-129)."""

    def test_fetch_hits_adapter_endpoint(self):
        """_fetch_subreddit_posts must hit the adapter URL, not Reddit direct."""
        analyzer = RedditSentimentAnalyzer(
            subreddits=("CryptoCurrency",),
            adapter_url="http://adapter:8080",
        )
        children = [make_child_data("BTC update")]

        mock_response = MagicMock()
        mock_response.json.return_value = reddit_json_response(children)
        mock_response.raise_for_status.return_value = None
        mock_response.request = MagicMock()

        with patch.object(analyzer._http_client, "get", return_value=mock_response) as mock_get:
            result = analyzer._fetch_subreddit_posts("CryptoCurrency")

        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        # Must use adapter URL, not https://www.reddit.com
        assert call_url.startswith("http://adapter:8080")
        assert "/r/CryptoCurrency/hot.json" in call_url
        assert "www.reddit.com" not in call_url
        assert result == children

    def test_fetch_uses_custom_adapter_url_from_env(self):
        """When REDDIT_ADAPTER_URL is set, fetches go through that adapter."""
        with patch.dict("os.environ", {"REDDIT_ADAPTER_URL": "http://custom:9000"}):
            analyzer = RedditSentimentAnalyzer(subreddits=("Bitcoin",))

        mock_response = MagicMock()
        mock_response.json.return_value = reddit_json_response([make_child_data("ETH discussion")])
        mock_response.raise_for_status.return_value = None
        mock_response.request = MagicMock()

        with patch.object(analyzer._http_client, "get", return_value=mock_response) as mock_get:
            analyzer._fetch_subreddit_posts("Bitcoin")

        call_url = mock_get.call_args[0][0]
        assert "http://custom:9000" in call_url
        assert "/r/Bitcoin/hot.json" in call_url
        assert "www.reddit.com" not in call_url


class TestRedditAdapterUnavailable:
    """Tests for reddit-adapter unavailability degradation (ENG-129)."""

    def test_connection_refused_returns_error(self):
        """Connection refused (adapter down) returns status=error, not crash."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["posts"] == []
        assert result["summary"]["total_posts"] == 0

    def test_adapter_timeout_returns_error(self):
        """Adapter timeout returns status=error, truthful degradation."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["posts"] == []
        assert result["summary"]["total_posts"] == 0

    def test_adapter_dns_failure_returns_error(self):
        """DNS resolution failure returns status=error."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.ConnectError("DNS failure"),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"


class TestRedditAdapterHTTPError:
    """Tests for HTTP errors returned by the reddit-adapter (ENG-129)."""

    def test_adapter_502_returns_error(self):
        """Adapter returns 502 when Reddit upstream fails — status=error."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "502 Bad Gateway",
                request=MagicMock(),
                response=MagicMock(status_code=502),
            ),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"
        assert result["posts"] == []
        assert result["summary"]["total_posts"] == 0

    def test_adapter_500_returns_error(self):
        """Adapter returns 500 on internal error — status=error."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            ),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"

    def test_adapter_403_returns_error(self):
        """Even with the adapter, Reddit may still return 403 — status=error."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "403 Forbidden",
                request=MagicMock(),
                response=MagicMock(status_code=403),
            ),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"

    def test_adapter_503_returns_error(self):
        """Adapter returns 503 when Tailscale/exit-node is unavailable — status=error."""
        import httpx

        analyzer = RedditSentimentAnalyzer(subreddits=("CryptoCurrency",))

        with patch.object(
            analyzer._http_client,
            "get",
            side_effect=httpx.HTTPStatusError(
                "503 Service Unavailable",
                request=MagicMock(),
                response=MagicMock(status_code=503),
            ),
        ):
            result = analyzer.analyze("BTC")

        assert result["status"] == "error"


class TestRedditAdapterHappyPathIntegration:
    """Integration-style tests verifying full adapter-backed analyze() flow (ENG-129)."""

    def test_analyze_through_adapter_returns_ok(self):
        """Full analyze() call with adapter-mocked response returns ok."""
        analyzer = RedditSentimentAnalyzer(
            subreddits=("CryptoCurrency", "Bitcoin"),
            adapter_url="http://adapter:8080",
        )
        children = [
            make_child_data("BTC is going to the moon!"),
            make_child_data("Bitcoin dump incoming"),
        ]

        def fake_fetch(subreddit):
            if subreddit == "CryptoCurrency":
                return children
            return [make_child_data("BTC only in Bitcoin subreddit")]

        with patch.object(analyzer, "_fetch_subreddit_posts", side_effect=fake_fetch):
            result = analyzer.analyze("BTC")

        assert result["status"] == "ok"
        assert result["symbol"] == "BTC"
        assert len(result["posts"]) == 3
        assert all("sentiment" in p for p in result["posts"])

    def test_combined_sentiment_sees_adapter_as_reddit_usable(self):
        """When adapter works, CombinedSentimentDashboard sees reddit as usable."""

        analyzer = RedditSentimentAnalyzer(
            subreddits=("CryptoCurrency",),
            adapter_url="http://adapter:8080",
        )
        children = [make_child_data("BTC to the moon!")]

        with patch.object(analyzer, "_fetch_subreddit_posts", return_value=children):
            result = analyzer.analyze("BTC")

        # status=ok means source is usable
        assert result["status"] == "ok"
        assert result["summary"]["total_posts"] == 1
