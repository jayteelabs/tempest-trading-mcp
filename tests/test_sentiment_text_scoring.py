"""Unit tests for the shared sentiment text scoring seam."""

import pytest

from tempest_mcp.sentiment.text_scoring import (
    KEYWORD_BOOST_TERMS,
    clamp_score,
    coerce_scoring_text,
    compute_keyword_boost,
    score_sentiment_text,
)


class FakeVader:
    def __init__(self, compound: float = 0.0) -> None:
        self.compound = compound
        self.seen_text: str | None = None

    def polarity_scores(self, text: str) -> dict[str, float]:
        self.seen_text = text
        return {"pos": 0.123456, "neu": 0.654321, "neg": 0.111111, "compound": self.compound}


def test_keyword_vocabulary_values_are_shared_contract() -> None:
    assert KEYWORD_BOOST_TERMS == (
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
    )


def test_overlap_aware_matching_prefers_phrase() -> None:
    assert compute_keyword_boost("DOGE to the moon") == pytest.approx(0.15)


@pytest.mark.parametrize(
    "text",
    [
        "DOGE to the moondust",
        "go to the moonshot",
        "alts are upside",
    ],
)
def test_multi_word_terms_require_word_boundaries(text: str) -> None:
    assert compute_keyword_boost(text) == pytest.approx(0.0)


def test_term_applies_once_per_record() -> None:
    assert compute_keyword_boost("BTC bullish bullish bullish") == pytest.approx(0.15)


def test_matching_is_case_insensitive() -> None:
    assert compute_keyword_boost("BITCOIN IS BULLISH") == compute_keyword_boost(
        "bitcoin is bullish"
    )


def test_boost_clamps_to_bounds() -> None:
    assert clamp_score(2.0) == 1.0
    assert clamp_score(-2.0) == -1.0


def test_coerce_scoring_text_handles_none_and_non_strings() -> None:
    assert coerce_scoring_text(None) == ""
    assert coerce_scoring_text(123) == "123"
    assert coerce_scoring_text("BTC") == "BTC"


def test_vader_fields_and_final_score_are_rounded_and_clamped() -> None:
    vader = FakeVader(compound=0.99999)

    score = score_sentiment_text("BTC is bullish", vader)

    assert vader.seen_text == "BTC is bullish"
    assert score.as_dict() == {
        "vader_pos": 0.1235,
        "vader_neu": 0.6543,
        "vader_neg": 0.1111,
        "vader_compound": 1.0,
        "keyword_boost": 0.15,
        "final_score": 1.0,
    }
