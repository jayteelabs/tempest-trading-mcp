"""Shared pure text-scoring helpers for sentiment analyzers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class VaderLike(Protocol):
    """Protocol for the subset of VADER used by the shared scorer."""

    def polarity_scores(self, text: str) -> Mapping[str, float]:
        """Return VADER polarity scores for the supplied text."""
        raise NotImplementedError


@dataclass(frozen=True)
class TextSentimentScore:
    """Rounded VADER + keyword sentiment envelope."""

    vader_pos: float
    vader_neu: float
    vader_neg: float
    vader_compound: float
    keyword_boost: float
    final_score: float

    def as_dict(self) -> dict[str, float]:
        """Return the source contract sentiment field shape."""
        return {
            "vader_pos": self.vader_pos,
            "vader_neu": self.vader_neu,
            "vader_neg": self.vader_neg,
            "vader_compound": self.vader_compound,
            "keyword_boost": self.keyword_boost,
            "final_score": self.final_score,
        }


KEYWORD_BOOST_TERMS: tuple[tuple[str, float], ...] = (
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


def coerce_scoring_text(value: Any) -> str:
    """Return a deterministic string for scorer input."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def clamp_score(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """Clamp a numeric sentiment score to the configured bounds."""
    return max(lo, min(hi, value))


def compute_keyword_boost(text: Any) -> float:
    """Compute deterministic keyword modifier for scorer input."""
    lower_text = coerce_scoring_text(text).lower()
    occupied_spans: list[tuple[int, int]] = []
    boost = 0.0

    def overlaps(span: tuple[int, int]) -> bool:
        start, end = span
        for other_start, other_end in occupied_spans:
            if start < other_end and end > other_start:
                return True
        return False

    for term, value in KEYWORD_BOOST_TERMS:
        alpha_phrase = term.replace(" ", "").isalpha()
        pattern = (
            re.compile(rf"\b{re.escape(term)}\b") if alpha_phrase else re.compile(re.escape(term))
        )
        for match in pattern.finditer(lower_text):
            span = match.span()
            if overlaps(span):
                continue
            occupied_spans.append(span)
            boost += value
            break

    return clamp_score(boost)


def score_sentiment_text(text: Any, vader: VaderLike) -> TextSentimentScore:
    """Score selected text using VADER plus deterministic keyword boosts."""
    scoring_text = coerce_scoring_text(text)
    vader_scores = vader.polarity_scores(scoring_text)
    keyword_boost = compute_keyword_boost(scoring_text)
    final_score = clamp_score(vader_scores["compound"] + keyword_boost)

    return TextSentimentScore(
        vader_pos=round(vader_scores["pos"], 4),
        vader_neu=round(vader_scores["neu"], 4),
        vader_neg=round(vader_scores["neg"], 4),
        vader_compound=round(vader_scores["compound"], 4),
        keyword_boost=round(keyword_boost, 4),
        final_score=round(final_score, 4),
    )
