"""
Persuasion and urgency detection for job posts.

Combines keyword matching with sentiment intensity. Uses VADER if available,
otherwise falls back to a neutral sentiment (0).
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

try:
    from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore
    _vader = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover - optional dependency
    _vader = None


URGENCY_KEYWORDS = [
    "apply now",
    "limited time",
    "urgent hiring",
    "hurry",
    "immediate joining",
]

PERSUASION_PHRASES = [
    "secure income",
    "guaranteed earning",
    "no experience needed",
    "easy money",
]


def _normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text.lower()).strip()


def _count_matches(text: str, patterns: List[str]) -> Tuple[int, List[str]]:
    hits = []
    count = 0
    for p in patterns:
        occurrences = len(re.findall(re.escape(p), text))
        if occurrences:
            hits.append(p)
            count += occurrences
    return count, hits


def _sentiment(text: str) -> float:
    if not _vader:
        return 0.0
    scores = _vader.polarity_scores(text)
    return scores.get("pos", 0.0) - scores.get("neg", 0.0)


def _score(count: int, text_len: int, sentiment_boost: float) -> float:
    if text_len <= 0:
        return 0.0
    base = count / max(text_len / 100.0, 1.0)  # normalize per ~100 chars
    score = base + max(sentiment_boost, 0.0)
    return max(0.0, min(1.0, score))


def analyze_persuasion_urgency(text: str) -> Dict[str, object]:
    """
    Detect urgency/persuasion cues in a job post.
    Returns:
    {
      "urgency_score": 0-1,
      "persuasion_score": 0-1,
      "matched_keywords": [list]
    }
    """
    normalized = _normalize(text or "")
    text_len = len(normalized)

    urg_count, urg_hits = _count_matches(normalized, URGENCY_KEYWORDS)
    pers_count, pers_hits = _count_matches(normalized, PERSUASION_PHRASES)

    sentiment_boost = _sentiment(normalized) * 0.2  # small influence

    urgency_score = _score(urg_count, text_len, sentiment_boost)
    persuasion_score = _score(pers_count, text_len, sentiment_boost)

    matched = sorted(set(urg_hits + pers_hits))

    return {
        "urgency_score": round(urgency_score, 4),
        "persuasion_score": round(persuasion_score, 4),
        "matched_keywords": matched,
    }


__all__ = ["analyze_persuasion_urgency"]
