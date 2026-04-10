"""
Suspicious phrase detection for fake job post scoring.
"""

from __future__ import annotations

from typing import List, Tuple

SUSPICIOUS_KEYWORDS = [
    "google partner",
    "urgent hiring",
    "limited slots",
    "no experience required",
    "pay",
    "fee",
]


def detect_suspicious_phrases(text: str, fake_score: int) -> Tuple[int, List[str]]:
    """
    Add penalties for suspicious phrases in job text.

    Args:
        text: Job post content.
        fake_score: Current accumulated fake score.

    Returns:
        (updated_fake_score, matched_keywords)
    """
    if not text:
        return fake_score, []

    lowered = text.lower()
    matched = []

    for kw in SUSPICIOUS_KEYWORDS:
        if kw in lowered:
            fake_score += 10
            matched.append(kw)

    return fake_score, matched


__all__ = ["detect_suspicious_phrases", "SUSPICIOUS_KEYWORDS"]
