"""
Payment-related risk rule for fake job detection.

The rule assigns a high penalty when posts mention fees or payments.
"""

from __future__ import annotations

from typing import List, Tuple

# Keywords that indicate payment/fees in job posts
PAYMENT_KEYWORDS = [
    "pay",
    "fee",
    "registration",
    "deposit",
    "onboarding fee",
]


def detect_payment_risk(text: str, fake_score: int) -> Tuple[int, List[str]]:
    """
    Increase fake_score if payment-related terms are present.

    Args:
        text: Job post text.
        fake_score: Current accumulated fake score.

    Returns:
        (updated_fake_score, matched_keywords)
    """
    if not text:
        return fake_score, []

    lowered = text.lower()
    matched = [kw for kw in PAYMENT_KEYWORDS if kw in lowered]

    if matched:
        fake_score += 50  # high penalty for payment requests

    return fake_score, matched


__all__ = ["detect_payment_risk", "PAYMENT_KEYWORDS"]
