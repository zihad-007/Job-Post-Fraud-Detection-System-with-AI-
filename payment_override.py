"""
High-priority payment detection rule for fake job posts.

If any payment-related term is present, the post is immediately classified
as "Fake", overriding other rules or model predictions.
"""

from __future__ import annotations

from typing import List, Tuple

PAYMENT_WORDS = [
    "pay",
    "deposit",
    "registration fee",
    "onboarding fee",
    "fee",
]


def detect_payment_override(text: str) -> Tuple[str, List[str]]:
    """
    Detect payment words and hard-override classification.

    Args:
        text: Job post content.

    Returns:
        (label, matched_payment_words)
        label: "Fake" if any payment word is found, otherwise "Unknown".
    """
    if not text:
        return "Unknown", []

    lowered = text.lower()
    matched = [kw for kw in PAYMENT_WORDS if kw in lowered]

    if matched:
        return "Fake", matched
    return "Unknown", []


__all__ = ["detect_payment_override", "PAYMENT_WORDS"]
