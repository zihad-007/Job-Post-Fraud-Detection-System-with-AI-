"""
Sensitive information detection for job fraud screening.

Detects requests for personal documents like passport, national ID, bank or
card details. Lightweight keyword/phrase matching with span highlights.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Canonical risky phrases (lowercase)
PHRASES = [
    "passport",
    "national id",
    "nid",
    "nid card",
    "id card",
    "personal id",
    "bank details",
    "bank account",
    "account number",
    "routing number",
    "credit card",
    "debit card",
    "card number",
    "cvv",
    "personal documents",
    "government id",
]

# Precompiled regex for fast search with word boundaries where sensible.
PATTERNS = [
    re.compile(r"\\bpassport\\b", re.IGNORECASE),
    re.compile(r"\\bnational\\s+id\\b", re.IGNORECASE),
    re.compile(r"\\bnid\\b", re.IGNORECASE),
    re.compile(r"\\bnid\\s+card\\b", re.IGNORECASE),
    re.compile(r"\\bid\\s+card\\b", re.IGNORECASE),
    re.compile(r"\\bpersonal\\s+id\\b", re.IGNORECASE),
    re.compile(r"\\bbank\\s+details\\b", re.IGNORECASE),
    re.compile(r"\\bbank\\s+account\\b", re.IGNORECASE),
    re.compile(r"\\baccount\\s+number\\b", re.IGNORECASE),
    re.compile(r"\\brouting\\s+number\\b", re.IGNORECASE),
    re.compile(r"\\bcredit\\s+card\\b", re.IGNORECASE),
    re.compile(r"\\bdebit\\s+card\\b", re.IGNORECASE),
    re.compile(r"\\bcard\\s+number\\b", re.IGNORECASE),
    re.compile(r"\\bcvv\\b", re.IGNORECASE),
    re.compile(r"\\bpersonal\\s+documents?\\b", re.IGNORECASE),
    re.compile(r"\\bgovernment\\s+id\\b", re.IGNORECASE),
]


def detect_sensitive_info(text: str) -> Dict[str, object]:
    """
    Detect sensitive document requests in a job description.

    Returns:
    {
      "sensitive_info_flag": bool,
      "detected_terms": [ {"phrase": str, "start": int, "end": int} ... ]
    }
    """
    detected: List[Dict[str, int | str]] = []
    for pattern in PATTERNS:
        for match in pattern.finditer(text or ""):
            detected.append(
                {
                    "phrase": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

    # deduplicate by span + phrase to avoid repeated overlaps
    seen = set()
    unique_detected = []
    for item in detected:
        key = (item["start"], item["end"], item["phrase"].lower())
        if key not in seen:
            seen.add(key)
            unique_detected.append(item)

    return {
        "sensitive_info_flag": bool(unique_detected),
        "detected_terms": unique_detected,
    }


__all__ = ["detect_sensitive_info", "PHRASES"]
