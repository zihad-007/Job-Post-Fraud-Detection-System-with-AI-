"""
Geo and language inconsistency detection for job postings.

Uses langdetect for language inference and regex-based phone country code
inspection to spot mismatches with declared job location.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

try:
    from langdetect import detect
except Exception:  # pragma: no cover - optional dependency
    detect = None


# Minimal dialing code map; extend as needed
DIAL_CODES = {
    "us": "+1",
    "ca": "+1",
    "uk": "+44",
    "gb": "+44",
    "in": "+91",
    "au": "+61",
    "bd": "+880",
}


def _normalize(text: str) -> str:
    return text.lower().strip() if text else ""


def _expected_lang_from_location(location: str) -> Optional[str]:
    loc = _normalize(location)
    if any(k in loc for k in ["united states", "usa", "us", "canada", "australia", "uk", "united kingdom", "england"]):
        return "en"
    if "india" in loc:
        return "en"  # common in job posts
    if "bangladesh" in loc or "bd" == loc:
        return "bn"
    return None


def _country_code_from_phone(contact: str) -> Optional[str]:
    if not contact:
        return None
    match = re.search(r"(\\+\\d{1,4})", contact.replace(" ", ""))
    return match.group(1) if match else None


def _expected_dial_code(location: str) -> Optional[str]:
    loc = _normalize(location)
    for country, code in DIAL_CODES.items():
        if country in loc or country.upper() in loc or code in loc:
            return code
    return None


def detect_geo_language_inconsistency(
    job_location: str,
    description: str,
    contact: str,
) -> Dict[str, object]:
    detected_lang = ""
    lang_flag = False
    phone_flag = False
    location_flag = False

    # Language detection
    if detect:
        try:
            detected_lang = detect(description or "")
        except Exception:
            detected_lang = ""
    expected_lang = _expected_lang_from_location(job_location)
    if expected_lang and detected_lang and detected_lang != expected_lang:
        lang_flag = True

    # Phone country code mismatch
    dial_code = _country_code_from_phone(contact)
    expected_code = _expected_dial_code(job_location)
    if dial_code and expected_code and dial_code != expected_code:
        phone_flag = True

    # Location mention consistency: if location string absent in description
    if job_location and job_location.lower() not in (description or "").lower():
        location_flag = True

    # Score heuristic
    flags = [lang_flag, phone_flag, location_flag]
    score = min(1.0, round(sum(flags) / 3.0, 4))

    return {
        "geo_inconsistency_score": score,
        "language_detected": detected_lang,
        "phone_country_mismatch": phone_flag,
    }


__all__ = ["detect_geo_language_inconsistency"]
