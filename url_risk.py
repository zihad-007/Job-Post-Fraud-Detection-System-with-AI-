"""
URL risk detection for job fraud screening.

Functions:
- extract_urls(description) -> list[str]
- assess_url_risk(description, company_name=None, check_reachability=False) -> dict
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests

# Regex: simplistic but fast; captures http(s) URLs and bare domains.
URL_REGEX = re.compile(
    r"(https?://[\\w.-]+(?:/[\\w\\-._~:/?#@!$&'()*+,;=%]*)?|\\b[\\w.-]+\\.[A-Za-z]{2,}\\b)",
    re.IGNORECASE,
)

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly"}
SUSPICIOUS_TLDS = {".xyz", ".top", ".click", ".online"}


def extract_urls(description: str) -> List[str]:
    """Return all URLs/domains from description."""
    return URL_REGEX.findall(description or "")


def _is_shortened(domain: str) -> bool:
    return domain in SHORTENERS


def _has_suspicious_tld(domain: str) -> bool:
    return any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS)


def _domain_from_url(url: str) -> str:
    parsed = urlparse(url if url.startswith("http") else "http://" + url)
    return parsed.hostname or ""


def _domain_mismatch(domain: str, company_name: Optional[str]) -> bool:
    if not company_name:
        return False
    token = company_name.lower().split()[0]
    return token not in domain.lower()


def _reachable(url: str, timeout: float = 3.0) -> bool:
    try:
        resp = requests.head(url if url.startswith("http") else "http://" + url, timeout=timeout, allow_redirects=True)
        if resp.status_code == 405:
            resp = requests.get(url, timeout=timeout, allow_redirects=True)
        return resp.status_code < 400
    except Exception:
        return False


def _reputation_stub(domain: str) -> bool:
    """
    Placeholder for domain reputation lookup.
    Returns False (not reputable) by default.
    Replace with real service integration as needed.
    """
    return False


def assess_url_risk(
    description: str,
    company_name: Optional[str] = None,
    check_reachability: bool = False,
) -> Dict[str, object]:
    urls = extract_urls(description)

    suspicious = []
    shortened_used = False
    score = 0.0

    for url in urls:
        domain = _domain_from_url(url)
        if not domain:
            continue

        short = _is_shortened(domain)
        bad_tld = _has_suspicious_tld(domain)
        mismatch = _domain_mismatch(domain, company_name)
        unreachable = False

        if check_reachability:
            reachable = _reachable(url)
            unreachable = not reachable

        reputation_bad = _reputation_stub(domain)

        reasons = []
        if short:
            reasons.append("shortener")
            shortened_used = True
        if bad_tld:
            reasons.append("suspicious_tld")
        if mismatch:
            reasons.append("domain_mismatch")
        if unreachable:
            reasons.append("unreachable")
        if reputation_bad:
            reasons.append("low_reputation")

        if reasons:
            suspicious.append({"url": url, "domain": domain, "reasons": reasons})

        # scoring heuristic: +0.3 shortener, +0.25 bad tld, +0.2 mismatch, +0.1 unreachable, +0.15 low rep
        score += (
            (0.3 if short else 0)
            + (0.25 if bad_tld else 0)
            + (0.2 if mismatch else 0)
            + (0.1 if unreachable else 0)
            + (0.15 if reputation_bad else 0)
        )

    # normalize score to 0-1
    url_risk_score = min(1.0, round(score, 4))

    return {
        "url_risk_score": url_risk_score,
        "suspicious_urls": suspicious,
        "shortened_url_used": shortened_used,
    }


__all__ = ["assess_url_risk", "extract_urls"]
