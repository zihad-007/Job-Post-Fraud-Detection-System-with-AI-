"""
External verification utilities for job fraud detection.

The functions are designed to be side-effect free and safe to call in
production pipelines. Network calls use short timeouts and fail closed
to avoid blocking the main flow.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, Optional
from urllib.parse import urlparse

import requests

# Optional dependency
try:
    import whois  # type: ignore
except Exception:  # pragma: no cover - only executed when missing/broken
    whois = None


# ---------- Helpers: extraction ----------

EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
FREE_EMAIL_PROVIDERS = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com", "aol.com"}


def extract_company_website(job_data: Dict) -> Optional[str]:
    """Return normalized company website URL if present in job data."""
    url = job_data.get("company_website") or job_data.get("website") or ""
    url = url.strip()
    if not url:
        return None
    if not urlparse(url).scheme:
        url = "https://" + url
    return url


def extract_email(job_data: Dict) -> Optional[str]:
    """Find an email in structured fields or free-form job description."""
    for key in ("contact_email", "email", "recruiter_email"):
        val = job_data.get(key)
        if val and EMAIL_REGEX.fullmatch(val.strip()):
            return val.strip().lower()

    description = job_data.get("description", "") or ""
    match = EMAIL_REGEX.search(description)
    return match.group(0).lower() if match else None


# ---------- Checks ----------

def is_website_reachable(url: str, timeout: float = 4.0) -> bool:
    """Return True if a HEAD/GET returns HTTP 200."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=timeout)
        if resp.status_code == 405:
            resp = requests.get(url, allow_redirects=True, timeout=timeout)
        return resp.status_code == 200
    except Exception:
        return False


def domain_age_ok(url: str, min_age_days: int = 30) -> bool:
    """
    Return True if domain is at least `min_age_days` old.
    Falls back to False if whois data is unavailable.
    """
    if not whois:
        return False
    try:
        domain = urlparse(url).hostname
        if not domain:
            return False
        record = whois.whois(domain)
        created = record.creation_date
        if isinstance(created, list):
            created = created[0]
        if not isinstance(created, datetime):
            return False
        return datetime.utcnow() - created >= timedelta(days=min_age_days)
    except Exception:
        return False


def email_domain_matches_company(email: str, company_url: Optional[str]) -> bool:
    if not email or not company_url:
        return False
    email_domain = email.split("@")[-1].lower()
    company_domain = (urlparse(company_url).hostname or "").lower()
    return email_domain == company_domain or email_domain.endswith("." + company_domain)


def is_free_provider(email: Optional[str]) -> bool:
    if not email:
        return False
    return email.split("@")[-1].lower() in FREE_EMAIL_PROVIDERS


def glassdoor_presence(company_name: Optional[str]) -> bool:
    """
    Placeholder for Glassdoor lookup.
    In production, replace with API or scraping logic.
    """
    if not company_name:
        return False
    # Stub: heuristic could be enhanced; currently deterministic False.
    return False


# ---------- Scoring ----------

def compute_score(flags: Dict[str, bool]) -> int:
    """
    Simple weighted score out of 100.
    Adjust weights as needed; current scheme favors website validation.
    """
    weights = {
        "website_valid": 35,
        "email_valid": 35,
        "glassdoor_found": 20,
        "free_email_used": -20,
    }
    score = 0
    for key, weight in weights.items():
        if key not in flags:
            continue
        score += weight if flags[key] else 0
    return max(0, min(100, score))


# ---------- Public API ----------

def verify_job(job_data: Dict) -> Dict[str, bool | int]:
    """
    Run external verification checks on a job posting dictionary.

    Expected job_data keys (best effort):
    - company_name
    - company_website or website
    - contact_email / email / recruiter_email
    - description
    """
    website = extract_company_website(job_data)
    email = extract_email(job_data)

    website_reachable = is_website_reachable(website) if website else False
    website_age_ok = domain_age_ok(website) if website else False

    website_valid = website_reachable and website_age_ok

    email_valid = email_domain_matches_company(email, website)
    free_email_used = is_free_provider(email)

    glassdoor_found = glassdoor_presence(job_data.get("company_name"))

    flags = {
        "website_valid": website_valid,
        "email_valid": email_valid,
        "free_email_used": free_email_used,
        "glassdoor_found": glassdoor_found,
    }
    score = compute_score(flags)

    return {
        **flags,
        "verification_score": score,
    }


__all__ = [
    "verify_job",
    "extract_company_website",
    "extract_email",
    "is_website_reachable",
    "domain_age_ok",
    "email_domain_matches_company",
    "is_free_provider",
    "glassdoor_presence",
    "compute_score",
]
