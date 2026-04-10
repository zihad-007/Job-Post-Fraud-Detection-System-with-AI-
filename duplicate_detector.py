"""
Duplicate and pattern detection for job postings using TF-IDF and regex.

Primary entrypoint: `detect_duplicate(new_job, existing_jobs)` where each job
is a dict containing at least a `description` field and optional text fields
with contact info.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Regex patterns kept simple for speed and coverage.
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
PHONE_REGEX = re.compile(r"\\+?\\d[\\d\\s().-]{6,}\\d")


def extract_contacts(text: str) -> Tuple[List[str], List[str]]:
    """Return (emails, phones) found in text."""
    emails = EMAIL_REGEX.findall(text or "")
    phones = [p.strip() for p in PHONE_REGEX.findall(text or "")]
    return emails, phones


def _gather_contacts(job: Dict) -> Tuple[List[str], List[str]]:
    """Aggregate contacts from common fields plus description."""
    fields = [
        job.get("contact_email", ""),
        job.get("email", ""),
        job.get("recruiter_email", ""),
        job.get("contact_phone", ""),
        job.get("phone", ""),
        job.get("recruiter_phone", ""),
        job.get("description", ""),
    ]
    combined = " ".join([f for f in fields if f])
    return extract_contacts(combined)


def _vectorize(corpus: List[str]) -> Tuple[TfidfVectorizer, any]:
    """Fit TF-IDF on corpus and return (vectorizer, matrix)."""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def _max_similarity(vectorizer: TfidfVectorizer, matrix, new_text: str) -> float:
    """Compute max cosine similarity between new_text and the fitted matrix."""
    if matrix.shape[0] == 0:
        return 0.0
    new_vec = vectorizer.transform([new_text])
    sims = cosine_similarity(new_vec, matrix)[0]
    return float(sims.max()) if sims.size else 0.0


def _contact_reused(new_contacts: Tuple[List[str], List[str]], corp_contacts: List[Tuple[List[str], List[str]]]) -> bool:
    """Check if any email/phone appears in more than one post."""
    new_emails, new_phones = set(map(str.lower, new_contacts[0])), set(map(str, new_contacts[1]))

    corpus_emails = set()
    corpus_phones = set()
    for emails, phones in corp_contacts:
        corpus_emails.update(map(str.lower, emails))
        corpus_phones.update(map(str, phones))

    reused_email = bool(new_emails & corpus_emails)
    reused_phone = bool(new_phones & corpus_phones)
    return reused_email or reused_phone


def detect_duplicate(new_job: Dict, existing_jobs: Iterable[Dict], similarity_threshold: float = 0.8) -> Dict[str, object]:
    """
    Compare a new job posting against existing posts.

    Args:
        new_job: dict with `description` plus optional contact fields.
        existing_jobs: iterable of job dicts with `description`.
        similarity_threshold: cosine similarity to mark as duplicate.

    Returns:
        {
          "is_duplicate": bool,
          "similarity_score": float,
          "reused_contact": bool
        }
    """
    existing_list = list(existing_jobs)
    corpus_texts = [job.get("description", "") for job in existing_list if job.get("description")]

    if corpus_texts:
        vectorizer, matrix = _vectorize(corpus_texts)
        similarity_score = _max_similarity(vectorizer, matrix, new_job.get("description", ""))
    else:
        similarity_score = 0.0

    is_duplicate = similarity_score >= similarity_threshold

    new_contacts = _gather_contacts(new_job)
    corpus_contacts = [_gather_contacts(job) for job in existing_list]
    reused_contact = _contact_reused(new_contacts, corpus_contacts)

    return {
        "is_duplicate": is_duplicate,
        "similarity_score": float(round(similarity_score, 4)),
        "reused_contact": reused_contact,
    }


__all__ = ["detect_duplicate", "extract_contacts"]
