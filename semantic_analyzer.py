"""
Semantic understanding module for job fraud detection using a transformer model.

Default approach: zero-shot classification with a pretrained BERT (or any
compatible) model from HuggingFace Transformers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None

# Labels used for zero-shot classification
CLASS_LABELS = ["legitimate job", "scam job", "suspicious tone"]

# Simple lexical cues for manipulative/indirect scam tone
SCAM_PHRASES = [
    "easy income",
    "quick money",
    "guaranteed income",
    "secure your future income",
    "no experience required",
    "work from home and earn",
    "limited time opportunity",
    "act now",
]


def _clean(text: str) -> str:
    text = text or ""
    # collapse whitespace, lowercase for lexical checks
    return re.sub(r"\\s+", " ", text).strip()


def load_zero_shot_pipeline(model_name: str = "facebook/bart-large-mnli"):
    """
    Return a transformers zero-shot classification pipeline.
    Caller must ensure `transformers` and model weights are available.
    """
    if pipeline is None:
        raise ImportError("transformers is not installed")
    return pipeline("zero-shot-classification", model=model_name)


def semantic_analysis(
    title: str,
    description: str,
    clf_pipeline=None,
    scam_threshold: float = 0.6,
    suspicious_threshold: float = 0.6,
) -> Dict[str, object]:
    """
    Analyze job text for scam / manipulative semantics.

    Args:
        title, description: strings to analyze.
        clf_pipeline: transformers pipeline; if None, caller should inject one.
        scam_threshold: probability above which we mark as suspicious.
        suspicious_threshold: probability above which we mark manipulative tone.
    """
    text = _clean(title) + " " + _clean(description)

    # Zero-shot classification scores
    scam_prob = 0.0
    suspicious_prob = 0.0
    if clf_pipeline:
        result = clf_pipeline(text, candidate_labels=CLASS_LABELS, multi_label=True)
        label_scores = dict(zip(result["labels"], result["scores"]))
        scam_prob = float(label_scores.get("scam job", 0.0))
        suspicious_prob = float(label_scores.get("suspicious tone", 0.0))

    # Lexical cues for indirect scam language
    lexical_hits = [p for p in SCAM_PHRASES if p in text.lower()]
    if lexical_hits:
        suspicious_prob = max(suspicious_prob, 0.5 + 0.1 * len(lexical_hits))

    semantic_score = max(scam_prob, suspicious_prob)
    is_suspicious = semantic_score >= scam_threshold or suspicious_prob >= suspicious_threshold

    return {
        "semantic_score": round(semantic_score, 4),
        "is_suspicious_semantic": is_suspicious,
    }


__all__ = ["semantic_analysis", "load_zero_shot_pipeline", "CLASS_LABELS"]
