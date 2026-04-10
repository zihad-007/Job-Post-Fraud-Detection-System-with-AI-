import json
import re
from pathlib import Path
from typing import Dict, List, Optional

# Optional ML dependencies (used when available)
try:
    import joblib  # type: ignore
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - defensive
    joblib = None
    np = None

# ---------------- Text Preprocessing ----------------
def preprocess_text(text: Optional[str]) -> str:
    """
    Normalize text while keeping non-English letters (Bangla, etc.) so
    multilingual inputs don't get stripped to empty.
    """
    if not text:
        return ""
    text = text.casefold()
    # remove urls, emails
    text = re.sub(r"http\S+|www\S+|\S+@\S+", " ", text)
    # keep unicode letters/numbers (including Bengali block); drop punctuation
    text = re.sub(r"[^\w\s\u0980-\u09FF]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------- Dataset / model metadata ----------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "job_fraud_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"
MODEL_METADATA: Dict[str, object] = {}
_MODEL_PIPELINE = None


def _load_model_metadata() -> None:
    global MODEL_METADATA
    if META_PATH.exists():
        try:
            MODEL_METADATA = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            MODEL_METADATA = {}


def _load_trained_pipeline():
    """
    Lazy-load the scikit-learn pipeline if present.
    Returns None when the trained model is absent or cannot be loaded.
    """
    global _MODEL_PIPELINE
    if _MODEL_PIPELINE is not None:
        return _MODEL_PIPELINE

    if not (joblib and MODEL_PATH.exists()):
        return None

    try:
        _MODEL_PIPELINE = joblib.load(MODEL_PATH)
    except Exception:
        _MODEL_PIPELINE = None
    return _MODEL_PIPELINE


_load_model_metadata()


# ---------------- Shared keyword lists ----------------
FRAUD_KEYWORDS = [
    "registration fee",
    "security deposit",
    "processing fee",
    "account activation fee",
    "earn money fast",
    "guaranteed profit",
    "investment required",
    "whatsapp",
    "contact on whatsapp",
    "telegram",
    "crypto",
    "bitcoin",
    "usdt",
    "eth",
    "no experience required",
    "work 2 hours",
    "daily payout",
    "click ads",
    "data entry work from home",
    "visa processing",
    # Bengali/other language cues
    "টাকা",
    "ফি",
    "রেজিস্ট্রেশন",
    "যোগাযোগ করুন",
    "হোয়াটসঅ্যাপ",
    "হোয়াটস অ্যাপ",
]


# ---------------- Stage 1: Job-post validation ----------------
JOB_KEYWORDS = [
    "job",
    "hiring",
    "salary",
    "company",
    "requirements",
    "requirement",
    "apply",
    "position",
    "responsibilities",
    "experience",
    "benefits",
    "role",
    "vacancy",
    "skills",
    "deadline",
    "location",
    "full time",
    "part time",
    "remote",
    "onsite",
]
MIN_WORDS = 12


def validate_job_post(text: str) -> Dict[str, object]:
    """
    Lightweight stage-1 gate that mixes rule-based filters with a
    keyword-density score to decide whether input looks like a job post.
    Returns a diagnostics dict that mirrors the fraud detail format.
    """
    raw = text or ""
    cleaned = preprocess_text(raw)
    words = cleaned.split()
    word_count = len(words)

    keyword_hits = [kw for kw in JOB_KEYWORDS if kw in cleaned]
    density = len(keyword_hits) / max(word_count, 1)

    # "ML-like" score: weighted sum approximating a tiny linear model
    score = (
        1.2 * len(keyword_hits)
        + 6.0 * density
        + (2.0 if "salary" in cleaned or "$" in raw.lower() else 0.0)
        + (1.0 if "apply" in cleaned or "apply now" in cleaned else 0.0)
    )

    reasons: List[str] = []
    if word_count < MIN_WORDS:
        reasons.append(f"Too short ({word_count} words); needs at least {MIN_WORDS}.")
    if len(keyword_hits) < 2:
        reasons.append("Not enough job-related keywords.")
    else:
        reasons.append(f"Matched keywords: {', '.join(keyword_hits[:5])}")
    reasons.append(f"Keyword density: {round(density*100, 1)}%")

    is_job = word_count >= MIN_WORDS and (len(keyword_hits) >= 2 or density >= 0.03) and score >= 3.0
    if not is_job:
        reasons.insert(0, "This does not appear to be a job posting.")

    return {
        "is_job": is_job,
        "label": "Job Post" if is_job else "Not a Job Post",
        "probability": round(min(score / 10.0, 1.0) * 100, 2),
        "rule_score": round(score, 2),
        "reasons": reasons,
        "keyword_hits": keyword_hits,
        "word_count": word_count,
        "keyword_density": round(density, 3),
    }


def rule_based_score(cleaned: str) -> int:
    """
    Lightweight heuristic scoring used when model is absent or as a guardrail.
    Returns an integer score; >=2 is considered suspicious.
    """
    patterns = [
        "registration fee",
        "security deposit",
        "processing fee",
        "activation fee",
        "training fee",
        "payable",
        "western union",
        "moneygram",
        "skrill",
        "bitcoin",
        "gift card",
        "whatsapp",
        "telegram",
        "viber",
        "wechat",
        "contact on whatsapp",
        "earn money fast",
        "guaranteed profit",
        "no experience required",
        "work 2 hours",
        "daily payout",
        "click ads",
        "data entry work from home",
        "remote job only",
        "urgent hiring",
        "limited seats",
        "pay to join",
        "investment required",
        "upfront payment",
        # Bengali cues
        "টাকা",
        "ফি",
        "রেজিস্ট্রেশন",
        "যোগাযোগ করুন",
        "হোয়াটসঅ্যাপ",
        "হোয়াটস অ্যাপ",
        "তাৎক্ষণিক",
        "জরুরি",
    ]
    score = sum(1 for p in patterns if p in cleaned)

    # add signal for personal email contact in description
    if "gmail.com" in cleaned or "yahoo.com" in cleaned or "outlook.com" in cleaned or "hotmail.com" in cleaned:
        score += 1
    # phone number patterns
    if any(token in cleaned for token in [" call ", " sms ", " text ", " whatsapp "]) and any(ch.isdigit() for ch in cleaned):
        score += 1
    # excessive money mentions
    if "$" in cleaned or "tk" in cleaned or "usd" in cleaned:
        score += 1
    return score


def structured_features(raw_text: str) -> Dict[str, float]:
    """
    Compute structured signals from raw text for explainability and hybrid scoring.
    """
    cleaned = preprocess_text(raw_text)
    signals = {
        "has_free_email": int(any(d in cleaned for d in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "proton.me"])),
        "has_crypto": int(any(k in cleaned for k in ["bitcoin", "crypto", "usdt", "btc", "eth", "tether"])),
        "has_payment": int(any(k in cleaned for k in ["registration fee", "security deposit", "processing fee", "activation fee", "pay to join", "upfront payment", "visa processing"])),
        "link_count": raw_text.count("http"),
        "digit_count": sum(ch.isdigit() for ch in raw_text),
        "money_symbol": int(any(cur in raw_text.lower() for cur in ["$", "usd", "tk", "bdt", "rs", "inr"])),
        "length_chars": len(raw_text),
        "length_words": len(raw_text.split()),
    }
    return signals


# ---------------- Fraud Prediction ----------------
def predict_fraud(text: str):
    details = predict_fraud_details(text)
    return details["label"]


def predict_fraud_details(text: str) -> Dict[str, object]:
    cleaned = preprocess_text(text)
    if not cleaned:
        return {"label": "Unknown", "probability": 0.0, "reasons": ["Empty text"]}

    # Run inexpensive rule/keyword analysis for explainability regardless of ML path
    keyword_hit = [k for k in FRAUD_KEYWORDS if k in cleaned]
    rule_score = rule_based_score(cleaned)
    struct = structured_features(text)

    # Simple heuristic probability from rule score only
    rule_prob = min(1.0, 0.25 * rule_score)  # each hit ~25% up to 100%
    if struct["has_crypto"]:
        rule_prob = max(rule_prob, 0.6)
    if struct["has_payment"]:
        rule_prob = max(rule_prob, 0.6)
    if struct["has_free_email"] and struct["link_count"] > 0:
        rule_prob = max(rule_prob, 0.5)

    combined_prob = rule_prob
    label = "Fake" if combined_prob >= 0.5 or rule_score >= 1 else "Real"

    # Try classical ML pipeline (tf-idf + logistic) if available
    ml_prob: Optional[float] = None
    pipeline = _load_trained_pipeline()
    if pipeline is not None and np is not None:
        try:
            proba = pipeline.predict_proba([text])[0]
            classes = pipeline.classes_
            fake_idx = int(np.where(classes == 1)[0][0]) if 1 in classes else int(np.argmax(classes))
            fake_prob = float(proba[fake_idx])
            ml_prob = fake_prob
            combined_prob = max(combined_prob, fake_prob)
            label = "Fake" if combined_prob >= 0.5 or rule_score >= 1 else "Real"
        except Exception:
            ml_prob = None

    # Ensure the UI always shows a small non-zero confidence for safe cases
    display_prob = combined_prob
    if label == "Real":
        display_prob = max(1 - combined_prob, 0.05) if ml_prob is not None else max(combined_prob, 0.05)

    reasons: List[str] = []
    if ml_prob is not None:
        reasons.append(f"Dataset-trained model score: {round(ml_prob * 100, 1)}% fake probability")
    if keyword_hit:
        reasons.append(f"Suspicious keywords: {', '.join(keyword_hit[:5])}")
    if struct["has_free_email"]:
        reasons.append("Uses free email domain")
    if struct["has_payment"]:
        reasons.append("Mentions payment/fee requirement")
    if struct["has_crypto"]:
        reasons.append("Crypto payment references")
    if struct["link_count"] > 0:
        reasons.append("Contains external links")
    if struct["digit_count"] > 10:
        reasons.append("Many numeric tokens (possible phone/IDs)")
    if struct["money_symbol"]:
        reasons.append("Mentions currency/salary terms")

    model_info = {
        "using_ml": ml_prob is not None,
        "trained_on": MODEL_METADATA.get("trained_on"),
        "dataset_name": MODEL_METADATA.get("dataset_name", "Fake Job Postings"),
        "dataset_rows": MODEL_METADATA.get("rows"),
        "fraudulent_rows": MODEL_METADATA.get("fraudulent_rows"),
        "version": MODEL_METADATA.get("version", "1.0"),
        "metrics": MODEL_METADATA.get("metrics", {}),
        "model_type": MODEL_METADATA.get("model_type"),
    }

    return {
        "label": label,
        "probability": round(display_prob * 100, 2),
        "display_probability": round(display_prob * 100, 2),
        "rule_prob": round((ml_prob if ml_prob is not None else rule_prob) * 100, 2),
        "reasons": reasons or ["No high-risk signals detected"],
        "rule_score": rule_score,
        "keywords": keyword_hit,
        "model_info": model_info,
    }
