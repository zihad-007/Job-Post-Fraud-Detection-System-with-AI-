"""
Anomaly detection for job fraud using Isolation Forest + rule-based checks.

Primary entrypoints:
- `train_isolation_forest(normal_jobs)` -> (model, scaler)
- `score_job(job, model, scaler)` -> result dict with anomaly score/flags
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------- Helpers ---------------- #

RELOCATE_REGEX = re.compile(r"relocat(e|ion)", re.IGNORECASE)


def _salary_midpoint(job: Dict) -> float:
    """
    Extract a single salary value.
    Accepts:
      - tuple/list (min, max)
      - dict with 'min'/'max'
      - scalar
    """
    salary = job.get("salary_range")
    if salary is None:
        return 0.0
    if isinstance(salary, (list, tuple)) and len(salary) == 2:
        low, high = salary
    elif isinstance(salary, dict):
        low, high = salary.get("min"), salary.get("max")
    else:
        low = high = salary
    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        return 0.0
    return (low + high) / 2.0 if high is not None else float(low)


def _is_remote(job: Dict) -> int:
    job_type = str(job.get("job_type", "")).lower()
    return 1 if "remote" in job_type else 0


def _experience(job: Dict) -> float:
    try:
        return float(job.get("experience_required", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _build_feature_vector(job: Dict) -> np.ndarray:
    return np.array(
        [
            _salary_midpoint(job),
            _experience(job),
            _is_remote(job),
        ],
        dtype=float,
    )


# ---------------- Rule Checks ---------------- #

def rule_checks(job: Dict) -> List[str]:
    """
    Evaluate deterministic rules; return list of triggered rule ids.
    """
    rules = []
    salary = _salary_midpoint(job)
    exp_years = _experience(job)
    desc = job.get("description", "") or ""
    is_remote = bool(_is_remote(job))

    # Extreme salary values
    if salary > 500_000:
        rules.append("salary_extremely_high")
    if 0 < salary < 10_000:
        rules.append("salary_extremely_low")

    # Entry-level with very high salary
    if exp_years <= 1 and salary >= 200_000:
        rules.append("entry_level_with_high_salary")

    # Senior experience but low pay
    if exp_years >= 10 and 0 < salary < 30_000:
        rules.append("senior_with_low_salary")

    # Remote role demanding relocation
    if is_remote and RELOCATE_REGEX.search(desc):
        rules.append("remote_with_relocation_requirement")

    return rules


# ---------------- Model Training ---------------- #

def train_isolation_forest(
    normal_jobs: Iterable[Dict],
    contamination: float = 0.05,
    random_state: int = 42,
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Fit Isolation Forest on normal job data.

    Returns model and scaler for reuse in scoring.
    """
    vectors = np.vstack([_build_feature_vector(job) for job in normal_jobs])
    scaler = StandardScaler()
    X = scaler.fit_transform(vectors)
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    model.fit(X)
    return model, scaler


# ---------------- Scoring ---------------- #

def score_job(job: Dict, model: IsolationForest, scaler: StandardScaler) -> Dict[str, object]:
    """
    Score a single job posting.
    Output format:
    {
      "anomaly_score": float,  # negative = more anomalous per isolation forest
      "is_anomalous": bool,
      "rule_flags": [str, ...]
    }
    """
    vec = _build_feature_vector(job).reshape(1, -1)
    X = scaler.transform(vec)
    score = float(model.decision_function(X)[0])  # higher = more normal
    is_anom = bool(model.predict(X)[0] == -1)

    flags = rule_checks(job)
    # Combine model + rules: if rules triggered, force anomalous
    if flags:
        is_anom = True

    return {
        "anomaly_score": score,
        "is_anomalous": is_anom,
        "rule_flags": flags,
    }


__all__ = [
    "train_isolation_forest",
    "score_job",
    "rule_checks",
]
