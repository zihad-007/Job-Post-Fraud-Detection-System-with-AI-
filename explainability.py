"""
Explainable AI utilities for job fraud detection models.

Supports SHAP (preferred) with a LIME fallback. Works with scikit-learn style
models that expose `predict_proba` or `decision_function`.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover
    LimeTabularExplainer = None


def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # binary case: convert to pseudo-prob via sigmoid
        if scores.ndim == 1:
            probs_pos = 1 / (1 + np.exp(-scores))
            return np.vstack([1 - probs_pos, probs_pos]).T
        # multi-class: softmax
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exps / exps.sum(axis=1, keepdims=True)
    raise ValueError("Model must implement predict_proba or decision_function")


def explain_with_shap(model, background: np.ndarray, sample: np.ndarray):
    """
    Compute SHAP values for a single sample using TreeExplainer if possible,
    else KernelExplainer.
    """
    if shap is None:
        return None
    try:
        explainer = shap.TreeExplainer(model, data=background, feature_perturbation="tree_path_dependent")
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(sample)
    return shap_values


def explain_with_lime(model, background: np.ndarray, sample: np.ndarray, feature_names: List[str], class_names: List[str]):
    if LimeTabularExplainer is None:
        return None
    explainer = LimeTabularExplainer(
        training_data=background,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        verbose=False,
    )
    exp = explainer.explain_instance(sample[0], model.predict_proba, num_features=len(feature_names))
    return exp.as_list(label=exp.available_labels()[0])


def explain_prediction(
    model,
    sample_vector: np.ndarray,
    feature_names: List[str],
    class_labels: List[str],
    background: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Generate prediction + per-feature contributions.
    """
    if sample_vector.ndim == 1:
        sample_vector = sample_vector.reshape(1, -1)

    probs = _predict_proba(model, sample_vector)
    pred_idx = int(np.argmax(probs, axis=1)[0])
    prediction = class_labels[pred_idx] if class_labels else str(pred_idx)
    confidence = float(probs[0, pred_idx])

    contributions = {}

    # SHAP preferred
    shap_values = None
    if background is not None:
        shap_values = explain_with_shap(model, background, sample_vector)

    if shap_values is not None:
        # shap_values could be list per class
        sv = shap_values[pred_idx] if isinstance(shap_values, list) else shap_values
        for name, value in zip(feature_names, sv[0]):
            contributions[name] = float(value)
    else:
        # LIME fallback
        lime_pairs = explain_with_lime(model, background or sample_vector, sample_vector, feature_names, class_labels)
        if lime_pairs:
            for name, value in lime_pairs:
                contributions[name] = float(value)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "feature_contributions": contributions,
    }


__all__ = ["explain_prediction", "explain_with_shap", "explain_with_lime"]
