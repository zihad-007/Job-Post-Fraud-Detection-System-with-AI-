"""
Train a text classifier on the Kaggle Fake Job Postings dataset.

Run:
    python train_model.py

Outputs:
    models/job_fraud_model.joblib      - scikit-learn Pipeline (Tfidf + LogisticRegression)
    models/model_metadata.json         - training metadata (dataset stats + metrics)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Mapping

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# Multiple data sources (all optional, only loaded when present)
DATA_SOURCES: List[Mapping[str, object]] = [
    {
        "path": Path("data/fake_job_postings.csv"),
        "text_cols": ["title", "company_profile", "description", "requirements", "benefits"],
        "label_col": "fraudulent",
        "name": "Kaggle Fake Job Postings",
    },
    {
        # Provided by user (absolute path because it sits outside repo)
        "path": Path(r"d:/SWE Design Capstone Project/dataset/fake_real_job_postings_3000x25.csv"),
        "text_cols": ["job_title", "job_description", "requirements", "benefits", "company_profile"],
        "label_col": "is_fake",
        "name": "Fake/Real Job Postings 3000x25",
    },
    {
        # Provided by user (absolute path because it sits outside repo)
        "path": Path(r"d:/SWE Design Capstone Project/dataset/Fake Postings.csv"),
        "text_cols": ["title", "description", "requirements", "company_profile", "benefits"],
        "label_col": "fraudulent",
        "name": "Synthetic Fake Postings",
    },
]
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "job_fraud_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"


def load_datasets(sources: Iterable[Mapping[str, object]]) -> pd.DataFrame:
    """
    Load and unify multiple job-posting datasets. Each source may have a different
    schema, so we map configurable text/label columns into a common format.
    Returns a concatenated dataframe with columns: text, fraudulent.
    """
    frames = []
    used_sources: List[str] = []

    for src in sources:
        path = Path(src["path"])  # type: ignore[arg-type]
        if not path.exists():
            continue

        text_cols: List[str] = list(src["text_cols"])  # type: ignore[assignment]
        label_col: str = str(src["label_col"])  # type: ignore[assignment]

        df = pd.read_csv(path)
        missing = [col for col in text_cols + [label_col] if col not in df.columns]
        if missing:
            raise ValueError(f"Columns {missing} missing in dataset {path}")

        df[text_cols] = df[text_cols].fillna("")
        df["text"] = df[text_cols].agg(" ".join, axis=1).str.strip()
        df = df[df["text"].str.len() > 20]
        df = df.drop_duplicates(subset=["text"])
        df["fraudulent"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)
        frames.append(df[["text", "fraudulent"]])
        used_sources.append(str(src.get("name", path.name)))

    if not frames:
        raise FileNotFoundError("No datasets found. Please check the configured paths.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["text"])
    combined.attrs["sources"] = used_sources
    return combined


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    n_jobs=-1,
                    solver="liblinear",
                ),
            ),
        ]
    )


def main() -> None:
    df = load_datasets(DATA_SOURCES)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["fraudulent"],
        test_size=0.2,
        random_state=42,
        stratify=df["fraudulent"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    used_sources = df.attrs.get("sources", [])
    metadata = {
        "dataset_name": " + ".join(used_sources) if used_sources else "Job Postings (combined)",
        "dataset_paths": [str(src["path"]) for src in DATA_SOURCES if Path(src["path"]).exists()],
        "rows": int(len(df)),
        "fraudulent_rows": int(df["fraudulent"].sum()),
        "real_rows": int((df["fraudulent"] == 0).sum()),
        "train_split": 0.8,
        "version": "2.0",
        "trained_on": datetime.now(timezone.utc).isoformat(),
        "model_type": "LogisticRegression + TF-IDF (1-2 grams)",
        "metrics": {
            "accuracy": round(float(acc), 4),
            "f1": round(float(f1), 4),
        },
    }
    META_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Saved model to", MODEL_PATH)
    print("Saved metadata to", META_PATH)
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    main()
