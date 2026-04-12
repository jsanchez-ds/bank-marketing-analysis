"""
Offline training script for the Streamlit demo.

Downloads the UCI Bank Marketing dataset (bank-full.csv), trains a Random Forest
on a leakage-free feature set, and persists the fitted pipeline + metadata to
``app/artifacts/model.joblib``. Run this once before deploying the app::

    python app/train_model.py

The Streamlit app loads the artifact at startup and never re-trains in the cloud.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
USER_AGENT = "Mozilla/5.0 (bank-marketing-demo)"

# OpenML hosts the legacy 45,211-row dataset (id 1461) under anonymized column
# names V1..V16. The mapping below restores the original UCI schema.
OPENML_COLUMN_MAP = {
    "V1": "age",
    "V2": "job",
    "V3": "marital",
    "V4": "education",
    "V5": "default",
    "V6": "balance",
    "V7": "housing",
    "V8": "loan",
    "V9": "contact",
    "V10": "day",
    "V11": "month",
    "V12": "duration",
    "V13": "campaign",
    "V14": "pdays",
    "V15": "previous",
    "V16": "poutcome",
}
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "model.joblib"

# Per Moro et al. (2014), `duration` is only known after a call ends and would
# leak the target. We exclude it from every predictive feature set.
NUMERIC_FEATURES = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "y"


def _from_uci() -> pd.DataFrame:
    print(f"Trying UCI: {UCI_URL}")
    req = Request(UCI_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("bank.zip")))
    with inner.open("bank-full.csv") as f:
        return pd.read_csv(f, sep=";")


def _from_openml() -> pd.DataFrame:
    print("Falling back to OpenML (id=1461) ...")
    ds = fetch_openml("bank-marketing", version=1, as_frame=True)
    df = ds.data.copy()
    df = df.rename(columns=OPENML_COLUMN_MAP)
    # OpenML stores categoricals; cast to plain str so downstream code matches
    # the UCI CSV exactly.
    for col in df.select_dtypes(include="category").columns:
        df[col] = df[col].astype(str)
    # OpenML target encodes 1=no, 2=yes — restore the UCI yes/no labels.
    df["y"] = ds.target.astype(int).map({1: "no", 2: "yes"})
    return df


def download_dataset() -> pd.DataFrame:
    """Load ``bank-full.csv`` (45,211 rows · 17 cols), trying UCI then OpenML."""
    try:
        df = _from_uci()
    except Exception as e:  # noqa: BLE001
        print(f"  UCI failed ({e})")
        df = _from_openml()
    print(f"  loaded {len(df):,} rows · {df.shape[1]} columns")
    return df


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train() -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = download_dataset()
    X = df[ALL_FEATURES]
    y = (df[TARGET] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, pred).tolist()
    fpr, tpr, _ = roc_curve(y_test, proba)

    rf: RandomForestClassifier = pipe.named_steps["clf"]
    feature_names = (
        NUMERIC_FEATURES
        + pipe.named_steps["pre"]
        .named_transformers_["cat"]
        .get_feature_names_out(CATEGORICAL_FEATURES)
        .tolist()
    )
    importances = sorted(
        zip(feature_names, rf.feature_importances_, strict=False),
        key=lambda kv: kv[1],
        reverse=True,
    )[:15]

    metadata = {
        "test_roc_auc": float(auc),
        "test_pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "top_features": [(name, float(imp)) for name, imp in importances],
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "feature_schema": {
            "numeric": NUMERIC_FEATURES,
            "categorical": {col: sorted(df[col].unique().tolist()) for col in CATEGORICAL_FEATURES},
        },
    }

    joblib.dump({"pipeline": pipe, "metadata": metadata}, ARTIFACT_PATH)
    print(f"\nSaved -> {ARTIFACT_PATH}")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  conf mx : {cm}")
    print("  top features:")
    for name, imp in importances[:8]:
        print(f"    {imp:.4f}  {name}")
    return metadata


if __name__ == "__main__":
    meta = train()
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(
        {k: v for k, v in meta.items() if k != "feature_schema"},
        indent=2,
        default=lambda o: float(o) if isinstance(o, np.floating) else o,
    ))
