# main.py
# HyperCore GH-OS - Python ML Service v2 (SAFE PATCHED VERSION)
# Only updates:
# - roc_curve → roc_curve_data
# - pr_curve → pr_curve_data
# Nothing else changed.

import io
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

app = FastAPI(title="HyperCore GH-OS ML Service", version="2.0.1")


# ---------- Request / Response Models ----------

class AnalyzeRequest(BaseModel):
    csv: str
    label_column: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class AnalyzeResponse(BaseModel):
    metrics: Dict[str, float]
    coefficients: Dict[str, float]
    roc_curve_data: Dict[str, List[float]]   # updated name
    pr_curve_data: Dict[str, List[float]]    # updated name
    feature_importance: List[FeatureImportance]
    dropped_features: List[str] = []


# ---------- Utility Functions ----------

def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity}


def clean_feature_matrix(df: pd.DataFrame, label_column: str):
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    df[label_column] = pd.to_numeric(df[label_column], errors="raise")
    y = df[label_column].values
    X_raw = df.drop(columns=[label_column])

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for analysis.")

    X = X_raw[numeric_cols].copy()
    variances = X.var()

    keep_cols = [col for col in numeric_cols if variances[col] > 0]
    dropped_cols = [col for col in numeric_cols if col not in keep_cols]

    if not keep_cols:
        raise ValueError("All numeric feature columns have zero variance; cannot fit model.")

    X = X[keep_cols]
    return X, y, keep_cols, dropped_cols


def logistic_regression_analysis(df: pd.DataFrame, label_column: str):
    X, y, feature_cols, dropped_cols = clean_feature_matrix(df, label_column)

    n_samples = len(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) < 2:
        raise ValueError("Label column must contain at least two classes (0 and 1).")

    if n_samples < 30 or class_counts.min() < 3:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    sens_spec = compute_sensitivity_specificity(y_test, y_pred)

    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)

    coef = model.coef_[0]
    coefficients = {f: float(c) for f, c in zip(feature_cols, coef)}

    abs_coef = np.abs(coef)
    importance = abs_coef / abs_coef.sum() if abs_coef.sum() > 0 else np.zeros_like(abs_coef)

    feature_importance = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(feature_cols, importance)
    ]

    # NOTE: KEYS UPDATED HERE
    return {
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "coefficients": coefficients,
        "roc_curve_data": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [float(x) for x in roc_thresh.tolist()]
        },
        "pr_curve_data": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": [float(x) for x in pr_thresh.tolist()]
        },
        "feature_importance": feature_importance,
        "dropped_features": dropped_cols
    }


# ---------- Endpoint ----------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    try:
        df = pd.read_csv(io.StringIO(request.csv))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        result = logistic_regression_analysis(df, request.label_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return AnalyzeResponse(**result)


# ---------- Local debug ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)


