# main.py
# HyperCore GH-OS - Python ML Service v2.1 (SANITIZED)
# - Strict data integrity
# - Safe logistic regression handling
# - JSON-safe (no inf / NaN)
# - roc_curve_data / pr_curve_data keys for Base44 compatibility

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

app = FastAPI(title="HyperCore GH-OS ML Service", version="2.1.0")


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
    roc_curve_data: Dict[str, List[float]]
    pr_curve_data: Dict[str, List[float]]
    feature_importance: List[FeatureImportance]
    dropped_features: List[str] = []


# ---------- Helpers to keep JSON SAFE ----------

def _sanitize_scalar(x: float) -> float:
    """Return a JSON-safe float (no inf/NaN)."""
    x = float(x)
    if np.isfinite(x):
        return x
    return 0.0


def _sanitize_array(arr: np.ndarray) -> List[float]:
    """Convert numpy array to list of JSON-safe floats."""
    return [float(v) if np.isfinite(v) else 0.0 for v in arr]


# ---------- Utility Functions ----------

def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def clean_feature_matrix(df: pd.DataFrame, label_column: str):
    """
    - Keeps only numeric feature columns
    - Drops constant / zero-variance columns (no signal)
    - Does NOT alter biomarker values
    """
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    # Convert label to numeric (must be 0/1 for binary classification)
    df[label_column] = pd.to_numeric(df[label_column], errors="raise")

    y = df[label_column].values
    X_raw = df.drop(columns=[label_column])

    # Keep only numeric columns
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns found for analysis.")

    X = X_raw[numeric_cols].copy()

    # Drop zero-variance / constant columns (no diagnostic signal)
    variances = X.var()
    keep_cols = [col for col in numeric_cols if variances[col] > 0]
    dropped_cols = [col for col in numeric_cols if col not in keep_cols]

    if not keep_cols:
        raise ValueError("All feature columns have zero variance; cannot fit a model.")

    X = X[keep_cols]

    return X, y, keep_cols, dropped_cols


def logistic_regression_analysis(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    """
    Runs logistic regression with:
    - strict data integrity (no artificial "cleaning")
    - safe handling for small or imbalanced datasets
    - JSON-safe outputs (no inf / NaN)
    """
    X, y, feature_cols, dropped_cols = clean_feature_matrix(df, label_column)

    n_samples = len(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) < 2:
        raise ValueError("Label column must contain at least two classes (e.g., 0 and 1).")

    # Decide split strategy
    # For very small datasets, avoid stratified split & keep more data in training
    if n_samples < 30 or class_counts.min() < 3:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback: non-stratified split if stratification fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

    # Fit logistic regression model
    try:
        model = LogisticRegression(max_iter=1000, solver="liblinear")
        model.fit(X_train, y_train)
    except Exception as e:
        raise ValueError(f"Logistic regression failed: {e}")

    # Predictions
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        raise ValueError(f"Failed to compute prediction probabilities: {e}")

    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
    except Exception as e:
        raise ValueError(f"Failed to compute ROC/PR metrics: {e}")

    acc = accuracy_score(y_test, y_pred)
    sens_spec = compute_sensitivity_specificity(y_test, y_pred)

    # ROC & PR curves
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)

    # Coefficients
    coef = model.coef_[0]
    coefficients = {feature: float(weight) for feature, weight in zip(feature_cols, coef)}

    # Feature importance as |coef| normalized
    abs_coef = np.abs(coef)
    if abs_coef.sum() > 0:
        importance = abs_coef / abs_coef.sum()
    else:
        importance = np.zeros_like(abs_coef)

    feature_importance = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(feature_cols, importance)
    ]

    # Build JSON-safe result
    result = {
        "metrics": {
            "roc_auc": _sanitize_scalar(roc_auc),
            "pr_auc": _sanitize_scalar(pr_auc),
            "accuracy": _sanitize_scalar(acc),
            "sensitivity": _sanitize_scalar(sens_spec["sensitivity"]),
            "specificity": _sanitize_scalar(sens_spec["specificity"]),
        },
        "coefficients": coefficients,
        "roc_curve_data": {
            "fpr": _sanitize_array(fpr),
            "tpr": _sanitize_array(tpr),
            "thresholds": _sanitize_array(roc_thresh),
        },
        "pr_curve_data": {
            "precision": _sanitize_array(prec),
            "recall": _sanitize_array(rec),
            "thresholds": _sanitize_array(pr_thresh),
        },
        "feature_importance": feature_importance,
        "dropped_features": dropped_cols,
    }

    return result


# ---------- Endpoint ----------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Accepts a CSV string and label column, returns real metric results
    for a binary classification problem.
    """
    try:
        df = pd.read_csv(io.StringIO(request.csv))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        result = logistic_regression_analysis(df, request.label_column)
    except ValueError as ve:
        # Data / ML-related issue → 400 with explanation
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        # Unexpected model error → 500
        raise HTTPException(status_code=500, detail=f"Unexpected analysis error: {e}")

    # Pydantic enforces output shape here
    return AnalyzeResponse(**result)


# ---------- Local debug ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
