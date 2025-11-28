# main.py
# HyperCore GH-OS - Python ML Service v1 (FIXED Pydantic Model)

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

app = FastAPI(title="HyperCore GH-OS ML Service", version="1.0.1")


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
    roc_curve: Dict[str, List[float]]
    pr_curve: Dict[str, List[float]]
    feature_importance: List[FeatureImportance]


# ---------- Util Functions ----------

def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity}


def logistic_regression_analysis(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not in dataset.")

    # convert label to numeric
    df[label_column] = pd.to_numeric(df[label_column], errors="raise")

    # Separate features and label
    y = df[label_column].values
    X = df.drop(columns=[label_column])

    # Only numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found.")

    # FIX: small dataset fallback (your sample had 5 rows)
    if df[label_column].value_counts().min() < 2:
        test_size = 0.5
        stratify = None
    else:
        test_size = 0.3
        stratify = y

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    # Model
    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    sens_spec = compute_sensitivity_specificity(y_test, y_pred)

    # Curves
    fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
    prec, rec, pr_thresh = precision_recall_curve(y_test, y_prob)

    # Coefficients
    coef = model.coef_[0]
    coefficients = {feature: float(weight) for feature, weight in zip(numeric_cols, coef)}

    # Importance
    abs_coef = np.abs(coef)
    if abs_coef.sum() > 0:
        importance = abs_coef / abs_coef.sum()
    else:
        importance = np.zeros_like(abs_coef)

    feature_importance = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(numeric_cols, importance)
    ]

    return {
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "coefficients": coefficients,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [float(x) for x in roc_thresh.tolist()],
        },
        "pr_curve": {
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": [float(x) for x in pr_thresh.tolist()],
        },
        "feature_importance": feature_importance,
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
        raise HTTPException(status_code=400, detail=f"Analysis error: {e}")

    return AnalyzeResponse(**result)


# ---------- Local debug ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

