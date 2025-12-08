# ================================================================
# main.py — DiviScan HyperCore Unified Backend v4 (Base44 Compatible)
# ================================================================
# Provides:
#   ✔ /analyze           – ML analysis API (pharma + hospital aware)
#   ✔ /generate_report   – Mode-aware narrative engine
#   ✔ /astra             – ASTRA persona logic
#   ✔ /health            – Railway uptime check
#
# Works with:
#   ✔ Base44 MVP frontend
#   ✔ Railway hosting
#   ✔ Future Firebase integration
#
# IMPORTANT:
#   This version FIXES Base44 JSON key requirements:
#       roc_curve_data  (NOT roc_curve)
#       pr_curve_data   (NOT pr_curve)
#
# ================================================================

import io
import json
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from openai import OpenAI

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

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI()

# -----------------------------
# FastAPI Application
# -----------------------------
app = FastAPI(
    title="DiviScan HyperCore Unified Backend",
    version="4.0.1",
    description="Unified backend for DiviScan HyperCore — now Base44 compatible.",
)

# -----------------------------
# Internal Knowledge Modules
# -----------------------------
COMPETITIVE_INTEL_CORE = """
DiviScan Competitive Intelligence Core

DiviScan covers all diagnostic domains simultaneously:
- Molecular diagnostics
- Multi-omics: genomics, proteomics, metabolomics, microbiomics
- Immune profiling
- Emerging disease detection + novelty scoring
- Trial intelligence + trial rescue

Main competitor gaps:
Cue: single pathogen only
Viome: microbiome only
Everlywell: mail-in labs only
Abbott: lab-based, slow
Freenome/Tempus: sequencing only, no point-of-care
Apple/Oura: physiology only

DiviScan advantages:
Real-time multi-omics; unknown phenotype detection; confounder intelligence;
federated learning; trial rescue engine; diagnostic OS architecture.
"""

CATEGORY_CREATION_CORE = """
Market Category: Multi-Omic Diagnostic OS (Diag-OS)

DiviScan is not a test or device — it is a diagnostic operating system.

Pillars:
- Multi-omic fusion
- CRISPR diagnostic analytics
- Quantum sensing (future hardware)
- Emerging phenotype detection
- Trial rescue intelligence
- Global federated learning
- Outbreak detection intelligence

Tagline:
"The Diagnostic OS for Humanity."
"""

HOSPITAL_PAIN_POINTS = """
Hospital Pain Points:
- Readmission risk and manual scoring overhead
- Physician burnout from fragmented diagnostic data
- Slow labs delaying treatment
- No multi-omic signal fusion
- No early-warning detection for organ-system drift
"""

PHARMA_PAIN_POINTS = """
Pharma Pain Points:
- Trial failures due to poor biomarker stratification
- Hidden confounders
- No emerging phenotype detection
- Slow trial forensics
"""

# -----------------------------
# Request / Response Models
# -----------------------------
class AnalyzeRequest(BaseModel):
    csv: str
    label_column: str
    pilot_mode: Optional[str] = "pharma"  # "pharma" or "hospital"


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
    pilot_mode: str


class ReportRequest(BaseModel):
    analysis_json: Dict[str, Any]
    mode: str  # "pharma", "hospital", "executive", etc.


class ReportResponse(BaseModel):
    narrative: str


class ASTRARequest(BaseModel):
    question: str
    mode: str


class ASTRAResponse(BaseModel):
    answer: str


# ================================================================
# UTILITY FUNCTIONS
# ================================================================
def compute_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }


def clean_feature_matrix(df: pd.DataFrame, label_column: str):
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found.")

    df[label_column] = pd.to_numeric(df[label_column], errors="raise")
    y = df[label_column].values
    X_raw = df.drop(columns=[label_column])

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Dataset has no numeric feature columns.")

    X = X_raw[numeric_cols].copy()
    variances = X.var()

    keep_cols = [c for c in numeric_cols if variances[c] > 0]
    dropped_cols = [c for c in numeric_cols if variances[c] == 0]

    if not keep_cols:
        raise ValueError("All numeric features have zero variance.")

    return X[keep_cols], y, keep_cols, dropped_cols


def logistic_regression_analysis(df: pd.DataFrame, label_column: str):
    X, y, features, dropped_cols = clean_feature_matrix(df, label_column)

    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Label must contain at least two classes (0 & 1).")

    # fallback for small datasets
    if len(y) < 30 or counts.min() < 3:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        except:
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

    abs_coef = np.abs(coef)
    importance = abs_coef / abs_coef.sum() if abs_coef.sum() else np.zeros_like(abs_coef)

    feature_importance = [
        {"feature": f, "importance": float(i)}
        for f, i in zip(features, importance)
    ]

    # FINAL — MUST MATCH Base44 KEY NAMES 🚨
    return {
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "coefficients": {f: float(c) for f, c in zip(features, coef)},
        "roc_curve_data": {   # FIXED KEY
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": [float(x) for x in roc_thresh.tolist()],
        },
        "pr_curve_data": {    # FIXED KEY
            "precision": prec.tolist(),
            "recall": rec.tolist(),
            "thresholds": [float(x) for x in pr_thresh.tolist()],
        },
        "feature_importance": feature_importance,
        "dropped_features": dropped_cols,
    }


# ================================================================
# /analyze ENDPOINT
# ================================================================
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        df = pd.read_csv(io.StringIO(req.csv))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV error: {e}")

    try:
        result = logistic_regression_analysis(df, req.label_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    result["pilot_mode"] = req.pilot_mode or "pharma"

    return AnalyzeResponse(**result)


# ================================================================
# /generate_report ENDPOINT
# ================================================================
def build_report_prompt(analysis_json, mode):
    mode = mode.lower()

    if mode == "pharma":
        context = PHARMA_PAIN_POINTS + "\n" + COMPETITIVE_INTEL_CORE + "\n" + CATEGORY_CREATION_CORE
    elif mode == "hospital":
        context = HOSPITAL_PAIN_POINTS + "\n" + COMPETITIVE_INTEL_CORE + "\n" + CATEGORY_CREATION_CORE
    else:
        context = COMPETITIVE_INTEL_CORE + "\n" + CATEGORY_CREATION_CORE

    return f"""
You are HyperCore — DiviScan's Diagnostic OS narrative engine.

Generate a {mode.upper()}-grade diagnostic intelligence narrative.

Internal context:
{context}

Analysis JSON:
{json.dumps(analysis_json, indent=2)}

Use these sections:

🔍 Key Insights
📈 Analytical Model
💡 Missed Opportunities
🧠 HyperCore Advantage
🧾 Recommended Next Steps
"""


@app.post("/generate_report", response_model=ReportResponse)
def generate_report(req: ReportRequest):
    prompt = build_report_prompt(req.analysis_json, req.mode)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        narrative = resp.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    return ReportResponse(narrative=narrative)


# ================================================================
# /astra ENDPOINT
# ================================================================
def build_astra_prefix(mode: str):
    m = mode.lower()
    if m in ["pharma", "trial"]:
        return "You are ASTRA in Pharma Mode — expert in biomarkers, trial rescue, and phenotyping."
    if m in ["hospital", "clinical"]:
        return "You are ASTRA in Hospital Mode — expert in diagnostic workflows and readmission risk."
    if m == "investor":
        return "You are ASTRA in Investor Mode — focus on ROI and defensibility, no SAFE internals."
    if m == "regulatory":
        return "You are ASTRA in Regulatory Mode — cautious, compliant, no overclaims."
    if m in ["unknown", "emerging"]:
        return "You are ASTRA in Unknown Disease Mode — describe novelty patterns carefully."
    return "You are ASTRA — DiviScan's multimodal clinical-intelligence assistant."


@app.post("/astra", response_model=ASTRAResponse)
def astra(req: ASTRARequest):
    prefix = build_astra_prefix(req.mode)

    prompt = f"""
{prefix}

Internal Knowledge:
{COMPETITIVE_INTEL_CORE}

Question:
{req.question}

Provide a precise, professional answer.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        answer = resp.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASTRA error: {e}")

    return ASTRAResponse(answer=answer)


# ================================================================
# /health ENDPOINT
# ================================================================
@app.get("/health")
def health():
    return {"status": "ok", "service": "DiviScan HyperCore Backend", "version": "4.0.1"}


# ================================================================
# LOCAL DEV ENTRY POINT
# ================================================================
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

