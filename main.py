# main.py
# DiviScan HyperCore – Unified Backend v4
#
# Supports:
# - /analyze          → ML analysis (pharma + hospital aware via pilot_mode)
# - /generate_report  → HyperCore narrative (pharma, hospital, exec, investor, etc.)
# - /astra            → ASTRA persona responses
# - /health           → health check for Railway / monitoring
#
# Designed for:
# - Railway deployment
# - Base44 front-end integration
# - Multiple pilot types (pharma, hospital, government in future)

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
# OpenAI client (uses env var OPENAI_API_KEY)
# -----------------------------

client = OpenAI()

app = FastAPI(
    title="DiviScan HyperCore Unified Backend",
    version="4.0.0",
    description=(
        "Unified backend for DiviScan HyperCore. "
        "Supports pharma + hospital pilots via pilot_mode and exposes ML + narrative APIs."
    ),
)

# -----------------------------
# Internal knowledge (competitive + category framing)
# -----------------------------

COMPETITIVE_INTEL_CORE = """
DiviScan Competitive Intelligence Core

Diagnostic Domains:
1. Molecular diagnostics
2. Multi-omics (genomics, proteomics, metabolomics, microbiomics)
3. Immune profiling (cytokines + chemokines)
4. Emerging disease detection + novelty scoring
5. Clinical trial intelligence + trial rescue

Competitor gaps:
- Cue Health: single pathogen
- Viome: microbiome only
- Everlywell: mail-in lab tests only
- Abbott: lab-based, slower, not multi-omic
- Freenome/Tempus: central lab sequencing, no point-of-care
- Apple/Oura: surface physiology, no deep biomarkers

DiviScan advantages:
- Real-time multi-omics (saliva and beyond)
- On-device inference (future device phase)
- Unknown phenotype detection
- Confounder intelligence (iron-cytokine, microbiome-endocrine, etc.)
- Federated learning across sites
- Trial rescue engine (for pharma)
- Diagnostic OS framework (for hospitals, pharma, and government)
"""

CATEGORY_CREATION_CORE = """
Market Category Creation Engine

DiviScan is not a diagnostic device.
It is the world's first:
Multi-Omic Diagnostic Operating System (Diag-OS).

Pillars:
1. Multi-omic fusion
2. Quantum sensing (future hardware iteration)
3. CRISPR diagnostic analytics
4. Phenotype drift detection
5. Unknown-disease novelty scoring
6. Trial rescue intelligence
7. Federated multi-site learning
8. Global health surveillance

Tagline:
"The Diagnostic OS for Humanity."

Category framing:
- Not just a test → a diagnostic OS
- Not just a device → an intelligence layer
- Not just biomarker analysis → multi-omic reasoning
- Not just clinical analytics → trial and care-pathway rescue
- Not just outbreak detection → emerging disease pattern detection
"""

HOSPITAL_PAIN_POINTS = """
Hospital & Clinician Pain Points (Internal Knowledge):

- Delayed diagnosis and fragmented evaluation across specialties
- High readmission rates and costly manual risk-scoring programs
- Physician burnout and cognitive overload
- Lack of multi-omic signal linking inflammation, microbiome, endocrine, and metabolic axes
- Insufficient decision support for triage and risk stratification
"""

PHARMA_PAIN_POINTS = """
Pharma & Trial Pain Points (Internal Knowledge):

- 20–40% of trial budgets wasted due to poor biomarker stratification
- Confounders (iron, microbiome, co-medications) undetected in standard analyses
- Failure to detect emerging phenotypes and responder/non-responder subgroups
- Slow, manual, post-hoc trial forensics instead of real-time rescue intelligence
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
    roc_curve: Dict[str, List[float]]
    pr_curve: Dict[str, List[float]]
    feature_importance: List[FeatureImportance]
    dropped_features: List[str] = []
    pilot_mode: str


class ReportRequest(BaseModel):
    analysis_json: Dict[str, Any]
    mode: str  # "pharma", "hospital", "executive", "investor"


class ReportResponse(BaseModel):
    narrative: str


class ASTRARequest(BaseModel):
    question: str
    mode: str  # "pharma", "clinical", "hospital", "regulatory", "investor", "executive", "unknown"


class ASTRAResponse(BaseModel):
    answer: str


# -----------------------------
# Utility Functions
# -----------------------------

def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity}


def clean_feature_matrix(df: pd.DataFrame, label_column: str):
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    # Convert label to numeric
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


def logistic_regression_analysis(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    X, y, feature_cols, dropped_cols = clean_feature_matrix(df, label_column)

    n_samples = len(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) < 2:
        raise ValueError("Label column must contain at least two classes (e.g., 0 and 1).")

    if n_samples < 30 or class_counts.min() < 3:
        X_train, X_test, y_train, y_test = X, X, y, y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except Exception:
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
        "dropped_features": dropped_cols,
    }


# -----------------------------
# /analyze endpoint
# -----------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Core ML analysis endpoint.
    Works for both pharma and hospital pilots.
    pilot_mode tells the front-end how to interpret the output and which UI to show.
    """
    try:
        df = pd.read_csv(io.StringIO(request.csv))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing failed: {e}")

    try:
        result = logistic_regression_analysis(df, request.label_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Attach pilot_mode for UI routing
    result["pilot_mode"] = request.pilot_mode or "pharma"

    return AnalyzeResponse(**result)


# -----------------------------
# /generate_report endpoint
# -----------------------------

def build_report_prompt(analysis_json: Dict[str, Any], mode: str) -> str:
    """
    Builds a mode-aware narrative prompt for HyperCore.
    mode: "pharma", "hospital", "executive", "investor"
    """
    mode = mode.lower()

    if mode == "pharma":
        context = f"{PHARMA_PAIN_POINTS}\n\n{COMPETITIVE_INTEL_CORE}\n\n{CATEGORY_CREATION_CORE}"
        audience = (
            "Pharmaceutical R&D, clinical operations, and trial leadership. "
            "Focus on trial performance, biomarkers, confounders, emerging subgroups, and trial rescue."
        )
    elif mode == "hospital":
        context = f"{HOSPITAL_PAIN_POINTS}\n\n{COMPETITIVE_INTEL_CORE}\n\n{CATEGORY_CREATION_CORE}"
        audience = (
            "Hospital leadership, clinical heads, and quality/improvement teams. "
            "Focus on diagnostic clarity, readmission reduction, burnout relief, and organ-system risk."
        )
    elif mode == "executive":
        context = f"{COMPETITIVE_INTEL_CORE}\n\n{CATEGORY_CREATION_CORE}"
        audience = (
            "Health system executives. Focus on ROI, strategic differentiation, and system-wide impact."
        )
    else:
        context = f"{COMPETITIVE_INTEL_CORE}\n\n{CATEGORY_CREATION_CORE}"
        audience = "Strategic stakeholder."

    prompt = f"""
You are HyperCore, the DiviScan diagnostic intelligence engine.

Audience: {audience}

Use this internal context:
{context}

Now, generate a {mode}-grade diagnostic intelligence narrative based on this analysis JSON:

{json.dumps(analysis_json, indent=2)}

Your narrative MUST include the following sections with headings (and use plain text, no markdown markup):

🔍 Key Insights
- Summarize the main diagnostic or trial signals.

📈 Analytical Model
- Explain how the model performed (ROC AUC, PR AUC, sensitivity, specificity).
- Describe which biomarkers or features drive separation.

💡 Missed Opportunities
- Call out what standard workflows or analytics would have missed.
- Highlight confounders or hidden subgroups (even if inferred).

🧠 HyperCore Advantage
- Explain how DiviScan HyperCore (and the Diagnostic OS framing) provides superior insight vs traditional systems.

🧾 Recommended Next Steps
- Provide 3–5 concrete actions the hospital or pharma team should take next.

Stay clinically responsible; do NOT make regulatory claims or overstate certainty.
    """
    return prompt


@app.post("/generate_report", response_model=ReportResponse)
def generate_report(request: ReportRequest):
    """
    Generate mode-aware narrative (pharma vs hospital vs executive etc.)
    """
    prompt = build_report_prompt(request.analysis_json, request.mode)

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        narrative = resp.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error in /generate_report: {e}")

    return ReportResponse(narrative=narrative)


# -----------------------------
# /astra endpoint
# -----------------------------

def build_astra_prefix(mode: str) -> str:
    m = mode.lower()
    if m in ["pharma", "trial", "r&d"]:
        return (
            "You are ASTRA in Pharma Mode. "
            "Answer as a clinical trial & biomarker intelligence specialist. "
            "Focus on trial rescue, biomarkers, confounders, emerging phenotypes, and portfolio ROI."
        )
    if m in ["hospital", "clinical", "clinician"]:
        return (
            "You are ASTRA in Hospital Clinical Mode. "
            "Answer as a diagnostic support and care-path optimization assistant. "
            "Focus on diagnostic clarity, readmission risk, organ-system patterns, and workflow relief."
        )
    if m in ["regulatory"]:
        return (
            "You are ASTRA in Regulatory Mode. "
            "Be conservative, compliance-focused, and avoid overclaims."
        )
    if m in ["investor"]:
        return (
            "You are ASTRA in Investor Mode. "
            "Highlight ROI, defensibility, data moat, and scale potential, without disclosing SAFE internals."
        )
    if m in ["executive"]:
        return (
            "You are ASTRA in Executive Mode. "
            "Be concise, strategic, and centered on system-wide impact."
        )
    if m in ["unknown", "emerging"]:
        return (
            "You are ASTRA in Unknown Disease Mode. "
            "Describe novelty and phenotype patterns without naming new diseases or making diagnostic claims."
        )
    return (
        "You are ASTRA, the DiviScan HyperCore assistant. "
        "Respond professionally with clinical and strategic awareness."
    )


@app.post("/astra", response_model=ASTRAResponse)
def astra(request: ASTRARequest):
    prefix = build_astra_prefix(request.mode)

    prompt = f"""
{prefix}

Use this internal knowledge:
{COMPETITIVE_INTEL_CORE}

Question:
{request.question}

Answer clearly and precisely. Do not reference internal SAFE note mechanics or investor-only terms unless in Investor Mode.
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        answer = resp.choices[0].message["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error in /astra: {e}")

    return ASTRAResponse(answer=answer)


# -----------------------------
# /health endpoint
# -----------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "DiviScan HyperCore Unified Backend",
        "version": "4.0.0",
    }


# -----------------------------
# Local Run / Railway Entry
# -----------------------------

if __name__ == "__main__":
    # Local dev: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
