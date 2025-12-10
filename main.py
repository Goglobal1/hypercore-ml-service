# main.py
# HyperCore GH-OS - Python ML Service v5.0
#
# This file is structured for:
# - Railway deployment
# - Base44 / DiviCore AI frontend
# - Codex/GitHub Copilot style backend implementation
#
# It contains:
# - /analyze                     → existing working ML analysis endpoint
# - /multi_omic_fusion           → multi-omic fusion engine
# - /confounder_detection        → confounder detection engine
# - /emerging_phenotype          → emerging phenotype / unknown disease engine
# - /responder_prediction        → responder vs non-responder prediction engine
# - /trial_rescue                → trial rescue analysis engine
# - /outbreak_detection          → outbreak / drift detection engine
# - /predictive_modeling         → predictive risk / timeline engine
# - /synthetic_cohort            → synthetic cohort generator
# - /digital_twin_simulation     → individual patient trajectory simulator
# - /population_risk             → population risk aggregator
# - /fluview_ingest              → FluView ingestion → CSV for modeling
# - /create_digital_twin         → dataset digital twin storage
#
# All new engines are defined with Pydantic models and TODOs for Codex.


import io
from typing import List, Dict, Any, Optional

import math
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

app = FastAPI(
    title="HyperCore GH-OS ML Service",
    version="5.0.0",
    description="Unified ML API for DiviScan HyperCore / DiviCore AI"
)

# ------------------------------------------------------------------------
# ------------- EXISTING /ANALYZE ENDPOINT (WORKING) ---------------------
# ------------------------------------------------------------------------

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


def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity}


def clean_feature_matrix(
    df: pd.DataFrame,
    label_column: str
) -> (pd.DataFrame, np.ndarray, List[str], List[str]):
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
    """
    X, y, feature_cols, dropped_cols = clean_feature_matrix(df, label_column)

    n_samples = len(y)
    unique_classes, class_counts = np.unique(y, return_counts=True)

    if len(unique_classes) < 2:
        raise ValueError("Label column must contain at least two classes (e.g., 0 and 1).")

    # Decide split strategy
    # For very small datasets, avoid stratified split & keep more data in training
    if n_samples < 30 or class_counts.min() < 3:
        # Small dataset mode: train on all data and evaluate on same set
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

    # Clean out any NaN/inf for JSON safety
    def clean_list(values: np.ndarray) -> List[float]:
        cleaned: List[float] = []
        for v in values:
            if v is None:
                continue
            if isinstance(v, float) or isinstance(v, np.floating):
                if not math.isfinite(float(v)):
                    continue
            cleaned.append(float(v))
        return cleaned

    fpr_list = clean_list(fpr)
    tpr_list = clean_list(tpr)
    roc_thr_list = clean_list(roc_thresh)
    prec_list = clean_list(prec)
    rec_list = clean_list(rec)
    pr_thr_list = clean_list(pr_thresh)

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

    result: Dict[str, Any] = {
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "coefficients": coefficients,
        "roc_curve_data": {
            "fpr": fpr_list,
            "tpr": tpr_list,
            "thresholds": roc_thr_list,
        },
        "pr_curve_data": {
            "precision": prec_list,
            "recall": rec_list,
            "thresholds": pr_thr_list,
        },
        "feature_importance": feature_importance,
        "dropped_features": dropped_cols,
    }

    return result


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Accepts a CSV string and label column, returns real metric results
    for a binary classification problem.

    This endpoint is already wired into the DiviCore / Base44 frontend.
    DO NOT CHANGE ITS INPUT OR OUTPUT SCHEMA.
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

    return AnalyzeResponse(**result)

# ------------------------------------------------------------------------
# ------------- NEW ENGINE REQUEST/RESPONSE MODELS -----------------------
# ------------------------------------------------------------------------

# 1) multi_omic_fusion

class MultiOmicFeatures(BaseModel):
    immune: List[float]
    metabolic: List[float]
    microbiome: List[float]


class MultiOmicFusionResult(BaseModel):
    fused_score: float
    domain_contributions: Dict[str, float]
    primary_driver: str
    confidence: float


# 2) confounder_detection

class ConfounderDetectionRequest(BaseModel):
    csv: str
    label_column: str


class ConfounderFlag(BaseModel):
    type: str
    features: Optional[List[str]] = None
    strength: Optional[float] = None
    variance: Optional[float] = None
    explanation: Optional[str] = None
    recommendation: Optional[str] = None


# 3) emerging_phenotype

class EmergingPhenotypeRequest(BaseModel):
    biomarker_profile: Dict[str, float]
    historical_library: List[Dict[str, Any]]


class EmergingPhenotypeResult(BaseModel):
    level: int
    novelty_score: float
    closest_match: Optional[Dict[str, Any]] = None
    explanation: str
    confidence: float
    requires_escalation: bool


# 4) responder_prediction

class ResponderPredictionRequest(BaseModel):
    baseline: Dict[str, float]
    week2: Dict[str, float]
    treatment_arm: str


class ResponderPredictionResult(BaseModel):
    responder_probability: List[float]
    early_signals: List[str]
    recommended_stratification: str
    confidence: float


# 5) trial_rescue

class TrialRescueRequest(BaseModel):
    csv: str
    arms: List[str]
    endpoints: List[str]


class TrialRescueResult(BaseModel):
    rescue_probability: float
    signals: List[Dict[str, Any]]
    cost_of_delay: Dict[str, float]
    recommended_actions: List[Dict[str, Any]]


# 6) outbreak_detection

class OutbreakDetectionRequest(BaseModel):
    csv: str
    regions: List[str]


class OutbreakAlert(BaseModel):
    region: Optional[str] = None
    type: str
    severity: str
    lead_time_days: Optional[int] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    recommendation: Optional[str] = None


# 7) predictive_modeling

class PredictiveModelingRequest(BaseModel):
    patient_data: Dict[str, Any]
    forecast_horizon_days: int = 30


class PredictiveModelingResult(BaseModel):
    hospitalization_risk: Dict[str, Any]
    deterioration_timeline: Dict[str, Any]
    community_surge: Dict[str, Any]


# 8) synthetic_cohort

class SyntheticCohortRequest(BaseModel):
    n_subjects: int
    real_data_distributions: Dict[str, Dict[str, float]]
    constraints: Optional[Dict[str, Any]] = None


class SyntheticCohortResult(BaseModel):
    synthetic_data: List[Dict[str, Any]]
    realism_score: float
    distribution_match: Dict[str, float]
    validation: Dict[str, Any]


# 9) digital_twin_simulation

class DigitalTwinSimulationRequest(BaseModel):
    patient_baseline: Dict[str, Any]
    treatment_plan: Dict[str, Any]
    simulation_horizon_days: int = 90


class DigitalTwinSimulationResult(BaseModel):
    timeline: List[Dict[str, Any]]
    predicted_outcome: str
    confidence: float
    key_inflection_points: List[int]


# 10) population_risk

class PopulationRiskRequest(BaseModel):
    analyses: List[Dict[str, Any]]
    region: str


class PopulationRiskResult(BaseModel):
    region: str
    risk_score: float
    trend: str
    confidence: float
    top_biomarkers: List[str]


# 11) fluview_ingest

class FluViewIngestionRequest(BaseModel):
    fluview_json: Dict[str, Any]
    label_engineering: str = "ili_spike"


class FluViewIngestionResult(BaseModel):
    csv: str
    dataset_id: str
    rows: int
    label_column: str


# 12) dataset_digital_twin_storage

class DigitalTwinStorageRequest(BaseModel):
    dataset_id: str
    analysis_id: str
    csv_content: str
    metadata: Optional[Dict[str, Any]] = None


class DigitalTwinStorageResult(BaseModel):
    digital_twin_id: str
    storage_url: str
    fingerprint: str
    indexed_in_global_learning: bool
    version: int


# ------------------------------------------------------------------------
# ------------- NEW ENGINE ENDPOINTS (TODO: IMPLEMENT) -------------------
# ------------------------------------------------------------------------

@app.post("/multi_omic_fusion", response_model=MultiOmicFusionResult)
def multi_omic_fusion_endpoint(features: MultiOmicFeatures):
    """
    TODO: Implement multi-omic fusion:
    - Take immune, metabolic, microbiome arrays
    - Apply domain-specific models and weighted fusion
    - Return fused_score, domain_contributions, primary_driver, confidence
    """
    raise HTTPException(status_code=501, detail="multi_omic_fusion not implemented yet")


@app.post("/confounder_detection", response_model=List[ConfounderFlag])
def confounder_detection_endpoint(request: ConfounderDetectionRequest):
    """
    TODO: Implement confounder detection:
    - Parse CSV into DataFrame
    - Detect interaction effects, site drift, demographic bias
    - Return list of ConfounderFlag
    """
    raise HTTPException(status_code=501, detail="confounder_detection not implemented yet")


@app.post("/emerging_phenotype", response_model=EmergingPhenotypeResult)
def emerging_phenotype_endpoint(request: EmergingPhenotypeRequest):
    """
    TODO: Implement emerging phenotype detection:
    - Build fingerprint from biomarker_profile
    - Compare against historical_library
    - Compute novelty score and level (1-3)
    - Return closest_match, explanation, requires_escalation
    """
    raise HTTPException(status_code=501, detail="emerging_phenotype not implemented yet")


@app.post("/responder_prediction", response_model=ResponderPredictionResult)
def responder_prediction_endpoint(request: ResponderPredictionRequest):
    """
    TODO: Implement responder prediction:
    - Train model from historical responders if available
    - Use baseline + week2 + treatment_arm
    - Return responder_probability per subject, early_signals, recommended_stratification
    """
    raise HTTPException(status_code=501, detail="responder_prediction not implemented yet")


@app.post("/trial_rescue", response_model=TrialRescueResult)
def trial_rescue_endpoint(request: TrialRescueRequest):
    """
    TODO: Implement trial rescue analysis:
    - Parse CSV
    - Detect arm drift, endpoint collapse, subgroup emergence
    - Compute rescue_probability, cost_of_delay, recommended_actions
    """
    raise HTTPException(status_code=501, detail="trial_rescue not implemented yet")


@app.post("/outbreak_detection", response_model=List[OutbreakAlert])
def outbreak_detection_endpoint(request: OutbreakDetectionRequest):
    """
    TODO: Implement outbreak detection:
    - Parse CSV
    - Time-series + regional anomaly detection
    - Return outbreak alerts and drift alerts
    """
    raise HTTPException(status_code=501, detail="outbreak_detection not implemented yet")


@app.post("/predictive_modeling", response_model=PredictiveModelingResult)
def predictive_modeling_endpoint(request: PredictiveModelingRequest):
    """
    TODO: Implement predictive modeling:
    - Forecast hospitalization risk, deterioration timeline, community surge
    """
    raise HTTPException(status_code=501, detail="predictive_modeling not implemented yet")


@app.post("/synthetic_cohort", response_model=SyntheticCohortResult)
def synthetic_cohort_endpoint(request: SyntheticCohortRequest):
    """
    TODO: Implement synthetic cohort generator:
    - Train VAE/GAN on real_data_distributions
    - Generate synthetic_data and compute realism_score and distribution_match
    """
    raise HTTPException(status_code=501, detail="synthetic_cohort not implemented yet")


@app.post("/digital_twin_simulation", response_model=DigitalTwinSimulationResult)
def digital_twin_simulation_endpoint(request: DigitalTwinSimulationRequest):
    """
    TODO: Implement digital twin simulation:
    - Simulate patient trajectory for simulation_horizon_days
    - Return timeline, predicted_outcome, confidence, key_inflection_points
    """
    raise HTTPException(status_code=501, detail="digital_twin_simulation not implemented yet")


@app.post("/population_risk", response_model=PopulationRiskResult)
def population_risk_endpoint(request: PopulationRiskRequest):
    """
    TODO: Implement population risk aggregator:
    - Aggregate risk from multiple analyses
    - Return region-level risk_score, trend, top_biomarkers
    """
    raise HTTPException(status_code=501, detail="population_risk not implemented yet")


@app.post("/fluview_ingest", response_model=FluViewIngestionResult)
def fluview_ingest_endpoint(request: FluViewIngestionRequest):
    """
    TODO: Implement FluView ingestion:
    - Take fluview_json from CDC
    - Convert to model-ready CSV
    - Return CSV, dataset_id, rows, label_column
    """
    raise HTTPException(status_code=501, detail="fluview_ingest not implemented yet")


@app.post("/create_digital_twin", response_model=DigitalTwinStorageResult)
def create_digital_twin_endpoint(request: DigitalTwinStorageRequest):
    """
    TODO: Implement dataset digital twin storage:
    - Store CSV in cloud storage (e.g., GCS or S3)
    - Compute fingerprint (SHA-256)
    - Create DigitalTwin record in DB (handled by frontend/Base44 or another service)
    - Return digital_twin_id, storage_url, fingerprint, indexed flag, version
    """
    raise HTTPException(status_code=501, detail="create_digital_twin not implemented yet")


# ------------------------------------------------------------------------
# ------------- HEALTH CHECK ---------------------------------------------
# ------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "HyperCore GH-OS ML Service",
        "version": "5.0.0"
    }


# ------------------------------------------------------------------------
# ------------- LOCAL DEBUG ENTRYPOINT -----------------------------------
# ------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
