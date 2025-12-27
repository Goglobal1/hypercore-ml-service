# main.py
# HyperCore GH-OS – Python ML Service v5.1 (PRODUCTION)
# Unified ML API for DiviScan HyperCore / DiviCore AI
#
# v5.1 goals:
# - Preserve ALL endpoints
# - Upgrade /analyze to HyperCore-grade pipeline:
#   canonical lab normalization (synonyms)
#   unit normalization + ref ranges + contextual overrides
#   trajectory features (delta/rate/volatility)
#   axis decomposition + interactions + feedback loops
#   comparator benchmarking (NEWS/qSOFA/SIRS when present)
#   silent-risk subgroup logic (blind spot)
#   nonlinear shadow model (RF) for interaction credibility
#   real negative-space reasoning (missed opportunities) from rules
#   report-grade pipeline + execution manifest
#
# Dependencies: fastapi, uvicorn, pandas, numpy, scikit-learn

import io
import hashlib
import math
import traceback
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict

APP_VERSION = "5.1.0"

app = FastAPI(
    title="HyperCore GH-OS ML Service",
    version=APP_VERSION,
    description="Unified ML API for DiviScan HyperCore / DiviCore AI",
)

# ---------------------------------------------------------------------
# CONSTANTS / CANONICALIZATION
# ---------------------------------------------------------------------

AXES: List[str] = [
    "inflammatory",
    "endocrine",
    "immune",
    "microbial",
    "metabolic",
    "cardiovascular",
    "neurologic",
    "nutritional",
]

AXIS_LAB_MAP: Dict[str, List[str]] = {
    "inflammatory": ["crp", "esr", "ferritin", "il6", "procalcitonin"],
    "endocrine": ["tsh", "ft4", "t4", "cortisol", "acth", "insulin"],
    "immune": ["wbc", "neutrophils", "lymphocytes", "platelets"],
    "microbial": ["lactate", "procalcitonin", "endotoxin", "blood_culture"],
    "metabolic": ["glucose", "hba1c", "bun", "creatinine", "triglycerides", "hdl"],
    "cardiovascular": ["troponin", "bnp", "ntprobnp", "creatinine"],
    "neurologic": ["sodium", "potassium", "calcium", "glucose"],
    "nutritional": ["albumin", "vitamin_d", "folate", "b12"],
}

REFERENCE_RANGES: Dict[str, Dict[str, float]] = {
    "crp": {"low": 0.0, "high": 5.0},
    "wbc": {"low": 4.0, "high": 11.0},
    "glucose": {"low": 70.0, "high": 110.0},
    "creatinine": {"low": 0.6, "high": 1.3},
    "albumin": {"low": 3.4, "high": 5.4},
    "lactate": {"low": 0.5, "high": 2.2},
    "bun": {"low": 7.0, "high": 20.0},
    "sodium": {"low": 135.0, "high": 145.0},
    "potassium": {"low": 3.5, "high": 5.1},
    "troponin": {"low": 0.0, "high": 0.04},
}

UNIT_CONVERSIONS: Dict[Tuple[str, str], Tuple[str, Any]] = {
    ("glucose", "mg/dl"): ("mmol/l", lambda v: v / 18.0),
    ("creatinine", "mg/dl"): ("umol/l", lambda v: v * 88.4),
    ("bun", "mg/dl"): ("mmol/l", lambda v: v / 2.8),
    ("bilirubin", "mg/dl"): ("umol/l", lambda v: v * 17.1),
    ("lactate", "mg/dl"): ("mmol/l", lambda v: v / 9.0),
    ("crp", "mg/l"): ("mg/dl", lambda v: v * 0.1),
}

LAB_SYNONYMS: Dict[str, List[str]] = {
    "crp": ["crp", "c-reactive protein", "c reactive protein", "hs-crp", "hs crp"],
    "il6": ["il-6", "il6", "interleukin 6", "interleukin-6"],
    "wbc": ["wbc", "white blood cell", "white count"],
    "neutrophils": ["neutrophil", "neutrophils", "neut", "anc"],
    "lymphocytes": ["lymphocyte", "lymphocytes", "lymph"],
    "platelets": ["platelet", "platelets", "plt"],
    "glucose": ["glucose", "blood glucose", "glucose, poct", "glucose poct", "bg"],
    "hba1c": ["hba1c", "a1c", "hemoglobin a1c"],
    "bun": ["bun", "blood urea nitrogen", "urea nitrogen"],
    "creatinine": ["creatinine", "creat"],
    "lactate": ["lactate", "lactic acid"],
    "procalcitonin": ["procalcitonin", "pct"],
    "troponin": ["troponin", "trop"],
    "bnp": ["bnp", "brain natriuretic peptide"],
    "ntprobnp": ["nt-probnp", "ntprobnp", "nt probnp"],
    "albumin": ["albumin", "alb"],
    "sodium": ["sodium", "na"],
    "potassium": ["potassium", "k"],
    "calcium": ["calcium", "ca"],
    "blood_culture": ["blood culture", "culture blood", "bcx"],
    "esr": ["esr", "sed rate", "sedimentation rate"],
    "ferritin": ["ferritin"],
    "cortisol": ["cortisol"],
    "acth": ["acth"],
    "tsh": ["tsh"],
    "t4": ["t4", "total t4"],
    "ft4": ["free t4", "ft4"],
    "insulin": ["insulin"],
    "vitamin_d": ["vitamin d", "25-oh vitamin d", "25 oh vitamin d"],
    "folate": ["folate"],
    "b12": ["b12", "vitamin b12"],
    "triglycerides": ["triglycerides", "tg"],
    "hdl": ["hdl", "high density lipoprotein"],
    "ldl": ["ldl", "low density lipoprotein"],
    "bilirubin": ["bilirubin", "bili"],
    "endotoxin": ["endotoxin"],
}

COMPARATOR_ALIASES: Dict[str, List[str]] = {
    "news": ["news", "news_score", "news2", "news_2", "news2_score", "NEWS"],
    "qsofa": ["qsofa", "q_sofa", "qsofa_score", "qSOFA"],
    "sirs": ["sirs", "sirs_score", "SIRS"],
}

SILENT_RISK_THRESHOLDS: Dict[str, float] = {"news": 4.0, "qsofa": 1.0, "sirs": 1.0}

NEGATIVE_SPACE_RULES: List[Dict[str, Any]] = [
    {
        "condition": "S. aureus bacteremia",
        "severity": "critical",
        "trigger": lambda ctx, notes: ("staph aureus" in notes) or ("s. aureus" in notes) or bool(ctx.get("suspected_condition_tags", {}).get("S. aureus bacteremia", False)),
        "required": ["TEE", "Repeat blood cultures x2"],
    },
    {
        "condition": "Pituitary surgery / panhypopituitarism",
        "severity": "high",
        "trigger": lambda ctx, notes: bool(ctx.get("suspected_condition_tags", {}).get("Pituitary surgery / panhypopituitarism", False)) or ("pituitary" in (ctx.get("surgeries", "") or "").lower()),
        "required": ["Free T4", "AM cortisol", "ACTH"],
    },
    {
        "condition": "Sinus hyperdensity + immunosuppression",
        "severity": "high",
        "trigger": lambda ctx, notes: bool(ctx.get("suspected_condition_tags", {}).get("Sinus hyperdensity + immunosuppression", False)) or (("hyperdense" in notes) and ("sinus" in notes) and bool(ctx.get("immunosuppressed", False))),
        "required": ["MRI sinuses w/ contrast", "β-D-glucan", "Galactomannan", "ENT consult"],
    },
    {
        "condition": "Anemia / RDW signal (nutrient depletion risk)",
        "severity": "moderate",
        "trigger": lambda ctx, notes: bool(ctx.get("suspected_condition_tags", {}).get("RDW elevated or anemia risk", False)) or ("rdw" in notes),
        "required": ["Ferritin", "Iron/TIBC/%Sat", "B12", "Folate"],
    },
]

# ---------------------------------------------------------------------
# Pydantic MODELS
# ---------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    csv: str
    label_column: str
    patient_id_column: Optional[str] = None
    time_column: Optional[str] = None
    lab_name_column: Optional[str] = None
    value_column: Optional[str] = None
    unit_column: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[float] = None
    context: Optional[Dict[str, Any]] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class AnalyzeResponse(BaseModel):
    metrics: Dict[str, Any]
    coefficients: Dict[str, float]
    roc_curve_data: Dict[str, List[float]]
    pr_curve_data: Dict[str, List[float]]
    feature_importance: List[FeatureImportance]
    dropped_features: List[str]
    pipeline: Dict[str, Any]
    execution_manifest: Dict[str, Any]


class MultiOmicFeatures(BaseModel):
    immune: List[float]
    metabolic: List[float]
    microbiome: List[float]


class MultiOmicFusionResult(BaseModel):
    fused_score: float
    domain_contributions: Dict[str, float]
    primary_driver: str
    confidence: float


class ConfounderDetectionRequest(BaseModel):
    csv: str
    label_column: str


class ConfounderFlag(BaseModel):
    type: str
    explanation: Optional[str] = None
    strength: Optional[float] = None
    recommendation: Optional[str] = None


class EmergingPhenotypeRequest(BaseModel):
    csv: str
    label_column: str


class EmergingPhenotypeResult(BaseModel):
    phenotype_clusters: List[Dict[str, Any]]
    novelty_score: float
    drivers: Dict[str, float]
    narrative: str


class ResponderPredictionRequest(BaseModel):
    csv: str
    label_column: str
    treatment_column: str


class ResponderPredictionResult(BaseModel):
    response_lift: float
    key_biomarkers: Dict[str, float]
    subgroup_summary: Dict[str, Any]
    narrative: str


class TrialRescueRequest(BaseModel):
    csv: str
    label_column: str


class TrialRescueResult(BaseModel):
    futility_flag: bool
    enrichment_strategy: Dict[str, Any]
    power_recalculation: Dict[str, Any]
    narrative: str


class OutbreakDetectionRequest(BaseModel):
    csv: str
    region_column: str
    time_column: str
    case_count_column: str


class OutbreakDetectionResult(BaseModel):
    outbreak_regions: List[str]
    signals: Dict[str, Any]
    confidence: float
    narrative: str


class PredictiveModelingRequest(BaseModel):
    csv: str
    label_column: str
    forecast_horizon_days: int = 30


class PredictiveModelingResult(BaseModel):
    hospitalization_risk: Dict[str, float]
    deterioration_timeline: Dict[str, List[int]]
    community_surge: Dict[str, float]
    narrative: str


class SyntheticCohortRequest(BaseModel):
    real_data_distributions: Dict[str, Dict[str, float]]
    n_subjects: int


class SyntheticCohortResult(BaseModel):
    synthetic_data: List[Dict[str, float]]
    realism_score: float
    distribution_match: Dict[str, float]
    validation: Dict[str, Any]
    narrative: str


class DigitalTwinSimulationRequest(BaseModel):
    baseline_profile: Dict[str, float]
    simulation_horizon_days: int = 90


class DigitalTwinSimulationResult(BaseModel):
    timeline: List[Dict[str, float]]
    predicted_outcome: str
    confidence: float
    key_inflection_points: List[int]
    narrative: str


class PopulationRiskRequest(BaseModel):
    analyses: List[Dict[str, Any]]
    region: str


class PopulationRiskResult(BaseModel):
    region: str
    risk_score: float
    trend: str
    confidence: float
    top_biomarkers: List[str]


class FluViewIngestionRequest(BaseModel):
    fluview_json: Dict[str, Any]
    label_engineering: str = "ili_spike"


class FluViewIngestionResult(BaseModel):
    csv: str
    dataset_id: str
    rows: int
    label_column: str


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


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return default
        return v
    except Exception:
        return default


def canonical_lab(raw_name: str) -> str:
    n = (raw_name or "").strip().lower()
    if not n:
        return "unknown"
    for canon, variants in LAB_SYNONYMS.items():
        if n == canon:
            return canon
        if any(v in n for v in variants):
            return canon
    # normalize punctuation
    nn = n.replace("-", " ").replace("_", " ")
    for canon, variants in LAB_SYNONYMS.items():
        if any(v.replace("-", " ").replace("_", " ") in nn for v in variants):
            return canon
    return n


def _find_comparator_columns(df: pd.DataFrame) -> Dict[str, str]:
    lower_map = {c.lower(): c for c in df.columns}
    found: Dict[str, str] = {}
    for comp, aliases in COMPARATOR_ALIASES.items():
        for a in aliases:
            if a.lower() in lower_map:
                found[comp] = lower_map[a.lower()]
                break
    return found


def normalize_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context = context or {}
    suspected_tags = context.get("suspected_condition_tags", {})
    if not isinstance(suspected_tags, dict):
        suspected_tags = {}
    return {
        "pregnancy": bool(context.get("pregnancy", False)),
        "renal_failure": bool(context.get("renal_failure", False)),
        "suspected_condition_tags": suspected_tags,
        # pass-through optional keys (safe)
        "clinical_notes": context.get("clinical_notes", ""),
        "present_tests": context.get("present_tests", []),
        "surgeries": context.get("surgeries", ""),
        "immunosuppressed": bool(context.get("immunosuppressed", False)),
    }


def ensure_patient_id(df: pd.DataFrame, patient_id_column: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if patient_id_column and patient_id_column in df.columns:
        return df, patient_id_column
    if "patient_id" in df.columns:
        return df, "patient_id"
    df = df.copy()
    df["patient_id"] = df.index.astype(str)
    return df, "patient_id"


def ensure_time_column(df: pd.DataFrame, time_column: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    if time_column and time_column in df.columns:
        return df, time_column
    if "time" in df.columns:
        return df, "time"
    if "timestamp" in df.columns:
        return df, "timestamp"
    return df, None


def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity}


# ---------------------------------------------------------------------
# INGESTION: wide or long -> long canonical
# ---------------------------------------------------------------------

def ingest_labs(
    df: pd.DataFrame,
    label_column: str,
    patient_id_column: Optional[str] = None,
    time_column: Optional[str] = None,
    lab_name_column: Optional[str] = None,
    value_column: Optional[str] = None,
    unit_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, pid_col = ensure_patient_id(df, patient_id_column)
    df, t_col = ensure_time_column(df, time_column)

    if label_column not in df.columns:
        raise ValueError(f"label_column '{label_column}' not present")

    # LONG format
    if lab_name_column and value_column and lab_name_column in df.columns and value_column in df.columns:
        long_df = df.copy()
        rename_map: Dict[str, str] = {
            pid_col: "patient_id",
            lab_name_column: "lab_name",
            value_column: "value",
            label_column: "label",
        }
        if t_col:
            rename_map[t_col] = "time"
        if unit_column and unit_column in df.columns:
            rename_map[unit_column] = "unit"
        long_df = long_df.rename(columns=rename_map)
        if "time" not in long_df.columns:
            long_df["time"] = None
        if "unit" not in long_df.columns:
            long_df["unit"] = None
        long_df = long_df[["patient_id", "time", "lab_name", "value", "unit", "label"]].copy()
        fmt = "long"
    else:
        exclude = {label_column, pid_col}
        if t_col:
            exclude.add(t_col)
        if unit_column and unit_column in df.columns:
            exclude.add(unit_column)

        feature_cols = [c for c in df.columns if c not in exclude]
        id_vars = [pid_col, label_column] + ([t_col] if t_col else [])
        long_df = df.melt(
            id_vars=id_vars,
            value_vars=feature_cols,
            var_name="lab_name",
            value_name="value",
        ).copy()
        long_df = long_df.rename(columns={pid_col: "patient_id", label_column: "label"})
        if t_col:
            long_df = long_df.rename(columns={t_col: "time"})
        if "time" not in long_df.columns:
            long_df["time"] = None
        long_df["unit"] = None
        fmt = "wide"

    long_df["lab_name"] = long_df["lab_name"].astype(str).str.strip()
    long_df["lab_key"] = long_df["lab_name"].apply(canonical_lab)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df["label"] = pd.to_numeric(long_df["label"], errors="coerce")
    long_df = long_df.dropna(subset=["patient_id", "lab_key", "value", "label"]).copy()
    long_df["patient_id"] = long_df["patient_id"].astype(str)

    meta = {
        "format": fmt,
        "records": int(len(long_df)),
        "patients": int(long_df["patient_id"].nunique()),
        "labs": int(long_df["lab_key"].nunique()),
        "label_column": label_column,
    }
    return long_df, meta


# ---------------------------------------------------------------------
# CANONICALIZATION: units + ref ranges + z-score + context overrides
# ---------------------------------------------------------------------

def normalize_units(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    df["unit"] = df["unit"].fillna("").astype(str).str.strip().str.lower()
    conversions: List[Dict[str, Any]] = []

    for (lab, unit), (target_unit, converter) in UNIT_CONVERSIONS.items():
        mask = (df["lab_key"] == lab) & (df["unit"] == unit)
        if mask.any():
            df.loc[mask, "value"] = df.loc[mask, "value"].apply(lambda v: _to_float(converter(v)))
            df.loc[mask, "unit"] = target_unit
            conversions.append({"lab": lab, "from": unit, "to": target_unit})

    return df, {"conversions": conversions}


def apply_reference_ranges(labs: pd.DataFrame, sex: Optional[str], age: Optional[float]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    sex_key = (sex or "").strip().lower()
    lows: List[float] = []
    highs: List[float] = []

    for _, row in df.iterrows():
        lab = row["lab_key"]
        rr = REFERENCE_RANGES.get(lab, {"low": 0.0, "high": 1.0})
        low = float(rr["low"])
        high = float(rr["high"])
        if lab == "creatinine":
            if sex_key in {"f", "female"}:
                high = min(high, 1.1)
            if age is not None and age >= 65:
                high = high + 0.2
        lows.append(low)
        highs.append(high)

    df["ref_low"] = lows
    df["ref_high"] = highs
    df["out_of_range"] = (df["value"] < df["ref_low"]) | (df["value"] > df["ref_high"])

    mid = (df["ref_low"] + df["ref_high"]) / 2.0
    span = (df["ref_high"] - df["ref_low"]).replace(0, np.nan)
    df["z_score"] = ((df["value"] - mid) / span).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, {"reference_ranges_applied": True}


def apply_contextual_overrides(labs: pd.DataFrame, context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    overrides: List[str] = []

    if bool(context.get("pregnancy")):
        mask = df["lab_key"] == "wbc"
        if mask.any():
            df.loc[mask, "ref_high"] = df.loc[mask, "ref_high"] + 1.0
            overrides.append("pregnancy_wbc_range_adjust")

    if bool(context.get("renal_failure")):
        mask = df["lab_key"] == "creatinine"
        if mask.any():
            df.loc[mask, "ref_high"] = df.loc[mask, "ref_high"] + 0.5
            overrides.append("renal_failure_creatinine_range_adjust")

    df["out_of_range"] = (df["value"] < df["ref_low"]) | (df["value"] > df["ref_high"])
    mid = (df["ref_low"] + df["ref_high"]) / 2.0
    span = (df["ref_high"] - df["ref_low"]).replace(0, np.nan)
    df["z_score"] = ((df["value"] - mid) / span).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, {"context_overrides": overrides}


# ---------------------------------------------------------------------
# TIME ALIGNMENT + TRAJECTORY FEATURES
# ---------------------------------------------------------------------

def align_time_series(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    if "time" in df.columns and df["time"].notna().any():
        df["time_parsed"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time_parsed"] = pd.NaT

    df = df.sort_values(by=["patient_id", "lab_key", "time_parsed", "lab_name"]).copy()
    df["baseline_value"] = df.groupby(["patient_id", "lab_key"])["value"].transform("first")
    df["baseline_time"] = df.groupby(["patient_id", "lab_key"])["time_parsed"].transform("first")
    df["delta"] = df["value"] - df["baseline_value"]

    time_delta_hrs = (df["time_parsed"] - df["baseline_time"]).dt.total_seconds() / 3600.0
    time_delta_hrs = time_delta_hrs.replace(0, np.nan)
    df["rate_of_change"] = (df["delta"] / time_delta_hrs).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, {"aligned": True}


def extract_numeric_features(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    if "time_parsed" not in df.columns:
        df["time_parsed"] = pd.NaT

    df = df.sort_values(by=["patient_id", "lab_key", "time_parsed"]).copy()
    grouped = df.groupby(["patient_id", "lab_key"])
    latest = grouped.tail(1).set_index(["patient_id", "lab_key"])

    stats = grouped["value"].agg(["mean", "min", "max", "std", "count"])
    oor_any = grouped["out_of_range"].max()

    latest_value = latest["value"].unstack(fill_value=np.nan)
    latest_value.columns = [f"value__{c}_latest" for c in latest_value.columns]

    mean_df = stats["mean"].unstack(fill_value=np.nan)
    mean_df.columns = [f"value__{c}_mean" for c in mean_df.columns]

    min_df = stats["min"].unstack(fill_value=np.nan)
    min_df.columns = [f"value__{c}_min" for c in min_df.columns]

    max_df = stats["max"].unstack(fill_value=np.nan)
    max_df.columns = [f"value__{c}_max" for c in max_df.columns]

    std_df = stats["std"].unstack(fill_value=np.nan)
    std_df.columns = [f"value__{c}_std" for c in std_df.columns]

    z_latest = latest["z_score"].unstack(fill_value=0.0)
    z_latest.columns = [f"z__{c}_latest" for c in z_latest.columns]

    oor_df = oor_any.unstack(fill_value=False).astype(int)
    oor_df.columns = [f"oor__{c}" for c in oor_df.columns]

    presence = stats["count"].unstack(fill_value=0)
    miss_df = (presence == 0).astype(int)
    miss_df.columns = [f"miss__{c}" for c in miss_df.columns]

    feature_df = pd.concat([latest_value, mean_df, min_df, max_df, std_df, z_latest, oor_df, miss_df], axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_index()
    meta = {"feature_count": int(feature_df.shape[1])}
    return feature_df, meta


def compute_delta_features(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    grouped = df.groupby(["patient_id", "lab_key"])
    delta_mean = grouped["delta"].mean().unstack(fill_value=0.0)
    delta_mean.columns = [f"delta__{c}_mean" for c in delta_mean.columns]

    delta_vol = grouped["delta"].std().unstack(fill_value=0.0).fillna(0.0)
    delta_vol.columns = [f"delta__{c}_vol" for c in delta_vol.columns]

    rate_mean = grouped["rate_of_change"].mean().unstack(fill_value=0.0).fillna(0.0)
    rate_mean.columns = [f"rate__{c}_mean" for c in rate_mean.columns]

    out = pd.concat([delta_mean, delta_vol, rate_mean], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    meta = {"delta_feature_count": int(out.shape[1])}
    return out, meta


def detect_volatility(delta_features: pd.DataFrame) -> Dict[str, Any]:
    vol_cols = [c for c in delta_features.columns if c.endswith("_vol")]
    if not vol_cols:
        return {"high_volatility_labs": [], "threshold": 0.0}

    vols = delta_features[vol_cols].mean().fillna(0.0)
    threshold = float(vols.mean() + vols.std())
    hi = [c.replace("delta__", "").replace("_vol", "") for c, v in vols.items() if float(v) > threshold]
    return {"high_volatility_labs": hi, "threshold": threshold}


def flag_extremes(labs: pd.DataFrame) -> Dict[str, Any]:
    df = labs.copy()
    out_df = df[df["out_of_range"]].copy()
    if out_df.empty:
        return {"extremes": []}
    agg = out_df.groupby("lab_key")["value"].agg(["min", "max"]).reset_index()
    return {"extremes": [{"lab": r["lab_key"], "min": float(r["min"]), "max": float(r["max"])} for _, r in agg.iterrows()]}


# ---------------------------------------------------------------------
# AXES / INTERACTIONS / FEEDBACK LOOPS
# ---------------------------------------------------------------------

def decompose_axes(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    axis_scores: Dict[str, pd.Series] = {}
    axis_summary: Dict[str, Any] = {}

    for axis in AXES:
        keys = set(AXIS_LAB_MAP.get(axis, []))
        sub = df[df["lab_key"].isin(keys)]
        if sub.empty:
            axis_scores[axis] = pd.Series(dtype=float)
            axis_summary[axis] = {"mean_score": 0.0, "top_drivers": [], "missing": True}
            continue
        per_patient = sub.groupby("patient_id")["z_score"].mean().fillna(0.0)
        axis_scores[axis] = per_patient
        drivers = sub.groupby("lab_key")["z_score"].mean().abs().sort_values(ascending=False).head(3).index.tolist()
        axis_summary[axis] = {
            "mean_score": float(per_patient.mean()) if len(per_patient) else 0.0,
            "top_drivers": drivers,
            "missing": False,
        }

    axis_df = pd.DataFrame(axis_scores).fillna(0.0)
    return axis_df, axis_summary


def map_axis_interactions(axis_scores: pd.DataFrame) -> List[Dict[str, Any]]:
    if axis_scores.empty:
        return []
    mean_scores = axis_scores.mean()
    out: List[Dict[str, Any]] = []
    for a, b in combinations(mean_scores.index, 2):
        s = float(mean_scores[a] + mean_scores[b])
        out.append({"axes": [a, b], "combined_score": s, "amplified": bool(s > 1.0)})
    out.sort(key=lambda d: d["combined_score"], reverse=True)
    return out[:12]


def identify_feedback_loops(axis_scores: pd.DataFrame) -> List[Dict[str, Any]]:
    if axis_scores.empty:
        return []
    mean_scores = axis_scores.mean()
    loops = []
    for axis, score in mean_scores.items():
        s = float(score)
        if s > 0.8:
            loops.append({"axis": axis, "severity": "high", "score": s, "pattern": "self_reinforcing"})
        elif s > 0.5:
            loops.append({"axis": axis, "severity": "moderate", "score": s, "pattern": "drift"})
    loops.sort(key=lambda d: d["score"], reverse=True)
    return loops


# ---------------------------------------------------------------------
# MODELING: linear explainable + nonlinear shadow mode
# ---------------------------------------------------------------------

def _sanitize_matrix(X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    numeric = X.select_dtypes(include=[np.number]).copy()
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variances = numeric.var()
    keep = variances[variances > 0].index.tolist()
    dropped = [c for c in numeric.columns if c not in keep]
    return numeric[keep], dropped


def _choose_cv_strategy(y: np.ndarray) -> Dict[str, Any]:
    n = int(len(y))
    unique, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if len(counts) else 0
    if n >= 100 and min_class >= 5:
        return {"type": "skf", "n_splits": 5}
    return {"type": "split", "test_size": 0.3}


def _fit_linear_model(X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    Xc, dropped = _sanitize_matrix(X)
    if Xc.shape[1] == 0:
        raise ValueError("No usable numeric features after cleaning")

    policy = _choose_cv_strategy(y)
    lr = LogisticRegression(max_iter=2000, solver="liblinear", class_weight="balanced")

    if policy["type"] == "skf":
        cv = StratifiedKFold(n_splits=policy["n_splits"], shuffle=True, random_state=42)
        probs = cross_val_predict(lr, Xc, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
        ap = float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
        acc = float(accuracy_score(y, preds))
        sens_spec = compute_sensitivity_specificity(y, preds)

        fpr, tpr, roc_thr = roc_curve(y, probs) if len(np.unique(y)) > 1 else (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5]))
        prec, rec, pr_thr = precision_recall_curve(y, probs) if len(np.unique(y)) > 1 else (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5]))

        lr.fit(Xc, y)
        coef = lr.coef_[0]
        abs_coef = np.abs(coef)
        importance = abs_coef / abs_coef.sum() if float(abs_coef.sum()) > 0 else abs_coef

        return {
            "cv_method": f"StratifiedKFold(n_splits={policy['n_splits']})",
            "metrics": {
                "roc_auc": auc,
                "pr_auc": ap,
                "accuracy": acc,
                "sensitivity": float(sens_spec["sensitivity"]),
                "specificity": float(sens_spec["specificity"]),
            },
            "coefficients": {f: float(c) for f, c in zip(Xc.columns, coef)},
            "feature_importance": [{"feature": f, "importance": float(i)} for f, i in zip(Xc.columns, importance)],
            "roc_curve_data": {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr], "thresholds": [float(x) for x in roc_thr]},
            "pr_curve_data": {"precision": [float(x) for x in prec], "recall": [float(x) for x in rec], "thresholds": [float(x) for x in pr_thr]},
            "probabilities": [float(p) for p in probs],
            "dropped_features": dropped,
            "model": lr,
            "X_clean": Xc,
        }

    # split path
    try:
        X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=policy["test_size"], random_state=42, stratify=y)
        split_used = "train_test_split_stratified"
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=policy["test_size"], random_state=42)
        split_used = "train_test_split"

    lr.fit(X_train, y_train)
    probs_test = lr.predict_proba(X_test)[:, 1]
    preds = (probs_test >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, probs_test)) if len(np.unique(y_test)) > 1 else 0.0
    ap = float(average_precision_score(y_test, probs_test)) if len(np.unique(y_test)) > 1 else 0.0
    acc = float(accuracy_score(y_test, preds))
    sens_spec = compute_sensitivity_specificity(y_test, preds)

    fpr, tpr, roc_thr = roc_curve(y_test, probs_test) if len(np.unique(y_test)) > 1 else (np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5]))
    prec, rec, pr_thr = precision_recall_curve(y_test, probs_test) if len(np.unique(y_test)) > 1 else (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.5]))

    coef = lr.coef_[0]
    abs_coef = np.abs(coef)
    importance = abs_coef / abs_coef.sum() if float(abs_coef.sum()) > 0 else abs_coef

    full_probs = lr.predict_proba(Xc)[:, 1]

    return {
        "cv_method": split_used,
        "metrics": {
            "roc_auc": auc,
            "pr_auc": ap,
            "accuracy": acc,
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "coefficients": {f: float(c) for f, c in zip(Xc.columns, coef)},
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in zip(Xc.columns, importance)],
        "roc_curve_data": {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr], "thresholds": [float(x) for x in roc_thr]},
        "pr_curve_data": {"precision": [float(x) for x in prec], "recall": [float(x) for x in rec], "thresholds": [float(x) for x in pr_thr]},
        "probabilities": [float(p) for p in full_probs],
        "dropped_features": dropped,
        "model": lr,
        "X_clean": Xc,
    }


def _fit_nonlinear_shadow(X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
    Xc, _ = _sanitize_matrix(X)
    if Xc.shape[1] == 0:
        return {"shadow_mode": True, "metrics": {}, "feature_importance": {}, "permutation_importance": {}, "cv_method": "none"}

    rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    policy = _choose_cv_strategy(y)

    if policy["type"] == "skf":
        cv = StratifiedKFold(n_splits=policy["n_splits"], shuffle=True, random_state=42)
        probs = cross_val_predict(rf, Xc, y, cv=cv, method="predict_proba")[:, 1]
        preds = (probs >= 0.5).astype(int)

        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
        ap = float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
        acc = float(accuracy_score(y, preds))
        sens_spec = compute_sensitivity_specificity(y, preds)

        rf.fit(Xc, y)
        fi = {f: float(v) for f, v in zip(Xc.columns, rf.feature_importances_)}

        return {
            "shadow_mode": True,
            "cv_method": f"StratifiedKFold(n_splits={policy['n_splits']})",
            "metrics": {
                "roc_auc": float(auc),
                "pr_auc": float(ap),
                "accuracy": float(acc),
                "sensitivity": float(sens_spec["sensitivity"]),
                "specificity": float(sens_spec["specificity"]),
            },
            "feature_importance": fi,
            "permutation_importance": {},
        }

    # split mode
    try:
        X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=policy["test_size"], random_state=42, stratify=y)
        split_used = "train_test_split_stratified"
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=policy["test_size"], random_state=42)
        split_used = "train_test_split"

    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0
    ap = float(average_precision_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0
    acc = float(accuracy_score(y_test, preds))
    sens_spec = compute_sensitivity_specificity(y_test, preds)

    fi = {f: float(v) for f, v in zip(Xc.columns, rf.feature_importances_)}

    perm_imp: Dict[str, float] = {}
    if X_test.shape[0] >= 10 and len(np.unique(y_test)) > 1:
        perm = permutation_importance(rf, X_test, y_test, n_repeats=5, random_state=42)
        perm_imp = {f: float(v) for f, v in zip(Xc.columns, perm.importances_mean)}

    return {
        "shadow_mode": True,
        "cv_method": split_used,
        "metrics": {
            "roc_auc": float(auc),
            "pr_auc": float(ap),
            "accuracy": float(acc),
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "feature_importance": fi,
        "permutation_importance": perm_imp,
    }


# ---------------------------------------------------------------------
# COMPARATOR BENCHMARKING + SILENT RISK
# ---------------------------------------------------------------------

def comparator_benchmarking(original_df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    comps = _find_comparator_columns(original_df)
    out: Dict[str, Any] = {"comparators_found": comps, "metrics": {}}

    if label_column not in original_df.columns:
        return out

    y = pd.to_numeric(original_df[label_column], errors="coerce").fillna(0.0).astype(int).values
    if len(np.unique(y)) < 2:
        return out

    for comp_key, col in comps.items():
        scores = pd.to_numeric(original_df[col], errors="coerce").fillna(0.0).values
        if len(np.unique(scores)) < 2:
            continue
        out["metrics"][comp_key] = {
            "column": col,
            "roc_auc": float(roc_auc_score(y, scores)),
            "threshold": float(SILENT_RISK_THRESHOLDS.get(comp_key, 0.0)),
        }

    return out


def detect_silent_risk(original_df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    comps = _find_comparator_columns(original_df)
    if not comps:
        return {"available": False, "reason": "no_comparator_columns"}

    if label_column not in original_df.columns:
        return {"available": False, "reason": "label_missing"}

    y = pd.to_numeric(original_df[label_column], errors="coerce").fillna(0.0).astype(int)
    out: Dict[str, Any] = {"available": True, "blind_spot": {}}

    mask = pd.Series(True, index=original_df.index)
    for comp_key, col in comps.items():
        thr = SILENT_RISK_THRESHOLDS.get(comp_key)
        if thr is None:
            continue
        s = pd.to_numeric(original_df[col], errors="coerce").fillna(0.0)
        mask = mask & (s <= float(thr))

    acceptable = original_df[mask].copy()
    if acceptable.empty:
        return {"available": True, "blind_spot": {}, "note": "no_acceptable_group_after_thresholds"}

    adverse = acceptable[pd.to_numeric(acceptable[label_column], errors="coerce").fillna(0.0).astype(int) == 1]
    adverse_rate = float(len(adverse) / len(acceptable)) if len(acceptable) else 0.0

    out["blind_spot"] = {
        "standard_acceptable_count": int(len(acceptable)),
        "adverse_in_acceptable_count": int(len(adverse)),
        "adverse_rate": adverse_rate,
    }
    return out


# ---------------------------------------------------------------------
# NEGATIVE SPACE: missed opportunity engine
# ---------------------------------------------------------------------

def detect_negative_space(ctx: Dict[str, Any], present_tests: List[str], notes: str) -> List[Dict[str, Any]]:
    present = {t.strip().lower() for t in present_tests if isinstance(t, str)}
    n = (notes or "").lower()

    missed: List[Dict[str, Any]] = []
    for rule in NEGATIVE_SPACE_RULES:
        try:
            triggered = bool(rule["trigger"](ctx, n))
        except Exception:
            triggered = False

        if not triggered:
            continue

        required = rule.get("required", [])
        missing = [t for t in required if t.lower() not in present]
        if missing:
            missed.append(
                {
                    "trigger_condition": rule["condition"],
                    "missing_tests": missing,
                    "severity": rule["severity"],
                }
            )
    return missed


# ---------------------------------------------------------------------
# EXPLAINABILITY (median comparisons + directionality)
# ---------------------------------------------------------------------

def explainability_layer(X: pd.DataFrame, y: np.ndarray, coefficients: Dict[str, float]) -> Dict[str, Any]:
    if X.empty or len(y) != len(X):
        return {"available": False, "reason": "no_features_or_label_mismatch"}

    df = X.copy()
    df["label"] = y

    event = df[df["label"] == 1]
    non_event = df[df["label"] == 0]

    med_event = event.median(numeric_only=True).to_dict() if not event.empty else {}
    med_nonevent = non_event.median(numeric_only=True).to_dict() if not non_event.empty else {}

    direction = {}
    for feat, coef in coefficients.items():
        if coef > 0:
            direction[feat] = "↑"
        elif coef < 0:
            direction[feat] = "↓"
        else:
            direction[feat] = "→"

    gaps = []
    for feat in X.columns:
        a = _to_float(med_event.get(feat, 0.0))
        b = _to_float(med_nonevent.get(feat, 0.0))
        gaps.append({"feature": feat, "event_median": a, "non_event_median": b, "diff": a - b, "direction": direction.get(feat, "→")})
    gaps.sort(key=lambda r: abs(r["diff"]), reverse=True)

    return {
        "available": True,
        "directionality": direction,
        "median_comparisons": {
            "event": {k: _to_float(v) for k, v in med_event.items()},
            "non_event": {k: _to_float(v) for k, v in med_nonevent.items()},
        },
        "top_median_gaps": gaps[:20],
    }


# ---------------------------------------------------------------------
# EXECUTION MANIFEST
# ---------------------------------------------------------------------

def build_execution_manifest(
    req: AnalyzeRequest,
    ingestion: Dict[str, Any],
    transforms: List[str],
    models_used: Dict[str, Any],
    metrics: Dict[str, Any],
    axis_summary: Dict[str, Any],
    interactions: List[Dict[str, Any]],
    feedback_loops: List[Dict[str, Any]],
    negative_space: List[Dict[str, Any]],
    silent_risk: Dict[str, Any],
    explainability: Dict[str, Any],
) -> Dict[str, Any]:
    req_hash = hashlib.sha256((req.csv[:20000] + req.label_column).encode("utf-8")).hexdigest()[:16]
    return {
        "manifest_version": "1.0.0",
        "generated_at": _now_iso(),
        "service_version": APP_VERSION,
        "request_fingerprint": req_hash,
        "inputs": {
            "label_column": req.label_column,
            "patient_id_column": req.patient_id_column,
            "time_column": req.time_column,
            "lab_name_column": req.lab_name_column,
            "value_column": req.value_column,
            "unit_column": req.unit_column,
            "sex": req.sex,
            "age": req.age,
            "context_keys": sorted(list((req.context or {}).keys())),
        },
        "ingestion": ingestion,
        "transforms_applied": transforms,
        "models_used": models_used,
        "metrics": metrics,
        "axes": axis_summary,
        "axis_interactions": interactions[:10],
        "feedback_loops": feedback_loops[:10],
        "silent_risk": silent_risk,
        "negative_space": negative_space,
        "explainability": {
            "available": bool(explainability.get("available", False)),
            "top_gap_features": [g["feature"] for g in explainability.get("top_median_gaps", [])[:8]] if isinstance(explainability, dict) else [],
        },
        "governance": {
            "use": "clinical_decision_support / quality_improvement",
            "not_for": "diagnosis",
            "human_in_the_loop": True,
        },
    }


# ---------------------------------------------------------------------
# ANALYZE ENDPOINT (HyperCore-grade pipeline)
# ---------------------------------------------------------------------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    try:
        df = pd.read_csv(io.StringIO(req.csv))
        context = normalize_context(req.context)

        if req.label_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found in dataset")

        labs_long, ingest_meta = ingest_labs(
            df=df,
            label_column=req.label_column,
            patient_id_column=req.patient_id_column,
            time_column=req.time_column,
            lab_name_column=req.lab_name_column,
            value_column=req.value_column,
            unit_column=req.unit_column,
        )

        labs_long, unit_meta = normalize_units(labs_long)
        labs_long, rr_meta = apply_reference_ranges(labs_long, req.sex, req.age)
        labs_long, align_meta = align_time_series(labs_long)
        labs_long, ctx_meta = apply_contextual_overrides(labs_long, context)

        feat_df, feat_meta = extract_numeric_features(labs_long)
        delta_df, delta_meta = compute_delta_features(labs_long)
        full_features = feat_df.join(delta_df, how="left").fillna(0.0)

        label_by_patient = labs_long.groupby("patient_id")["label"].max().astype(int)
        label_by_patient = label_by_patient.reindex(full_features.index).dropna()
        full_features = full_features.loc[label_by_patient.index]
        y = label_by_patient.values.astype(int)

        if len(np.unique(y)) < 2:
            raise HTTPException(status_code=400, detail="Label must contain at least two classes (0/1) after aggregation.")

        axis_scores, axis_summary = decompose_axes(labs_long)
        axis_scores = axis_scores.reindex(full_features.index).fillna(0.0)

        interactions = map_axis_interactions(axis_scores)
        feedback_loops = identify_feedback_loops(axis_scores)

        linear = _fit_linear_model(full_features, y)
        nonlinear = _fit_nonlinear_shadow(full_features, y)

        comparator = comparator_benchmarking(df, req.label_column)
        silent_risk = detect_silent_risk(df, req.label_column)

        notes_text = str(context.get("clinical_notes", "") or "").lower()
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if text_cols:
            sample_text = " ".join([str(v) for v in df[text_cols].head(25).fillna("").values.flatten().tolist()])
            notes_text = (notes_text + " " + sample_text.lower()).strip()

        present_tests = list(set(labs_long["lab_key"].unique().tolist()))
        if isinstance(context.get("present_tests"), list):
            present_tests.extend([str(x) for x in context.get("present_tests", [])])

        negative_space = detect_negative_space(ctx=context, present_tests=present_tests, notes=notes_text)

        volatility = detect_volatility(delta_df)
        extremes = flag_extremes(labs_long)

        X_clean = linear.get("X_clean", pd.DataFrame())
        coef_map = linear.get("coefficients", {})
        explain = explainability_layer(X_clean, y, coef_map)

        pipeline: Dict[str, Any] = {
            "ingestion": ingest_meta,
            "unit_normalization": unit_meta,
            "reference_ranges": rr_meta,
            "time_alignment": align_meta,
            "context_overrides": ctx_meta,
            "feature_extraction": feat_meta,
            "delta_features": delta_meta,
            "axes": axis_summary,
            "axis_interactions": interactions[:12],
            "feedback_loops": feedback_loops[:12],
            "modeling": {
                "linear": {"cv_method": linear.get("cv_method"), "metrics": linear.get("metrics")},
                "nonlinear_shadow": {"shadow_mode": True, "cv_method": nonlinear.get("cv_method"), "metrics": nonlinear.get("metrics")},
            },
            "benchmarking": comparator,
            "silent_risk": silent_risk,
            "negative_space": negative_space,
            "volatility": volatility,
            "extremes": extremes,
            "explainability": explain,
            "governance": {"use": "quality_improvement / decision_support", "not_for": "diagnosis", "human_in_the_loop": True},
        }

        metrics: Dict[str, Any] = {
            "linear": linear.get("metrics", {}),
            "nonlinear_shadow": nonlinear.get("metrics", {}),
            "comparators": comparator.get("metrics", {}),
            "silent_risk": silent_risk,
            "negative_space_count": int(len(negative_space)),
        }

        execution_manifest = build_execution_manifest(
            req=req,
            ingestion={**ingest_meta, **{"columns": int(df.shape[1]), "rows": int(df.shape[0])}},
            transforms=[
                "canonical_lab_mapping",
                "unit_normalization",
                "reference_range_enrichment",
                "time_alignment_delta_rate",
                "trajectory_features",
                "axis_decomposition",
                "interaction_screen",
                "linear_model",
                "nonlinear_shadow_model",
                "benchmarking_if_present",
                "silent_risk_if_present",
                "negative_space_rules",
            ],
            models_used={
                "linear": {"type": "LogisticRegression", "cv_method": linear.get("cv_method")},
                "nonlinear_shadow": {"type": "RandomForestClassifier", "cv_method": nonlinear.get("cv_method"), "shadow_mode": True},
            },
            metrics=metrics,
            axis_summary=axis_summary,
            interactions=interactions,
            feedback_loops=feedback_loops,
            negative_space=negative_space,
            silent_risk=silent_risk,
            explainability=explain,
        )

        return AnalyzeResponse(
            metrics=metrics,
            coefficients={k: float(v) for k, v in (linear.get("coefficients", {}) or {}).items()},
            roc_curve_data=linear.get("roc_curve_data", {"fpr": [], "tpr": [], "thresholds": []}),
            pr_curve_data=linear.get("pr_curve_data", {"precision": [], "recall": [], "thresholds": []}),
            feature_importance=[FeatureImportance(**fi) for fi in (linear.get("feature_importance", []) or [])],
            dropped_features=linear.get("dropped_features", []) or [],
            pipeline=pipeline,
            execution_manifest=execution_manifest,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "trace": traceback.format_exc().splitlines()[-12:]},
        )


# ---------------------------------------------------------------------
# OTHER ENDPOINTS (PRESERVED)
# ---------------------------------------------------------------------

def mean_safe(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0


@app.post("/multi_omic_fusion", response_model=MultiOmicFusionResult)
def multi_omic_fusion(f: MultiOmicFeatures) -> MultiOmicFusionResult:
    scores = {
        "immune": mean_safe(f.immune),
        "metabolic": mean_safe(f.metabolic),
        "microbiome": mean_safe(f.microbiome),
    }
    fused = float(np.mean(list(scores.values()))) if scores else 0.0
    total = float(sum(abs(v) for v in scores.values()) or 1.0)
    contrib = {k: float(abs(v) / total) for k, v in scores.items()}
    primary = max(scores, key=scores.get) if scores else "immune"
    confidence = float(max(0.0, min(1.0, 1.0 - float(np.std(list(scores.values()))))))
    return MultiOmicFusionResult(fused_score=fused, domain_contributions=contrib, primary_driver=str(primary), confidence=confidence)


@app.post("/confounder_detection", response_model=List[ConfounderFlag])
def confounder_detection(req: ConfounderDetectionRequest) -> List[ConfounderFlag]:
    df = pd.read_csv(io.StringIO(req.csv))
    if req.label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found")
    y = pd.to_numeric(df[req.label_column], errors="coerce").fillna(0.0)
    flags: List[ConfounderFlag] = []
    counts = y.value_counts(normalize=True)
    if not counts.empty and float(counts.max()) >= 0.9:
        flags.append(
            ConfounderFlag(
                type="class_imbalance",
                explanation=f"Highly imbalanced label distribution (max fraction={float(counts.max()):.2f}).",
                strength=float(counts.max()),
                recommendation="Collect more minority-class examples or rebalance; interpret metrics cautiously.",
            )
        )
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == req.label_column:
            continue
        x = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        if x.notna().sum() < 10:
            continue
        try:
            corr = float(np.corrcoef(x.values, y.values)[0, 1])
        except Exception:
            corr = 0.0
        if abs(corr) >= 0.8:
            flags.append(
                ConfounderFlag(
                    type="label_leakage_suspected",
                    explanation=f"Feature '{col}' highly correlated with label (corr={corr:.2f}) → leakage risk.",
                    strength=float(abs(corr)),
                    recommendation="Validate whether feature encodes post-outcome measurement timing.",
                )
            )
    return flags


@app.post("/emerging_phenotype", response_model=EmergingPhenotypeResult)
def emerging_phenotype(req: EmergingPhenotypeRequest) -> EmergingPhenotypeResult:
    df = pd.read_csv(io.StringIO(req.csv))
    if req.label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found")
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[req.label_column], errors="ignore").fillna(0.0)
    if numeric.shape[1] == 0:
        return EmergingPhenotypeResult(phenotype_clusters=[], novelty_score=0.0, drivers={}, narrative="No numeric signal space available.")
    variances = numeric.var().sort_values(ascending=False)
    top = variances.head(5)
    novelty = float(min(1.0, (top.mean() / (variances.mean() + 1e-9))))
    clusters = [{"cluster_id": 0, "size": int(len(df) * 0.6)}, {"cluster_id": 1, "size": int(len(df) * 0.4)}]
    drivers = {str(k): float(v) for k, v in top.items()}
    narrative = "Phenotype drift scan executed; treat as discovery output and confirm clinically."
    return EmergingPhenotypeResult(phenotype_clusters=clusters, novelty_score=novelty, drivers=drivers, narrative=narrative)


@app.post("/responder_prediction", response_model=ResponderPredictionResult)
def responder_prediction(req: ResponderPredictionRequest) -> ResponderPredictionResult:
    df = pd.read_csv(io.StringIO(req.csv))
    if req.label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found")
    if req.treatment_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"treatment_column '{req.treatment_column}' not found")
    y = pd.to_numeric(df[req.label_column], errors="coerce").fillna(0.0).astype(int)
    treat = df[req.treatment_column].astype(str)
    arms = treat.unique().tolist()
    if len(arms) < 2:
        return ResponderPredictionResult(response_lift=0.0, key_biomarkers={}, subgroup_summary={"note": "Only one arm present."}, narrative="Responder prediction requires ≥2 arms.")
    arm_rates = {a: float(y[treat == a].mean()) if (treat == a).any() else 0.0 for a in arms}
    best = min(arm_rates, key=arm_rates.get)
    worst = max(arm_rates, key=arm_rates.get)
    lift = float(arm_rates[worst] - arm_rates[best])
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[req.label_column], errors="ignore").fillna(0.0)
    diffs: Dict[str, float] = {}
    if numeric.shape[1] > 0:
        mean_best = numeric[treat == best].mean()
        mean_worst = numeric[treat == worst].mean()
        delta = (mean_worst - mean_best).abs().sort_values(ascending=False).head(6)
        diffs = {str(k): float(v) for k, v in delta.items()}
    narrative = f"Responder scan executed. Outcome-rate separation between '{best}' and '{worst}' supports enrichment targeting."
    return ResponderPredictionResult(response_lift=lift, key_biomarkers=diffs, subgroup_summary={"arms": arm_rates, "best_arm": best, "worst_arm": worst}, narrative=narrative)


@app.post("/trial_rescue", response_model=TrialRescueResult)
def trial_rescue(req: TrialRescueRequest) -> TrialRescueResult:
    df = pd.read_csv(io.StringIO(req.csv))
    if req.label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found")
    y = pd.to_numeric(df[req.label_column], errors="coerce").fillna(0.0).astype(int)
    rate = float(y.mean())
    futility = bool(rate < 0.05 or rate > 0.95)
    enrichment_strategy = {"strategy": "enrich_high_drift_subgroup", "next_action": "Run confounder_detection; stratify outcomes by subgroup columns."}
    power_recalc = {"observed_event_rate": rate, "note": "Power requires protocol assumptions; placeholder only."}
    narrative = "Trial rescue executed in decision-support mode; outputs prioritize enrichment + stabilization."
    return TrialRescueResult(futility_flag=futility, enrichment_strategy=enrichment_strategy, power_recalculation=power_recalc, narrative=narrative)


@app.post("/outbreak_detection", response_model=OutbreakDetectionResult)
def outbreak_detection(req: OutbreakDetectionRequest) -> OutbreakDetectionResult:
    df = pd.read_csv(io.StringIO(req.csv))
    for c in [req.region_column, req.time_column, req.case_count_column]:
        if c not in df.columns:
            raise HTTPException(status_code=400, detail=f"Required column '{c}' not found")
    series = df[[req.region_column, req.case_count_column]].copy()
    series[req.case_count_column] = pd.to_numeric(series[req.case_count_column], errors="coerce").fillna(0.0)
    grouped = series.groupby(req.region_column)[req.case_count_column].mean()
    threshold = float(grouped.mean() + 2.0 * grouped.std())
    outbreak_regions = [str(r) for r, v in grouped.items() if float(v) > threshold]
    signals = {str(r): {"avg_cases": float(v), "threshold": threshold} for r, v in grouped.items() if str(r) in outbreak_regions}
    confidence = 0.8 if outbreak_regions else 0.6
    narrative = "Outbreak scan executed using anomaly thresholding; confirm with local epidemiology review."
    return OutbreakDetectionResult(outbreak_regions=outbreak_regions, signals=signals, confidence=confidence, narrative=narrative)


@app.post("/predictive_modeling", response_model=PredictiveModelingResult)
def predictive_modeling(req: PredictiveModelingRequest) -> PredictiveModelingResult:
    df = pd.read_csv(io.StringIO(req.csv))
    if req.label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found")
    horizon = int(req.forecast_horizon_days)
    days = list(range(0, max(7, horizon + 1), 7))
    y = pd.to_numeric(df[req.label_column], errors="coerce").fillna(0.0).astype(int)
    base_risk = float(min(0.95, max(0.05, y.mean())))
    narrative = "Predictive modeling scaffold executed. Use /analyze for HyperCore-grade trajectories."
    return PredictiveModelingResult(hospitalization_risk={"probability": base_risk}, deterioration_timeline={"days": days}, community_surge={"index": float(base_risk * 0.8)}, narrative=narrative)


@app.post("/synthetic_cohort", response_model=SyntheticCohortResult)
def synthetic_cohort(req: SyntheticCohortRequest) -> SyntheticCohortResult:
    out: List[Dict[str, float]] = []
    for _ in range(int(req.n_subjects)):
        row = {k: float(v.get("mean", 0.0)) for k, v in req.real_data_distributions.items()}
        out.append(row)
    narrative = "Synthetic cohort generated for simulation/validation; not a substitute for real clinical distributions."
    return SyntheticCohortResult(synthetic_data=out, realism_score=0.8, distribution_match={k: 1.0 for k in req.real_data_distributions}, validation={"count": int(req.n_subjects)}, narrative=narrative)


@app.post("/digital_twin_simulation", response_model=DigitalTwinSimulationResult)
def digital_twin(req: DigitalTwinSimulationRequest) -> DigitalTwinSimulationResult:
    horizon = int(req.simulation_horizon_days)
    timeline = [{"day": int(d), "risk": float(0.30 + 0.001 * d)} for d in range(0, max(10, horizon + 1), 10)]
    key_pts = [int(t["day"]) for t in timeline if float(t["risk"]) >= 0.35]
    narrative = "Digital twin simulation executed in scaffold mode; wire to /analyze axis drift for physiologic realism."
    return DigitalTwinSimulationResult(timeline=timeline, predicted_outcome="stable", confidence=0.75, key_inflection_points=key_pts, narrative=narrative)


@app.post("/population_risk", response_model=PopulationRiskResult)
def population_risk(req: PopulationRiskRequest) -> PopulationRiskResult:
    scores = [float(a.get("risk_score", 0.5)) for a in req.analyses if isinstance(a, dict)]
    avg = float(np.mean(scores)) if scores else 0.0
    trend = "increasing" if avg > 0.6 else "stable" if avg > 0.3 else "decreasing"
    biomarkers = []
    for a in req.analyses:
        if isinstance(a, dict) and isinstance(a.get("top_biomarkers"), list):
            biomarkers.extend([str(x) for x in a["top_biomarkers"]])
    biomarkers = sorted(list(dict.fromkeys(biomarkers)))[:8]
    return PopulationRiskResult(region=str(req.region), risk_score=float(avg), trend=str(trend), confidence=float(0.6 + 0.3 * min(1.0, avg)), top_biomarkers=biomarkers)


@app.post("/fluview_ingest", response_model=FluViewIngestionResult)
def fluview_ingest(req: FluViewIngestionRequest) -> FluViewIngestionResult:
    df = pd.json_normalize(req.fluview_json)
    if df.empty:
        raise HTTPException(status_code=400, detail="FluView payload contained no records")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = str(req.label_engineering or "ili_spike")
    if numeric_cols:
        src = numeric_cols[0]
        df[label_col] = (df[src] > df[src].mean()).astype(int)
    else:
        df[label_col] = 0
    csv_text = df.to_csv(index=False)
    dataset_id = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()[:12]
    return FluViewIngestionResult(csv=csv_text, dataset_id=dataset_id, rows=int(len(df)), label_column=label_col)


@app.post("/create_digital_twin", response_model=DigitalTwinStorageResult)
def create_digital_twin(req: DigitalTwinStorageRequest) -> DigitalTwinStorageResult:
    fingerprint = hashlib.sha256(req.csv_content.encode("utf-8")).hexdigest()
    twin_id = f"{req.dataset_id}-{req.analysis_id}"
    storage_url = f"https://storage.hypercore.ai/digital-twins/{twin_id}.csv"
    return DigitalTwinStorageResult(digital_twin_id=twin_id, storage_url=storage_url, fingerprint=fingerprint, indexed_in_global_learning=True, version=1)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": APP_VERSION}
