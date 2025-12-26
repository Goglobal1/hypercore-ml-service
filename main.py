# main.py
# HyperCore GH-OS – Python ML Service v5.1 (REPORT-CONTRACT BUILD)
# - Keeps /analyze response schema stable for Base44
# - Adds /report endpoint that outputs HyperCore-style report sections consistently
# - Adds comparator benchmarking (NEWS/qSOFA/SIRS), silent-risk, shadow nonlinear model,
#   stratified K-fold CV when eligible, trajectory features, loop engine, negative-space engine,
#   and audit-grade execution manifest.

from __future__ import annotations

import io
import json
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
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

# ---------------------------------------------------------------------
# APP
# ---------------------------------------------------------------------

APP_VERSION = "5.1.0"

app = FastAPI(
    title="HyperCore GH-OS ML Service",
    version=APP_VERSION,
    description="Unified ML API for DiviScan HyperCore / DiviCore AI",
)

# ---------------------------------------------------------------------
# UTIL
# ---------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def json_hash(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, default=str))

def to_builtin(x: Any) -> Any:
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): to_builtin(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_builtin(v) for v in x]
    return x

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default

# ---------------------------------------------------------------------
# HYPERCORE CONSTANTS (minimal but extensible)
# ---------------------------------------------------------------------

AXES = [
    "inflammatory",
    "endocrine",
    "immune",
    "microbial",
    "metabolic",
    "cardiovascular",
    "neurologic",
    "nutritional",
]

# Axis routing by canonical lab keys (lowercase)
AXIS_LAB_MAP: Dict[str, List[str]] = {
    "inflammatory": ["crp", "esr", "ferritin", "il6", "il-6", "tnf", "procalcitonin", "pct", "d_dimer", "ddimer"],
    "endocrine": ["tsh", "t4", "ft4", "t3", "cortisol", "acth", "insulin"],
    "immune": ["wbc", "neutrophils", "neut", "lymphocytes", "lymph", "platelets", "plt"],
    "microbial": ["lactate", "procalcitonin", "pct", "endotoxin", "culture", "pcr"],
    "metabolic": ["glucose", "a1c", "hba1c", "bun", "creatinine", "triglycerides", "hdl", "ldl"],
    "cardiovascular": ["troponin", "bnp", "ntprobnp", "creatinine", "lactate"],
    "neurologic": ["sodium", "na", "potassium", "k", "calcium", "ca", "glucose"],
    "nutritional": ["albumin", "vitamin_d", "b12", "folate", "iron", "tibc", "transferrin_saturation", "rdw"],
}

# Reference ranges (minimal; fill progressively)
REFERENCE_RANGES = {
    "crp": {"low": 0.0, "high": 5.0},
    "wbc": {"low": 4.0, "high": 11.0},
    "glucose": {"low": 70.0, "high": 110.0},
    "creatinine": {"low": 0.6, "high": 1.3},
    "albumin": {"low": 3.4, "high": 5.4},
    "lactate": {"low": 0.5, "high": 2.2},
    "troponin": {"low": 0.0, "high": 0.04},
}

# Unit conversion rules (minimal; extend as needed)
UNIT_CONVERSIONS = {
    # unit -> (target_unit, factor)
    "g/l": ("mg/dl", 100.0),
    "mg/l": ("mg/dl", 0.1),
}

# Comparator score thresholds for blind-spot logic
COMPARATOR_THRESHOLDS = {"NEWS": 4.0, "qSOFA": 1.0, "SIRS": 1.0}

# Negative-space rules (deterministic “what was NOT ordered” logic)
REQUIRED_IF = [
    # trigger_tag -> required tests (strings)
    ("staph_aureus_bacteremia", ["TEE", "Repeat blood cultures x2"]),
    ("pituitary_surgery", ["Free T4 (dose by FT4, not TSH)"]),
    ("sinus_hyperdensity_immunosuppression", ["MRI brain/sinuses w/ contrast", "β-D-glucan", "Galactomannan", "ENT consult / culture"]),
    ("anemia_or_high_rdw", ["Ferritin", "Iron/TIBC/%Sat", "B12", "Folate"]),
    ("recurrent_infection_or_inflammation", ["Stool PCR panel", "Calprotectin (if GI symptoms)"]),
]

# ---------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------

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

# Report inputs (adds multi-modal inputs; all optional)
class NoteBlock(BaseModel):
    source: str = "unknown"
    timestamp: Optional[str] = None
    text: str

class ImagingFinding(BaseModel):
    modality: str  # CT/XR/Echo/MRI
    study_date: Optional[str] = None
    impression: str
    key_values: Dict[str, Any] = Field(default_factory=dict)

class ReportRequest(BaseModel):
    csv: str
    label_column: str

    # optional schema hints for “long format” CSV
    patient_id_column: Optional[str] = None
    time_column: Optional[str] = None
    lab_name_column: Optional[str] = None
    value_column: Optional[str] = None
    unit_column: Optional[str] = None

    # demographics/context
    sex: Optional[str] = None
    age: Optional[float] = None
    context: Dict[str, Any] = Field(default_factory=dict)

    # multi-modal
    notes: List[NoteBlock] = Field(default_factory=list)
    imaging: List[ImagingFinding] = Field(default_factory=list)
    meds: List[str] = Field(default_factory=list)

    # governance mode tags
    usage_mode: str = "quality_improvement"  # e.g. quality_improvement, retrospective_audit
    not_for: str = "diagnosis"

class ReportResponse(BaseModel):
    report_schema_version: str
    generated_at: str
    engine_version: str
    input_fingerprint: str

    # core HyperCore sections
    multi_axis_driver_mapping: Dict[str, Any]
    crosstalk_loops: List[Dict[str, Any]]
    diagnostic_models: List[Dict[str, Any]]
    baseline_benchmarking: Dict[str, Any]
    silent_risk: Dict[str, Any]
    missed_opportunities: List[Dict[str, Any]]
    risk_forecast: Dict[str, Any]
    therapeutic_modeling: Dict[str, Any]
    simulation: Dict[str, Any]
    explainability: Dict[str, Any]

    # audit/repro
    execution_manifest: Dict[str, Any]
    governance: Dict[str, Any]

# ---------------------------------------------------------------------
# BASELINE /ANALYZE (KEEP STABLE)
# ---------------------------------------------------------------------

def compute_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {"sensitivity": sensitivity, "specificity": specificity}

def clean_feature_matrix_wide(df: pd.DataFrame, label_column: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], List[str]]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    df = df.copy()
    df[label_column] = pd.to_numeric(df[label_column], errors="raise")
    y = df[label_column].values
    X_raw = df.drop(columns=[label_column])

    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns")

    X = X_raw[numeric_cols].copy()
    variances = X.var()
    keep = variances[variances > 0].index.tolist()
    dropped = [c for c in numeric_cols if c not in keep]
    if not keep:
        raise ValueError("All features have zero variance")

    return X[keep], y, keep, dropped

def logistic_regression_analysis(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    X, y, feature_cols, dropped_cols = clean_feature_matrix_wide(df, label_column)

    if len(np.unique(y)) < 2:
        raise ValueError("Label must contain at least two classes")

    if len(y) < 30:
        X_train = X_test = X
        y_train = y_test = y
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )

    model = LogisticRegression(max_iter=1000, solver="liblinear")
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    pr_auc = average_precision_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0
    acc = accuracy_score(y_test, y_pred)
    sens_spec = compute_sensitivity_specificity(y_test, y_pred)

    fpr, tpr, roc_thr = roc_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else (np.array([0,1]), np.array([0,1]), np.array([0.5]))
    prec, rec, pr_thr = precision_recall_curve(y_test, y_prob) if len(np.unique(y_test)) > 1 else (np.array([1,0]), np.array([0,1]), np.array([0.5]))

    coef = model.coef_[0]
    abs_coef = np.abs(coef)
    importance = abs_coef / abs_coef.sum() if abs_coef.sum() > 0 else abs_coef

    return {
        "metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "accuracy": float(acc),
            "sensitivity": float(sens_spec["sensitivity"]),
            "specificity": float(sens_spec["specificity"]),
        },
        "coefficients": {f: float(c) for f, c in zip(feature_cols, coef)},
        "roc_curve_data": {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr], "thresholds": [float(x) for x in roc_thr]},
        "pr_curve_data": {"precision": [float(x) for x in prec], "recall": [float(x) for x in rec], "thresholds": [float(x) for x in pr_thr]},
        "feature_importance": [{"feature": f, "importance": float(i)} for f, i in zip(feature_cols, importance)],
        "dropped_features": dropped_cols,
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    try:
        df = pd.read_csv(io.StringIO(req.csv))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")
    try:
        result = logistic_regression_analysis(df, req.label_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AnalyzeResponse(**result)

# ---------------------------------------------------------------------
# HYPERCORE REPORT PIPELINE (multi-modal, deterministic schema)
# ---------------------------------------------------------------------

def ensure_patient_id(df: pd.DataFrame, patient_id_column: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if patient_id_column and patient_id_column in df.columns:
        return df, patient_id_column
    if "patient_id" in df.columns:
        return df, "patient_id"
    out = df.copy()
    out["patient_id"] = out.index.astype(str)
    return out, "patient_id"

def ensure_time_column(df: pd.DataFrame, time_column: Optional[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    if time_column and time_column in df.columns:
        return df, time_column
    if "time" in df.columns:
        return df, "time"
    if "timestamp" in df.columns:
        return df, "timestamp"
    return df, None

def ingest_labs(
    df: pd.DataFrame,
    label_column: str,
    patient_id_column: Optional[str],
    time_column: Optional[str],
    lab_name_column: Optional[str],
    value_column: Optional[str],
    unit_column: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, pid_col = ensure_patient_id(df, patient_id_column)
    df, t_col = ensure_time_column(df, time_column)

    # Long format preferred if provided
    if lab_name_column and value_column and lab_name_column in df.columns and value_column in df.columns:
        long_df = df.copy()
        rename_map: Dict[str, str] = {lab_name_column: "lab_name", value_column: "value", pid_col: "patient_id"}
        if t_col:
            rename_map[t_col] = "time"
        if unit_column and unit_column in df.columns:
            rename_map[unit_column] = "unit"
        long_df = long_df.rename(columns=rename_map)
        if "time" not in long_df.columns:
            long_df["time"] = None
        if "unit" not in long_df.columns:
            long_df["unit"] = None
        fmt = "long"
        keep_cols = ["patient_id", "time", "lab_name", "value", "unit"]
        if label_column in long_df.columns:
            keep_cols.append(label_column)
        long_df = long_df[keep_cols].copy()
    else:
        # Wide: melt numeric cols except label/pid/time/unit
        exclude = {label_column, pid_col}
        if t_col:
            exclude.add(t_col)
        if unit_column and unit_column in df.columns:
            exclude.add(unit_column)
        feature_cols = [c for c in df.columns if c not in exclude]
        id_vars = [pid_col] + ([t_col] if t_col else []) + ([label_column] if label_column in df.columns else [])
        long_df = df.melt(id_vars=id_vars, value_vars=feature_cols, var_name="lab_name", value_name="value")
        rename_map2: Dict[str, str] = {pid_col: "patient_id"}
        if t_col:
            rename_map2[t_col] = "time"
        long_df = long_df.rename(columns=rename_map2)
        if "time" not in long_df.columns:
            long_df["time"] = None
        if unit_column and unit_column in df.columns:
            long_df["unit"] = df[unit_column].iloc[0]
        else:
            long_df["unit"] = None
        fmt = "wide"

    long_df["lab_name"] = long_df["lab_name"].astype(str).str.strip().str.lower()
    # canonicalize some common keys
    long_df["lab_name"] = long_df["lab_name"].str.replace(" ", "_").str.replace("-", "_")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])

    meta = {
        "format": fmt,
        "records": int(len(long_df)),
        "patients": int(long_df["patient_id"].nunique()),
        "labs": int(long_df["lab_name"].nunique()),
        "label_present": bool(label_column in long_df.columns),
    }
    return long_df, meta

def normalize_units(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    df["unit"] = df.get("unit", None)
    if "unit" not in df.columns:
        df["unit"] = None
    df["unit"] = df["unit"].fillna("").astype(str).str.strip().str.lower()

    applied: List[Dict[str, Any]] = []
    for unit, (target_unit, factor) in UNIT_CONVERSIONS.items():
        mask = df["unit"] == unit.lower()
        if mask.any():
            df.loc[mask, "value"] = df.loc[mask, "value"].astype(float) * float(factor)
            df.loc[mask, "unit"] = target_unit
            applied.append({"from": unit, "to": target_unit, "factor": float(factor), "count": int(mask.sum())})
    return df, {"conversions": applied}

def apply_reference_ranges(labs: pd.DataFrame, sex: Optional[str], age: Optional[float], context: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    df["lab_key"] = df["lab_name"].astype(str).str.lower()

    sex_key = (sex or "").strip().lower()
    # Basic contextual overrides for reference highs (extend as needed)
    overrides: List[str] = []

    def pick_range(lab: str) -> Tuple[float, float]:
        base = REFERENCE_RANGES.get(lab, {"low": 0.0, "high": 1.0})
        low = float(base["low"])
        high = float(base["high"])
        if lab == "creatinine" and sex_key in {"f", "female"}:
            high = min(high, 1.1)
        if lab == "creatinine" and age is not None and age > 65:
            high += 0.2
        return low, high

    df["ref_low"] = df["lab_key"].apply(lambda lab: pick_range(lab)[0])
    df["ref_high"] = df["lab_key"].apply(lambda lab: pick_range(lab)[1])

    # context overrides (examples)
    if context.get("pregnancy") is True:
        mask = df["lab_key"] == "wbc"
        df.loc[mask, "ref_high"] = df.loc[mask, "ref_high"] + 1.0
        overrides.append("pregnancy_wbc_ref_high")
    if context.get("renal_failure") is True:
        mask = df["lab_key"] == "creatinine"
        df.loc[mask, "ref_high"] = df.loc[mask, "ref_high"] + 0.5
        overrides.append("renal_failure_creatinine_ref_high")

    df["out_of_range"] = (df["value"] < df["ref_low"]) | (df["value"] > df["ref_high"])
    mid = (df["ref_low"] + df["ref_high"]) / 2.0
    span = (df["ref_high"] - df["ref_low"]).replace(0.0, np.nan) / 2.0
    df["z_score"] = ((df["value"] - mid) / span).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, {"reference_ranges_applied": True, "context_overrides": overrides}

def align_time_series(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    if "time" in df.columns and df["time"].notna().any():
        df["time_parsed"] = pd.to_datetime(df["time"], errors="coerce")
    else:
        df["time_parsed"] = pd.NaT
    df = df.sort_values(by=["patient_id", "lab_name", "time_parsed"])

    df["baseline_value"] = df.groupby(["patient_id", "lab_name"])["value"].transform("first")
    df["baseline_time"] = df.groupby(["patient_id", "lab_name"])["time_parsed"].transform("first")
    df["delta"] = df["value"] - df["baseline_value"]

    dt_hours = (df["time_parsed"] - df["baseline_time"]).dt.total_seconds() / 3600.0
    dt_hours = dt_hours.replace(0, np.nan)
    df["rate_of_change"] = (df["delta"] / dt_hours).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df, {"aligned": True}

def extract_features(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Builds patient-level matrix with:
    - latest, mean, min, max, std
    - latest z-score
    - out_of_range flag
    - missingness indicator
    """
    df = labs.copy()
    if "time_parsed" not in df.columns:
        df["time_parsed"] = pd.NaT
    df = df.sort_values(by=["patient_id", "lab_name", "time_parsed"])

    latest = df.groupby(["patient_id", "lab_name"]).tail(1).set_index(["patient_id", "lab_name"])
    grouped = df.groupby(["patient_id", "lab_name"])

    stats = grouped["value"].agg(["mean", "min", "max", "std", "count"])
    out_any = grouped["out_of_range"].max()
    latest_val = latest["value"].unstack()
    latest_z = latest["z_score"].unstack()
    latest_val.columns = [f"{c}_latest" for c in latest_val.columns]
    latest_z.columns = [f"{c}_latest_z" for c in latest_z.columns]

    mats: List[pd.DataFrame] = [latest_val, latest_z]

    for stat_name in ["mean", "min", "max", "std"]:
        m = stats[stat_name].unstack()
        m.columns = [f"{c}_{stat_name}" for c in m.columns]
        mats.append(m)

    oor = out_any.unstack().fillna(False).astype(int)
    oor.columns = [f"{c}_out_of_range" for c in oor.columns]
    mats.append(oor)

    presence = stats["count"].unstack().fillna(0)
    missing = (presence == 0).astype(int)
    missing.columns = [f"{c}_missing" for c in missing.columns]
    mats.append(missing)

    feature_df = pd.concat(mats, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_index()
    meta = {"feature_count": int(feature_df.shape[1])}
    return feature_df, meta

def trajectory_features(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Trajectory primitives:
    - slope (per lab, per patient) via simple linear fit on time index
    - volatility (std)
    - persistence (max consecutive direction)
    """
    df = labs.copy()
    # Build time index for each series
    df = df.sort_values(by=["patient_id", "lab_name", "time_parsed"])
    feats: Dict[str, Dict[str, float]] = {}
    for (pid, lab), grp in df.groupby(["patient_id", "lab_name"]):
        series = grp["value"].astype(float).values
        if len(series) == 0:
            continue
        # time axis: if time_parsed exists, use ordinal; else use index
        if grp["time_parsed"].notna().any():
            t = grp["time_parsed"].astype("int64").values.astype(float)
        else:
            t = np.arange(len(series), dtype=float)

        # slope
        if len(series) >= 2 and np.std(t) > 0:
            slope = float(np.polyfit(t, series, 1)[0])
        else:
            slope = 0.0

        # volatility
        vol = float(np.std(series)) if len(series) >= 2 else 0.0

        # persistence: longest run of increasing or decreasing deltas
        diffs = np.diff(series)
        if len(diffs) == 0:
            persistence = 0
        else:
            signs = np.sign(diffs)
            best = cur = 1
            for i in range(1, len(signs)):
                if signs[i] == signs[i - 1] and signs[i] != 0:
                    cur += 1
                    best = max(best, cur)
                else:
                    cur = 1
            persistence = int(best)

        feats.setdefault(pid, {})
        feats[pid][f"{lab}_slope"] = slope
        feats[pid][f"{lab}_volatility"] = vol
        feats[pid][f"{lab}_persistence"] = float(persistence)

    if not feats:
        return pd.DataFrame(), {"trajectory_features": 0}

    traj_df = pd.DataFrame.from_dict(feats, orient="index").fillna(0.0)
    traj_df.index.name = "patient_id"
    return traj_df, {"trajectory_features": int(traj_df.shape[1])}

def axis_decomposition(labs: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = labs.copy()
    axis_scores: Dict[str, pd.Series] = {}
    axis_drivers: Dict[str, Dict[str, Any]] = {}

    for axis in AXES:
        keys = set(AXIS_LAB_MAP.get(axis, []))
        axis_df = df[df["lab_key"].isin(keys)]
        if axis_df.empty:
            axis_scores[axis] = pd.Series(dtype=float)
            axis_drivers[axis] = {"missing": True, "top_drivers": []}
            continue
        scores = axis_df.groupby("patient_id")["z_score"].mean()
        axis_scores[axis] = scores
        strength = axis_df.groupby("lab_name")["z_score"].mean().abs().sort_values(ascending=False)
        axis_drivers[axis] = {"missing": False, "top_drivers": list(strength.head(3).index)}

    axis_mat = pd.DataFrame(axis_scores).fillna(0.0)
    summary = {
        axis: {
            "mean_score": float(axis_mat[axis].mean()) if axis in axis_mat.columns else 0.0,
            "top_drivers": axis_drivers.get(axis, {}).get("top_drivers", []),
            "missing": axis_drivers.get(axis, {}).get("missing", True),
        }
        for axis in AXES
    }
    return axis_mat, summary

def interaction_graph(axis_mat: pd.DataFrame) -> List[Dict[str, Any]]:
    if axis_mat.empty:
        return []
    mean_scores = axis_mat.mean()
    out: List[Dict[str, Any]] = []
    for a, b in combinations(mean_scores.index, 2):
        s = float(mean_scores[a] + mean_scores[b])
        out.append({"axes": [a, b], "combined_score": s, "amplified": bool(s > 1.0)})
    out.sort(key=lambda d: d["combined_score"], reverse=True)
    return out[:12]

def feedback_loops(context: Dict[str, Any], labs: pd.DataFrame, notes: List[NoteBlock], imaging: List[ImagingFinding], meds: List[str]) -> List[Dict[str, Any]]:
    """
    Deterministic loop engine. Extend these rules to match your report library.
    """
    loops: List[Dict[str, Any]] = []
    imaging_text = " ".join([i.impression for i in imaging]).lower()
    notes_text = " ".join([n.text for n in notes]).lower()
    meds_text = " ".join(meds).lower()

    lab_keys = set(labs["lab_key"].unique())

    # Example loop: infection + adrenal axis + glucose instability
    if ("cortisol" in lab_keys or "acth" in lab_keys) and ("glucose" in lab_keys) and ("pneumonia" in imaging_text or "infection" in notes_text):
        loops.append({
            "loop": "infection ↔ adrenal axis ↔ glucose instability",
            "clinical_risk": "adrenal crisis recurrence + hypoglycemia",
            "priority": "critical",
            "evidence": ["cortisol/ACTH present", "glucose present", "infection signal in notes/imaging"],
        })

    # Example loop: steroids + sinus hyperdensity → fungal risk
    if ("steroid" in meds_text or "prednisone" in meds_text or "hydrocortisone" in meds_text) and ("sphenoid" in imaging_text and "hyperdense" in imaging_text):
        loops.append({
            "loop": "immune suppression ↔ sinus disease ↔ invasive fungal risk",
            "clinical_risk": "invasive fungal sinusitis",
            "priority": "high",
            "evidence": ["steroid exposure", "hyperdense sphenoid sinus signal"],
        })

    return loops

def baseline_benchmarking(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"comparators": {}, "available": False}
    if label_column not in df.columns:
        return out
    y = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int).values
    if len(np.unique(y)) < 2:
        return out

    for col, thr in COMPARATOR_THRESHOLDS.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(float)
            if np.std(s) > 0:
                out["comparators"][col] = {
                    "roc_auc": float(roc_auc_score(y, s)),
                    "threshold": float(thr),
                    "count": int(len(s)),
                }
    out["available"] = bool(out["comparators"])
    return out

def silent_risk_detection(df: pd.DataFrame, label_column: str) -> Dict[str, Any]:
    """
    Blind-spot logic: among patients considered acceptable by standard score,
    compute adverse outcome prevalence and provide medians/missingness.
    """
    result: Dict[str, Any] = {"by_score": {}, "available": False}
    if label_column not in df.columns:
        return result
    y = pd.to_numeric(df[label_column], errors="coerce").fillna(0).astype(int)

    for col, thr in COMPARATOR_THRESHOLDS.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").fillna(np.inf)
        acceptable = df[s <= thr]
        if len(acceptable) == 0:
            continue
        adverse = acceptable[pd.to_numeric(acceptable[label_column], errors="coerce").fillna(0).astype(int) == 1]
        numeric = acceptable.select_dtypes(include=[np.number])
        result["by_score"][col] = {
            "threshold": float(thr),
            "acceptable_n": int(len(acceptable)),
            "adverse_n": int(len(adverse)),
            "adverse_rate": float(len(adverse) / len(acceptable)),
            "median_table": to_builtin(numeric.median().to_dict()),
            "missingness_table": to_builtin(acceptable.isna().mean().to_dict()),
        }
    result["available"] = bool(result["by_score"])
    return result

def negative_space_engine(labs: pd.DataFrame, notes: List[NoteBlock], imaging: List[ImagingFinding], meds: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deterministic missed opportunities: trigger tags -> required tests.
    """
    present_tests = set([str(x).lower() for x in labs["lab_name"].unique()])

    imaging_text = " ".join([i.impression for i in imaging]).lower()
    notes_text = " ".join([n.text for n in notes]).lower()
    meds_text = " ".join(meds).lower()

    tags = {
        "staph_aureus_bacteremia": ("staph aureus" in notes_text) or ("staphylococcus aureus" in notes_text),
        "pituitary_surgery": any("pituitary" in str(s).lower() or "hypophy" in str(s).lower() for s in context.get("surgeries", [])),
        "sinus_hyperdensity_immunosuppression": ("hyperdense" in imaging_text and "sinus" in imaging_text) and ("steroid" in meds_text or "prednisone" in meds_text),
        "anemia_or_high_rdw": ("rdw" in present_tests) or ("hemoglobin" in present_tests),
        "recurrent_infection_or_inflammation": ("recurrent" in notes_text) or ("chronic" in notes_text) or ("crp" in present_tests),
    }

    missed: List[Dict[str, Any]] = []
    for trigger, tests in REQUIRED_IF:
        if tags.get(trigger, False):
            missing = [t for t in tests if t.lower() not in present_tests]
            if missing:
                missed.append({
                    "trigger": trigger,
                    "missing": missing,
                    "severity": "high" if trigger in {"staph_aureus_bacteremia", "sinus_hyperdensity_immunosuppression"} else "moderate",
                })
    return missed

@dataclass
class ModelRun:
    name: str
    kind: str
    metrics: Dict[str, float]
    feature_importance: List[Dict[str, Any]]
    coefficients: Dict[str, float]
    curves: Dict[str, Any]
    cv_method: str
    stability: Dict[str, Any]

def model_suite(X: pd.DataFrame, y: np.ndarray) -> Tuple[ModelRun, ModelRun]:
    """
    Linear model + shadow nonlinear model + CV when eligible.
    """
    # Clean
    Xc = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    variances = Xc.var()
    keep = variances[variances > 0].index.tolist()
    dropped = [c for c in Xc.columns if c not in keep]
    Xc = Xc[keep] if keep else Xc

    # stability rule
    stability: Dict[str, Any] = {"n": int(len(y)), "rule": None, "eligible_kfold": False}
    if len(y) >= 100 and len(np.unique(y)) == 2:
        # K-fold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        stability["rule"] = "StratifiedKFold(k=5)"
        stability["eligible_kfold"] = True
        cv_method = "StratifiedKFold(5)"
    else:
        cv_method = "train_test_split"
        stability["rule"] = "train_test_split"

    # ---------- Linear ----------
    lin = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
    if stability["eligible_kfold"]:
        probs = cross_val_predict(lin, Xc, y, cv=cv, method="predict_proba")[:, 1]
        lin.fit(Xc, y)
        y_pred = (probs >= 0.5).astype(int)
        y_eval = y
        lin_coeff = lin.coef_[0]
        fpr, tpr, roc_thr = roc_curve(y_eval, probs)
        prec, rec, pr_thr = precision_recall_curve(y_eval, probs)
    else:
        X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=0.3, random_state=42, stratify=y) if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else train_test_split(Xc, y, test_size=0.3, random_state=42)
        lin.fit(X_train, y_train)
        probs = lin.predict_proba(X_test)[:, 1] if len(X_test) else lin.predict_proba(X_train)[:, 1]
        y_eval = y_test if len(X_test) else y_train
        y_pred = (probs >= 0.5).astype(int)
        lin_coeff = lin.coef_[0]
        if len(np.unique(y_eval)) > 1:
            fpr, tpr, roc_thr = roc_curve(y_eval, probs)
            prec, rec, pr_thr = precision_recall_curve(y_eval, probs)
        else:
            fpr, tpr, roc_thr = np.array([0,1]), np.array([0,1]), np.array([0.5])
            prec, rec, pr_thr = np.array([1,0]), np.array([0,1]), np.array([0.5])

    lin_metrics = {
        "roc_auc": float(roc_auc_score(y_eval, probs)) if len(np.unique(y_eval)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(y_eval, probs)) if len(np.unique(y_eval)) > 1 else 0.0,
        "accuracy": float(accuracy_score(y_eval, y_pred)),
        **compute_sensitivity_specificity(y_eval, y_pred),
    }

    abs_coef = np.abs(lin_coeff)
    lin_imp = abs_coef / abs_coef.sum() if abs_coef.sum() > 0 else abs_coef
    lin_fi = [{"feature": f, "importance": float(i)} for f, i in zip(Xc.columns, lin_imp)]
    lin_fi.sort(key=lambda d: d["importance"], reverse=True)

    lin_run = ModelRun(
        name="logistic_regression",
        kind="linear",
        metrics=lin_metrics,
        feature_importance=lin_fi[:50],
        coefficients={f: float(c) for f, c in zip(Xc.columns, lin_coeff)},
        curves={
            "roc_curve_data": {"fpr": [float(x) for x in fpr], "tpr": [float(x) for x in tpr], "thresholds": [float(x) for x in roc_thr]},
            "pr_curve_data": {"precision": [float(x) for x in prec], "recall": [float(x) for x in rec], "thresholds": [float(x) for x in pr_thr]},
            "dropped_features": dropped,
        },
        cv_method=cv_method,
        stability=stability,
    )

    # ---------- Nonlinear (shadow) ----------
    # Use HistGradientBoosting for interactions; fallback to RF if needed
    # We'll use RF + permutation importance because it's stable with existing deps.
    nl = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    if stability["eligible_kfold"]:
        nl_probs = cross_val_predict(nl, Xc, y, cv=cv, method="predict_proba")[:, 1]
        nl.fit(Xc, y)
        nl_eval = y
        nl_pred = (nl_probs >= 0.5).astype(int)
        nl_cv = "StratifiedKFold(5)"
    else:
        # same split as above is not guaranteed; acceptable for shadow comparison
        X_train, X_test, y_train, y_test = train_test_split(Xc, y, test_size=0.3, random_state=42, stratify=y) if len(np.unique(y)) == 2 and min(np.bincount(y)) >= 2 else train_test_split(Xc, y, test_size=0.3, random_state=42)
        nl.fit(X_train, y_train)
        nl_probs = nl.predict_proba(X_test)[:, 1] if len(X_test) else nl.predict_proba(X_train)[:, 1]
        nl_eval = y_test if len(X_test) else y_train
        nl_pred = (nl_probs >= 0.5).astype(int)
        nl_cv = "train_test_split"

    nl_metrics = {
        "roc_auc": float(roc_auc_score(nl_eval, nl_probs)) if len(np.unique(nl_eval)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(nl_eval, nl_probs)) if len(np.unique(nl_eval)) > 1 else 0.0,
        "accuracy": float(accuracy_score(nl_eval, nl_pred)),
        **compute_sensitivity_specificity(nl_eval, nl_pred),
    }

    # permutation importance (where feasible)
    fi_list: List[Dict[str, Any]] = []
    try:
        if stability["eligible_kfold"]:
            # approximate: use full dataset for permutation (still informative)
            perm = permutation_importance(nl, Xc, y, n_repeats=5, random_state=42)
            vals = perm.importances_mean
        else:
            # use test fold if exists, else full
            vals = nl.feature_importances_
        fi_list = [{"feature": f, "importance": float(v)} for f, v in zip(Xc.columns, vals)]
        fi_list.sort(key=lambda d: d["importance"], reverse=True)
    except Exception:
        fi_list = []

    nl_run = ModelRun(
        name="random_forest",
        kind="nonlinear_shadow",
        metrics=nl_metrics,
        feature_importance=fi_list[:50],
        coefficients={},  # nonlinear
        curves={},
        cv_method=nl_cv,
        stability={"shadow_mode": True, **stability},
    )

    return lin_run, nl_run

def report_forecast(axis_mat: pd.DataFrame) -> Dict[str, Any]:
    """
    Multi-horizon risk forecast (heuristic, deterministic, reproducible).
    """
    if axis_mat.empty:
        return {"risk_72h": 0.0, "risk_30d": 0.0, "risk_90d": 0.0, "risk_6m": 0.0, "risk_12m": 0.0}

    mean_score = float(axis_mat.mean().mean())
    # logistic transform
    def sig(z: float) -> float:
        return float(1.0 / (1.0 + math.exp(-z)))

    return {
        "risk_72h": sig(mean_score),
        "risk_30d": sig(mean_score - 0.2),
        "risk_90d": sig(mean_score - 0.4),
        "risk_6m": sig(mean_score - 0.55),
        "risk_12m": sig(mean_score - 0.7),
    }

def therapeutic_model(axis_summary: Dict[str, Any], loops: List[Dict[str, Any]], missed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministic “what changes the trajectory” recommendations (non-prescriptive).
    """
    priorities: List[Dict[str, Any]] = []

    # axis-based priorities
    for axis, info in axis_summary.items():
        score = safe_float(info.get("mean_score", 0.0))
        if score > 0.6:
            priorities.append({
                "type": "axis_target",
                "axis": axis,
                "priority": "high" if score > 0.9 else "moderate",
                "rationale": f"{axis} axis elevated (mean_score={score:.2f})",
                "suggested_actions": [f"Confirm top drivers: {', '.join(info.get('top_drivers', [])[:3])}"],
            })

    # loop-based priorities
    for loop in loops:
        priorities.append({
            "type": "feedback_loop",
            "priority": loop.get("priority", "moderate"),
            "rationale": loop.get("loop"),
            "clinical_risk": loop.get("clinical_risk"),
            "suggested_actions": ["Verify loop evidence; treat as escalation candidate in human review"],
        })

    # missed opportunities (negative space)
    for mo in missed:
        priorities.append({
            "type": "missed_opportunity",
            "priority": mo.get("severity", "moderate"),
            "rationale": f"Missing follow-ups triggered by {mo.get('trigger')}",
            "suggested_actions": mo.get("missing", []),
        })

    priorities.sort(key=lambda d: {"critical": 0, "high": 1, "moderate": 2, "low": 3}.get(str(d.get("priority")).lower(), 9))
    return {"priorities": priorities[:15]}

def simulation_layer(forecast: Dict[str, Any], lin: ModelRun, nl: ModelRun) -> Dict[str, Any]:
    """
    Deterministic simulation outputs:
    - model comparison
    - forecast deltas
    """
    return {
        "model_comparison": {
            "linear_auc": lin.metrics.get("roc_auc", 0.0),
            "nonlinear_auc": nl.metrics.get("roc_auc", 0.0),
            "delta_auc": float(nl.metrics.get("roc_auc", 0.0) - lin.metrics.get("roc_auc", 0.0)),
            "shadow_mode": True,
        },
        "counterfactual_notes": [
            "Nonlinear model is shadow-mode: used for comparison only; no automated decision output.",
            "Forecasts are heuristic transforms of axis load; intended for retrospective intelligence and triage discussion.",
        ],
        "forecast": forecast,
    }

def drivers_from_axes(axis_summary: Dict[str, Any], labs: pd.DataFrame) -> Dict[str, Any]:
    """
    Build consistent driver mapping object.
    """
    # Top abnormal labs by mean |z_score|
    z_strength = labs.groupby("lab_name")["z_score"].mean().abs().sort_values(ascending=False)
    top_labs = [{"signal": k, "mean_abs_z": float(v)} for k, v in z_strength.head(10).items()]

    return {
        "axes": axis_summary,
        "top_lab_drivers": top_labs,
    }

def patient_friendly_summary(axis_summary: Dict[str, Any], loops: List[Dict[str, Any]], forecast: Dict[str, Any]) -> str:
    """
    Short, consistent narrative seed (Firebase can expand this).
    """
    # pick top 2 axes by mean_score
    items = [(a, safe_float(v.get("mean_score", 0.0))) for a, v in axis_summary.items()]
    items.sort(key=lambda x: x[1], reverse=True)
    top_axes = [a for a, _ in items[:2] if _ > 0]

    loop_flag = "yes" if len(loops) > 0 else "no"
    return (
        f"Summary: Highest signal load appears in: {', '.join(top_axes) or 'no dominant axis detected'}. "
        f"Feedback-loop patterns detected: {loop_flag}. "
        f"Estimated risk trend (heuristic): 72h={forecast.get('risk_72h', 0.0):.2f}, 30d={forecast.get('risk_30d', 0.0):.2f}, 90d={forecast.get('risk_90d', 0.0):.2f}. "
        f"This is decision-support only and requires clinical review."
    )

def build_execution_manifest(req: ReportRequest, input_hash: str, stage_hashes: Dict[str, str], lin: ModelRun, nl: ModelRun) -> Dict[str, Any]:
    return {
        "manifest_version": "1.0.0",
        "generated_at": utc_now_iso(),
        "engine_version": APP_VERSION,
        "input_fingerprint": input_hash,
        "stages": stage_hashes,
        "models": {
            "linear": {"name": lin.name, "cv_method": lin.cv_method, "stability": lin.stability, "metrics": lin.metrics},
            "nonlinear_shadow": {"name": nl.name, "cv_method": nl.cv_method, "stability": nl.stability, "metrics": nl.metrics},
        },
        "governance": {"usage_mode": req.usage_mode, "not_for": req.not_for},
    }

# ---------------------------------------------------------------------
# /report endpoint (the contract you need Base44 + Firebase to render)
# ---------------------------------------------------------------------

@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest):
    # Parse CSV
    try:
        df = pd.read_csv(io.StringIO(req.csv))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if req.label_column not in df.columns:
        raise HTTPException(status_code=400, detail=f"label_column '{req.label_column}' not found in CSV")

    input_fingerprint = sha256_text(req.csv + "|" + req.label_column)

    # Ingest → canonicalize → features
    labs, ingest_meta = ingest_labs(
        df=df,
        label_column=req.label_column,
        patient_id_column=req.patient_id_column,
        time_column=req.time_column,
        lab_name_column=req.lab_name_column,
        value_column=req.value_column,
        unit_column=req.unit_column,
    )
    stage_hashes: Dict[str, str] = {"ingest": json_hash(ingest_meta)}

    labs, unit_meta = normalize_units(labs)
    stage_hashes["unit_norm"] = json_hash(unit_meta)

    labs, rr_meta = apply_reference_ranges(labs, req.sex, req.age, req.context or {})
    stage_hashes["ref_ranges"] = json_hash(rr_meta)

    labs, ts_meta = align_time_series(labs)
    stage_hashes["time_align"] = json_hash(ts_meta)

    feature_df, feat_meta = extract_features(labs)
    stage_hashes["features"] = json_hash(feat_meta)

    traj_df, traj_meta = trajectory_features(labs)
    stage_hashes["trajectory"] = json_hash(traj_meta)

    # Combine patient feature matrix
    X = feature_df.join(traj_df, how="left").fillna(0.0)

    # Labels per patient
    # If label is per-row, take max per patient
    y_series = (
        labs[["patient_id", req.label_column]]
        .dropna()
        .drop_duplicates()
        .groupby("patient_id")[req.label_column]
        .max()
    )
    # Align
    y_series = y_series.reindex(X.index).dropna()
    X = X.loc[y_series.index]
    y = pd.to_numeric(y_series, errors="coerce").fillna(0).astype(int).values

    if len(np.unique(y)) < 2:
        raise HTTPException(status_code=400, detail="Label must contain at least two classes (0/1) after patient aggregation.")

    # Axes + interactions + loops
    axis_mat, axis_summary = axis_decomposition(labs)
    axis_mat = axis_mat.reindex(X.index).fillna(0.0)
    interactions = interaction_graph(axis_mat)
    loops = feedback_loops(req.context or {}, labs, req.notes, req.imaging, req.meds)

    # Models (linear + shadow nonlinear)
    lin_run, nl_run = model_suite(X, y)

    # Baseline benchmarking + silent risk
    bench = baseline_benchmarking(df, req.label_column)
    silent = silent_risk_detection(df, req.label_column)

    # Missed opportunities (negative space)
    missed = negative_space_engine(labs, req.notes, req.imaging, req.meds, req.context or {})

    # Forecast + therapeutic modeling + simulation
    forecast = report_forecast(axis_mat)
    therapy = therapeutic_model(axis_summary, loops, missed)
    sim = simulation_layer(forecast, lin_run, nl_run)

    # Explainability block (structured)
    explain = {
        "linear": {
            "directionality": {k: ("↑" if v > 0 else "↓" if v < 0 else "→") for k, v in lin_run.coefficients.items()},
            "top_features": lin_run.feature_importance[:15],
        },
        "nonlinear_shadow": {
            "top_features": nl_run.feature_importance[:15],
            "shadow_mode": True,
        },
        "feature_medians": {
            "event": to_builtin(pd.DataFrame(X).assign(label=y).query("label==1").median(numeric_only=True).to_dict()),
            "non_event": to_builtin(pd.DataFrame(X).assign(label=y).query("label==0").median(numeric_only=True).to_dict()),
        },
        "missingness": to_builtin(pd.DataFrame(X).isna().mean().to_dict()),
    }

    # Diagnostic models list (report-friendly)
    diag_models = [
        {
            "model": "logistic_regression",
            "role": "explainable_primary",
            "metrics": lin_run.metrics,
            "cv_method": lin_run.cv_method,
            "stability": lin_run.stability,
        },
        {
            "model": "random_forest",
            "role": "interaction_shadow_comparator",
            "metrics": nl_run.metrics,
            "cv_method": nl_run.cv_method,
            "stability": nl_run.stability,
            "shadow_mode": True,
        },
    ]

    # Multi-axis drivers mapping (report section)
    driver_mapping = drivers_from_axes(axis_summary, labs)

    # Governance tags
    governance = {
        "use": req.usage_mode,
        "not_for": req.not_for,
        "human_in_the_loop": True,
        "non_diagnostic": True,
    }

    # Execution manifest
    execution_manifest = build_execution_manifest(req, input_fingerprint, stage_hashes, lin_run, nl_run)

    # Return ReportSchema v1
    return ReportResponse(
        report_schema_version="1.0.0",
        generated_at=utc_now_iso(),
        engine_version=APP_VERSION,
        input_fingerprint=input_fingerprint,
        multi_axis_driver_mapping=to_builtin(driver_mapping),
        crosstalk_loops=to_builtin(loops),
        diagnostic_models=to_builtin(diag_models),
        baseline_benchmarking=to_builtin(bench),
        silent_risk=to_builtin(silent),
        missed_opportunities=to_builtin(missed),
        risk_forecast=to_builtin(forecast),
        therapeutic_modeling=to_builtin(therapy),
        simulation=to_builtin(sim),
        explainability=to_builtin(explain),
        execution_manifest=to_builtin(execution_manifest),
        governance=to_builtin(governance),
    )

# ---------------------------------------------------------------------
# KEEP EXISTING ENGINE ENDPOINTS (as-is), but you can wire Base44 to /report first.
# ---------------------------------------------------------------------

class MultiOmicFeatures(BaseModel):
    immune: List[float]
    metabolic: List[float]
    microbiome: List[float]

class MultiOmicFusionResult(BaseModel):
    fused_score: float
    domain_contributions: Dict[str, float]
    primary_driver: str
    confidence: float

def mean_safe(x: List[float]) -> float:
    return float(np.mean(x)) if x else 0.0

@app.post("/multi_omic_fusion", response_model=MultiOmicFusionResult)
def multi_omic_fusion(f: MultiOmicFeatures) -> MultiOmicFusionResult:
    scores = {
        "immune": mean_safe(f.immune),
        "metabolic": mean_safe(f.metabolic),
        "microbiome": mean_safe(f.microbiome),
    }
    fused = mean_safe(list(scores.values()))
    total = sum(abs(v) for v in scores.values()) or 1.0
    contrib = {k: abs(v) / total for k, v in scores.items()}
    primary = max(scores, key=scores.get)
    confidence = max(0.0, min(1.0, 1.0 - float(np.std(list(scores.values())))))
    return MultiOmicFusionResult(
        fused_score=float(fused),
        domain_contributions={k: float(v) for k, v in contrib.items()},
        primary_driver=str(primary),
        confidence=float(confidence),
    )

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": APP_VERSION}

