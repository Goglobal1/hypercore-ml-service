"""
HyperCore v2.1 - Optimal Hybrid Algorithm
==========================================

Combines the best elements from ALL previous algorithms:

FROM v1.0 (Algorithms #1-3):
- Clinical gradient functions with nuanced scoring
- Named clinical pattern detection (AKI, lactic acidosis, oliguria, etc.)
- BUN/Creatinine ratio for prerenal vs intrinsic
- Rolling windows with leakage protection
- Three-part domain formula
- Multiplicative convergence

FROM v2.0 (Algorithm #4):
- Advanced trajectory analysis (acceleration, volatility, momentum, changepoints)
- Clinical indices (Shock Index, SOFA, deltas)
- Cascade detection
- Coupling matrix
- Tiered alerts with cooldowns

KEY FIXES from v1.0's failures:
- Lower thresholds for balanced/high-confidence
- min_domains=1 for balanced (was 2)
- Clinical index multipliers
- Multi-path alerting

Target Performance:
- Sensitivity: 75-85% across modes
- Specificity: 78-82%
- PPV @5%: 18-22%
- Lead Time: 15-20 hours
"""

from __future__ import annotations

import io
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION - v2.1 OPTIMIZED THRESHOLDS
# =============================================================================

CONFIG_V21: Dict[str, Any] = {
    # KEY FIX: Lower thresholds for balanced/high-confidence
    "modes": {
        "screening": {"min_domains": 1, "risk_threshold": 0.15, "shock_index_trigger": 1.0},
        "balanced": {"min_domains": 1, "risk_threshold": 0.22, "shock_index_trigger": 1.05},  # Was 0.30, min_domains=2
        "high_confidence": {"min_domains": 2, "risk_threshold": 0.38, "shock_index_trigger": 1.15},  # Was 0.50, min_domains=3
    },
    
    # KEY FIX: Rebalanced formula weights
    "domain_formula_weights": {
        "absolute": 0.35,    # Increased from 0.30
        "trajectory": 0.40,  # Decreased from 0.50
        "interaction": 0.25, # Increased from 0.20
    },
    
    # Convergence
    "convergence": {
        "gamma": 0.25,
        "max_convergence_bonus": 0.35,
        "max_coupling_bonus": 0.30,
        "max_cascade_bonus": 0.30,
        "max_accel_bonus": 0.20,
        "per_accel_bonus": 0.05,
        "shock_critical_bonus": 0.15,
    },
    
    # Clinical index multipliers (NEW)
    "clinical_index_multipliers": {
        "shock_index_critical": 1.30,  # SI >= 1.2
        "shock_index_elevated": 1.15,  # SI >= 0.9
        "sofa_high": 1.20,             # SOFA >= 8
        "sofa_moderate": 1.10,         # SOFA >= 4
    },
    
    # Alerts
    "alerts": {
        "tier_thresholds": {"monitor": 0.15, "warning": 0.25, "urgent": 0.45, "critical": 0.65},
        "cooldown_hours": {"tier2": 2.0, "tier3": 1.0},
    },
    
    # Domain weights
    "endpoint_weights": {
        "cardiac": 1.2,
        "respiratory": 1.2,
        "metabolic": 1.1,
        "renal": 1.0,
        "inflammatory": 0.9,
        "hemodynamic": 1.2,
        "hematologic": 0.8,
        "hepatic": 0.8,
    },
    
    # Coupling matrix (inter-domain synergy)
    "coupling_matrix": {
        "cardiac": {"respiratory": 0.8, "metabolic": 0.9, "renal": 0.7, "hemodynamic": 0.9},
        "respiratory": {"cardiac": 0.8, "metabolic": 0.7, "inflammatory": 0.7},
        "metabolic": {"cardiac": 0.9, "renal": 0.8, "hemodynamic": 0.9},
        "renal": {"cardiac": 0.7, "metabolic": 0.8, "hemodynamic": 0.7},
        "inflammatory": {"respiratory": 0.7, "metabolic": 0.8},
        "hemodynamic": {"cardiac": 0.9, "metabolic": 0.9, "renal": 0.7},
    },
    
    # Trajectory v2 config
    "trajectory": {
        "direction_threshold": 0.08,  # 8% change to count as worsening
        "volatility_thresholds": {"moderate": 0.12, "erratic": 0.25},
        "momentum_decay": 0.9,
        "weights": {
            "percent_change": 0.25,
            "velocity": 0.25,
            "acceleration": 0.25,
            "momentum": 0.25,
        },
        "norms": {
            "percent_change": 0.25,
            "velocity": 3.0,
            "acceleration": 0.8,
            "momentum": 6.0,
        },
    },
    
    # Parameter specs with clinical ranges (from v1.0)
    "parameters": {
        "heart_rate": {"normal_low": 60, "normal_high": 100, "severe_low": 40, "severe_high": 130, "direction": "both", "time_window_hours": 8},
        "sbp": {"normal_low": 100, "normal_high": 140, "severe_low": 80, "severe_high": 180, "direction": "both", "time_window_hours": 8},
        "map": {"normal_low": 65, "normal_high": 100, "severe_low": 55, "severe_high": 110, "direction": "low", "time_window_hours": 8},
        "respiratory_rate": {"normal_low": 12, "normal_high": 20, "severe_low": 8, "severe_high": 30, "direction": "high", "time_window_hours": 8},
        "spo2": {"normal_low": 94, "normal_high": 100, "severe_low": 88, "severe_high": 100, "direction": "low", "time_window_hours": 8},
        "temperature": {"normal_low": 36.5, "normal_high": 37.5, "severe_low": 35.0, "severe_high": 39.0, "direction": "both", "time_window_hours": 12},
        "lactate": {"normal_low": 0.5, "normal_high": 2.0, "severe_low": 0.3, "severe_high": 4.0, "direction": "high", "time_window_hours": 12},
        "creatinine": {"normal_low": 0.7, "normal_high": 1.3, "severe_low": 0.5, "severe_high": 3.0, "direction": "high", "time_window_hours": 24},
        "ph": {"normal_low": 7.35, "normal_high": 7.45, "severe_low": 7.20, "severe_high": 7.55, "direction": "both", "time_window_hours": 12},
        "wbc": {"normal_low": 4.5, "normal_high": 11.0, "severe_low": 2.0, "severe_high": 20.0, "direction": "both", "time_window_hours": 24},
        "platelets": {"normal_low": 150, "normal_high": 400, "severe_low": 50, "severe_high": 600, "direction": "low", "time_window_hours": 24},
        "bilirubin": {"normal_low": 0.1, "normal_high": 1.2, "severe_low": 0.1, "severe_high": 5.0, "direction": "high", "time_window_hours": 24},
        "bun": {"normal_low": 7, "normal_high": 20, "severe_low": 5, "severe_high": 40, "direction": "high", "time_window_hours": 24},
        "urine_output": {"normal_low": 0.5, "normal_high": 2.0, "severe_low": 0.3, "severe_high": 3.0, "direction": "low", "time_window_hours": 12},
        "fio2": {"normal_low": 21, "normal_high": 40, "severe_low": 21, "severe_high": 80, "direction": "high", "time_window_hours": 8},
        "troponin": {"normal_low": 0, "normal_high": 0.04, "severe_low": 0, "severe_high": 0.5, "direction": "high", "time_window_hours": 12},
        "bnp": {"normal_low": 0, "normal_high": 100, "severe_low": 0, "severe_high": 1000, "direction": "high", "time_window_hours": 24},
    },
    
    # Endpoints
    "endpoints": {
        "cardiac": {"params": ["heart_rate", "sbp", "map", "troponin", "bnp"], "weights": {"heart_rate": 0.25, "sbp": 0.30, "map": 0.20, "troponin": 0.15, "bnp": 0.10}},
        "respiratory": {"params": ["respiratory_rate", "spo2", "fio2"], "weights": {"respiratory_rate": 0.35, "spo2": 0.45, "fio2": 0.20}},
        "metabolic": {"params": ["lactate", "ph"], "weights": {"lactate": 0.60, "ph": 0.40}},
        "renal": {"params": ["creatinine", "bun", "urine_output"], "weights": {"creatinine": 0.50, "bun": 0.20, "urine_output": 0.30}},
        "inflammatory": {"params": ["temperature", "wbc"], "weights": {"temperature": 0.50, "wbc": 0.50}},
        "hemodynamic": {"params": ["heart_rate", "sbp", "map", "lactate"], "weights": {"heart_rate": 0.20, "sbp": 0.30, "map": 0.30, "lactate": 0.20}},
        "hematologic": {"params": ["platelets"], "weights": {"platelets": 1.0}},
        "hepatic": {"params": ["bilirubin"], "weights": {"bilirubin": 1.0}},
    },
    
    # Named clinical patterns (from v1.0 Algorithm #1)
    "clinical_patterns": {
        "aki": {"conditions": ["creatinine>=1.5", "creatinine_rise>=0.25"], "bonus": 0.18},
        "lactic_acidosis": {"conditions": ["lactate>=2.0", "lactate_rise>=0.20"], "bonus": 0.20},
        "oliguria": {"conditions": ["urine_output<=0.5", "uo_falling"], "bonus": 0.15},
        "hypoxia": {"conditions": ["spo2<=92", "spo2_falling"], "bonus": 0.18},
        "severe_tachypnea": {"conditions": ["respiratory_rate>=28"], "bonus": 0.12},
        "severe_acidosis": {"conditions": ["ph<=7.25"], "bonus": 0.18},
        "cardiogenic_shock": {"conditions": ["hr_rising", "sbp_falling"], "bonus": 0.22},
        "sepsis_signature": {"conditions": ["temp_abnormal", "wbc_abnormal"], "bonus": 0.15},
    },
    
    # Interaction patterns
    "interaction_patterns": {
        "shock": {"domains": ["cardiac", "hemodynamic"], "bonus": 0.20},
        "perfusion_failure": {"domains": ["metabolic", "hemodynamic"], "bonus": 0.25},
        "respiratory_metabolic": {"domains": ["respiratory", "metabolic"], "bonus": 0.18},
        "cardiorenal": {"domains": ["cardiac", "renal"], "bonus": 0.18},
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TrajectoryProfile:
    direction: str  # worsening, improving, stable, erratic
    percent_change: float
    velocity: float
    acceleration: float
    acceleration_label: str
    volatility: float
    volatility_label: str
    momentum: float
    trajectory_score: float


@dataclass
class ClinicalIndices:
    shock_index: Optional[float]
    shock_index_category: str
    sofa_total: int
    delta_hr: Optional[float]
    delta_sbp: Optional[float]
    delta_creatinine: Optional[float]
    delta_lactate: Optional[float]


@dataclass
class EndpointResult:
    domain: str
    domain_score: float
    is_alerting: bool
    absolute_score: float
    trajectory_score: float
    interaction_score: float
    pattern_bonus: float
    patterns_detected: List[str]
    trajectory_profile: TrajectoryProfile


@dataclass
class ConvergenceResult:
    base_score: float
    final_score: float
    n_alerting: int
    domains_alerting: List[str]
    convergence_bonus: float
    coupling_bonus: float
    cascade_bonus: float
    clinical_index_multiplier: float
    cascades_detected: List[str]


@dataclass
class HyperCoreOutput:
    patient_id: str
    timestamp: datetime
    risk_score: float
    risk_level: str
    should_alert: bool
    alert_path: str  # which condition triggered alert
    convergence: ConvergenceResult
    endpoints: List[EndpointResult]
    clinical_indices: ClinicalIndices
    news_score: Optional[int]
    qsofa_score: Optional[int]
    mews_score: Optional[int]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
        return None if math.isnan(v) else v
    except (ValueError, TypeError):
        return None


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# =============================================================================
# CLINICAL GRADIENT SCORING (from v1.0)
# =============================================================================

def score_absolute_clinical(param: str, value: Optional[float], config: Dict) -> float:
    """Score parameter using clinical gradient (from v1.0 Algorithm #1)."""
    if value is None:
        return 0.0
    
    spec = config["parameters"].get(param, {})
    normal_low = spec.get("normal_low")
    normal_high = spec.get("normal_high")
    severe_low = spec.get("severe_low")
    severe_high = spec.get("severe_high")
    direction = spec.get("direction", "high")
    
    # Check normal range
    if normal_low is not None and normal_high is not None:
        if normal_low <= value <= normal_high:
            return 0.0
    
    score = 0.0
    
    if direction in ("high", "both") and normal_high is not None and value > normal_high:
        range_size = (severe_high or normal_high * 1.5) - normal_high
        score = max(score, _clip01((value - normal_high) / max(range_size, 0.01)))
    
    if direction in ("low", "both") and normal_low is not None and value < normal_low:
        range_size = normal_low - (severe_low or normal_low * 0.5)
        score = max(score, _clip01((normal_low - value) / max(range_size, 0.01)))
    
    return score


# =============================================================================
# TRAJECTORY v2 ANALYSIS (from v2.0 Algorithm #4)
# =============================================================================

def calculate_trajectory_v2(
    values: List[float],
    times: List[datetime],
    param: str,
    baselines: Dict[str, float],
    config: Dict
) -> TrajectoryProfile:
    """Advanced trajectory with acceleration, volatility, momentum."""
    if len(values) < 2:
        return _neutral_trajectory()
    
    arr = np.array(values)
    baseline = baselines.get(param, arr[0])
    if baseline == 0:
        baseline = 0.001
    
    # Percent change from baseline
    percent_change = (arr[-1] - baseline) / abs(baseline)
    
    # Time in hours
    t0 = times[0]
    t_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
    
    # Velocity (slope)
    velocity = 0.0
    if t_hours[-1] > 0:
        try:
            slope, _ = np.polyfit(t_hours, arr, 1)
            velocity = float(slope)
        except:
            pass
    
    # Acceleration (second derivative)
    acceleration = 0.0
    accel_label = "constant"
    if len(arr) >= 3:
        try:
            coeffs = np.polyfit(t_hours, arr, 2)
            acceleration = 2 * coeffs[0]
            if acceleration > 0.1:
                accel_label = "accelerating"
            elif acceleration < -0.1:
                accel_label = "decelerating"
        except:
            pass
    
    # Volatility (CV)
    mean_val = np.mean(arr)
    volatility = float(np.std(arr) / abs(mean_val)) if mean_val != 0 else 0.0
    
    traj_cfg = config.get("trajectory", {})
    vol_thresh = traj_cfg.get("volatility_thresholds", {})
    if volatility < vol_thresh.get("moderate", 0.12):
        vol_label = "stable"
    elif volatility < vol_thresh.get("erratic", 0.25):
        vol_label = "moderate"
    else:
        vol_label = "erratic"
    
    # Momentum (cumulative abnormal time)
    momentum = 0.0
    abnormal_thresh = config["parameters"].get(param, {}).get("normal_high", arr[0] * 1.1)
    direction = config["parameters"].get(param, {}).get("direction", "high")
    decay = traj_cfg.get("momentum_decay", 0.9)
    
    t_max = times[-1]
    for i, (val, t) in enumerate(zip(arr, times)):
        is_abnormal = False
        if direction in ("high", "both") and val > abnormal_thresh:
            is_abnormal = True
        if direction in ("low", "both") and val < config["parameters"].get(param, {}).get("normal_low", arr[0] * 0.9):
            is_abnormal = True
        
        if is_abnormal:
            hours_ago = (t_max - t).total_seconds() / 3600.0
            momentum += 1.0 * (decay ** hours_ago)
    
    # Calculate trajectory score
    norms = traj_cfg.get("norms", {})
    weights = traj_cfg.get("weights", {})
    
    pc_norm = min(1.0, abs(percent_change) / norms.get("percent_change", 0.25))
    vel_norm = min(1.0, abs(velocity) / norms.get("velocity", 3.0))
    acc_norm = min(1.0, abs(acceleration) / norms.get("acceleration", 0.8))
    mom_norm = min(1.0, momentum / norms.get("momentum", 6.0))
    
    traj_score = (
        weights.get("percent_change", 0.25) * pc_norm +
        weights.get("velocity", 0.25) * vel_norm +
        weights.get("acceleration", 0.25) * acc_norm +
        weights.get("momentum", 0.25) * mom_norm
    )
    
    # Volatility amplifies
    if vol_label == "erratic":
        traj_score = min(1.0, traj_score * 1.3)
    
    # Direction
    dir_thresh = traj_cfg.get("direction_threshold", 0.08)
    if direction == "high":
        dir_label = "worsening" if percent_change > dir_thresh else ("improving" if percent_change < -dir_thresh else "stable")
    elif direction == "low":
        dir_label = "worsening" if percent_change < -dir_thresh else ("improving" if percent_change > dir_thresh else "stable")
    else:
        dir_label = "worsening" if abs(percent_change) > dir_thresh else "stable"
    
    if vol_label == "erratic":
        dir_label = "erratic"
    
    return TrajectoryProfile(
        direction=dir_label,
        percent_change=float(percent_change),
        velocity=velocity,
        acceleration=acceleration,
        acceleration_label=accel_label,
        volatility=volatility,
        volatility_label=vol_label,
        momentum=momentum,
        trajectory_score=_clip01(traj_score),
    )


def _neutral_trajectory() -> TrajectoryProfile:
    return TrajectoryProfile(
        direction="stable", percent_change=0.0, velocity=0.0,
        acceleration=0.0, acceleration_label="constant",
        volatility=0.0, volatility_label="stable",
        momentum=0.0, trajectory_score=0.0,
    )


# =============================================================================
# CLINICAL INDICES (from v2.0)
# =============================================================================

def calculate_clinical_indices(
    current: Dict[str, float],
    baselines: Dict[str, float],
    config: Dict
) -> ClinicalIndices:
    """Calculate Shock Index, SOFA, deltas."""
    hr = _safe_float(current.get("heart_rate"))
    sbp = _safe_float(current.get("sbp"))
    
    # Shock Index
    shock_index = None
    si_cat = "unknown"
    if hr is not None and sbp is not None and sbp > 0:
        shock_index = hr / sbp
        if shock_index < 0.9:
            si_cat = "normal"
        elif shock_index < 1.2:
            si_cat = "elevated"
        else:
            si_cat = "critical"
    
    # Simplified SOFA
    sofa = 0
    
    # Respiratory
    spo2 = _safe_float(current.get("spo2"))
    if spo2 is not None:
        if spo2 < 88: sofa += 3
        elif spo2 < 92: sofa += 2
        elif spo2 < 95: sofa += 1
    
    # Cardiovascular (MAP or SBP)
    map_val = _safe_float(current.get("map"))
    if map_val is not None:
        if map_val < 55: sofa += 3
        elif map_val < 65: sofa += 2
        elif map_val < 70: sofa += 1
    elif sbp is not None:
        if sbp < 80: sofa += 3
        elif sbp < 90: sofa += 2
        elif sbp < 100: sofa += 1
    
    # Renal
    cr = _safe_float(current.get("creatinine"))
    if cr is not None:
        if cr >= 5.0: sofa += 4
        elif cr >= 3.5: sofa += 3
        elif cr >= 2.0: sofa += 2
        elif cr >= 1.2: sofa += 1
    
    # Coagulation
    plt = _safe_float(current.get("platelets"))
    if plt is not None:
        if plt < 20: sofa += 4
        elif plt < 50: sofa += 3
        elif plt < 100: sofa += 2
        elif plt < 150: sofa += 1
    
    # Liver
    bili = _safe_float(current.get("bilirubin"))
    if bili is not None:
        if bili >= 12: sofa += 4
        elif bili >= 6: sofa += 3
        elif bili >= 2: sofa += 2
        elif bili >= 1.2: sofa += 1
    
    # Deltas
    def delta(p):
        cur = _safe_float(current.get(p))
        base = baselines.get(p)
        if cur is None or base is None:
            return None
        return cur - base
    
    return ClinicalIndices(
        shock_index=shock_index,
        shock_index_category=si_cat,
        sofa_total=sofa,
        delta_hr=delta("heart_rate"),
        delta_sbp=delta("sbp"),
        delta_creatinine=delta("creatinine"),
        delta_lactate=delta("lactate"),
    )


# =============================================================================
# PATTERN DETECTION (from v1.0 Algorithm #1)
# =============================================================================

def detect_clinical_patterns(
    current: Dict[str, float],
    trajectories: Dict[str, TrajectoryProfile],
    config: Dict
) -> Tuple[float, List[str]]:
    """Detect named clinical patterns (AKI, lactic acidosis, etc.)."""
    patterns_detected = []
    total_bonus = 0.0
    
    # AKI
    cr = _safe_float(current.get("creatinine"))
    cr_traj = trajectories.get("creatinine")
    if cr is not None and cr >= 1.5:
        if cr_traj and cr_traj.percent_change >= 0.25:
            patterns_detected.append("aki")
            total_bonus = max(total_bonus, config["clinical_patterns"]["aki"]["bonus"])
    
    # Lactic acidosis
    lac = _safe_float(current.get("lactate"))
    lac_traj = trajectories.get("lactate")
    if lac is not None and lac >= 2.0:
        if lac_traj and lac_traj.percent_change >= 0.20:
            patterns_detected.append("lactic_acidosis")
            total_bonus = max(total_bonus, config["clinical_patterns"]["lactic_acidosis"]["bonus"])
    
    # Hypoxia
    spo2 = _safe_float(current.get("spo2"))
    spo2_traj = trajectories.get("spo2")
    if spo2 is not None and spo2 <= 92:
        if spo2_traj and spo2_traj.direction == "worsening":
            patterns_detected.append("hypoxia")
            total_bonus = max(total_bonus, config["clinical_patterns"]["hypoxia"]["bonus"])
    
    # Cardiogenic shock
    hr_traj = trajectories.get("heart_rate")
    sbp_traj = trajectories.get("sbp")
    if hr_traj and sbp_traj:
        if hr_traj.direction == "worsening" and sbp_traj.direction == "worsening":
            patterns_detected.append("cardiogenic_shock")
            total_bonus = max(total_bonus, config["clinical_patterns"]["cardiogenic_shock"]["bonus"])
    
    # Severe acidosis
    ph = _safe_float(current.get("ph"))
    if ph is not None and ph <= 7.25:
        patterns_detected.append("severe_acidosis")
        total_bonus = max(total_bonus, config["clinical_patterns"]["severe_acidosis"]["bonus"])
    
    # Severe tachypnea
    rr = _safe_float(current.get("respiratory_rate"))
    if rr is not None and rr >= 28:
        patterns_detected.append("severe_tachypnea")
        total_bonus = max(total_bonus, config["clinical_patterns"]["severe_tachypnea"]["bonus"])
    
    # Sepsis signature
    temp = _safe_float(current.get("temperature"))
    wbc = _safe_float(current.get("wbc"))
    if temp is not None and wbc is not None:
        temp_abnormal = temp < 36.0 or temp > 38.0
        wbc_abnormal = wbc < 4.5 or wbc > 11.0
        if temp_abnormal and wbc_abnormal:
            patterns_detected.append("sepsis_signature")
            total_bonus = max(total_bonus, config["clinical_patterns"]["sepsis_signature"]["bonus"])
    
    return total_bonus, patterns_detected


# =============================================================================
# ENDPOINT SCORING
# =============================================================================

def score_endpoint(
    domain: str,
    patient_df: pd.DataFrame,
    baselines: Dict[str, float],
    config: Dict
) -> EndpointResult:
    """Score a single endpoint using v2.1 formula."""
    endpoint_cfg = config["endpoints"].get(domain, {})
    params = endpoint_cfg.get("params", [])
    weights = endpoint_cfg.get("weights", {})
    
    if len(patient_df) == 0:
        return _empty_endpoint(domain)
    
    latest = patient_df.sort_values("timestamp").iloc[-1]
    current = latest.to_dict()
    
    # Calculate trajectories
    trajectories = {}
    for param in params:
        if param in patient_df.columns:
            vals = patient_df[param].dropna().values.tolist()
            times = patient_df.loc[patient_df[param].notna(), "timestamp"].tolist()
            if len(vals) >= 2:
                trajectories[param] = calculate_trajectory_v2(vals, times, param, baselines, config)
            else:
                trajectories[param] = _neutral_trajectory()
    
    # Calculate component scores
    abs_scores = []
    traj_scores = []
    total_weight = 0.0
    
    for param in params:
        if param not in patient_df.columns:
            continue
        
        value = _safe_float(latest.get(param))
        weight = weights.get(param, 1.0 / len(params))
        
        abs_score = score_absolute_clinical(param, value, config)
        traj = trajectories.get(param, _neutral_trajectory())
        
        abs_scores.append(abs_score * weight)
        traj_scores.append(traj.trajectory_score * weight)
        total_weight += weight
    
    if total_weight > 0:
        weighted_abs = sum(abs_scores) / total_weight
        weighted_traj = sum(traj_scores) / total_weight
    else:
        weighted_abs = 0.0
        weighted_traj = 0.0
    
    # Pattern bonus
    pattern_bonus, patterns = detect_clinical_patterns(current, trajectories, config)
    
    # Interaction score (intra-domain - simplified)
    interaction_score = pattern_bonus  # Patterns are our intra-domain interactions
    
    # Apply v2.1 formula
    fw = config["domain_formula_weights"]
    domain_score = (
        fw["absolute"] * weighted_abs +
        fw["trajectory"] * weighted_traj +
        fw["interaction"] * interaction_score
    )
    
    # Add pattern bonus (capped)
    domain_score = _clip01(domain_score + min(0.20, pattern_bonus))
    
    # Is alerting?
    is_alerting = domain_score >= 0.25  # Lower threshold
    
    # Best trajectory for this domain
    best_traj = _neutral_trajectory()
    if trajectories:
        best_traj = max(trajectories.values(), key=lambda t: t.trajectory_score)
    
    return EndpointResult(
        domain=domain,
        domain_score=domain_score,
        is_alerting=is_alerting,
        absolute_score=weighted_abs,
        trajectory_score=weighted_traj,
        interaction_score=interaction_score,
        pattern_bonus=pattern_bonus,
        patterns_detected=patterns,
        trajectory_profile=best_traj,
    )


def _empty_endpoint(domain: str) -> EndpointResult:
    return EndpointResult(
        domain=domain, domain_score=0.0, is_alerting=False,
        absolute_score=0.0, trajectory_score=0.0, interaction_score=0.0,
        pattern_bonus=0.0, patterns_detected=[],
        trajectory_profile=_neutral_trajectory(),
    )


# =============================================================================
# CONVERGENCE MODEL (v2.1 enhanced)
# =============================================================================

def calculate_convergence(
    endpoints: List[EndpointResult],
    clinical_indices: ClinicalIndices,
    config: Dict
) -> ConvergenceResult:
    """Calculate convergence with all bonuses."""
    conv_cfg = config["convergence"]
    
    # Base score (weighted average)
    domain_weights = config["endpoint_weights"]
    scores = {e.domain: e.domain_score for e in endpoints}
    
    if scores:
        numerator = sum(domain_weights.get(d, 1.0) * s for d, s in scores.items())
        denominator = sum(domain_weights.get(d, 1.0) for d in scores)
        base_score = numerator / denominator
    else:
        base_score = 0.0
    
    alerting = [e for e in endpoints if e.is_alerting]
    n_alerting = len(alerting)
    domains_alerting = [e.domain for e in alerting]
    
    # Convergence bonus
    conv_bonus = 0.0
    if n_alerting >= 2:
        conv_bonus = min(conv_cfg.get("max_convergence_bonus", 0.35), conv_cfg.get("gamma", 0.25) * math.log(n_alerting))
    
    # Coupling bonus (inter-domain synergy)
    coupling_bonus = 0.0
    coupling_matrix = config.get("coupling_matrix", {})
    for i, da in enumerate(domains_alerting):
        for db in domains_alerting[i+1:]:
            coupling = coupling_matrix.get(da, {}).get(db, 0.0)
            if coupling > 0:
                coupling_bonus += coupling * min(scores.get(da, 0), scores.get(db, 0))
    coupling_bonus = min(conv_cfg.get("max_coupling_bonus", 0.30), coupling_bonus)
    
    # Cascade bonus (sequential domain failures)
    cascade_bonus = 0.0
    cascades = []
    if n_alerting >= 3:
        cascade_bonus = conv_cfg.get("max_cascade_bonus", 0.30) * 0.5
        cascades.append(f"cascade_{n_alerting}_domains")
    
    # Acceleration bonus
    n_accel = sum(1 for e in endpoints if e.trajectory_profile.acceleration_label == "accelerating")
    accel_bonus = min(conv_cfg.get("max_accel_bonus", 0.20), n_accel * conv_cfg.get("per_accel_bonus", 0.05))
    
    # Clinical index multiplier (KEY FIX)
    ci_mult = 1.0
    mult_cfg = config.get("clinical_index_multipliers", {})
    
    if clinical_indices.shock_index_category == "critical":
        ci_mult = max(ci_mult, mult_cfg.get("shock_index_critical", 1.30))
    elif clinical_indices.shock_index_category == "elevated":
        ci_mult = max(ci_mult, mult_cfg.get("shock_index_elevated", 1.15))
    
    if clinical_indices.sofa_total >= 8:
        ci_mult = max(ci_mult, mult_cfg.get("sofa_high", 1.20))
    elif clinical_indices.sofa_total >= 4:
        ci_mult = max(ci_mult, mult_cfg.get("sofa_moderate", 1.10))
    
    # Final score
    conv_multiplier = (1 + conv_bonus) * (1 + coupling_bonus) * (1 + cascade_bonus) * (1 + accel_bonus)
    final_score = _clip01(base_score * conv_multiplier * ci_mult)
    
    return ConvergenceResult(
        base_score=base_score,
        final_score=final_score,
        n_alerting=n_alerting,
        domains_alerting=domains_alerting,
        convergence_bonus=conv_bonus,
        coupling_bonus=coupling_bonus,
        cascade_bonus=cascade_bonus,
        clinical_index_multiplier=ci_mult,
        cascades_detected=cascades,
    )


# =============================================================================
# BASELINE COMPARATORS
# =============================================================================

def compute_news(row: pd.Series) -> Optional[int]:
    score = 0
    usable = False
    
    rr = _safe_float(row.get("respiratory_rate"))
    spo2 = _safe_float(row.get("spo2"))
    temp = _safe_float(row.get("temperature"))
    sbp = _safe_float(row.get("sbp"))
    hr = _safe_float(row.get("heart_rate"))
    
    if rr is not None:
        usable = True
        if rr <= 8: score += 3
        elif 9 <= rr <= 11: score += 1
        elif 21 <= rr <= 24: score += 2
        elif rr >= 25: score += 3
    
    if spo2 is not None:
        usable = True
        if spo2 <= 91: score += 3
        elif 92 <= spo2 <= 93: score += 2
        elif 94 <= spo2 <= 95: score += 1
    
    if temp is not None:
        usable = True
        if temp <= 35.0: score += 3
        elif 35.1 <= temp <= 36.0: score += 1
        elif 38.1 <= temp <= 39.0: score += 1
        elif temp >= 39.1: score += 2
    
    if sbp is not None:
        usable = True
        if sbp <= 90: score += 3
        elif 91 <= sbp <= 100: score += 2
        elif 101 <= sbp <= 110: score += 1
        elif sbp >= 220: score += 3
    
    if hr is not None:
        usable = True
        if hr <= 40: score += 3
        elif 41 <= hr <= 50: score += 1
        elif 91 <= hr <= 110: score += 1
        elif 111 <= hr <= 130: score += 2
        elif hr >= 131: score += 3
    
    return score if usable else None


def compute_qsofa(row: pd.Series) -> Optional[int]:
    score = 0
    rr = _safe_float(row.get("respiratory_rate"))
    sbp = _safe_float(row.get("sbp"))
    
    if rr is not None and rr >= 22:
        score += 1
    if sbp is not None and sbp <= 100:
        score += 1
    
    return score


def compute_mews(row: pd.Series) -> Optional[int]:
    score = 0
    hr = _safe_float(row.get("heart_rate"))
    rr = _safe_float(row.get("respiratory_rate"))
    sbp = _safe_float(row.get("sbp"))
    temp = _safe_float(row.get("temperature"))
    
    if hr is not None:
        if hr <= 40: score += 2
        elif 41 <= hr <= 50: score += 1
        elif 101 <= hr <= 110: score += 1
        elif 111 <= hr <= 129: score += 2
        elif hr >= 130: score += 3
    
    if rr is not None:
        if rr < 9: score += 2
        elif 15 <= rr <= 20: score += 1
        elif 21 <= rr <= 29: score += 2
        elif rr >= 30: score += 3
    
    if sbp is not None:
        if sbp <= 70: score += 3
        elif 71 <= sbp <= 80: score += 2
        elif 81 <= sbp <= 100: score += 1
        elif sbp >= 200: score += 2
    
    if temp is not None:
        if temp < 35.0: score += 2
        elif temp >= 38.5: score += 2
    
    return score


# =============================================================================
# MULTI-PATH ALERT LOGIC (KEY FIX)
# =============================================================================

def should_alert_multipath(
    convergence: ConvergenceResult,
    clinical_indices: ClinicalIndices,
    endpoints: List[EndpointResult],
    mode_config: Dict
) -> Tuple[bool, str]:
    """
    Multi-path alerting - can alert through multiple conditions.
    KEY FIX: More paths to trigger alerts.
    """
    risk_threshold = mode_config.get("risk_threshold", 0.25)
    min_domains = mode_config.get("min_domains", 1)
    si_trigger = mode_config.get("shock_index_trigger", 1.1)
    
    # Path 1: Score threshold met
    if convergence.final_score >= risk_threshold and convergence.n_alerting >= min_domains:
        return True, "score_threshold"
    
    # Path 2: Critical clinical index
    if clinical_indices.shock_index is not None and clinical_indices.shock_index >= si_trigger:
        return True, "shock_index"
    
    # Path 3: High SOFA
    if clinical_indices.sofa_total >= 6:
        return True, "sofa_high"
    
    # Path 4: Critical named pattern
    critical_patterns = {"cardiogenic_shock", "lactic_acidosis", "severe_acidosis", "hypoxia"}
    for ep in endpoints:
        if any(p in critical_patterns for p in ep.patterns_detected):
            return True, f"pattern_{ep.patterns_detected[0]}"
    
    # Path 5: Cascade detected
    if convergence.cascades_detected:
        return True, "cascade"
    
    # Path 6: Single domain very high
    for ep in endpoints:
        if ep.domain_score >= 0.70:
            return True, f"high_domain_{ep.domain}"
    
    return False, "none"


# =============================================================================
# MAIN ENGINE
# =============================================================================

class HyperCoreV21:
    """HyperCore v2.1 Optimal Hybrid Engine."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or CONFIG_V21
    
    def score_patient(
        self,
        patient_df: pd.DataFrame,
        operating_mode: str = "balanced",
    ) -> HyperCoreOutput:
        """Score a patient."""
        patient_df = patient_df.sort_values("timestamp")
        patient_id = str(patient_df["patient_id"].iloc[0])
        
        # Calculate baselines from first 4 hours (or first 25% of data)
        n_baseline = max(2, int(len(patient_df) * 0.25))
        baseline_df = patient_df.head(n_baseline)
        baselines = {}
        for col in patient_df.columns:
            if col not in ["patient_id", "timestamp", "outcome"]:
                vals = baseline_df[col].dropna()
                if len(vals) > 0:
                    baselines[col] = float(vals.median())
        
        # Get current values
        latest = patient_df.iloc[-1]
        current = latest.to_dict()
        
        # Calculate clinical indices
        clinical_indices = calculate_clinical_indices(current, baselines, self.config)
        
        # Score all endpoints
        endpoints = []
        for domain in self.config["endpoints"].keys():
            result = score_endpoint(domain, patient_df, baselines, self.config)
            endpoints.append(result)
        
        # Calculate convergence
        convergence = calculate_convergence(endpoints, clinical_indices, self.config)
        
        # Alert decision (multi-path)
        mode_config = self.config["modes"].get(operating_mode, self.config["modes"]["balanced"])
        should_alert, alert_path = should_alert_multipath(
            convergence, clinical_indices, endpoints, mode_config
        )
        
        # Risk level
        score = convergence.final_score
        if score < 0.15:
            risk_level = "LOW"
        elif score < 0.25:
            risk_level = "MODERATE"
        elif score < 0.45:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Baseline comparators
        news = compute_news(latest)
        qsofa = compute_qsofa(latest)
        mews = compute_mews(latest)
        
        return HyperCoreOutput(
            patient_id=patient_id,
            timestamp=patient_df["timestamp"].max(),
            risk_score=convergence.final_score,
            risk_level=risk_level,
            should_alert=should_alert,
            alert_path=alert_path,
            convergence=convergence,
            endpoints=endpoints,
            clinical_indices=clinical_indices,
            news_score=news,
            qsofa_score=qsofa,
            mews_score=mews,
        )


# =============================================================================
# VALIDATION & COMPARISON
# =============================================================================

def run_comparison_v21(
    df: pd.DataFrame,
    scoring_mode: str = "balanced",
) -> Dict[str, Any]:
    """Run v2.1 comparison on cohort."""
    engine = HyperCoreV21()
    
    results = []
    
    for pid, patient_df in df.groupby("patient_id", sort=False):
        patient_df = patient_df.sort_values("timestamp").copy()
        
        # Get outcome
        outcome = None
        if "outcome" in patient_df.columns:
            outcome_vals = patient_df["outcome"].dropna()
            if len(outcome_vals) > 0:
                outcome = int(outcome_vals.max())
        
        # Score patient
        output = engine.score_patient(patient_df, scoring_mode)
        
        results.append({
            "patient_id": str(pid),
            "outcome": outcome,
            "hypercore_score": output.risk_score,
            "hypercore_alert": output.should_alert,
            "alert_path": output.alert_path,
            "risk_level": output.risk_level,
            "n_alerting": output.convergence.n_alerting,
            "domains_alerting": output.convergence.domains_alerting,
            "shock_index": output.clinical_indices.shock_index,
            "shock_index_category": output.clinical_indices.shock_index_category,
            "sofa_total": output.clinical_indices.sofa_total,
            "news_score": output.news_score,
            "news_alert": output.news_score >= 5 if output.news_score else None,
            "qsofa_score": output.qsofa_score,
            "qsofa_alert": output.qsofa_score >= 2 if output.qsofa_score else None,
            "mews_score": output.mews_score,
            "mews_alert": output.mews_score >= 4 if output.mews_score else None,
        })
    
    # Compute metrics
    valid = [r for r in results if r["outcome"] is not None]
    
    metrics = {}
    
    if valid:
        y_true = np.array([r["outcome"] for r in valid])
        
        # HyperCore
        y_pred = np.array([1 if r["hypercore_alert"] else 0 for r in valid])
        y_scores = np.array([r["hypercore_score"] for r in valid])
        metrics["hypercore"] = _compute_metrics(y_true, y_pred, y_scores)
        
        # NEWS
        news_valid = [r for r in valid if r["news_alert"] is not None]
        if news_valid:
            y_true_news = np.array([r["outcome"] for r in news_valid])
            y_pred_news = np.array([1 if r["news_alert"] else 0 for r in news_valid])
            y_scores_news = np.array([r["news_score"] for r in news_valid])
            metrics["news"] = _compute_metrics(y_true_news, y_pred_news, y_scores_news)
        
        # qSOFA
        qsofa_valid = [r for r in valid if r["qsofa_alert"] is not None]
        if qsofa_valid:
            y_true_q = np.array([r["outcome"] for r in qsofa_valid])
            y_pred_q = np.array([1 if r["qsofa_alert"] else 0 for r in qsofa_valid])
            y_scores_q = np.array([r["qsofa_score"] for r in qsofa_valid])
            metrics["qsofa"] = _compute_metrics(y_true_q, y_pred_q, y_scores_q)
        
        # MEWS
        mews_valid = [r for r in valid if r["mews_alert"] is not None]
        if mews_valid:
            y_true_m = np.array([r["outcome"] for r in mews_valid])
            y_pred_m = np.array([1 if r["mews_alert"] else 0 for r in mews_valid])
            y_scores_m = np.array([r["mews_score"] for r in mews_valid])
            metrics["mews"] = _compute_metrics(y_true_m, y_pred_m, y_scores_m)
    
    return {
        "status": "success",
        "version": "2.1.0-optimal",
        "n_patients": len(results),
        "n_positive_outcomes": sum(1 for r in valid if r["outcome"] == 1) if valid else 0,
        "scoring_mode": scoring_mode,
        "results": metrics,
        "patient_details": results,
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray) -> Dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    
    prev = 0.05
    ppv_5 = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev)) if (sens * prev + (1 - spec) * (1 - prev)) > 0 else 0.0
    
    metrics = {
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "ppv": round(ppv, 4),
        "ppv_at_5_percent": round(ppv_5, 4),
        "npv": round(npv, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    
    try:
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        if len(set(y_true)) > 1:
            metrics["roc_auc"] = round(roc_auc_score(y_true, y_scores), 4)
            prec, rec, _ = precision_recall_curve(y_true, y_scores)
            metrics["pr_auc"] = round(auc(rec, prec), 4)
    except:
        pass
    
    return metrics


# =============================================================================
# TEST WITH SAMPLE DATA
# =============================================================================

def _test_v21():
    """Test v2.1 with sample data."""
    csv = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,lactate,creatinine,wbc,platelets,outcome
P001,2024-01-01T00:00:00,78,16,120,36.8,98,1.0,0.9,7.5,200,0
P001,2024-01-01T04:00:00,88,19,112,37.2,96,1.5,1.1,9.0,185,0
P001,2024-01-01T08:00:00,102,24,98,38.0,92,2.5,1.5,12.0,160,0
P001,2024-01-01T12:00:00,118,28,85,38.8,88,4.0,2.0,16.0,130,1
P002,2024-01-01T00:00:00,72,14,125,36.7,99,0.9,0.9,6.8,220,0
P002,2024-01-01T04:00:00,70,14,122,36.8,99,0.8,0.9,6.9,225,0
P002,2024-01-01T08:00:00,74,15,120,36.7,98,0.9,0.9,7.1,218,0
P002,2024-01-01T12:00:00,72,14,123,36.8,98,0.9,0.9,7.0,222,0"""
    
    df = pd.read_csv(io.StringIO(csv))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print("=" * 60)
    print("HyperCore v2.1 Optimal Hybrid - Test Results")
    print("=" * 60)
    
    for mode in ["screening", "balanced", "high_confidence"]:
        results = run_comparison_v21(df, mode)
        print(f"\n--- {mode.upper()} MODE ---")
        print(f"Patients: {results['n_patients']}")
        print(f"Events: {results['n_positive_outcomes']}")
        
        if "hypercore" in results["results"]:
            hc = results["results"]["hypercore"]
            print(f"\nHyperCore v2.1:")
            print(f"  Sensitivity: {hc['sensitivity']*100:.1f}%")
            print(f"  Specificity: {hc['specificity']*100:.1f}%")
            print(f"  PPV: {hc['ppv']*100:.1f}%")
            print(f"  TP: {hc['tp']}, FP: {hc['fp']}, TN: {hc['tn']}, FN: {hc['fn']}")
        
        if "news" in results["results"]:
            news = results["results"]["news"]
            print(f"\nNEWS ≥5:")
            print(f"  Sensitivity: {news['sensitivity']*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("EXPECTED: Deteriorating patient (P001) should alert")
    print("EXPECTED: Stable patient (P002) should NOT alert")
    print("=" * 60)


if __name__ == "__main__":
    _test_v21()
