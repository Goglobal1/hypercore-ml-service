# bug_fixes.py
# Location: app/core/bug_fixes.py
"""
Bug fixes for HyperCore ML Service endpoints.
Import these functions and use them in the respective endpoints.
"""

import numpy as np
import pandas as pd
import re
from typing import Dict, List, Any


# =============================================================================
# FIX 1: /responder_prediction - Inverted best_arm/worst_arm labels
# =============================================================================

def fix_responder_subgroup_summary(arm_rates: dict) -> dict:
    """
    Correctly identify best and worst performing treatment arms.
    Best arm = highest response rate, Worst arm = lowest response rate
    """
    if not arm_rates:
        return {"arms": {}, "best_arm": None, "worst_arm": None}

    return {
        "arms": arm_rates,
        "best_arm": max(arm_rates, key=arm_rates.get),
        "worst_arm": min(arm_rates, key=arm_rates.get)
    }


# =============================================================================
# FIX 2: /synthetic_cohort - No randomization (all patients identical)
# =============================================================================

def generate_synthetic_cohort(real_data_distributions: Dict[str, Dict[str, float]], n_subjects: int) -> List[Dict[str, Any]]:
    """
    Generate synthetic patients with realistic variation.
    Samples from normal distribution based on mean/std, clipped to min/max.
    """
    synthetic_data = []

    for i in range(n_subjects):
        record = {}
        for var, dist in real_data_distributions.items():
            mean = dist.get("mean", 0)
            std = dist.get("std", 1)
            min_val = dist.get("min", float("-inf"))
            max_val = dist.get("max", float("inf"))

            value = np.random.normal(mean, std)
            value = np.clip(value, min_val, max_val)
            record[var] = round(float(value), 2)

        synthetic_data.append(record)

    return synthetic_data


# =============================================================================
# FIX 3: /outbreak_detection - Threshold too high
# =============================================================================

def detect_outbreak_regions(df: pd.DataFrame, region_column: str = "region", time_column: str = "time", case_column: str = "cases", threshold_multiplier: float = 1.5, min_percent_increase: float = 50.0) -> List[Dict[str, Any]]:
    """
    Detect outbreak regions using multiple criteria.
    """
    outbreak_regions = []
    regions = df[region_column].unique()

    for region in regions:
        region_data = df[df[region_column] == region].sort_values(time_column)
        cases = region_data[case_column].values

        if len(cases) < 3:
            continue

        baseline_n = max(3, len(cases) // 3)
        baseline = np.mean(cases[:baseline_n])
        recent = np.mean(cases[-baseline_n:])

        if baseline > 0:
            multiplier = recent / baseline
            percent_increase = ((recent - baseline) / baseline) * 100
        else:
            multiplier = recent if recent > 0 else 0
            percent_increase = 100 if recent > 0 else 0

        consecutive_increases = 0
        for i in range(1, len(cases)):
            if cases[i] > cases[i-1]:
                consecutive_increases += 1
            else:
                consecutive_increases = 0

        is_outbreak = (multiplier >= threshold_multiplier or percent_increase >= min_percent_increase or consecutive_increases >= 5)

        if is_outbreak:
            outbreak_regions.append({
                "region": region,
                "baseline_cases": round(baseline, 1),
                "recent_cases": round(recent, 1),
                "multiplier": round(multiplier, 2),
                "percent_increase": round(percent_increase, 1),
                "consecutive_increase_days": consecutive_increases,
                "severity": "HIGH" if multiplier >= 3 or percent_increase >= 100 else "MEDIUM"
            })

    return outbreak_regions


# =============================================================================
# FIX 4: /confounder_detection - Missing obvious confounders
# =============================================================================

def detect_confounders_improved(df: pd.DataFrame, label_column: str, treatment_column: str, correlation_threshold: float = 0.3) -> List[Dict[str, Any]]:
    """
    Detect confounders by checking correlation with BOTH treatment and outcome.
    """
    confounders = []

    if df[treatment_column].dtype == 'object':
        treatment_encoded = pd.factorize(df[treatment_column])[0]
    else:
        treatment_encoded = df[treatment_column]

    if df[label_column].dtype == 'object':
        outcome_encoded = pd.factorize(df[label_column])[0]
    else:
        outcome_encoded = df[label_column]

    for col in df.columns:
        if col in [label_column, treatment_column, "patient_id"]:
            continue

        try:
            if df[col].dtype == 'object':
                col_encoded = pd.factorize(df[col])[0]
            else:
                col_encoded = df[col].astype(float)

            treatment_corr = np.corrcoef(col_encoded, treatment_encoded)[0, 1]
            outcome_corr = np.corrcoef(col_encoded, outcome_encoded)[0, 1]

            if np.isnan(treatment_corr):
                treatment_corr = 0
            if np.isnan(outcome_corr):
                outcome_corr = 0

            if abs(treatment_corr) > correlation_threshold and abs(outcome_corr) > correlation_threshold:
                strength = (abs(treatment_corr) + abs(outcome_corr)) / 2
                confounders.append({
                    "type": "statistical_confounder",
                    "variable": col,
                    "treatment_correlation": round(treatment_corr, 3),
                    "outcome_correlation": round(outcome_corr, 3),
                    "strength": round(strength, 3),
                    "explanation": f"'{col}' correlates with both treatment ({treatment_corr:.2f}) and outcome ({outcome_corr:.2f})",
                    "recommendation": f"Stratify analysis by '{col}' or include as covariate"
                })
            elif abs(treatment_corr) > 0.5:
                confounders.append({
                    "type": "imbalanced_covariate",
                    "variable": col,
                    "treatment_correlation": round(treatment_corr, 3),
                    "outcome_correlation": round(outcome_corr, 3),
                    "strength": round(abs(treatment_corr), 3),
                    "explanation": f"'{col}' is imbalanced across treatment arms (correlation: {treatment_corr:.2f})",
                    "recommendation": f"Check randomization; consider stratified analysis by '{col}'"
                })
        except Exception:
            continue

    confounders.sort(key=lambda x: x.get("strength", 0), reverse=True)
    return confounders


# =============================================================================
# FIX 5: /multi_omic_fusion - confidence always 0
# =============================================================================

def calculate_multi_omic_confidence(domain_contributions: Dict[str, float]) -> float:
    """
    Calculate confidence score for multi-omic fusion.
    """
    contributions = list(domain_contributions.values())

    if not contributions or sum(contributions) == 0:
        return 0.1

    total = sum(contributions)
    normalized = [c / total for c in contributions]

    coverage = sum(1 for c in normalized if c > 0.01) / len(normalized)
    max_contribution = max(normalized)
    balance = 1 - (max_contribution - (1 / len(normalized)))

    confidence = (coverage * 0.4) + (balance * 0.6)
    confidence = max(0.1, min(0.95, confidence))

    return round(confidence, 2)


# =============================================================================
# FIX 6: /population_risk - empty top_biomarkers
# =============================================================================

def identify_top_biomarkers(analyses: List[Dict[str, Any]], n_top: int = 5) -> List[str]:
    """
    Identify top biomarkers from population analyses using coefficient of variation.
    """
    if not analyses:
        return []

    df = pd.DataFrame(analyses)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude = ["patient_id", "id", "age", "sex", "gender"]
    numeric_cols = [c for c in numeric_cols if c.lower() not in exclude]

    if not numeric_cols:
        return []

    biomarker_stats = []
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) < 2:
            continue
        mean = values.mean()
        std = values.std()
        cv = (std / mean) if mean != 0 else 0
        range_ratio = (values.max() - values.min()) / mean if mean != 0 else 0
        biomarker_stats.append({"name": col, "cv": abs(cv), "range_ratio": abs(range_ratio), "score": abs(cv) * 0.6 + abs(range_ratio) * 0.4})

    biomarker_stats.sort(key=lambda x: x["score"], reverse=True)
    return [b["name"] for b in biomarker_stats[:n_top]]


# =============================================================================
# FIX 7: /patient_report - Incomplete jargon simplification
# =============================================================================

MEDICAL_JARGON_DICTIONARY = {
    "inflammatory": "swelling and irritation in your body",
    "rheumatoid arthritis": "a condition where your immune system attacks your joints",
    "autoimmune": "when your body's defense system attacks healthy cells by mistake",
    "chronic": "long-lasting",
    "acute": "sudden and short-term",
    "systemic": "affecting your whole body",
    "dmard": "disease-modifying medicine",
    "dmards": "disease-modifying medicines",
    "biologic": "a special medicine made from living cells",
    "biologics": "special medicines made from living cells",
    "anti-tnf": "medicine that reduces inflammation",
    "il-6 inhibitor": "medicine that blocks a substance causing swelling",
    "corticosteroid": "strong anti-swelling medicine",
    "immunosuppressant": "medicine that calms down your immune system",
    "c-reactive protein": "a blood test that shows inflammation",
    "crp": "a blood test for inflammation",
    "interleukin-6": "a substance in your blood that causes swelling",
    "il-6": "a substance that causes swelling",
    "il6": "a substance that causes swelling",
    "esr": "a blood test that shows inflammation",
    "hemoglobin": "the part of blood that carries oxygen",
    "hba1c": "a test showing your average blood sugar over 3 months",
    "creatinine": "a test showing how well your kidneys work",
    "wbc": "white blood cells that fight infection",
    "cytopenia": "low blood cell count",
    "cytopenias": "low blood cell counts",
    "hepatic": "liver",
    "renal": "kidney",
    "cardiovascular": "heart and blood vessels",
    "pulmonary": "lung",
    "elevated": "higher than normal",
    "decreased": "lower than normal",
    "inadequate response": "the medicine isn't working well enough",
    "partial response": "the medicine is helping but not completely",
    "remission": "the disease is under control",
    "flare": "the symptoms got worse",
}

def simplify_medical_text(text: str, reading_level: str = "6th_grade") -> str:
    """Replace medical jargon with patient-friendly language."""
    simplified = text.lower()
    sorted_terms = sorted(MEDICAL_JARGON_DICTIONARY.items(), key=lambda x: len(x[0]), reverse=True)

    for medical_term, simple_term in sorted_terms:
        pattern = re.compile(re.escape(medical_term), re.IGNORECASE)
        simplified = pattern.sub(simple_term, simplified)

    sentences = simplified.split('. ')
    sentences = [s.capitalize() for s in sentences]
    return '. '.join(sentences)

def generate_key_findings(clinical_signals: List[Dict], reading_level: str = "6th_grade") -> List[str]:
    """Generate patient-friendly key findings from clinical signals."""
    findings = []

    for signal in clinical_signals:
        name = signal.get("signal_name", "").lower()
        status = signal.get("status", "").lower()
        value = signal.get("value", "")
        normal_range = signal.get("normal_range", "")

        if status in ["elevated", "high"]:
            finding = f"Your {simplify_medical_text(name)} level is higher than normal"
            if value and normal_range:
                finding += f" ({value} vs normal {normal_range})"
        elif status in ["decreased", "low"]:
            finding = f"Your {simplify_medical_text(name)} level is lower than normal"
        else:
            finding = f"Your {simplify_medical_text(name)} was tested"

        findings.append(finding)

    return findings
