"""
HyperCore Comparison Utilities
==============================
Standard early warning score calculators and comparison metrics
for benchmarking HyperCore against NEWS, qSOFA, MEWS.
"""

from typing import Dict, List, Any


def calculate_news_score(hr: float, rr: float, sbp: float, temp: float, spo2: float,
                         consciousness: str = 'A', on_oxygen: bool = False) -> int:
    """
    Calculate NEWS (National Early Warning Score).
    Standard clinical early warning score used in UK NHS.

    Args:
        hr: Heart rate (bpm)
        rr: Respiratory rate (breaths/min)
        sbp: Systolic blood pressure (mmHg)
        temp: Temperature (Celsius)
        spo2: Oxygen saturation (%)
        consciousness: AVPU scale - 'A'lert, 'V'oice, 'P'ain, 'U'nresponsive
        on_oxygen: Whether patient is on supplemental oxygen

    Returns:
        NEWS score (0-20)
    """
    score = 0

    # Respiratory rate (3, 1, 0, 2, 3)
    if rr <= 8: score += 3
    elif rr <= 11: score += 1
    elif rr <= 20: score += 0
    elif rr <= 24: score += 2
    else: score += 3

    # SpO2 (3, 2, 1, 0)
    if spo2 <= 91: score += 3
    elif spo2 <= 93: score += 2
    elif spo2 <= 95: score += 1
    else: score += 0

    # Supplemental oxygen (+2)
    if on_oxygen: score += 2

    # Temperature (3, 1, 0, 1, 2)
    if temp <= 35.0: score += 3
    elif temp <= 36.0: score += 1
    elif temp <= 38.0: score += 0
    elif temp <= 39.0: score += 1
    else: score += 2

    # Systolic BP (3, 2, 1, 0, 3)
    if sbp <= 90: score += 3
    elif sbp <= 100: score += 2
    elif sbp <= 110: score += 1
    elif sbp <= 219: score += 0
    else: score += 3

    # Heart rate (3, 1, 0, 1, 2, 3)
    if hr <= 40: score += 3
    elif hr <= 50: score += 1
    elif hr <= 90: score += 0
    elif hr <= 110: score += 1
    elif hr <= 130: score += 2
    else: score += 3

    # Consciousness (AVPU) - any non-Alert = 3
    if consciousness and consciousness.upper() != 'A':
        score += 3

    return score


def calculate_qsofa_score(rr: float, sbp: float, gcs: int = 15) -> int:
    """
    Calculate qSOFA (Quick SOFA) score.
    Simplified sepsis screening tool.

    Args:
        rr: Respiratory rate (breaths/min)
        sbp: Systolic blood pressure (mmHg)
        gcs: Glasgow Coma Scale (3-15, default 15)

    Returns:
        qSOFA score (0-3)
    """
    score = 0
    if rr >= 22: score += 1
    if sbp <= 100: score += 1
    if gcs < 15: score += 1
    return score


def calculate_mews_score(hr: float, rr: float, sbp: float, temp: float,
                         consciousness: str = 'A') -> int:
    """
    Calculate MEWS (Modified Early Warning Score).

    Args:
        hr: Heart rate (bpm)
        rr: Respiratory rate (breaths/min)
        sbp: Systolic blood pressure (mmHg)
        temp: Temperature (Celsius)
        consciousness: AVPU scale - 'A'lert, 'V'oice, 'P'ain, 'U'nresponsive

    Returns:
        MEWS score (0-14)
    """
    score = 0

    # Respiratory rate
    if rr < 9: score += 2
    elif rr <= 14: score += 0
    elif rr <= 20: score += 1
    elif rr <= 29: score += 2
    else: score += 3

    # Heart rate
    if hr < 40: score += 2
    elif hr <= 50: score += 1
    elif hr <= 100: score += 0
    elif hr <= 110: score += 1
    elif hr <= 129: score += 2
    else: score += 3

    # Systolic BP
    if sbp < 70: score += 3
    elif sbp < 80: score += 2
    elif sbp < 100: score += 1
    elif sbp <= 199: score += 0
    else: score += 2

    # Temperature
    if temp < 35: score += 2
    elif temp <= 38.4: score += 0
    else: score += 2

    # Consciousness (AVPU)
    consciousness = consciousness.upper() if consciousness else 'A'
    if consciousness == 'A': score += 0
    elif consciousness == 'V': score += 1
    elif consciousness == 'P': score += 2
    else: score += 3

    return score


def calculate_comparison_metrics(predictions: List[bool], outcomes: List[bool]) -> Dict[str, Any]:
    """
    Calculate sensitivity, specificity, PPV from predictions vs outcomes.

    Args:
        predictions: List of boolean predictions (True = alert)
        outcomes: List of boolean actual outcomes (True = deterioration)

    Returns:
        Dictionary with sensitivity, specificity, PPV, NPV, confusion matrix
    """
    tp = fp = tn = fn = 0

    for pred, actual in zip(predictions, outcomes):
        if pred and actual: tp += 1
        elif pred and not actual: fp += 1
        elif not pred and not actual: tn += 1
        else: fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    # PPV at 5% prevalence (standardized comparison)
    # PPV = (sens * prev) / (sens * prev + (1 - spec) * (1 - prev))
    prevalence = 0.05
    denom = sensitivity * prevalence + (1 - specificity) * (1 - prevalence)
    ppv_at_5 = (sensitivity * prevalence) / denom if denom > 0 else 0

    return {
        'sensitivity': round(sensitivity, 3),
        'specificity': round(specificity, 3),
        'ppv': round(ppv, 3),
        'ppv_at_5_percent': round(ppv_at_5, 3),
        'npv': round(npv, 3),
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'total': tp + fp + tn + fn
    }
