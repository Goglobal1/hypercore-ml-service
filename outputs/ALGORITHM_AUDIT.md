# HyperCore Algorithm Audit Report

**Date:** 2026-03-31
**Status:** BUG FOUND - Threshold Mismatch

---

## EXECUTIVE SUMMARY

**A critical threshold mismatch between `/compare` and the main scoring function was found.**

This explains the sensitivity discrepancy:
- Claimed: 78% sensitivity (balanced mode)
- Actual with bug: 31% sensitivity
- Expected after fix: Should improve significantly

---

## BUG IDENTIFIED

### Location: `/compare` endpoint (main.py line 20866-20870)

**Current (WRONG):**
```python
mode_config = {
    'screening': {'risk_threshold': 0.15, 'min_domains': 1},
    'balanced': {'risk_threshold': 0.30, 'min_domains': 2},  # <-- BUG: 0.30
    'high_confidence': {'risk_threshold': 0.50, 'min_domains': 3}  # <-- BUG: 0.50
}
```

**Should match OPERATING_MODES (main.py line 874-926):**
```python
OPERATING_MODES = {
    "high_confidence": {
        "min_domains": 3,
        "alert_threshold": 0.15,  # <-- Correct: 0.15
    },
    "balanced": {
        "min_domains": 2,
        "alert_threshold": 0.15,  # <-- Correct: 0.15
    },
    "screening": {
        "min_domains": 1,
        "alert_threshold": 0.15,  # <-- Correct: 0.15
    }
}
```

### Impact

| Mode | /compare threshold | OPERATING_MODES threshold | Error |
|------|-------------------|--------------------------|-------|
| screening | 0.15 | 0.15 | OK |
| balanced | **0.30** | 0.15 | **2x stricter** |
| high_confidence | **0.50** | 0.15 | **3.3x stricter** |

This explains why:
- Screening mode: 59.5% sensitivity (threshold matches)
- Balanced mode: 31% sensitivity (threshold is 2x stricter than designed)
- High confidence mode: 2.4% sensitivity (threshold is 3.3x stricter than designed)

---

## SYSTEM ARCHITECTURE

### Endpoints That Use Hybrid Scoring

| Endpoint | Uses hybrid scoring? | Notes |
|----------|---------------------|-------|
| `/analyze` | No | Uses separate analysis |
| `/early_risk_discovery` | **YES** | Primary entry point |
| `/trajectory/analyze` | Partial | Different scoring |
| `/predictive_modeling` | No | ML-based |
| `/predict/time-to-harm` | Partial | Domain-specific |
| `/compare` | **YES** | Has threshold bug |
| `/scoring_modes` | Info only | Returns mode descriptions |

### Core Scoring Functions

| Function | Location | Description |
|----------|----------|-------------|
| `calculate_hybrid_risk_score()` | main.py:932 | **Main scoring function** |
| `calculate_news_score()` | comparison_utils.py:11 | NEWS calculation |
| `calculate_qsofa_score()` | comparison_utils.py:76 | qSOFA calculation |
| `calculate_mews_score()` | comparison_utils.py:96 | MEWS calculation |
| `calculate_comparison_metrics()` | comparison_utils.py:150 | Confusion matrix metrics |
| `calculate_risk_score()` | risk_calculator.py:211 | Alert system scoring |

---

## HYBRID SCORING ALGORITHM ANALYSIS

### Signal Generation (Lines 1017-1059)

For each biomarker, score is calculated as:

**1. Absolute Thresholds:**
- Critical breach: +0.25 + critical_bonus (0.15)
- Warning breach: +0.20

**2. Trajectory Analysis:**
- Rising concern (>15-25% increase): +0.30
- Falling concern (>15-30% decrease): +0.30

**3. Domain Convergence:**
- 2 domains alerting: 1.0 + domain_bonus_2 (0.12) = 1.12x multiplier
- 3 domains alerting: 1.0 + domain_bonus_3 (0.20) = 1.20x multiplier
- 4+ domains alerting: 1.0 + domain_bonus_3 + 0.1 = 1.30x multiplier

### Alerting Logic (Lines 1098-1107)

```python
# Current (correct in main function):
meets_min_domains = num_domains_alerting >= min_domains
meets_alert_criteria = meets_min_domains and final_score >= alert_threshold
# Where alert_threshold = 0.15 for all modes
```

### Biomarker Domains

| Domain | Biomarkers | Weight Range |
|--------|------------|--------------|
| hemodynamic | heart_rate, sbp, dbp, map | 0.8-1.3 |
| respiratory | respiratory_rate, spo2, fio2 | 1.0-1.5 |
| inflammatory | lactate, crp, procalcitonin, wbc, temperature | 0.8-1.5 |
| renal | creatinine, bun, gfr | 1.0-1.3 |
| cardiac | troponin, bnp | 1.2-1.3 |
| hepatic | alt, ast, bilirubin | 1.0-1.2 |
| hematologic | platelets, hemoglobin, inr | 1.0-1.2 |
| metabolic | glucose, potassium, sodium | 0.9-1.1 |

---

## FIX REQUIRED

### Change in `/compare` endpoint

**From:**
```python
mode_config = {
    'screening': {'risk_threshold': 0.15, 'min_domains': 1},
    'balanced': {'risk_threshold': 0.30, 'min_domains': 2},
    'high_confidence': {'risk_threshold': 0.50, 'min_domains': 3}
}
```

**To:**
```python
mode_config = {
    'screening': {'risk_threshold': 0.15, 'min_domains': 1},
    'balanced': {'risk_threshold': 0.15, 'min_domains': 2},
    'high_confidence': {'risk_threshold': 0.15, 'min_domains': 3}
}
```

---

## EXPECTED RESULTS AFTER FIX

With correct thresholds (0.15 for all modes), the primary differentiator becomes `min_domains`:

| Mode | Threshold | Min Domains | Expected Sensitivity |
|------|-----------|-------------|---------------------|
| screening | 0.15 | 1 | ~60% (current) |
| balanced | 0.15 | 2 | ~60%? (was 31%) |
| high_confidence | 0.15 | 3 | Higher than 2.4% |

**Note:** The domain requirement is the main filter, not the risk threshold.

---

## STILL OPEN QUESTION

Even with the fix, will we achieve 78% sensitivity?

The `expected_metrics` in `OPERATING_MODES` claims:
- balanced: 78% sensitivity
- screening: 87.8% sensitivity

These may have been calibrated with different thresholds or data. After fixing the bug, we need to re-validate and potentially:
1. Lower the min_domains requirement
2. Adjust trajectory thresholds
3. Lower the alert_threshold below 0.15

---

*Report generated: 2026-03-31*
