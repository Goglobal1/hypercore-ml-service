# Risk Scoring Guide

This document explains how the HyperCore Alert System automatically calculates risk scores from raw biomarker values.

---

## Overview

When a patient intake request is submitted without an explicit `risk_score`, the system automatically calculates one using:

1. **Biomarker Thresholds** - Predefined warning and critical levels
2. **Direction Awareness** - Whether high or low values are dangerous
3. **Weight Factors** - Relative importance of each biomarker
4. **Composite Scoring** - Weighted average across all provided values

---

## Calculation Algorithm

### Step 1: Normalize Biomarker Names

Input biomarker names are normalized to handle variations:

```
"WBC" → "wbc"
"white_blood_cell" → "wbc"
"c-reactive-protein" → "crp"
"heart_rate" → "heart_rate"
"HR" → "heart_rate"
```

### Step 2: Calculate Individual Scores

For each biomarker, a score (0.0 - 1.0) is calculated based on thresholds:

**Rising Direction** (bad when high - e.g., lactate, WBC):
```
if value >= critical: score = 1.0
elif value >= warning: score = 0.5 + 0.5 * (value - warning) / (critical - warning)
else: score = 0.0 (or small value if close to warning)
```

**Falling Direction** (bad when low - e.g., SpO2, GFR):
```
if value <= critical: score = 1.0
elif value <= warning: score = 0.5 + 0.5 * (warning - value) / (warning - critical)
else: score = 0.0 (or small value if close to warning)
```

### Step 3: Apply Weights

Each biomarker has a weight (0.0 - 1.0) indicating clinical importance:

| Biomarker | Weight | Rationale |
|-----------|--------|-----------|
| lactate | 1.0 | Primary sepsis marker |
| troponin | 1.0 | Primary cardiac marker |
| creatinine | 1.0 | Primary kidney marker |
| GCS | 1.0 | Primary neuro marker |
| SpO2 | 1.0 | Primary respiratory marker |
| WBC | 0.6 | Supporting marker |
| CRP | 0.7 | Supporting marker |
| temperature | 0.5 | Supporting marker |
| heart_rate | 0.5-0.7 | Context dependent |

### Step 4: Composite Score

```python
total_weighted_score = sum(score * weight for each biomarker)
total_weight = sum(weight for each biomarker)
risk_score = total_weighted_score / total_weight
```

### Step 5: Multi-Critical Boost

If multiple biomarkers are critical, apply a boost:
- 3+ critical: +15% (cap at 1.0)
- 2 critical: +10% (cap at 1.0)
- 4+ warnings: +5% (cap at 1.0)

---

## Biomarker Thresholds by Domain

### Sepsis

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| lactate | 2.0 mmol/L | 4.0 mmol/L | rising | 1.0 |
| WBC | 10.0 K/uL | 12.0 K/uL | rising | 0.6 |
| CRP | 50.0 mg/L | 100.0 mg/L | rising | 0.7 |
| procalcitonin | 0.5 ng/mL | 2.0 ng/mL | rising | 0.8 |
| temperature | 37.8 C | 38.3 C | rising | 0.5 |
| heart_rate | 100 bpm | 120 bpm | rising | 0.5 |
| respiratory_rate | 20 /min | 24 /min | rising | 0.5 |
| MAP | 70 mmHg | 65 mmHg | falling | 0.9 |

### Cardiac

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| troponin | 0.01 ng/mL | 0.04 ng/mL | rising | 1.0 |
| troponin_i | 0.01 ng/mL | 0.04 ng/mL | rising | 1.0 |
| troponin_t | 0.03 ng/mL | 0.1 ng/mL | rising | 1.0 |
| BNP | 100 pg/mL | 400 pg/mL | rising | 0.9 |
| NT-proBNP | 300 pg/mL | 900 pg/mL | rising | 0.9 |
| heart_rate | 110 bpm | 130 bpm | rising | 0.7 |
| systolic_bp | 100 mmHg | 90 mmHg | falling | 0.8 |
| CK-MB | 5.0 ng/mL | 25.0 ng/mL | rising | 0.7 |

### Kidney

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| creatinine | 1.3 mg/dL | 2.0 mg/dL | rising | 1.0 |
| BUN | 25 mg/dL | 40 mg/dL | rising | 0.8 |
| potassium | 5.0 mEq/L | 5.5 mEq/L | rising | 0.9 |
| GFR | 60 mL/min | 30 mL/min | falling | 0.9 |
| urine_output | 1.0 mL/kg/h | 0.5 mL/kg/h | falling | 0.8 |

### Respiratory

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| SpO2 | 94% | 90% | falling | 1.0 |
| PaO2 | 80 mmHg | 60 mmHg | falling | 0.9 |
| respiratory_rate | 24 /min | 30 /min | rising | 0.8 |
| FiO2 | 0.4 | 0.6 | rising | 0.7 |
| PaO2/FiO2 | 300 | 200 | falling | 0.9 |
| PaCO2 | 45 mmHg | 50 mmHg | rising | 0.6 |

### Hepatic

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| ALT | 200 U/L | 1000 U/L | rising | 0.8 |
| AST | 200 U/L | 1000 U/L | rising | 0.8 |
| bilirubin | 2.0 mg/dL | 4.0 mg/dL | rising | 0.9 |
| INR | 1.5 | 2.0 | rising | 0.9 |
| albumin | 3.0 g/dL | 2.5 g/dL | falling | 0.7 |
| ammonia | 50 umol/L | 100 umol/L | rising | 0.8 |

### Neurological

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| GCS | 12 | 8 | falling | 1.0 |
| ICP | 15 mmHg | 22 mmHg | rising | 0.9 |
| CPP | 70 mmHg | 60 mmHg | falling | 0.9 |

### Metabolic

| Biomarker | Warning | Critical | Direction | Weight |
|-----------|---------|----------|-----------|--------|
| glucose (high) | 180 mg/dL | 400 mg/dL | rising | 0.8 |
| glucose (low) | 70 mg/dL | 50 mg/dL | falling | 0.9 |
| pH | 7.32 | 7.25 | falling | 0.9 |
| pH (high) | 7.48 | 7.55 | rising | 0.8 |
| potassium | 5.0 mEq/L | 6.0 mEq/L | rising | 0.9 |
| potassium (low) | 3.5 mEq/L | 3.0 mEq/L | falling | 0.8 |
| lactate | 2.0 mmol/L | 4.0 mmol/L | rising | 0.9 |
| anion_gap | 14 mEq/L | 20 mEq/L | rising | 0.7 |

---

## Supported Biomarker Aliases

The system accepts many common aliases:

### Sepsis Markers
- `lactate`, `lactic_acid`
- `WBC`, `white_blood_cell`, `leukocytes`
- `CRP`, `c_reactive_protein`, `c-reactive-protein`
- `procalcitonin`, `PCT`
- `temperature`, `temp`

### Cardiac Markers
- `troponin`, `troponin_i`, `troponin_t`, `tropi`, `tropt`
- `BNP`, `nt_probnp`, `ntprobnp`
- `CK_MB`, `CKMB`

### Vital Signs
- `heart_rate`, `HR`, `pulse`
- `respiratory_rate`, `RR`, `resp_rate`
- `blood_pressure_systolic`, `bp_systolic`, `systolic_bp`, `SBP`
- `blood_pressure_diastolic`, `bp_diastolic`, `diastolic_bp`, `DBP`
- `SpO2`, `oxygen_saturation`, `o2_sat`

### Kidney Markers
- `creatinine`, `creat`
- `BUN`, `blood_urea_nitrogen`
- `GFR`, `eGFR`
- `potassium`, `K`

### Liver Markers
- `ALT`, `SGPT`
- `AST`, `SGOT`
- `bilirubin`, `bili`, `total_bilirubin`
- `albumin`, `alb`

---

## Examples

### Example 1: Normal Sepsis Labs

```json
{
  "patient_id": "P001",
  "risk_domain": "sepsis",
  "lab_data": {
    "lactate": 1.0,
    "WBC": 7.5,
    "CRP": 5.0
  }
}
```

**Result:**
- lactate: 1.0 < 2.0 (warning) → score = 0.0, weight = 1.0
- WBC: 7.5 < 10.0 (warning) → score = 0.0, weight = 0.6
- CRP: 5.0 < 50.0 (warning) → score = 0.0, weight = 0.7

**Composite:** (0 + 0 + 0) / (1.0 + 0.6 + 0.7) = **0.0**
**State:** S0 (Stable)

---

### Example 2: Critical Sepsis

```json
{
  "patient_id": "P002",
  "risk_domain": "sepsis",
  "lab_data": {
    "lactate": 5.0,
    "WBC": 16.0,
    "CRP": 150.0,
    "procalcitonin": 3.0
  }
}
```

**Result:**
- lactate: 5.0 >= 4.0 (critical) → score = 1.0, weight = 1.0
- WBC: 16.0 >= 12.0 (critical) → score = 1.0, weight = 0.6
- CRP: 150.0 >= 100.0 (critical) → score = 1.0, weight = 0.7
- procalcitonin: 3.0 >= 2.0 (critical) → score = 1.0, weight = 0.8

**Composite:** (1.0 + 0.6 + 0.7 + 0.8) / (1.0 + 0.6 + 0.7 + 0.8) = **1.0**
**Multi-critical boost:** 4 critical → no additional boost needed (already 1.0)
**State:** S3 (Critical)

---

### Example 3: Mixed Warning/Critical

```json
{
  "patient_id": "P003",
  "risk_domain": "sepsis",
  "lab_data": {
    "lactate": 3.0,
    "WBC": 14.0,
    "CRP": 80.0
  }
}
```

**Result:**
- lactate: 3.0 between 2.0-4.0 → score = 0.75, weight = 1.0
- WBC: 14.0 >= 12.0 (critical) → score = 1.0, weight = 0.6
- CRP: 80.0 between 50-100 → score = 0.65, weight = 0.7

**Composite:** (0.75 + 0.6 + 0.455) / (1.0 + 0.6 + 0.7) = **0.785**
**State:** S3 (Critical, just above 0.75)

---

### Example 4: Falling Biomarker (SpO2)

```json
{
  "patient_id": "P004",
  "risk_domain": "respiratory",
  "vitals": {
    "SpO2": 91
  }
}
```

**Result:**
- SpO2: 91 between 90-94 (falling direction) → score = 0.75, weight = 1.0

**Composite:** 0.75 / 1.0 = **0.75**
**State:** S2 (Escalating) or S3 if exactly at boundary

---

## State Mapping

| Risk Score | State | Name | Action |
|------------|-------|------|--------|
| 0.00 - 0.25 | S0 | Stable | Routine monitoring |
| 0.25 - 0.50 | S1 | Watch | Enhanced monitoring |
| 0.50 - 0.75 | S2 | Escalating | Clinical evaluation |
| 0.75 - 1.00 | S3 | Critical | Immediate intervention |

Note: Exact thresholds vary by domain. Sepsis uses more aggressive thresholds (0.25/0.50/0.75).

---

## Fallback Behavior

1. **Unknown Domain:** Falls back to sepsis thresholds (most comprehensive)
2. **Unknown Biomarker:** Skipped, doesn't contribute to score
3. **Non-numeric Values:** Skipped
4. **Empty Data:** Returns risk_score = 0.0

---

## Customization

Site-specific thresholds can be configured via `SiteConfig`:

```python
from app.core.alert_system.config import SiteConfig, BiomarkerThreshold

site_config = SiteConfig(
    site_id="hospital_abc",
    site_name="ABC Hospital",
    biomarker_overrides={
        "sepsis": {
            "lactate": BiomarkerThreshold(
                warning=1.5,  # More aggressive
                critical=3.0,
                unit="mmol/L",
                direction="rising",
                weight=1.0
            )
        }
    }
)
```

---

**Last Updated:** 2026-03-19
