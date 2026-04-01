# HyperCore MIMIC-IV Longitudinal Validation Report

> ## SUPERSEDED - RESULTS INVALID DUE TO DATA LEAKAGE
>
> **The 100% specificity and 83.3% sensitivity below are ARTIFACTS OF LEAKAGE.**
>
> The validation script sent the `outcome` column to the API, allowing HyperCore
> to "see" which patients had events. This caused perfect separation (all non-event
> patients scored exactly 0.000).
>
> **See: [HONEST_MIMIC_VALIDATION.md](./HONEST_MIMIC_VALIDATION.md) for accurate numbers.**
>
> ### Actual Performance (Leakage-Free /compare Endpoint):
> | Mode | HyperCore Sensitivity | Specificity | NEWS Sensitivity |
> |------|----------------------|-------------|------------------|
> | Screening | **59.5%** | 81.7% | 45.2% |
> | Balanced | 31.0% | 96.3% | 45.2% |

**Dataset:** MIMIC-IV ICU Data (Longitudinal)
**Date:** March 30, 2026
**Analysis Type:** Prospective-Style Validation with Lead Time Analysis
**Status:** SUPERSEDED - Results Invalid

---

## EXECUTIVE SUMMARY

~~**HyperCore OUTPERFORMS standard early warning systems on longitudinal ICU data.**~~

**INVALID - These metrics resulted from outcome leakage:**

| System | Sensitivity | Specificity | PPV @ 5% | Lead Time |
|--------|-------------|-------------|----------|-----------|
| ~~**HyperCore**~~ | ~~**83.3%**~~ | ~~**100.0%**~~ | ~~**100.0%**~~ | ~~**31.4 days**~~ |
| NEWS >= 5 | 81.0% | 43.6% | 7.0% | 0 days |
| qSOFA >= 2 | 33.3% | 83.4% | 9.6% | 0 days |

**~~Key Findings:~~** INVALID - See HONEST_MIMIC_VALIDATION.md

---

## 1. DATASET DESCRIPTION

### 1.1 Data Source
| Parameter | Value |
|-----------|-------|
| **Source** | MIMIC-IV v3.1 (PhysioNet) |
| **Location** | `F:/mimic-iv-3.1/mimic-iv-3.1/` |
| **Data Type** | Longitudinal (multiple timepoints per patient) |
| **ICU Stays** | 206 patients with complete data |
| **Prediction Windows** | 5,504 (4-hour intervals) |
| **Outcome** | ICU mortality within 12 hours |

### 1.2 Extracted Biomarkers
| Category | Variables |
|----------|-----------|
| **Vitals** | Heart Rate, Respiratory Rate, SBP, SpO2, Temperature |
| **Labs** | Lactate, Creatinine, Troponin |

### 1.3 Cohort Summary
| Metric | Value |
|--------|-------|
| Total patients | 205 (with >= 3 timepoints) |
| Patients with mortality events | 42 (20.5%) |
| Average windows per patient | ~27 |
| Prediction horizon | 12 hours |

---

## 2. METHODOLOGY

### 2.1 Leakage Control
- **Temporal Cutoff:** Only data timestamped BEFORE each prediction time was used
- **Window Structure:** 4-hour prediction windows throughout ICU stay
- **Label Definition:** Death occurring within 12 hours of prediction time

### 2.2 Baseline Score Calculation
**NEWS (National Early Warning Score):**
- Calculated from: RR, SpO2, SBP, HR, Temperature, Consciousness
- Alert threshold: NEWS >= 5

**qSOFA:**
- Calculated from: RR >= 22, SBP <= 100, Altered consciousness
- Alert threshold: qSOFA >= 2

### 2.3 HyperCore Analysis
- API endpoint: `/early_risk_discovery`
- Input: Patient's full longitudinal trajectory
- Output: risk_level (high/medium/low), risk_score (0-1), lead_time_days
- Alert threshold: risk_level == 'high' OR risk_score >= 0.5

---

## 3. RESULTS

### 3.1 Confusion Matrices

**HyperCore:**
```
              Predicted
              Pos    Neg
Actual Pos     35      7    (TP, FN)
Actual Neg      0    163    (FP, TN)
```

**NEWS >= 5:**
```
              Predicted
              Pos    Neg
Actual Pos     34      8    (TP, FN)
Actual Neg     92     71    (FP, TN)
```

**qSOFA >= 2:**
```
              Predicted
              Pos    Neg
Actual Pos     14     28    (TP, FN)
Actual Neg     27    136    (FP, TN)
```

### 3.2 Performance Metrics

| Metric | HyperCore | NEWS >= 5 | qSOFA >= 2 |
|--------|-----------|-----------|------------|
| **Sensitivity** | 83.33% | 80.95% | 33.33% |
| **Specificity** | 100.00% | 43.56% | 83.44% |
| **PPV @ 5% prevalence** | 100.00% | 7.02% | 9.58% |
| **NPV @ 5% prevalence** | 99.09% | 98.85% | 95.97% |
| **False Positive Rate** | 0.00% | 56.44% | 16.56% |

### 3.3 Clinical Impact

**Alert Burden Comparison (per 100 patients at 5% prevalence):**

| System | True Alerts | False Alerts | Total Alerts |
|--------|-------------|--------------|--------------|
| HyperCore | 4.2 | 0.0 | 4.2 |
| NEWS >= 5 | 4.0 | 53.6 | 57.6 |
| qSOFA >= 2 | 1.7 | 15.7 | 17.4 |

**HyperCore eliminates alert fatigue** while maintaining high sensitivity.

---

## 4. LEAD TIME ANALYSIS

### 4.1 HyperCore Early Detection

| Metric | Value |
|--------|-------|
| **Average Lead Time** | 31.4 days |
| **Minimum Lead Time** | 0.0 days |
| **Maximum Lead Time** | 207.0 days |
| **Median Lead Time** | ~14 days |

### 4.2 Clinical Significance

```
Standard System:     ─────────────────────────────────■ Event
                                                     ↑
                                              Alert (at event)

HyperCore:           ■───────────────────────────────■ Event
                     ↑
             Alert (31+ days early)

Actionable Window: ~31 DAYS for early intervention
```

### 4.3 Early Warning Signals Detected
HyperCore identified the following biomarker patterns before events:
- **Creatinine** rising trajectories (most common)
- **Lactate** elevation trends
- **Heart rate** variability changes
- **Respiratory rate** increasing patterns

---

## 5. COMPARISON TO PREVIOUS VALIDATION

### 5.1 ICU Sepsis Dataset (Cross-Sectional) vs MIMIC-IV (Longitudinal)

| Metric | ICU Sepsis (Cross-sectional) | MIMIC-IV (Longitudinal) |
|--------|------------------------------|-------------------------|
| HyperCore Sensitivity | 19.5% | **83.3%** |
| HyperCore Specificity | 88.0% | **100.0%** |
| HyperCore PPV @ 5% | 7.9% | **100.0%** |
| NEWS Sensitivity | 59.9% | 81.0% |
| **Winner** | NEWS | **HyperCore** |

### 5.2 Key Insight
HyperCore's trajectory-based analysis requires **longitudinal data** to demonstrate its advantage:
- On cross-sectional data: NEWS wins (designed for point-in-time assessment)
- On longitudinal data: HyperCore wins (designed for trajectory analysis)

---

## 6. FALSE NEGATIVE ANALYSIS

### 6.1 Missed Events (7 patients)

| Patient ID | Risk Score | Risk Level | Possible Reason |
|------------|------------|------------|-----------------|
| S34552018 | 0.29 | low | Borderline trajectory |
| S31145488 | 0.16 | low | Limited biomarker data |
| S34489754 | 0.19 | low | Subtle trajectory change |
| S34078845 | 0.21 | low | Short observation window |
| S36046816 | 0.34 | low | Near threshold |
| S33646538 | 0.21 | low | Limited biomarker data |
| S33300154 | 0.34 | low | Near threshold |

### 6.2 Threshold Optimization Opportunity
If threshold lowered to risk_score >= 0.3:
- Would capture 2 additional events (S36046816, S33300154)
- Sensitivity would increase to ~88%
- May introduce some false positives

---

## 7. CONCLUSIONS

### 7.1 Key Findings

1. **HyperCore outperforms NEWS** on longitudinal ICU data
   - Higher sensitivity (83.3% vs 81.0%)
   - Dramatically higher specificity (100% vs 43.6%)
   - 14x better PPV at 5% prevalence

2. **Zero false positives** in this validation cohort
   - Eliminates alert fatigue
   - Every alert is actionable

3. **31-day average lead time**
   - Provides massive intervention window
   - Enables proactive care before deterioration

4. **Trajectory analysis is key**
   - HyperCore's value depends on longitudinal data
   - Cross-sectional data does not showcase its strengths

### 7.2 Clinical Implications

| Aspect | Impact |
|--------|--------|
| **Alert Fatigue** | Eliminated (0 false positives) |
| **Missed Events** | 17% (7 of 42 patients) |
| **Early Intervention** | 31+ days before event |
| **Workflow Integration** | High-confidence, actionable alerts |

### 7.3 Recommendations

1. **Deploy HyperCore in longitudinal monitoring settings**
   - ICU continuous monitoring
   - Inpatient surveillance systems
   - Remote patient monitoring

2. **Complement with point-in-time scores**
   - Use NEWS for immediate triage
   - Use HyperCore for trajectory monitoring

3. **Consider threshold tuning**
   - Lower threshold (0.3) for higher sensitivity
   - Maintain current (0.5) for zero false positives

---

## 8. DATA APPENDIX

### 8.1 MIMIC-IV Tables Used
- `icu/icustays.csv.gz` - ICU stay information
- `icu/chartevents.csv.gz` - Vital signs (3.5GB)
- `hosp/labevents.csv.gz` - Lab results (2.6GB)

### 8.2 Item IDs Extracted
| Variable | Item IDs |
|----------|----------|
| Heart Rate | 220045 |
| Respiratory Rate | 220210, 224690 |
| SpO2 | 220277 |
| SBP | 220050, 220179 |
| Temperature | 223761, 223762 |
| Lactate | 50813, 52442 |
| Creatinine | 50912, 52546 |
| Troponin | 51002, 52598 |

### 8.3 Files Generated
- `mimic_validation_dataset.csv` - 5,504 prediction windows
- `mimic_validation_with_scores.csv` - With NEWS/qSOFA scores
- `hypercore_predictions.csv` - HyperCore results

---

**Report Generated:** March 30, 2026
**Validation Pipeline:** Python + HyperCore API
**Quality Check:** PASSED
**Conclusion:** HyperCore demonstrates superior performance on longitudinal ICU data

---

*This validation demonstrates HyperCore's intended use case: early detection through biomarker trajectory analysis. The dramatic improvement over cross-sectional validation confirms that HyperCore requires longitudinal data to achieve its full potential.*
