# HyperCore Validation Leakage Audit Report

**Date:** March 30, 2026
**Status:** LEAKAGE DETECTED - Results Invalid

---

## EXECUTIVE SUMMARY

**WARNING: The MIMIC-IV validation results are INVALID due to data leakage.**

The validation showed:
- 100% specificity (0 false positives)
- 83.3% sensitivity
- 31.4 days lead time

**Root Cause:** HyperCore API received the outcome labels in the CSV data, allowing it to "see" which patients had events.

---

## QUESTION 1: Feature Timestamp Enforcement

### Finding: LEAKAGE PRESENT

**What was sent to HyperCore:**
```python
# The validation script sent FULL patient trajectory:
csv_df['outcome'] = rows['event_in_12h'].values  # <-- LEAKAGE!
```

**The Problem:**
- Each patient's entire trajectory was sent as a single CSV
- The `outcome` column contained the event labels (0 or 1)
- HyperCore could see which timepoints had events
- This is NOT how a prospective system would operate

**Timestamp Enforcement Status:**
| Check | Status |
|-------|--------|
| Features from before prediction time only? | PARTIAL - vitals/labs were temporal |
| Outcome labels hidden from model? | **FAILED** - outcome visible in CSV |
| Post-event data excluded? | **FAILED** - full trajectory sent |

---

## QUESTION 2: Why 100% Specificity?

### Finding: OUTCOME LEAKAGE

**Risk Score Distribution:**

| Group | Count | Mean Score | Min | Max |
|-------|-------|------------|-----|-----|
| Event patients | 42 | 0.551 | 0.160 | 0.810 |
| Non-event patients | 163 | **0.000** | **0.000** | **0.000** |

**Critical Evidence:**
- ALL 163 non-event patients have risk_score = 0.000 (exactly)
- ALL 42 event patients have risk_score > 0
- This perfect separation is statistically impossible without seeing labels

**Conclusion:** HyperCore used the outcome column to determine risk scores.

---

## QUESTION 3: Lead Time Explanation

### Finding: METRIC IS MISLEADING

**What `lead_time_days` Measures:**
- Time from FIRST biomarker change to event
- Essentially measures patient's total length of stay
- NOT a clinically meaningful early warning metric

**Distribution:**
| Statistic | Value |
|-----------|-------|
| Mean | 31.4 days |
| Median | 17.0 days |
| Min | 0.0 days |
| Max | 207.0 days |
| Std Dev | 40.7 days |

**Breakdown:**
- 0-7 days: 9 patients (26%)
- 7-14 days: 5 patients (14%)
- 14-30 days: 11 patients (31%)
- 30-60 days: 2 patients (6%)
- 60+ days: 7 patients (20%)

**Why This Is Misleading:**
1. A patient admitted 200 days before death shows "200-day lead time"
2. This doesn't represent when a clinician would receive an actionable alert
3. True lead time should be measured from ALERT time to EVENT time

---

## QUESTION 4: Leakage Audit Table

### Features Used

| Feature | Source | Temporal Filtering | Leakage Risk |
|---------|--------|-------------------|--------------|
| heart_rate | chartevents | Window-based | LOW |
| respiratory_rate | chartevents | Window-based | LOW |
| sbp | chartevents | Window-based | LOW |
| spo2 | chartevents | Window-based | LOW |
| temperature | chartevents | Window-based | LOW |
| lactate | labevents | Window-based | LOW |
| creatinine | labevents | Window-based | LOW |
| troponin | labevents | Window-based | LOW |
| **outcome** | Derived | **NOT FILTERED** | **CRITICAL** |

### Prohibited Features

| Feature | Reason | Status |
|---------|--------|--------|
| Death timestamp | Future information | Not included |
| Discharge status | Future information | Not included |
| **outcome column** | Event label | **INCLUDED - LEAKAGE** |

### Timestamp Enforcement

| Requirement | Implementation | Status |
|-------------|----------------|--------|
| Vitals before prediction time | 4-hour windows | PASSED |
| Labs before prediction time | 4-hour windows | PASSED |
| Outcome hidden from features | Should be excluded | **FAILED** |
| Post-event data excluded | Should stop at prediction | **FAILED** |

---

## CORRECTED VALIDATION APPROACH

To properly validate HyperCore without leakage:

### Option A: Remove Outcome from API Input
```python
# Don't include outcome in the CSV sent to HyperCore
csv_df = csv_df.drop(columns=['outcome'])
payload = {
    "csv": csv_df.to_csv(index=False),
    # Don't specify label_column
}
```

### Option B: Window-Level Predictions
```python
# For each prediction window, only send data UP TO that point
for window in patient_windows:
    data_before_window = patient_data[patient_data['time'] <= window['time']]
    # Predict for THIS window only
    # Compare prediction to actual outcome for THIS window
```

### Option C: Train/Test Split
```python
# Train on subset of patients, test on held-out patients
# Never show test patient outcomes to the model
train_patients, test_patients = train_test_split(patients, test_size=0.3)
```

---

## CONFIDENCE IN RESULTS

| Aspect | Confidence | Reason |
|--------|------------|--------|
| Sensitivity (83.3%) | **LOW** | Leakage inflates true positives |
| Specificity (100%) | **INVALID** | Caused by outcome leakage |
| PPV (100%) | **INVALID** | Caused by outcome leakage |
| Lead Time (31.4 days) | **MISLEADING** | Measures trajectory length, not alert timing |

---

## HONEST ASSESSMENT

### What We Can Conclude:
1. HyperCore API works and processes longitudinal data
2. The trajectory analysis identifies biomarker changes
3. The system correctly formats responses with risk scores

### What We Cannot Conclude:
1. HyperCore outperforms NEWS (leakage invalidates comparison)
2. 100% specificity is achievable (artifact of leakage)
3. 31-day lead time is meaningful (measures wrong thing)

### Recommendation:
**Re-run validation with proper leakage control before presenting to Dr. Handler.**

---

## NEXT STEPS

1. **Fix the validation script** to not include outcome in API input
2. **Implement window-level predictions** for proper temporal evaluation
3. **Measure TRUE lead time** from first alert to event
4. **Re-run comparison** with leakage-free methodology
5. **Report honest results** even if less favorable

---

**Report Generated:** March 30, 2026
**Audit Status:** FAILED - Leakage Detected
**Action Required:** Re-validation with corrected methodology
