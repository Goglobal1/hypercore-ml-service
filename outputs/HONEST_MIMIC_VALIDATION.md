# HyperCore MIMIC-IV Validation - HONEST RESULTS

**Date:** 2026-03-31
**Methodology:** Leakage-free /compare endpoint
**Dataset:** 206 patients, 42 events (20.4% prevalence)

---

## EXECUTIVE SUMMARY

**Previous claims were INFLATED due to data leakage and methodological issues.**

The /compare endpoint provides leakage-free validation by:
1. NOT sending outcome labels to the scoring algorithm
2. Using the actual HyperCore hybrid scoring (not simplified thresholds)
3. Calculating real confusion matrices from predictions vs outcomes

---

## ACTUAL RESULTS

### Screening Mode (risk >= 0.15, domains >= 1)

| System | Sensitivity | Specificity | PPV |
|--------|-------------|-------------|-----|
| **HyperCore** | **59.5%** | 81.7% | 45.5% |
| NEWS >= 5 | 45.2% | 85.4% | 44.2% |
| qSOFA >= 2 | 11.9% | 98.8% | 71.4% |
| MEWS >= 4 | 28.6% | 92.7% | 50.0% |

**HyperCore Confusion Matrix:**
- TP: 25, FP: 30, TN: 134, FN: 17

### Balanced Mode (risk >= 0.30, domains >= 2)

| System | Sensitivity | Specificity | PPV |
|--------|-------------|-------------|-----|
| **HyperCore** | 31.0% | **96.3%** | **68.4%** |
| NEWS >= 5 | 45.2% | 85.4% | 44.2% |
| qSOFA >= 2 | 11.9% | 98.8% | 71.4% |
| MEWS >= 4 | 28.6% | 92.7% | 50.0% |

**HyperCore Confusion Matrix:**
- TP: 13, FP: 6, TN: 158, FN: 29

### High Confidence Mode (risk >= 0.50, domains >= 3)

| System | Sensitivity | Specificity | PPV |
|--------|-------------|-------------|-----|
| **HyperCore** | 2.4% | 99.4% | 50.0% |
| NEWS >= 5 | 45.2% | 85.4% | 44.2% |
| qSOFA >= 2 | 11.9% | 98.8% | 71.4% |
| MEWS >= 4 | 28.6% | 92.7% | 50.0% |

**HyperCore Confusion Matrix:**
- TP: 1, FP: 1, TN: 163, FN: 41

---

## COMPARISON TO PREVIOUS CLAIMS

| Metric | Previous Claim | Actual Result | Status |
|--------|---------------|---------------|--------|
| Sensitivity | 78-90% | **59.5%** (screening) | OVERESTIMATED |
| Specificity | 100% | **81.7%** (screening) | INVALID (leakage) |
| NEWS sensitivity | 24% | **45.2%** | UNDERESTIMATED |
| "3.7x better than NEWS" | Yes | **1.3x** (screening) | OVERSTATED |

---

## WHAT THIS MEANS

### HyperCore's Actual Advantage (Screening Mode)
- **+14.3 percentage points** better sensitivity than NEWS (59.5% vs 45.2%)
- **1.3x** more events caught (not 3.7x as claimed)
- Trades some specificity for higher sensitivity

### Operating Mode Trade-offs
| Mode | Best For | Sensitivity | False Positives |
|------|----------|-------------|-----------------|
| Screening | Don't miss events | 59.5% | Higher (30 FP) |
| Balanced | Clinical use | 31.0% | Lower (6 FP) |
| High Confidence | Rule-in only | 2.4% | Minimal (1 FP) |

---

## WHY PREVIOUS RESULTS WERE WRONG

### 1. Outcome Leakage (100% Specificity)
The validation script sent the `outcome` column to the API, allowing HyperCore to "see" which patients had events. This caused perfect separation (all non-event patients scored exactly 0.000).

### 2. Simplified Threshold Logic (90% Sensitivity)
The "corrected" report used a simple ">20% biomarker rise" threshold locally, NOT the actual HyperCore hybrid scoring algorithm.

### 3. Inflated Claims (78% Sensitivity)
Various reports mixed methodologies and cherry-picked favorable numbers.

---

## HONEST ASSESSMENT

### HyperCore Does Work
- In screening mode, it genuinely outperforms NEWS on sensitivity (59.5% vs 45.2%)
- The multi-domain trajectory analysis catches deterioration that point-in-time scores miss
- The three operating modes provide clinical flexibility

### But Claims Were Overstated
- Not 3.7x better - more like 1.3x better on sensitivity
- Not 78% sensitivity - more like 59.5% in screening mode
- Previous 100% specificity was an artifact of leakage

### Clinical Utility
- **Screening mode** is best for hospitals that prioritize catching every deteriorating patient
- **Balanced mode** reduces alert fatigue but misses 69% of events
- **High confidence mode** is essentially unusable (2.4% sensitivity)

---

## RECOMMENDATIONS

1. **Use screening mode** if sensitivity is priority
2. **Acknowledge honest numbers** in any documentation or demos
3. **Don't claim 78% sensitivity** - the real number is ~60%
4. **The /compare endpoint** is now the authoritative source for validation metrics

---

## METHODOLOGY NOTE

The /compare endpoint:
- Sends patient trajectories WITHOUT outcome labels
- Calculates HyperCore risk scores using the production hybrid algorithm
- Applies mode-specific thresholds (risk + domain count)
- Calculates NEWS/qSOFA/MEWS using standard clinical formulas
- Computes confusion matrices and metrics from predictions vs outcomes

This is a proper leakage-free validation methodology.

---

*Report generated: 2026-03-31*
*Endpoint tested: /compare (local, confirmed working)*
