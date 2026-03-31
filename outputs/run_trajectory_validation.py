"""
PROSPECTIVE TRAJECTORY VALIDATION
=================================
Apply HyperCore's trajectory detection logic (>20% biomarker rise) prospectively.
No API calls needed - we apply the same thresholds locally.

This validates the CONCEPT of trajectory-based early warning, not the API.
"""
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Configuration
VALIDATION_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\mimic_validation_with_scores.csv'
OUTPUT_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\trajectory_predictions.csv'

# HyperCore's trajectory threshold (from main.py line 4877)
RISING_THRESHOLD = 20  # >20% increase = rising pattern

# Biomarkers to analyze (same as HyperCore uses)
BIOMARKER_COLS = ['heart_rate', 'respiratory_rate', 'sbp', 'spo2', 'lactate', 'creatinine', 'troponin']

# Critical thresholds for concerning rises (clinical relevance)
CRITICAL_THRESHOLDS = {
    'lactate': {'baseline': 2.0, 'rise_pct': 20},      # Lactate >2 or rising >20%
    'creatinine': {'baseline': 1.5, 'rise_pct': 25},   # Creatinine >1.5 or rising >25%
    'troponin': {'baseline': 0.04, 'rise_pct': 50},    # Troponin elevated or rising
    'heart_rate': {'baseline': 100, 'rise_pct': 20},   # HR >100 or rising >20%
    'respiratory_rate': {'baseline': 22, 'rise_pct': 25},  # RR >22 or rising
}

def detect_trajectory_risk(patient_data, up_to_window):
    """
    Apply HyperCore's trajectory detection logic:
    - Flag patient as high risk if ANY biomarker shows >20% rise
    - Also flag if critical thresholds exceeded

    Returns: (alert, risk_score, signals)
    """
    # Get data up to prediction window
    data = patient_data[patient_data['window_num'] <= up_to_window].sort_values('window_num')

    if len(data) < 2:
        return 0, 0.0, []

    signals = []
    risk_score = 0.0

    for biomarker in BIOMARKER_COLS:
        if biomarker not in data.columns:
            continue

        values = data[biomarker].dropna().values
        if len(values) < 2:
            continue

        first_val = values[0]
        last_val = values[-1]

        if first_val == 0:
            first_val = 0.001  # Avoid division by zero

        pct_change = ((last_val - first_val) / abs(first_val)) * 100

        # Check for rising pattern (HyperCore threshold)
        if pct_change > RISING_THRESHOLD:
            signals.append({
                'biomarker': biomarker,
                'pattern': 'rising',
                'pct_change': round(pct_change, 1),
                'first': round(first_val, 2),
                'last': round(last_val, 2)
            })
            risk_score += 0.2  # Add to risk score

        # Check critical thresholds
        if biomarker in CRITICAL_THRESHOLDS:
            thresh = CRITICAL_THRESHOLDS[biomarker]
            if last_val > thresh['baseline']:
                signals.append({
                    'biomarker': biomarker,
                    'pattern': 'elevated',
                    'value': round(last_val, 2),
                    'threshold': thresh['baseline']
                })
                risk_score += 0.15
            if pct_change > thresh['rise_pct']:
                risk_score += 0.1  # Additional risk for concerning rise

    # Cap risk score at 1.0
    risk_score = min(risk_score, 1.0)

    # Alert if risk score >= 0.3 (at least 1-2 concerning signals)
    alert = 1 if risk_score >= 0.3 else 0

    return alert, risk_score, signals

def main():
    print("=" * 70)
    print("PROSPECTIVE TRAJECTORY VALIDATION (HyperCore Logic)")
    print("=" * 70)
    print()
    print("METHODOLOGY:")
    print("  - Apply HyperCore's >20% rise detection prospectively")
    print("  - Also check critical biomarker thresholds")
    print("  - NO outcome data used in prediction")
    print("  - Alert if risk_score >= 0.3")
    print()

    # Load data
    df = pd.read_csv(VALIDATION_FILE)
    print(f"Loaded {len(df)} prediction windows from {df['patient_id'].nunique()} patients")
    print(f"Events (death within 12h): {df['event_in_12h'].sum()}")
    print()

    # Get unique patients
    patients = df['patient_id'].unique()

    results = []

    print("Running trajectory analysis...")

    for i, patient_id in enumerate(patients):
        patient_data = df[df['patient_id'] == patient_id].copy()

        # Get patient's outcome
        has_event = patient_data['event_in_12h'].max()

        # Determine prediction window (one before event, or last window)
        if has_event == 1:
            event_window = patient_data[patient_data['event_in_12h'] == 1]['window_num'].min()
            predict_window = event_window - 1 if event_window > 1 else event_window
        else:
            predict_window = patient_data['window_num'].max()

        # Need at least 3 windows for trajectory
        if predict_window < 3:
            continue

        # Apply trajectory detection
        alert, risk_score, signals = detect_trajectory_risk(patient_data, predict_window)

        # Get NEWS/qSOFA at prediction window
        window_data = patient_data[patient_data['window_num'] == predict_window]
        news_alert = window_data['news_alert'].values[0] if len(window_data) > 0 else 0
        qsofa_alert = window_data['qsofa_alert'].values[0] if len(window_data) > 0 else 0

        results.append({
            'patient_id': patient_id,
            'predict_window': predict_window,
            'actual_event': has_event,
            'trajectory_risk_score': risk_score,
            'trajectory_alert': alert,
            'num_signals': len(signals),
            'news_alert': news_alert,
            'qsofa_alert': qsofa_alert
        })

        if i % 50 == 0:
            print(f"  Processed: {i}/{len(patients)} patients")

    print(f"\nProcessed: {len(results)} patients")

    # Create results dataframe
    results_df = pd.DataFrame(results)

    print(f"Patients with events: {results_df['actual_event'].sum()}")
    print(f"Trajectory alerts: {results_df['trajectory_alert'].sum()}")
    print()

    # Leakage check
    print("-" * 50)
    print("LEAKAGE CHECK:")
    event_scores = results_df[results_df['actual_event']==1]['trajectory_risk_score']
    no_event_scores = results_df[results_df['actual_event']==0]['trajectory_risk_score']
    print(f"  Event patients - mean: {event_scores.mean():.3f}, range: [{event_scores.min():.3f}, {event_scores.max():.3f}]")
    print(f"  Non-event patients - mean: {no_event_scores.mean():.3f}, range: [{no_event_scores.min():.3f}, {no_event_scores.max():.3f}]")
    print("-" * 50)

    # Calculate metrics for Trajectory Detection
    actual = results_df['actual_event'].values
    alert = results_df['trajectory_alert'].values

    tp = ((actual == 1) & (alert == 1)).sum()
    tn = ((actual == 0) & (alert == 0)).sum()
    fp = ((actual == 0) & (alert == 1)).sum()
    fn = ((actual == 1) & (alert == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # PPV at 5% prevalence
    prevalence = 0.05
    if sensitivity > 0 or (1-specificity) > 0:
        ppv_5pct = (sensitivity * prevalence) / ((sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)))
    else:
        ppv_5pct = 0

    print("\n" + "=" * 70)
    print("TRAJECTORY DETECTION RESULTS (HyperCore Logic)")
    print("=" * 70)
    print(f"Confusion Matrix:")
    print(f"  TP: {tp}, FN: {fn}")
    print(f"  FP: {fp}, TN: {tn}")
    print(f"\nMetrics:")
    print(f"  Sensitivity: {sensitivity*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  PPV @ 5%:    {ppv_5pct*100:.2f}%")

    # Calculate NEWS metrics
    news_alert_arr = results_df['news_alert'].values

    news_tp = ((actual == 1) & (news_alert_arr == 1)).sum()
    news_fn = ((actual == 1) & (news_alert_arr == 0)).sum()
    news_fp = ((actual == 0) & (news_alert_arr == 1)).sum()
    news_tn = ((actual == 0) & (news_alert_arr == 0)).sum()

    news_sens = news_tp / (news_tp + news_fn) if (news_tp + news_fn) > 0 else 0
    news_spec = news_tn / (news_tn + news_fp) if (news_tn + news_fp) > 0 else 0
    if news_sens > 0 or (1-news_spec) > 0:
        news_ppv = (news_sens * prevalence) / ((news_sens * prevalence) + ((1 - news_spec) * (1 - prevalence)))
    else:
        news_ppv = 0

    print("\n" + "=" * 70)
    print("NEWS >= 5 RESULTS")
    print("=" * 70)
    print(f"Confusion Matrix:")
    print(f"  TP: {news_tp}, FN: {news_fn}")
    print(f"  FP: {news_fp}, TN: {news_tn}")
    print(f"\nMetrics:")
    print(f"  Sensitivity: {news_sens*100:.2f}%")
    print(f"  Specificity: {news_spec*100:.2f}%")
    print(f"  PPV @ 5%:    {news_ppv*100:.2f}%")

    # Calculate qSOFA metrics
    qsofa_alert_arr = results_df['qsofa_alert'].values

    qsofa_tp = ((actual == 1) & (qsofa_alert_arr == 1)).sum()
    qsofa_fn = ((actual == 1) & (qsofa_alert_arr == 0)).sum()
    qsofa_fp = ((actual == 0) & (qsofa_alert_arr == 1)).sum()
    qsofa_tn = ((actual == 0) & (qsofa_alert_arr == 0)).sum()

    qsofa_sens = qsofa_tp / (qsofa_tp + qsofa_fn) if (qsofa_tp + qsofa_fn) > 0 else 0
    qsofa_spec = qsofa_tn / (qsofa_tn + qsofa_fp) if (qsofa_tn + qsofa_fp) > 0 else 0
    if qsofa_sens > 0 or (1-qsofa_spec) > 0:
        qsofa_ppv = (qsofa_sens * prevalence) / ((qsofa_sens * prevalence) + ((1 - qsofa_spec) * (1 - prevalence)))
    else:
        qsofa_ppv = 0

    print("\n" + "=" * 70)
    print("qSOFA >= 2 RESULTS")
    print("=" * 70)
    print(f"Confusion Matrix:")
    print(f"  TP: {qsofa_tp}, FN: {qsofa_fn}")
    print(f"  FP: {qsofa_fp}, TN: {qsofa_tn}")
    print(f"\nMetrics:")
    print(f"  Sensitivity: {qsofa_sens*100:.2f}%")
    print(f"  Specificity: {qsofa_spec*100:.2f}%")
    print(f"  PPV @ 5%:    {qsofa_ppv*100:.2f}%")

    # Combined system (trajectory OR NEWS)
    combined_alert = ((results_df['trajectory_alert'] == 1) | (results_df['news_alert'] == 1)).astype(int).values

    comb_tp = ((actual == 1) & (combined_alert == 1)).sum()
    comb_fn = ((actual == 1) & (combined_alert == 0)).sum()
    comb_fp = ((actual == 0) & (combined_alert == 1)).sum()
    comb_tn = ((actual == 0) & (combined_alert == 0)).sum()

    comb_sens = comb_tp / (comb_tp + comb_fn) if (comb_tp + comb_fn) > 0 else 0
    comb_spec = comb_tn / (comb_tn + comb_fp) if (comb_tn + comb_fp) > 0 else 0
    if comb_sens > 0 or (1-comb_spec) > 0:
        comb_ppv = (comb_sens * prevalence) / ((comb_sens * prevalence) + ((1 - comb_spec) * (1 - prevalence)))
    else:
        comb_ppv = 0

    # Final comparison
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (Leakage-Free)")
    print("=" * 70)
    print(f"\n| System               | Sensitivity | Specificity | PPV @ 5% |")
    print(f"|----------------------|-------------|-------------|----------|")
    print(f"| Trajectory (HyperCore)| {sensitivity*100:10.1f}% | {specificity*100:10.1f}% | {ppv_5pct*100:7.1f}% |")
    print(f"| NEWS >= 5            | {news_sens*100:10.1f}% | {news_spec*100:10.1f}% | {news_ppv*100:7.1f}% |")
    print(f"| qSOFA >= 2           | {qsofa_sens*100:10.1f}% | {qsofa_spec*100:10.1f}% | {qsofa_ppv*100:7.1f}% |")
    print(f"| **Trajectory + NEWS**| {comb_sens*100:10.1f}% | {comb_spec*100:10.1f}% | {comb_ppv*100:7.1f}% |")

    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Verdict
    print("\n" + "-" * 70)
    if sensitivity > news_sens and ppv_5pct >= news_ppv:
        print("VERDICT: Trajectory Detection OUTPERFORMS NEWS")
    elif sensitivity > news_sens:
        print("VERDICT: Trajectory has HIGHER SENSITIVITY than NEWS")
        print(f"  (Sensitivity: {sensitivity*100:.1f}% vs {news_sens*100:.1f}%)")
    elif ppv_5pct > news_ppv:
        print("VERDICT: Trajectory has HIGHER PPV than NEWS")
    elif abs(sensitivity - news_sens) < 0.05 and specificity > news_spec:
        print("VERDICT: Trajectory COMPARABLE sensitivity with BETTER specificity")
    elif sensitivity < news_sens and ppv_5pct < news_ppv:
        print("VERDICT: NEWS OUTPERFORMS Trajectory Detection")
        print("  (This is an honest assessment)")
    else:
        print("VERDICT: MIXED results")

    # Value of combination
    if comb_sens > news_sens and comb_sens > sensitivity:
        print(f"\nCOMBINED VALUE: Trajectory + NEWS catches {comb_sens*100:.1f}% of events")
        print(f"  vs NEWS alone: {news_sens*100:.1f}%")
        print(f"  vs Trajectory alone: {sensitivity*100:.1f}%")

    print("-" * 70)

if __name__ == "__main__":
    main()
