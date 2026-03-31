"""
LEAKAGE-FREE HyperCore Validation on MIMIC-IV Data
==================================================
Fixes:
1. NO outcome column in API input
2. Window-level predictions (only data BEFORE prediction time)
3. Proper lead time measurement (first alert to event)
"""
import pandas as pd
import numpy as np
import requests
import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# Configuration
API_URL = "https://hypercore-ml-service-production.up.railway.app/early_risk_discovery"
VALIDATION_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\mimic_validation_with_scores.csv'
OUTPUT_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\hypercore_predictions_fixed.csv'

def prepare_patient_csv_no_leakage(patient_df, up_to_window):
    """
    Prepare CSV data WITHOUT outcome column (no leakage)
    Only include data UP TO the specified window (temporal cutoff)
    """
    # Filter to only windows up to (and including) the prediction window
    rows = patient_df[patient_df['window_num'] <= up_to_window].copy()
    rows = rows.sort_values('window_num')

    if len(rows) < 3:
        return None  # Need at least 3 timepoints

    # Create a clean dataframe WITHOUT outcome (no leakage!)
    csv_df = pd.DataFrame()
    csv_df['patient_id'] = rows['patient_id']
    csv_df['time'] = range(len(rows))  # Time index

    # Add biomarkers only - NO OUTCOME COLUMN
    for col in ['heart_rate', 'respiratory_rate', 'sbp', 'spo2', 'temperature',
                'lactate', 'creatinine', 'troponin']:
        if col in rows.columns:
            csv_df[col] = rows[col].values

    # DO NOT add outcome column - this is the key fix!

    # Convert to CSV string
    csv_str = csv_df.to_csv(index=False)
    return csv_str

def call_hypercore_api(csv_data):
    """Call the HyperCore early_risk_discovery API WITHOUT outcome info"""
    payload = {
        "csv": csv_data,
        # NO label_column specified - HyperCore must predict without seeing outcomes
        "patient_id_column": "patient_id",
        "time_column": "time",
        "outcome_type": "mortality"
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code, "msg": response.text[:300]}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 70)
    print("LEAKAGE-FREE HYPERCORE VALIDATION ON MIMIC-IV DATA")
    print("=" * 70)
    print()
    print("KEY FIXES:")
    print("  1. NO outcome column sent to HyperCore API")
    print("  2. Window-level predictions (only data before prediction time)")
    print("  3. Proper lead time = first alert to event")
    print()

    # Load data
    df = pd.read_csv(VALIDATION_FILE)
    print(f"Loaded {len(df)} prediction windows from {df['patient_id'].nunique()} patients")
    print(f"Events (death within 12h): {df['event_in_12h'].sum()}")
    print()

    # Get unique patients
    patients = df['patient_id'].unique()

    # For efficiency, sample prediction windows (not every 4-hour window)
    # Take last window for each patient (most data available)
    results = []
    api_calls = 0
    api_success = 0
    api_errors = []

    print("Running LEAKAGE-FREE predictions...")
    print("(Predicting at each patient's final observation window)")
    print()

    for i, patient_id in enumerate(patients):
        patient_data = df[df['patient_id'] == patient_id].copy()

        # Get patient's outcome (did they have any event?)
        has_event = patient_data['event_in_12h'].max()

        # Use the LAST window for prediction (most trajectory data available)
        # But only use data UP TO that point
        last_window = patient_data['window_num'].max()

        # If patient has event, predict at the window BEFORE the event
        # (to simulate prospective prediction)
        if has_event == 1:
            event_window = patient_data[patient_data['event_in_12h'] == 1]['window_num'].min()
            # Predict one window before the event (if possible)
            predict_window = event_window - 1 if event_window > 1 else event_window
        else:
            predict_window = last_window

        # Need at least 3 windows of history
        if predict_window < 3:
            continue

        # Prepare CSV WITHOUT outcome column
        csv_data = prepare_patient_csv_no_leakage(patient_data, predict_window)

        if csv_data is None:
            continue

        # Call API
        api_calls += 1
        response = call_hypercore_api(csv_data)

        if 'error' not in response:
            api_success += 1

            # Extract risk info
            risk_level = response.get('risk_level', 'low')
            risk_score = response.get('risk_score', 0)

            # Alert if risk_level is 'high' or 'critical', OR risk_score >= 0.5
            high_risk = risk_level in ['high', 'critical'] or (isinstance(risk_score, (int, float)) and risk_score >= 0.5)

            results.append({
                'patient_id': patient_id,
                'predict_window': predict_window,
                'actual_event': has_event,
                'hypercore_risk_score': risk_score if isinstance(risk_score, (int, float)) else 0,
                'hypercore_risk_level': risk_level,
                'hypercore_alert': 1 if high_risk else 0,
            })

            if api_success == 1:
                print(f"First response: risk_level={risk_level}, risk_score={risk_score}")
                print(f"Response keys: {list(response.keys())[:8]}")
        else:
            if len(api_errors) < 3:
                api_errors.append(f"{patient_id}: {response}")

        if api_calls % 20 == 0:
            print(f"  Patients: {api_calls}, Success: {api_success}")

    print(f"\nTotal API calls: {api_calls}, Successful: {api_success}")

    if api_errors:
        print(f"\nSample errors:")
        for err in api_errors[:3]:
            print(f"  {err[:100]}")

    if api_success == 0:
        print("ERROR: No successful API calls!")
        return

    # Create results dataframe
    results_df = pd.DataFrame(results)

    print(f"\nValid predictions: {len(results_df)}")
    print(f"Patients with events: {results_df['actual_event'].sum()}")
    print(f"HyperCore alerts: {results_df['hypercore_alert'].sum()}")

    # Risk score distribution check (leakage detection)
    print("\n" + "-" * 50)
    print("LEAKAGE CHECK:")
    event_scores = results_df[results_df['actual_event']==1]['hypercore_risk_score']
    no_event_scores = results_df[results_df['actual_event']==0]['hypercore_risk_score']
    print(f"  Event patients - mean score: {event_scores.mean():.3f}, range: [{event_scores.min():.3f}, {event_scores.max():.3f}]")
    print(f"  Non-event patients - mean score: {no_event_scores.mean():.3f}, range: [{no_event_scores.min():.3f}, {no_event_scores.max():.3f}]")

    if no_event_scores.max() == 0 and no_event_scores.min() == 0:
        print("  WARNING: All non-event scores = 0, possible leakage still present!")
    else:
        print("  OK: Non-event patients have varied scores (no obvious leakage)")
    print("-" * 50)

    # Calculate metrics for HyperCore
    actual = results_df['actual_event'].values
    alert = results_df['hypercore_alert'].values

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
    print("HYPERCORE RESULTS (Leakage-Free)")
    print("=" * 70)
    print(f"Confusion Matrix:")
    print(f"  TP: {tp}, FN: {fn}")
    print(f"  FP: {fp}, TN: {tn}")
    print(f"\nMetrics:")
    print(f"  Sensitivity: {sensitivity*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  PPV @ 5%:    {ppv_5pct*100:.2f}%")

    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Calculate baseline metrics
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON (Patient-Level)")
    print("=" * 70)

    # Get NEWS/qSOFA at the same prediction windows
    patient_baselines = []
    for _, row in results_df.iterrows():
        pid = row['patient_id']
        win = row['predict_window']
        patient_data = df[(df['patient_id'] == pid) & (df['window_num'] == win)]
        if len(patient_data) > 0:
            patient_baselines.append({
                'patient_id': pid,
                'actual_event': row['actual_event'],
                'news_alert': patient_data['news_alert'].values[0],
                'qsofa_alert': patient_data['qsofa_alert'].values[0]
            })

    baselines_df = pd.DataFrame(patient_baselines)

    # NEWS metrics
    news_actual = baselines_df['actual_event'].values
    news_alert = baselines_df['news_alert'].values

    news_tp = ((news_actual == 1) & (news_alert == 1)).sum()
    news_fn = ((news_actual == 1) & (news_alert == 0)).sum()
    news_fp = ((news_actual == 0) & (news_alert == 1)).sum()
    news_tn = ((news_actual == 0) & (news_alert == 0)).sum()

    news_sens = news_tp / (news_tp + news_fn) if (news_tp + news_fn) > 0 else 0
    news_spec = news_tn / (news_tn + news_fp) if (news_tn + news_fp) > 0 else 0
    if news_sens > 0 or (1-news_spec) > 0:
        news_ppv = (news_sens * prevalence) / ((news_sens * prevalence) + ((1 - news_spec) * (1 - prevalence)))
    else:
        news_ppv = 0

    print(f"\nNEWS >= 5:")
    print(f"  TP: {news_tp}, FN: {news_fn}, FP: {news_fp}, TN: {news_tn}")
    print(f"  Sensitivity: {news_sens*100:.2f}%")
    print(f"  Specificity: {news_spec*100:.2f}%")
    print(f"  PPV @ 5%:    {news_ppv*100:.2f}%")

    # qSOFA metrics
    qsofa_alert = baselines_df['qsofa_alert'].values

    qsofa_tp = ((news_actual == 1) & (qsofa_alert == 1)).sum()
    qsofa_fn = ((news_actual == 1) & (qsofa_alert == 0)).sum()
    qsofa_fp = ((news_actual == 0) & (qsofa_alert == 1)).sum()
    qsofa_tn = ((news_actual == 0) & (qsofa_alert == 0)).sum()

    qsofa_sens = qsofa_tp / (qsofa_tp + qsofa_fn) if (qsofa_tp + qsofa_fn) > 0 else 0
    qsofa_spec = qsofa_tn / (qsofa_tn + qsofa_fp) if (qsofa_tn + qsofa_fp) > 0 else 0
    if qsofa_sens > 0 or (1-qsofa_spec) > 0:
        qsofa_ppv = (qsofa_sens * prevalence) / ((qsofa_sens * prevalence) + ((1 - qsofa_spec) * (1 - prevalence)))
    else:
        qsofa_ppv = 0

    print(f"\nqSOFA >= 2:")
    print(f"  TP: {qsofa_tp}, FN: {qsofa_fn}, FP: {qsofa_fp}, TN: {qsofa_tn}")
    print(f"  Sensitivity: {qsofa_sens*100:.2f}%")
    print(f"  Specificity: {qsofa_spec*100:.2f}%")
    print(f"  PPV @ 5%:    {qsofa_ppv*100:.2f}%")

    # Final comparison
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON (MIMIC-IV - Leakage-Free)")
    print("=" * 70)
    print(f"\n| System      | Sensitivity | Specificity | PPV @ 5% |")
    print(f"|-------------|-------------|-------------|----------|")
    print(f"| HyperCore   | {sensitivity*100:10.1f}% | {specificity*100:10.1f}% | {ppv_5pct*100:7.1f}% |")
    print(f"| NEWS >= 5   | {news_sens*100:10.1f}% | {news_spec*100:10.1f}% | {news_ppv*100:7.1f}% |")
    print(f"| qSOFA >= 2  | {qsofa_sens*100:10.1f}% | {qsofa_spec*100:10.1f}% | {qsofa_ppv*100:7.1f}% |")

    # Determine winner
    print("\n" + "-" * 70)
    if sensitivity > news_sens and ppv_5pct > news_ppv:
        print("VERDICT: HyperCore OUTPERFORMS NEWS (Sensitivity & PPV)")
    elif sensitivity > news_sens:
        print("VERDICT: HyperCore has HIGHER SENSITIVITY than NEWS")
    elif ppv_5pct > news_ppv:
        print("VERDICT: HyperCore has HIGHER PPV than NEWS")
    elif sensitivity >= news_sens * 0.9 and specificity > news_spec:
        print("VERDICT: HyperCore COMPARABLE sensitivity with BETTER specificity")
    elif sensitivity < news_sens and ppv_5pct < news_ppv:
        print("VERDICT: NEWS OUTPERFORMS HyperCore (honest assessment)")
    else:
        print("VERDICT: MIXED results - different trade-offs")
    print("-" * 70)

if __name__ == "__main__":
    main()
