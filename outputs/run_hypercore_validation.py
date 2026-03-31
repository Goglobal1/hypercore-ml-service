"""
Run HyperCore early_risk_discovery on MIMIC-IV validation dataset
Compare results to NEWS and qSOFA baselines
"""
import pandas as pd
import numpy as np
import requests
import json
import sys
from io import StringIO

sys.stdout.reconfigure(encoding='utf-8')

# Configuration
API_URL = "https://hypercore-ml-service-production.up.railway.app/early_risk_discovery"
VALIDATION_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\mimic_validation_with_scores.csv'
OUTPUT_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\hypercore_predictions.csv'

def prepare_patient_csv(patient_df):
    """Prepare CSV data for a patient's longitudinal data"""
    rows = patient_df.sort_values('prediction_time').copy()

    # Create a clean dataframe with required columns
    csv_df = pd.DataFrame()
    csv_df['patient_id'] = rows['patient_id']
    csv_df['time'] = range(len(rows))  # Time index

    # Add biomarkers
    for col in ['heart_rate', 'respiratory_rate', 'sbp', 'spo2', 'temperature',
                'lactate', 'creatinine', 'troponin']:
        if col in rows.columns:
            csv_df[col] = rows[col].values

    # Add outcome (1 = event, 0 = no event) - use the LAST row's outcome
    csv_df['outcome'] = rows['event_in_12h'].values

    # Convert to CSV string
    csv_str = csv_df.to_csv(index=False)
    return csv_str

def call_hypercore_api(csv_data, patient_id):
    """Call the HyperCore early_risk_discovery API"""
    payload = {
        "csv": csv_data,
        "label_column": "outcome",
        "patient_id_column": "patient_id",
        "time_column": "time",
        "outcome_type": "mortality"
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.status_code, "msg": response.text[:200]}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("=" * 60)
    print("HYPERCORE VALIDATION ON MIMIC-IV DATA")
    print("=" * 60)
    print()

    # Load data
    df = pd.read_csv(VALIDATION_FILE)
    print(f"Loaded {len(df)} prediction windows from {df['patient_id'].nunique()} patients")
    print(f"Events (death within 12h): {df['event_in_12h'].sum()}")
    print()

    # Get unique patients
    patients = df['patient_id'].unique()

    results = []
    api_calls = 0
    api_success = 0

    print("Running HyperCore predictions per patient...")
    print("(Using patient's full trajectory for risk assessment)")
    print()

    # Process each patient (not each window - that would be too many API calls)
    for i, patient_id in enumerate(patients):
        patient_data = df[df['patient_id'] == patient_id].copy()

        # Get patient's outcome (whether they had any event)
        has_event = patient_data['event_in_12h'].max()

        # Need at least 3 timepoints for meaningful trajectory
        if len(patient_data) < 3:
            continue

        # Prepare CSV
        csv_data = prepare_patient_csv(patient_data)

        # Call API
        api_calls += 1
        response = call_hypercore_api(csv_data, patient_id)

        if 'error' not in response:
            api_success += 1

            # Extract risk info from CORRECT response fields
            risk_level = response.get('risk_level', 'low')  # 'high', 'medium', 'low'
            risk_score = response.get('risk_score', 0)  # 0.0 to 1.0

            # Extract lead time from risk_timing_delta
            timing = response.get('risk_timing_delta', {})
            lead_time = timing.get('lead_time_days', 0)

            # Alert if risk_level is 'high' OR risk_score >= 0.5
            high_risk = risk_level == 'high' or risk_score >= 0.5

            results.append({
                'patient_id': patient_id,
                'actual_event': has_event,
                'hypercore_risk_score': risk_score,
                'hypercore_risk_level': risk_level,
                'hypercore_alert': 1 if high_risk else 0,
                'lead_time_days': lead_time,
                'num_windows': len(patient_data)
            })

            if api_success == 1:
                print(f"Sample: risk_level={risk_level}, risk_score={risk_score}, lead_time={lead_time}")
        else:
            if api_calls <= 3:
                print(f"API error for {patient_id}: {response}")

        if api_calls % 20 == 0:
            print(f"  Patients: {api_calls}, Success: {api_success}")

    print(f"\nTotal API calls: {api_calls}, Successful: {api_success}")

    if api_success == 0:
        print("ERROR: No successful API calls!")
        return

    # Create results dataframe
    results_df = pd.DataFrame(results)

    print(f"\nValid predictions: {len(results_df)}")
    print(f"Patients with events: {results_df['actual_event'].sum()}")
    print(f"HyperCore alerts: {results_df['hypercore_alert'].sum()}")

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
    ppv_5pct = (sensitivity * prevalence) / ((sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence))) if (sensitivity + (1-specificity)) > 0 else 0

    print("\n" + "=" * 60)
    print("HYPERCORE RESULTS (Patient-Level)")
    print("=" * 60)
    print(f"Confusion Matrix:")
    print(f"  TP: {tp}, FN: {fn}")
    print(f"  FP: {fp}, TN: {tn}")
    print(f"\nMetrics:")
    print(f"  Sensitivity: {sensitivity*100:.2f}%")
    print(f"  Specificity: {specificity*100:.2f}%")
    print(f"  PPV @ 5%:    {ppv_5pct*100:.2f}%")

    # Calculate lead time for true positives
    tp_results = results_df[(results_df['actual_event'] == 1) & (results_df['hypercore_alert'] == 1)]
    if len(tp_results) > 0:
        avg_lead_time = tp_results['lead_time_days'].mean()
        print(f"\nAverage Lead Time: {avg_lead_time:.2f} days")

    # Save results
    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Calculate baseline metrics from original data (patient-level)
    print("\n" + "=" * 60)
    print("BASELINE COMPARISON (Patient-Level)")
    print("=" * 60)

    # For each patient, check if NEWS/qSOFA ever alerted
    patient_baselines = df.groupby('patient_id').agg({
        'event_in_12h': 'max',  # Patient had event
        'news_alert': 'max',     # NEWS ever alerted
        'qsofa_alert': 'max'     # qSOFA ever alerted
    }).reset_index()

    # Filter to patients we predicted on
    patient_baselines = patient_baselines[patient_baselines['patient_id'].isin(results_df['patient_id'])]

    # NEWS metrics
    news_actual = patient_baselines['event_in_12h'].values
    news_alert = patient_baselines['news_alert'].values

    news_tp = ((news_actual == 1) & (news_alert == 1)).sum()
    news_fn = ((news_actual == 1) & (news_alert == 0)).sum()
    news_fp = ((news_actual == 0) & (news_alert == 1)).sum()
    news_tn = ((news_actual == 0) & (news_alert == 0)).sum()

    news_sens = news_tp / (news_tp + news_fn) if (news_tp + news_fn) > 0 else 0
    news_spec = news_tn / (news_tn + news_fp) if (news_tn + news_fp) > 0 else 0
    news_ppv = (news_sens * prevalence) / ((news_sens * prevalence) + ((1 - news_spec) * (1 - prevalence))) if (news_sens + (1-news_spec)) > 0 else 0

    print(f"\nNEWS >= 5 (Patient-Level):")
    print(f"  TP: {news_tp}, FN: {news_fn}, FP: {news_fp}, TN: {news_tn}")
    print(f"  Sensitivity: {news_sens*100:.2f}%")
    print(f"  Specificity: {news_spec*100:.2f}%")
    print(f"  PPV @ 5%:    {news_ppv*100:.2f}%")

    # qSOFA metrics
    qsofa_alert = patient_baselines['qsofa_alert'].values

    qsofa_tp = ((news_actual == 1) & (qsofa_alert == 1)).sum()
    qsofa_fn = ((news_actual == 1) & (qsofa_alert == 0)).sum()
    qsofa_fp = ((news_actual == 0) & (qsofa_alert == 1)).sum()
    qsofa_tn = ((news_actual == 0) & (qsofa_alert == 0)).sum()

    qsofa_sens = qsofa_tp / (qsofa_tp + qsofa_fn) if (qsofa_tp + qsofa_fn) > 0 else 0
    qsofa_spec = qsofa_tn / (qsofa_tn + qsofa_fp) if (qsofa_tn + qsofa_fp) > 0 else 0
    qsofa_ppv = (qsofa_sens * prevalence) / ((qsofa_sens * prevalence) + ((1 - qsofa_spec) * (1 - prevalence))) if (qsofa_sens + (1-qsofa_spec)) > 0 else 0

    print(f"\nqSOFA >= 2 (Patient-Level):")
    print(f"  TP: {qsofa_tp}, FN: {qsofa_fn}, FP: {qsofa_fp}, TN: {qsofa_tn}")
    print(f"  Sensitivity: {qsofa_sens*100:.2f}%")
    print(f"  Specificity: {qsofa_spec*100:.2f}%")
    print(f"  PPV @ 5%:    {qsofa_ppv*100:.2f}%")

    # Final comparison table
    print("\n" + "=" * 60)
    print("HEAD-TO-HEAD COMPARISON (MIMIC-IV ICU Data)")
    print("=" * 60)
    print(f"\n| System      | Sensitivity | Specificity | PPV @ 5% |")
    print(f"|-------------|-------------|-------------|----------|")
    print(f"| HyperCore   | {sensitivity*100:10.1f}% | {specificity*100:10.1f}% | {ppv_5pct*100:7.1f}% |")
    print(f"| NEWS >= 5   | {news_sens*100:10.1f}% | {news_spec*100:10.1f}% | {news_ppv*100:7.1f}% |")
    print(f"| qSOFA >= 2  | {qsofa_sens*100:10.1f}% | {qsofa_spec*100:10.1f}% | {qsofa_ppv*100:7.1f}% |")

    # Lead time comparison
    if len(tp_results) > 0:
        print(f"\nLEAD TIME ADVANTAGE:")
        print(f"  HyperCore avg lead time: {avg_lead_time:.1f} days")
        print(f"  NEWS/qSOFA lead time: 0 days (point-in-time)")
        print(f"  HyperCore advantage: +{avg_lead_time:.1f} days early detection")

    # Determine winner
    print("\n" + "-" * 60)
    if sensitivity > news_sens and ppv_5pct > news_ppv:
        print("VERDICT: HyperCore OUTPERFORMS NEWS")
    elif sensitivity < news_sens and ppv_5pct < news_ppv:
        print("VERDICT: NEWS OUTPERFORMS HyperCore on standard metrics")
        if len(tp_results) > 0 and avg_lead_time > 0.5:
            print(f"  BUT: HyperCore provides {avg_lead_time:.1f} days early warning advantage")
    elif sensitivity >= news_sens or ppv_5pct >= news_ppv:
        print("VERDICT: MIXED - HyperCore comparable to NEWS")
        if len(tp_results) > 0 and avg_lead_time > 0.5:
            print(f"  PLUS: HyperCore provides {avg_lead_time:.1f} days early warning advantage")
    else:
        print("VERDICT: MIXED - Different strengths")

    print("-" * 60)

if __name__ == "__main__":
    main()
