"""
MULTI-DOMAIN CONVERGENCE VALIDATION
====================================
HyperCore's true value: Cross-domain synthesis to filter false positives.

Key insight: Single-domain trajectory changes are often noise.
Multi-domain convergence indicates true systemic deterioration.
"""
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

# Configuration
VALIDATION_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\mimic_validation_with_scores.csv'
OUTPUT_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\multidomain_predictions.csv'

# =============================================================================
# STEP 1: DEFINE BIOMARKER DOMAINS
# =============================================================================
BIOMARKER_DOMAINS = {
    "cardiac": ["troponin", "bnp", "ck_mb", "myoglobin"],
    "renal": ["creatinine", "bun", "gfr", "cystatin_c"],
    "inflammatory": ["crp", "lactate", "procalcitonin", "il_6", "ferritin"],
    "hepatic": ["alt", "ast", "bilirubin", "albumin", "inr"],
    "hematologic": ["wbc", "platelets", "hemoglobin", "d_dimer"],
    "metabolic": ["glucose", "sodium", "potassium", "bicarbonate"],
    "hemodynamic": ["sbp", "dbp", "map", "heart_rate"],
    "respiratory": ["spo2", "respiratory_rate", "pao2", "fio2"]
}

# Map dataset columns to domain biomarkers
COLUMN_TO_BIOMARKER = {
    'heart_rate': 'heart_rate',
    'respiratory_rate': 'respiratory_rate',
    'sbp': 'sbp',
    'dbp': 'dbp',
    'spo2': 'spo2',
    'temperature': 'temperature',
    'lactate': 'lactate',
    'creatinine': 'creatinine',
    'troponin': 'troponin',
    'wbc': 'wbc',
    'crp': 'crp'
}

# Thresholds for concerning changes
RISING_THRESHOLD = 0.20  # 20% increase
FALLING_THRESHOLD = -0.15  # 15% decrease (for protective markers like SpO2)

# Markers where FALLING is concerning
FALLING_CONCERNING = ['spo2', 'sbp', 'dbp', 'map', 'platelets', 'hemoglobin', 'albumin', 'gfr']

# =============================================================================
# STEP 2: DOMAIN-LEVEL ANALYSIS
# =============================================================================
def analyze_domains(patient_data, up_to_window):
    """
    Analyze each domain independently, then check for convergence.
    """
    # Get data up to prediction window
    data = patient_data[patient_data['window_num'] <= up_to_window].sort_values('window_num')

    if len(data) < 2:
        return {}

    domain_alerts = {}

    for domain, biomarkers in BIOMARKER_DOMAINS.items():
        domain_signals = []
        biomarkers_checked = 0

        for biomarker in biomarkers:
            # Check if this biomarker exists in our data (via column mapping)
            col_name = None
            for col, mapped in COLUMN_TO_BIOMARKER.items():
                if mapped == biomarker and col in data.columns:
                    col_name = col
                    break

            if col_name is None:
                continue

            values = data[col_name].dropna().values
            if len(values) < 2:
                continue

            biomarkers_checked += 1

            # Calculate trajectory using first half vs second half
            mid = len(values) // 2
            if mid == 0:
                mid = 1

            baseline = np.mean(values[:mid])
            current = np.mean(values[mid:])

            if baseline == 0:
                baseline = 0.001

            pct_change = (current - baseline) / abs(baseline)

            # Check if concerning
            is_concerning = False
            direction = None

            if biomarker in FALLING_CONCERNING:
                # Falling is concerning
                if pct_change < FALLING_THRESHOLD:
                    is_concerning = True
                    direction = "falling"
            else:
                # Rising is concerning
                if pct_change > RISING_THRESHOLD:
                    is_concerning = True
                    direction = "rising"

            if is_concerning:
                domain_signals.append({
                    "biomarker": biomarker,
                    "change_pct": round(pct_change * 100, 1),
                    "direction": direction,
                    "baseline": round(baseline, 2),
                    "current": round(current, 2)
                })

        # Domain alerts if ANY biomarker shows trajectory
        if domain_signals:
            domain_alerts[domain] = {
                "alert": True,
                "signals": domain_signals,
                "num_signals": len(domain_signals),
                "biomarkers_available": biomarkers_checked,
                "confidence": len(domain_signals) / max(biomarkers_checked, 1)
            }

    return domain_alerts

# =============================================================================
# STEP 3: CONVERGENCE SCORING
# =============================================================================
def calculate_convergence_score(domain_alerts):
    """
    Score based on how many domains are converging.
    More domains = higher confidence = fewer false positives.
    """
    alerting_domains = [d for d, info in domain_alerts.items() if info.get("alert")]
    num_domains = len(alerting_domains)

    # Calculate total signals across domains
    total_signals = sum(info.get("num_signals", 0) for info in domain_alerts.values())

    if num_domains >= 4:
        risk_level = "critical"
        confidence = "very_high"
        fp_risk = "very_low"
    elif num_domains == 3:
        risk_level = "high"
        confidence = "high"
        fp_risk = "low"
    elif num_domains == 2:
        risk_level = "moderate"
        confidence = "medium"
        fp_risk = "moderate"
    elif num_domains == 1:
        risk_level = "watch"
        confidence = "low"
        fp_risk = "high"
    else:
        risk_level = "stable"
        confidence = "high"
        fp_risk = "n/a"

    return {
        "num_domains": num_domains,
        "domains_involved": alerting_domains,
        "total_signals": total_signals,
        "risk_level": risk_level,
        "confidence": confidence,
        "false_positive_risk": fp_risk
    }

# =============================================================================
# STEP 4: ALERT DECISION
# =============================================================================
def should_alert(domain_alerts, min_domains=2):
    """
    Only alert if MULTIPLE domains show convergent deterioration.
    """
    convergence = calculate_convergence_score(domain_alerts)
    return convergence["num_domains"] >= min_domains, convergence

# =============================================================================
# MAIN VALIDATION
# =============================================================================
def run_validation(df, min_domains_threshold):
    """Run validation with specified minimum domains for alert."""
    patients = df['patient_id'].unique()
    results = []

    for patient_id in patients:
        patient_data = df[df['patient_id'] == patient_id].copy()

        # Get patient's outcome
        has_event = patient_data['event_in_12h'].max()

        # Determine prediction window
        if has_event == 1:
            event_window = patient_data[patient_data['event_in_12h'] == 1]['window_num'].min()
            predict_window = event_window - 1 if event_window > 1 else event_window
        else:
            predict_window = patient_data['window_num'].max()

        if predict_window < 3:
            continue

        # Analyze domains
        domain_alerts = analyze_domains(patient_data, predict_window)
        alert, convergence = should_alert(domain_alerts, min_domains=min_domains_threshold)

        # Get NEWS/qSOFA at prediction window
        window_data = patient_data[patient_data['window_num'] == predict_window]
        news_alert = window_data['news_alert'].values[0] if len(window_data) > 0 else 0
        qsofa_alert = window_data['qsofa_alert'].values[0] if len(window_data) > 0 else 0

        results.append({
            'patient_id': patient_id,
            'actual_event': has_event,
            'multidomain_alert': 1 if alert else 0,
            'num_domains': convergence['num_domains'],
            'domains_involved': ','.join(convergence['domains_involved']),
            'risk_level': convergence['risk_level'],
            'news_alert': news_alert,
            'qsofa_alert': qsofa_alert
        })

    return pd.DataFrame(results)

def calculate_metrics(results_df, alert_col):
    """Calculate performance metrics for a given alert column."""
    actual = results_df['actual_event'].values
    alert = results_df[alert_col].values

    tp = ((actual == 1) & (alert == 1)).sum()
    tn = ((actual == 0) & (alert == 0)).sum()
    fp = ((actual == 0) & (alert == 1)).sum()
    fn = ((actual == 1) & (alert == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    prevalence = 0.05
    if sensitivity > 0 or (1-specificity) > 0:
        ppv = (sensitivity * prevalence) / ((sensitivity * prevalence) + ((1 - specificity) * (1 - prevalence)))
    else:
        ppv = 0

    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv_5pct': ppv,
        'fp_rate': fp_rate
    }

def main():
    print("=" * 70)
    print("MULTI-DOMAIN CONVERGENCE VALIDATION")
    print("=" * 70)
    print()
    print("HYPOTHESIS: Requiring multiple domains to show deterioration")
    print("            will eliminate false positives while catching true events.")
    print()

    # Load data
    df = pd.read_csv(VALIDATION_FILE)
    print(f"Loaded {len(df)} windows from {df['patient_id'].nunique()} patients")
    print(f"Events: {df['event_in_12h'].sum()}")
    print()

    # Test different domain thresholds
    thresholds_to_test = [1, 2, 3, 4]
    all_results = {}

    print("Testing domain thresholds...")
    print("-" * 50)

    for min_domains in thresholds_to_test:
        results_df = run_validation(df, min_domains)
        metrics = calculate_metrics(results_df, 'multidomain_alert')
        all_results[min_domains] = {
            'results_df': results_df,
            'metrics': metrics
        }
        print(f"  {min_domains}+ domains: Sens={metrics['sensitivity']*100:.1f}%, "
              f"Spec={metrics['specificity']*100:.1f}%, "
              f"PPV={metrics['ppv_5pct']*100:.1f}%, "
              f"Alerts={results_df['multidomain_alert'].sum()}")

    # Calculate NEWS metrics (using 1-domain results df for consistent patient set)
    news_metrics = calculate_metrics(all_results[1]['results_df'], 'news_alert')
    qsofa_metrics = calculate_metrics(all_results[1]['results_df'], 'qsofa_alert')

    print()
    print(f"  NEWS >= 5:  Sens={news_metrics['sensitivity']*100:.1f}%, "
          f"Spec={news_metrics['specificity']*100:.1f}%, "
          f"PPV={news_metrics['ppv_5pct']*100:.1f}%")
    print(f"  qSOFA >= 2: Sens={qsofa_metrics['sensitivity']*100:.1f}%, "
          f"Spec={qsofa_metrics['specificity']*100:.1f}%, "
          f"PPV={qsofa_metrics['ppv_5pct']*100:.1f}%")

    # ==========================================================================
    # COMPREHENSIVE COMPARISON TABLE
    # ==========================================================================
    print()
    print("=" * 70)
    print("MULTI-DOMAIN THRESHOLD COMPARISON")
    print("=" * 70)
    print()
    print("| Threshold        | Sensitivity | Specificity | PPV @ 5% | FP Rate | Alerts |")
    print("|------------------|-------------|-------------|----------|---------|--------|")

    for min_domains in thresholds_to_test:
        m = all_results[min_domains]['metrics']
        alerts = all_results[min_domains]['results_df']['multidomain_alert'].sum()
        print(f"| {min_domains}+ domains        | {m['sensitivity']*100:10.1f}% | {m['specificity']*100:10.1f}% | {m['ppv_5pct']*100:7.1f}% | {m['fp_rate']*100:6.1f}% | {alerts:6} |")

    print("|------------------|-------------|-------------|----------|---------|--------|")
    alerts_news = all_results[1]['results_df']['news_alert'].sum()
    alerts_qsofa = all_results[1]['results_df']['qsofa_alert'].sum()
    print(f"| NEWS >= 5        | {news_metrics['sensitivity']*100:10.1f}% | {news_metrics['specificity']*100:10.1f}% | {news_metrics['ppv_5pct']*100:7.1f}% | {news_metrics['fp_rate']*100:6.1f}% | {alerts_news:6} |")
    print(f"| qSOFA >= 2       | {qsofa_metrics['sensitivity']*100:10.1f}% | {qsofa_metrics['specificity']*100:10.1f}% | {qsofa_metrics['ppv_5pct']*100:7.1f}% | {qsofa_metrics['fp_rate']*100:6.1f}% | {alerts_qsofa:6} |")

    # ==========================================================================
    # FIND OPTIMAL THRESHOLD
    # ==========================================================================
    print()
    print("=" * 70)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 70)
    print()
    print("Criteria: Sensitivity >= 80%, Specificity > NEWS, PPV > NEWS")
    print()

    optimal_threshold = None
    for min_domains in thresholds_to_test:
        m = all_results[min_domains]['metrics']

        sens_ok = m['sensitivity'] >= 0.80
        spec_better = m['specificity'] > news_metrics['specificity']
        ppv_better = m['ppv_5pct'] > news_metrics['ppv_5pct']

        status = []
        if sens_ok:
            status.append("Sens>=80%")
        if spec_better:
            status.append("Spec>NEWS")
        if ppv_better:
            status.append("PPV>NEWS")

        is_optimal = sens_ok and spec_better and ppv_better
        marker = "***OPTIMAL***" if is_optimal else ""

        print(f"  {min_domains}+ domains: {', '.join(status) if status else 'None'} {marker}")

        if is_optimal and optimal_threshold is None:
            optimal_threshold = min_domains

    # ==========================================================================
    # DETAILED ANALYSIS OF BEST THRESHOLD
    # ==========================================================================
    print()
    print("=" * 70)

    # If no optimal found, use 2 domains as default
    best_threshold = optimal_threshold if optimal_threshold else 2
    best_m = all_results[best_threshold]['metrics']
    best_df = all_results[best_threshold]['results_df']

    if optimal_threshold:
        print(f"OPTIMAL THRESHOLD FOUND: {optimal_threshold}+ domains")
    else:
        print(f"NO OPTIMAL THRESHOLD - Using {best_threshold}+ domains (best balance)")
    print("=" * 70)
    print()

    print(f"Performance at {best_threshold}+ domain threshold:")
    print(f"  Sensitivity: {best_m['sensitivity']*100:.1f}% (catches {best_m['tp']}/{best_m['tp']+best_m['fn']} events)")
    print(f"  Specificity: {best_m['specificity']*100:.1f}% (avoids {best_m['tn']}/{best_m['tn']+best_m['fp']} false alerts)")
    print(f"  PPV @ 5%:    {best_m['ppv_5pct']*100:.1f}%")
    print(f"  FP Rate:     {best_m['fp_rate']*100:.1f}%")
    print()

    print("Comparison to NEWS:")
    sens_diff = (best_m['sensitivity'] - news_metrics['sensitivity']) * 100
    spec_diff = (best_m['specificity'] - news_metrics['specificity']) * 100
    ppv_diff = (best_m['ppv_5pct'] - news_metrics['ppv_5pct']) * 100

    print(f"  Sensitivity: {'+' if sens_diff >= 0 else ''}{sens_diff:.1f}% vs NEWS")
    print(f"  Specificity: {'+' if spec_diff >= 0 else ''}{spec_diff:.1f}% vs NEWS")
    print(f"  PPV @ 5%:    {'+' if ppv_diff >= 0 else ''}{ppv_diff:.1f}% vs NEWS")

    # ==========================================================================
    # DOMAIN DISTRIBUTION ANALYSIS
    # ==========================================================================
    print()
    print("=" * 70)
    print("DOMAIN INVOLVEMENT ANALYSIS")
    print("=" * 70)
    print()

    # Analyze which domains are most predictive
    event_patients = best_df[best_df['actual_event'] == 1]
    non_event_patients = best_df[best_df['actual_event'] == 0]

    print(f"Average domains involved:")
    print(f"  Event patients:     {event_patients['num_domains'].mean():.2f} domains")
    print(f"  Non-event patients: {non_event_patients['num_domains'].mean():.2f} domains")
    print()

    # Count domain involvement
    all_domains = []
    for domains_str in best_df['domains_involved']:
        if domains_str:
            all_domains.extend(domains_str.split(','))

    domain_counts = pd.Series(all_domains).value_counts()
    print("Most frequently alerting domains:")
    for domain, count in domain_counts.head(5).items():
        print(f"  {domain}: {count} patients")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    best_df.to_csv(OUTPUT_FILE, index=False)
    print()
    print(f"Results saved to: {OUTPUT_FILE}")

    # ==========================================================================
    # FINAL VERDICT
    # ==========================================================================
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print()

    if optimal_threshold:
        print(f"SUCCESS: {optimal_threshold}+ domain threshold achieves:")
        print(f"  - Sensitivity: {best_m['sensitivity']*100:.1f}% (>= 80% target)")
        print(f"  - Specificity: {best_m['specificity']*100:.1f}% (> NEWS {news_metrics['specificity']*100:.1f}%)")
        print(f"  - PPV @ 5%:    {best_m['ppv_5pct']*100:.1f}% (> NEWS {news_metrics['ppv_5pct']*100:.1f}%)")
        print()
        print("Multi-domain convergence SUCCESSFULLY eliminates false positives")
        print("while maintaining high sensitivity for true events.")
    else:
        # Find best trade-off
        print("NO threshold meets ALL criteria. Analysis of trade-offs:")
        print()
        for min_domains in thresholds_to_test:
            m = all_results[min_domains]['metrics']
            print(f"  {min_domains}+ domains:")
            print(f"    Sensitivity: {m['sensitivity']*100:.1f}% {'OK' if m['sensitivity'] >= 0.80 else 'LOW'}")
            print(f"    Specificity: {m['specificity']*100:.1f}% {'> NEWS' if m['specificity'] > news_metrics['specificity'] else '< NEWS'}")
            print(f"    PPV @ 5%:    {m['ppv_5pct']*100:.1f}% {'> NEWS' if m['ppv_5pct'] > news_metrics['ppv_5pct'] else '< NEWS'}")
            print()

        # Recommend based on clinical priority
        print("RECOMMENDATION based on clinical priority:")
        print()
        print("  If SENSITIVITY is priority (don't miss events):")
        print(f"    Use 1+ domains: {all_results[1]['metrics']['sensitivity']*100:.1f}% sensitivity")
        print()
        print("  If SPECIFICITY is priority (fewer false alarms):")
        best_spec_threshold = max(thresholds_to_test, key=lambda t: all_results[t]['metrics']['specificity'])
        print(f"    Use {best_spec_threshold}+ domains: {all_results[best_spec_threshold]['metrics']['specificity']*100:.1f}% specificity")
        print()
        print("  If BALANCED performance:")
        # Find threshold with best F1-like balance
        best_balanced = max(thresholds_to_test,
                          key=lambda t: 2 * all_results[t]['metrics']['sensitivity'] * all_results[t]['metrics']['specificity'] /
                                        (all_results[t]['metrics']['sensitivity'] + all_results[t]['metrics']['specificity'] + 0.001))
        bm = all_results[best_balanced]['metrics']
        print(f"    Use {best_balanced}+ domains: Sens={bm['sensitivity']*100:.1f}%, Spec={bm['specificity']*100:.1f}%")

    print()
    print("-" * 70)

if __name__ == "__main__":
    main()
