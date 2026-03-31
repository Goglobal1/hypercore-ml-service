"""
HYBRID MULTI-SIGNAL VALIDATION
==============================
Combines trajectory analysis + absolute thresholds + domain convergence.
Optimized for the biomarkers available in MIMIC-IV data.
"""
import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

VALIDATION_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\mimic_validation_with_scores.csv'
OUTPUT_FILE = r'C:\Users\letsa\Documents\hypercore-ml-service\outputs\hybrid_predictions.csv'

# =============================================================================
# BIOMARKER DEFINITIONS (optimized for available data)
# =============================================================================
BIOMARKER_CONFIG = {
    'heart_rate': {
        'domain': 'hemodynamic',
        'critical_high': 120,
        'critical_low': 50,
        'warning_high': 100,
        'warning_low': 60,
        'rise_concerning': 0.15,  # 15% rise is concerning
        'fall_concerning': None,
        'weight': 1.0
    },
    'respiratory_rate': {
        'domain': 'respiratory',
        'critical_high': 30,
        'critical_low': 8,
        'warning_high': 22,
        'warning_low': 10,
        'rise_concerning': 0.20,
        'fall_concerning': -0.30,
        'weight': 1.2  # RR is very predictive
    },
    'spo2': {
        'domain': 'respiratory',
        'critical_high': None,
        'critical_low': 90,
        'warning_high': None,
        'warning_low': 94,
        'rise_concerning': None,
        'fall_concerning': -0.05,  # 5% drop is concerning
        'weight': 1.5  # SpO2 drops are very serious
    },
    'sbp': {
        'domain': 'hemodynamic',
        'critical_high': 180,
        'critical_low': 90,
        'warning_high': 160,
        'warning_low': 100,
        'rise_concerning': None,
        'fall_concerning': -0.15,  # 15% drop is concerning
        'weight': 1.3
    },
    'lactate': {
        'domain': 'inflammatory',
        'critical_high': 4.0,
        'critical_low': None,
        'warning_high': 2.0,
        'warning_low': None,
        'rise_concerning': 0.25,  # 25% rise
        'fall_concerning': None,
        'weight': 1.5  # Lactate is very predictive
    },
    'creatinine': {
        'domain': 'renal',
        'critical_high': 3.0,
        'critical_low': None,
        'warning_high': 1.5,
        'warning_low': None,
        'rise_concerning': 0.25,  # 25% rise (AKI definition)
        'fall_concerning': None,
        'weight': 1.2
    },
    'troponin': {
        'domain': 'cardiac',
        'critical_high': 0.1,
        'critical_low': None,
        'warning_high': 0.04,
        'warning_low': None,
        'rise_concerning': 0.20,
        'fall_concerning': None,
        'weight': 1.3
    }
}

def calculate_signal_score(patient_data, up_to_window):
    """
    Calculate a composite risk score based on:
    1. Current values vs thresholds (like NEWS)
    2. Trajectory changes (HyperCore's contribution)
    3. Domain convergence (multi-signal synthesis)
    """
    data = patient_data[patient_data['window_num'] <= up_to_window].sort_values('window_num')

    if len(data) < 2:
        return 0.0, {}, []

    signals = []
    domain_scores = {}
    total_score = 0.0
    total_weight = 0.0

    for biomarker, config in BIOMARKER_CONFIG.items():
        if biomarker not in data.columns:
            continue

        values = data[biomarker].dropna().values
        if len(values) < 1:
            continue

        domain = config['domain']
        weight = config['weight']

        current_val = values[-1]
        signal_score = 0.0
        signal_reasons = []

        # Check absolute thresholds (current value)
        if config['critical_high'] and current_val > config['critical_high']:
            signal_score += 0.4
            signal_reasons.append(f"critical_high ({current_val:.1f}>{config['critical_high']})")
        elif config['warning_high'] and current_val > config['warning_high']:
            signal_score += 0.2
            signal_reasons.append(f"warning_high ({current_val:.1f}>{config['warning_high']})")

        if config['critical_low'] and current_val < config['critical_low']:
            signal_score += 0.4
            signal_reasons.append(f"critical_low ({current_val:.1f}<{config['critical_low']})")
        elif config['warning_low'] and current_val < config['warning_low']:
            signal_score += 0.2
            signal_reasons.append(f"warning_low ({current_val:.1f}<{config['warning_low']})")

        # Check trajectory (if we have multiple values)
        if len(values) >= 2:
            mid = max(1, len(values) // 2)
            baseline = np.mean(values[:mid])
            recent = np.mean(values[mid:])

            if baseline != 0:
                pct_change = (recent - baseline) / abs(baseline)

                # Rising concern
                if config['rise_concerning'] and pct_change > config['rise_concerning']:
                    signal_score += 0.3
                    signal_reasons.append(f"rising +{pct_change*100:.0f}%")

                # Falling concern
                if config['fall_concerning'] and pct_change < config['fall_concerning']:
                    signal_score += 0.3
                    signal_reasons.append(f"falling {pct_change*100:.0f}%")

        # Apply weight and add to domain
        weighted_score = signal_score * weight
        if weighted_score > 0:
            if domain not in domain_scores:
                domain_scores[domain] = 0.0
            domain_scores[domain] = max(domain_scores[domain], weighted_score)

            signals.append({
                'biomarker': biomarker,
                'domain': domain,
                'score': weighted_score,
                'reasons': signal_reasons,
                'current': current_val
            })

        total_score += weighted_score
        total_weight += weight

    # Normalize
    if total_weight > 0:
        normalized_score = total_score / total_weight
    else:
        normalized_score = 0.0

    # Apply domain convergence bonus
    num_domains_alerting = sum(1 for d, s in domain_scores.items() if s > 0.1)
    if num_domains_alerting >= 3:
        normalized_score *= 1.3  # 30% boost for 3+ domains
    elif num_domains_alerting >= 2:
        normalized_score *= 1.15  # 15% boost for 2 domains

    # Cap at 1.0
    normalized_score = min(normalized_score, 1.0)

    return normalized_score, domain_scores, signals

def run_validation(df, alert_threshold):
    """Run validation with specified alert threshold."""
    patients = df['patient_id'].unique()
    results = []

    for patient_id in patients:
        patient_data = df[df['patient_id'] == patient_id].copy()
        has_event = patient_data['event_in_12h'].max()

        if has_event == 1:
            event_window = patient_data[patient_data['event_in_12h'] == 1]['window_num'].min()
            predict_window = event_window - 1 if event_window > 1 else event_window
        else:
            predict_window = patient_data['window_num'].max()

        if predict_window < 3:
            continue

        score, domain_scores, signals = calculate_signal_score(patient_data, predict_window)
        alert = 1 if score >= alert_threshold else 0

        window_data = patient_data[patient_data['window_num'] == predict_window]
        news_alert = window_data['news_alert'].values[0] if len(window_data) > 0 else 0
        qsofa_alert = window_data['qsofa_alert'].values[0] if len(window_data) > 0 else 0

        results.append({
            'patient_id': patient_id,
            'actual_event': has_event,
            'hybrid_score': score,
            'hybrid_alert': alert,
            'num_domains': len([d for d, s in domain_scores.items() if s > 0]),
            'news_alert': news_alert,
            'qsofa_alert': qsofa_alert
        })

    return pd.DataFrame(results)

def calculate_metrics(results_df, alert_col):
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

    return {
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv_5pct': ppv,
        'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0
    }

def main():
    print("=" * 70)
    print("HYBRID MULTI-SIGNAL VALIDATION")
    print("=" * 70)
    print()
    print("APPROACH: Combine absolute thresholds + trajectories + domain convergence")
    print()

    df = pd.read_csv(VALIDATION_FILE)
    print(f"Loaded {len(df)} windows from {df['patient_id'].nunique()} patients")
    print(f"Events: {df['event_in_12h'].sum()}")
    print()

    # Test different thresholds
    thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    all_results = {}

    print("Testing alert thresholds...")
    print("-" * 60)

    for thresh in thresholds:
        results_df = run_validation(df, thresh)
        metrics = calculate_metrics(results_df, 'hybrid_alert')
        all_results[thresh] = {'df': results_df, 'metrics': metrics}
        alerts = results_df['hybrid_alert'].sum()
        print(f"  Score >= {thresh:.2f}: Sens={metrics['sensitivity']*100:.1f}%, "
              f"Spec={metrics['specificity']*100:.1f}%, "
              f"PPV={metrics['ppv_5pct']*100:.1f}%, Alerts={alerts}")

    # NEWS baseline
    news_metrics = calculate_metrics(all_results[0.20]['df'], 'news_alert')
    qsofa_metrics = calculate_metrics(all_results[0.20]['df'], 'qsofa_alert')

    print()
    print(f"  NEWS >= 5:  Sens={news_metrics['sensitivity']*100:.1f}%, "
          f"Spec={news_metrics['specificity']*100:.1f}%, "
          f"PPV={news_metrics['ppv_5pct']*100:.1f}%")
    print(f"  qSOFA >= 2: Sens={qsofa_metrics['sensitivity']*100:.1f}%, "
          f"Spec={qsofa_metrics['specificity']*100:.1f}%, "
          f"PPV={qsofa_metrics['ppv_5pct']*100:.1f}%")

    # Find optimal threshold
    print()
    print("=" * 70)
    print("FINDING OPTIMAL THRESHOLD")
    print("=" * 70)
    print()
    print("Criteria: Sensitivity >= 70%, Specificity > NEWS, PPV > NEWS")
    print()

    optimal = None
    for thresh in thresholds:
        m = all_results[thresh]['metrics']
        sens_ok = m['sensitivity'] >= 0.70
        spec_better = m['specificity'] > news_metrics['specificity']
        ppv_better = m['ppv_5pct'] > news_metrics['ppv_5pct']

        checks = []
        if sens_ok: checks.append("Sens>=70%")
        if spec_better: checks.append("Spec>NEWS")
        if ppv_better: checks.append("PPV>NEWS")

        is_optimal = sens_ok and spec_better and ppv_better
        marker = " ***OPTIMAL***" if is_optimal else ""

        print(f"  {thresh:.2f}: {', '.join(checks) if checks else 'None'}{marker}")

        if is_optimal and optimal is None:
            optimal = thresh

    # Comparison table
    print()
    print("=" * 70)
    print("COMPREHENSIVE COMPARISON")
    print("=" * 70)
    print()
    print("| System              | Sensitivity | Specificity | PPV @ 5% | Alerts |")
    print("|---------------------|-------------|-------------|----------|--------|")

    for thresh in [0.15, 0.20, 0.25, 0.30]:
        m = all_results[thresh]['metrics']
        alerts = all_results[thresh]['df']['hybrid_alert'].sum()
        print(f"| Hybrid >= {thresh:.2f}       | {m['sensitivity']*100:10.1f}% | {m['specificity']*100:10.1f}% | {m['ppv_5pct']*100:7.1f}% | {alerts:6} |")

    print("|---------------------|-------------|-------------|----------|--------|")
    news_alerts = all_results[0.20]['df']['news_alert'].sum()
    qsofa_alerts = all_results[0.20]['df']['qsofa_alert'].sum()
    print(f"| NEWS >= 5           | {news_metrics['sensitivity']*100:10.1f}% | {news_metrics['specificity']*100:.1f}% | {news_metrics['ppv_5pct']*100:7.1f}% | {news_alerts:6} |")
    print(f"| qSOFA >= 2          | {qsofa_metrics['sensitivity']*100:10.1f}% | {qsofa_metrics['specificity']*100:10.1f}% | {qsofa_metrics['ppv_5pct']*100:7.1f}% | {qsofa_alerts:6} |")

    # Best threshold analysis
    print()
    print("=" * 70)

    if optimal:
        print(f"OPTIMAL THRESHOLD FOUND: Score >= {optimal}")
        best_m = all_results[optimal]['metrics']
    else:
        # Find best balanced threshold
        best_thresh = max(thresholds,
                         key=lambda t: (all_results[t]['metrics']['sensitivity'] +
                                       all_results[t]['metrics']['specificity'] +
                                       all_results[t]['metrics']['ppv_5pct']))
        print(f"NO OPTIMAL THRESHOLD - Best balanced: Score >= {best_thresh}")
        best_m = all_results[best_thresh]['metrics']
        optimal = best_thresh

    print("=" * 70)
    print()

    best_df = all_results[optimal]['df']

    print(f"Performance at score >= {optimal}:")
    print(f"  Sensitivity: {best_m['sensitivity']*100:.1f}% (catches {best_m['tp']}/{best_m['tp']+best_m['fn']} events)")
    print(f"  Specificity: {best_m['specificity']*100:.1f}%")
    print(f"  PPV @ 5%:    {best_m['ppv_5pct']*100:.1f}%")
    print()

    print("vs NEWS:")
    print(f"  Sensitivity: {(best_m['sensitivity']-news_metrics['sensitivity'])*100:+.1f}%")
    print(f"  Specificity: {(best_m['specificity']-news_metrics['specificity'])*100:+.1f}%")
    print(f"  PPV @ 5%:    {(best_m['ppv_5pct']-news_metrics['ppv_5pct'])*100:+.1f}%")

    # Hybrid + NEWS combination
    print()
    print("=" * 70)
    print("HYBRID + NEWS COMBINATION")
    print("=" * 70)
    print()

    # Try combining: Alert if Hybrid OR NEWS
    combined_alert = ((best_df['hybrid_alert'] == 1) | (best_df['news_alert'] == 1)).astype(int).values
    actual = best_df['actual_event'].values

    comb_tp = ((actual == 1) & (combined_alert == 1)).sum()
    comb_fn = ((actual == 1) & (combined_alert == 0)).sum()
    comb_fp = ((actual == 0) & (combined_alert == 1)).sum()
    comb_tn = ((actual == 0) & (combined_alert == 0)).sum()

    comb_sens = comb_tp / (comb_tp + comb_fn) if (comb_tp + comb_fn) > 0 else 0
    comb_spec = comb_tn / (comb_tn + comb_fp) if (comb_tn + comb_fp) > 0 else 0
    comb_ppv = (comb_sens * 0.05) / ((comb_sens * 0.05) + ((1 - comb_spec) * 0.95)) if comb_sens > 0 else 0

    print(f"Hybrid OR NEWS:")
    print(f"  Sensitivity: {comb_sens*100:.1f}%")
    print(f"  Specificity: {comb_spec*100:.1f}%")
    print(f"  PPV @ 5%:    {comb_ppv*100:.1f}%")

    # Try combining: Alert if Hybrid AND NEWS
    both_alert = ((best_df['hybrid_alert'] == 1) & (best_df['news_alert'] == 1)).astype(int).values

    both_tp = ((actual == 1) & (both_alert == 1)).sum()
    both_fn = ((actual == 1) & (both_alert == 0)).sum()
    both_fp = ((actual == 0) & (both_alert == 1)).sum()
    both_tn = ((actual == 0) & (both_alert == 0)).sum()

    both_sens = both_tp / (both_tp + both_fn) if (both_tp + both_fn) > 0 else 0
    both_spec = both_tn / (both_tn + both_fp) if (both_tn + both_fp) > 0 else 0
    both_ppv = (both_sens * 0.05) / ((both_sens * 0.05) + ((1 - both_spec) * 0.95)) if both_sens > 0 else 0

    print()
    print(f"Hybrid AND NEWS (both must agree):")
    print(f"  Sensitivity: {both_sens*100:.1f}%")
    print(f"  Specificity: {both_spec*100:.1f}%")
    print(f"  PPV @ 5%:    {both_ppv*100:.1f}%")

    # Save
    best_df.to_csv(OUTPUT_FILE, index=False)
    print()
    print(f"Results saved to: {OUTPUT_FILE}")

    # Final verdict
    print()
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    print()

    if best_m['sensitivity'] > news_metrics['sensitivity'] and best_m['specificity'] > news_metrics['specificity']:
        print("SUCCESS: Hybrid scoring BEATS NEWS on BOTH sensitivity and specificity!")
    elif best_m['sensitivity'] > news_metrics['sensitivity']:
        print(f"PARTIAL: Hybrid has HIGHER sensitivity ({best_m['sensitivity']*100:.1f}% vs {news_metrics['sensitivity']*100:.1f}%)")
        print(f"         but lower specificity ({best_m['specificity']*100:.1f}% vs {news_metrics['specificity']*100:.1f}%)")
    elif best_m['specificity'] > news_metrics['specificity'] and best_m['ppv_5pct'] > news_metrics['ppv_5pct']:
        print(f"PARTIAL: Hybrid has HIGHER specificity and PPV")
        print(f"         Specificity: {best_m['specificity']*100:.1f}% vs {news_metrics['specificity']*100:.1f}%")
        print(f"         PPV @ 5%: {best_m['ppv_5pct']*100:.1f}% vs {news_metrics['ppv_5pct']*100:.1f}%")
    else:
        print("HONEST: Hybrid does not clearly beat NEWS in this dataset")

    print()
    print("RECOMMENDATION:")
    if both_ppv > news_metrics['ppv_5pct'] * 1.5:
        print(f"  Use Hybrid AND NEWS for high-confidence alerts (PPV: {both_ppv*100:.1f}%)")
    if comb_sens > news_metrics['sensitivity']:
        print(f"  Use Hybrid OR NEWS for screening (Sensitivity: {comb_sens*100:.1f}%)")

    print()
    print("-" * 70)

if __name__ == "__main__":
    main()
