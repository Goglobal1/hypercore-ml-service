"""
FAST OPTIMIZATION - Find Best HyperCore Configuration
======================================================
Systematic search for optimal performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Load data
print("Loading MIMIC-IV data...")
predictions_df = pd.read_csv('hybrid_predictions.csv')
validation_df = pd.read_csv('mimic_validation_dataset.csv')
patient_outcomes = predictions_df[['patient_id', 'actual_event']].drop_duplicates()

n_patients = len(patient_outcomes)
n_events = patient_outcomes['actual_event'].sum()
n_non_events = n_patients - n_events
actual_prevalence = n_events / n_patients

print(f"\n{'='*60}")
print("DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total patients: {n_patients}")
print(f"Events: {n_events}")
print(f"Non-events: {n_non_events}")
print(f"Actual prevalence: {actual_prevalence*100:.1f}%")

# Baselines
BASELINES = {
    "NEWS >= 5": {"sensitivity": 0.244, "specificity": 0.854, "ppv_5pct": 0.081},
    "qSOFA >= 2": {"sensitivity": 0.073, "specificity": 0.988, "ppv_5pct": 0.240},
    "MEWS >= 4": {"sensitivity": 0.30, "specificity": 0.80, "ppv_5pct": 0.073},
    "Epic DI": {"sensitivity": 0.65, "specificity": 0.80, "ppv_5pct": 0.146},
}

# Biomarker config
BIOMARKER_CONFIG = {
    'heart_rate': {'domain': 'hemodynamic', 'critical_high': 130, 'critical_low': 45, 'warning_high': 110, 'warning_low': 55, 'weight': 1.2},
    'respiratory_rate': {'domain': 'respiratory', 'critical_high': 28, 'critical_low': 8, 'warning_high': 24, 'warning_low': 10, 'weight': 1.3},
    'sbp': {'domain': 'hemodynamic', 'critical_high': 200, 'critical_low': 85, 'warning_high': 180, 'warning_low': 95, 'weight': 1.4},
    'spo2': {'domain': 'respiratory', 'critical_high': None, 'critical_low': 88, 'warning_high': None, 'warning_low': 92, 'weight': 1.5},
    'lactate': {'domain': 'inflammatory', 'critical_high': 4.0, 'critical_low': None, 'warning_high': 2.5, 'warning_low': None, 'weight': 1.8},
    'creatinine': {'domain': 'renal', 'critical_high': 3.5, 'critical_low': None, 'warning_high': 2.0, 'warning_low': None, 'weight': 1.3},
    'temperature': {'domain': 'inflammatory', 'critical_high': 39.5, 'critical_low': 35.0, 'warning_high': 38.5, 'warning_low': 36.0, 'weight': 1.0},
    'troponin': {'domain': 'cardiac', 'critical_high': 0.5, 'critical_low': None, 'warning_high': 0.1, 'warning_low': None, 'weight': 1.6}
}

DOMAINS = ['hemodynamic', 'respiratory', 'inflammatory', 'renal', 'cardiac']


def calculate_ppv(sens, spec, prevalence):
    """PPV at given prevalence."""
    if sens == 0:
        return 0
    tp = sens * prevalence
    fp = (1 - spec) * (1 - prevalence)
    return tp / (tp + fp) if (tp + fp) > 0 else 0


def evaluate_patient(patient_data: pd.DataFrame, config: Dict) -> Dict:
    """Score a patient."""
    domain_alerts = {d: {'alerting': False, 'strength': 0, 'critical': False} for d in DOMAINS}

    if len(patient_data) == 0:
        return {'should_alert': False}

    latest = patient_data.iloc[-1]
    first = patient_data.iloc[0] if len(patient_data) >= 2 else latest
    has_trajectory = len(patient_data) >= 2

    total_strength = 0
    critical_count = 0

    for biomarker, bio_config in BIOMARKER_CONFIG.items():
        if biomarker not in patient_data.columns:
            continue

        value = latest.get(biomarker)
        if pd.isna(value):
            continue

        domain = bio_config['domain']
        weight = bio_config.get('weight', 1.0)
        signal_strength = 0
        is_critical = False

        # Absolute thresholds
        if bio_config.get('critical_high') and value > bio_config['critical_high']:
            signal_strength = 0.9
            is_critical = True
        elif bio_config.get('critical_low') and value < bio_config['critical_low']:
            signal_strength = 0.9
            is_critical = True
        elif bio_config.get('warning_high') and value > bio_config['warning_high']:
            signal_strength = 0.5
        elif bio_config.get('warning_low') and value < bio_config['warning_low']:
            signal_strength = 0.5

        # Trajectory
        traj_threshold = config.get('trajectory_threshold', 0.25)
        if has_trajectory and not pd.isna(first.get(biomarker)) and first.get(biomarker) > 0:
            first_val = first.get(biomarker)
            pct_change = (value - first_val) / first_val

            if biomarker in ['heart_rate', 'respiratory_rate', 'lactate', 'creatinine', 'troponin']:
                if pct_change > traj_threshold:
                    signal_strength = max(signal_strength, min(0.7, pct_change))
                    if pct_change > 0.50:
                        is_critical = True
            elif biomarker in ['sbp', 'spo2']:
                if pct_change < -0.15:
                    signal_strength = max(signal_strength, min(0.7, abs(pct_change)))
                    if pct_change < -0.25:
                        is_critical = True

        if signal_strength > 0:
            domain_alerts[domain]['alerting'] = True
            domain_alerts[domain]['strength'] = max(domain_alerts[domain]['strength'], signal_strength * weight)
            if is_critical:
                domain_alerts[domain]['critical'] = True
                critical_count += 1
            total_strength += signal_strength * weight

    alerting_domains = sum(1 for d in domain_alerts.values() if d['alerting'])

    # Score calculation
    base_score = 0
    if alerting_domains > 0:
        base_score = (total_strength / alerting_domains) * config.get('absolute_weight', 0.6)

    # Domain bonuses
    if alerting_domains >= 3:
        base_score += config.get('domain_bonus_3', 0.30)
    elif alerting_domains >= 2:
        base_score += config.get('domain_bonus_2', 0.15)

    # Critical bonus
    if critical_count > 0:
        base_score += config.get('critical_bonus', 0.20) * min(critical_count, 3)

    # Alert decision
    min_domains = config.get('min_domains', 2)
    threshold = config.get('alert_threshold', 0.25)
    require_critical = config.get('require_critical', False)

    if require_critical:
        should_alert = base_score >= threshold and (critical_count >= 1 or alerting_domains >= min_domains)
    else:
        should_alert = base_score >= threshold and alerting_domains >= min_domains

    return {'should_alert': should_alert, 'score': base_score, 'domains': alerting_domains, 'critical': critical_count}


def evaluate_config(config: Dict) -> Dict:
    """Evaluate config on all patients."""
    tp = fn = fp = tn = 0

    for patient_id in patient_outcomes['patient_id'].unique():
        patient_data = validation_df[validation_df['patient_id'] == patient_id].copy()
        actual_event = patient_outcomes[patient_outcomes['patient_id'] == patient_id]['actual_event'].iloc[0]

        if len(patient_data) == 0:
            continue

        if 'prediction_time' in patient_data.columns:
            patient_data = patient_data.sort_values('prediction_time')

        result = evaluate_patient(patient_data, config)

        if actual_event == 1 and result['should_alert']:
            tp += 1
        elif actual_event == 1 and not result['should_alert']:
            fn += 1
        elif actual_event == 0 and result['should_alert']:
            fp += 1
        else:
            tn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv_5pct = calculate_ppv(sensitivity, specificity, 0.05)
    ppv_actual = calculate_ppv(sensitivity, specificity, actual_prevalence)

    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv_5pct': ppv_5pct,
        'ppv_actual': ppv_actual,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn
    }


# GRID SEARCH
print(f"\n{'='*60}")
print("SYSTEMATIC OPTIMIZATION")
print(f"{'='*60}")

results = []
search_id = 0

# Focused search space
min_domains_options = [1, 2, 3]
require_critical_options = [True, False]
threshold_options = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
critical_bonus_options = [0.15, 0.20, 0.25, 0.30]
domain_bonus_3_options = [0.20, 0.30, 0.40]
trajectory_options = [0.20, 0.30, 0.40]

total = len(min_domains_options) * len(require_critical_options) * len(threshold_options) * \
        len(critical_bonus_options) * len(domain_bonus_3_options) * len(trajectory_options)
print(f"Testing {total} configurations...")

for min_domains in min_domains_options:
    for require_critical in require_critical_options:
        for threshold in threshold_options:
            for critical_bonus in critical_bonus_options:
                for domain_bonus_3 in domain_bonus_3_options:
                    for traj_threshold in trajectory_options:
                        search_id += 1

                        config = {
                            'min_domains': min_domains,
                            'require_critical': require_critical,
                            'alert_threshold': threshold,
                            'critical_bonus': critical_bonus,
                            'domain_bonus_2': 0.12,
                            'domain_bonus_3': domain_bonus_3,
                            'absolute_weight': 0.60,
                            'trajectory_threshold': traj_threshold,
                        }

                        metrics = evaluate_config(config)
                        metrics['config'] = config
                        results.append(metrics)

print(f"Evaluated {search_id} configurations")

# Find best configs for different objectives
print(f"\n{'='*60}")
print("OPTIMAL CONFIGURATIONS")
print(f"{'='*60}")

# Best overall (balanced)
best_balanced = max(results, key=lambda x: x['sensitivity'] * 0.35 + x['specificity'] * 0.35 + x['ppv_5pct'] * 0.3)
print("\n[1] BEST BALANCED (Equal weight to all metrics)")
print(f"    Sensitivity: {best_balanced['sensitivity']*100:.1f}%")
print(f"    Specificity: {best_balanced['specificity']*100:.1f}%")
print(f"    PPV @ 5%: {best_balanced['ppv_5pct']*100:.1f}%")
print(f"    Config: min_domains={best_balanced['config']['min_domains']}, threshold={best_balanced['config']['alert_threshold']}")

# Best for beating NEWS on ALL metrics
news_beaters = [r for r in results if
                r['sensitivity'] > 0.244 and r['specificity'] > 0.854 and r['ppv_5pct'] > 0.081]
if news_beaters:
    best_news = max(news_beaters, key=lambda x: x['sensitivity'] * 0.4 + x['specificity'] * 0.3 + x['ppv_5pct'] * 0.3)
    print("\n[2] BEATS NEWS ON ALL METRICS")
    print(f"    Sensitivity: {best_news['sensitivity']*100:.1f}% (vs NEWS 24.4%)")
    print(f"    Specificity: {best_news['specificity']*100:.1f}% (vs NEWS 85.4%)")
    print(f"    PPV @ 5%: {best_news['ppv_5pct']*100:.1f}% (vs NEWS 8.1%)")
    print(f"    Config: {best_news['config']}")
else:
    print("\n[2] BEATS NEWS ON ALL METRICS: Not found")

# Best for beating qSOFA PPV
qsofa_ppv_beaters = [r for r in results if r['ppv_5pct'] > 0.24 and r['sensitivity'] > 0.073]
if qsofa_ppv_beaters:
    best_qsofa = max(qsofa_ppv_beaters, key=lambda x: x['ppv_5pct'] * 0.5 + x['sensitivity'] * 0.3 + x['specificity'] * 0.2)
    print("\n[3] BEATS qSOFA PPV (24%) WITH BETTER SENSITIVITY")
    print(f"    Sensitivity: {best_qsofa['sensitivity']*100:.1f}% (vs qSOFA 7.3%)")
    print(f"    Specificity: {best_qsofa['specificity']*100:.1f}%")
    print(f"    PPV @ 5%: {best_qsofa['ppv_5pct']*100:.1f}% (vs qSOFA 24.0%)")
    print(f"    Config: {best_qsofa['config']}")
else:
    print("\n[3] BEATS qSOFA PPV: Not found")

# Best sensitivity (for screening)
best_sens = max(results, key=lambda x: x['sensitivity'])
print("\n[4] MAXIMUM SENSITIVITY (Screening Mode)")
print(f"    Sensitivity: {best_sens['sensitivity']*100:.1f}%")
print(f"    Specificity: {best_sens['specificity']*100:.1f}%")
print(f"    PPV @ 5%: {best_sens['ppv_5pct']*100:.1f}%")

# Best PPV (for precision)
best_ppv = max(results, key=lambda x: x['ppv_5pct'] if x['sensitivity'] > 0.10 else 0)
print("\n[5] MAXIMUM PPV (Precision Mode)")
print(f"    Sensitivity: {best_ppv['sensitivity']*100:.1f}%")
print(f"    Specificity: {best_ppv['specificity']*100:.1f}%")
print(f"    PPV @ 5%: {best_ppv['ppv_5pct']*100:.1f}%")

# COMPARISON TABLE
print(f"\n{'='*60}")
print("FINAL COMPARISON TABLE")
print(f"{'='*60}")

print(f"\n{'System':<25} {'Sensitivity':>12} {'Specificity':>12} {'PPV @ 5%':>12}")
print("-" * 65)
for name, baseline in BASELINES.items():
    print(f"{name:<25} {baseline['sensitivity']*100:>11.1f}% {baseline['specificity']*100:>11.1f}% {baseline['ppv_5pct']*100:>11.1f}%")

print("-" * 65)
print(f"{'HyperCore Balanced':<25} {best_balanced['sensitivity']*100:>11.1f}% {best_balanced['specificity']*100:>11.1f}% {best_balanced['ppv_5pct']*100:>11.1f}%")

if news_beaters:
    print(f"{'HyperCore (beats NEWS)':<25} {best_news['sensitivity']*100:>11.1f}% {best_news['specificity']*100:>11.1f}% {best_news['ppv_5pct']*100:>11.1f}%")

if qsofa_ppv_beaters:
    print(f"{'HyperCore (high PPV)':<25} {best_qsofa['sensitivity']*100:>11.1f}% {best_qsofa['specificity']*100:>11.1f}% {best_qsofa['ppv_5pct']*100:>11.1f}%")

# TIERED CONFIGURATION
print(f"\n{'='*60}")
print("RECOMMENDED TIERED CONFIGURATION")
print(f"{'='*60}")

print("""
TIER 1: CRITICAL (High Confidence)
  - Use: ICU escalation, rapid response triggers
  - Optimize for: PPV (minimize false alarms)
""")
if qsofa_ppv_beaters:
    c = best_qsofa['config']
    print(f"  Config: min_domains={c['min_domains']}, threshold={c['alert_threshold']}, require_critical={c['require_critical']}")
    print(f"  Performance: {best_qsofa['sensitivity']*100:.1f}% sens, {best_qsofa['specificity']*100:.1f}% spec, {best_qsofa['ppv_5pct']*100:.1f}% PPV")

print("""
TIER 2: HIGH (Balanced)
  - Use: Standard early warning
  - Optimize for: Balance of sensitivity and specificity
""")
c = best_balanced['config']
print(f"  Config: min_domains={c['min_domains']}, threshold={c['alert_threshold']}, require_critical={c['require_critical']}")
print(f"  Performance: {best_balanced['sensitivity']*100:.1f}% sens, {best_balanced['specificity']*100:.1f}% spec, {best_balanced['ppv_5pct']*100:.1f}% PPV")

print("""
TIER 3: WATCH (Screening)
  - Use: Don't miss any deterioration
  - Optimize for: Maximum sensitivity
""")
c = best_sens['config']
print(f"  Config: min_domains={c['min_domains']}, threshold={c['alert_threshold']}, require_critical={c['require_critical']}")
print(f"  Performance: {best_sens['sensitivity']*100:.1f}% sens, {best_sens['specificity']*100:.1f}% spec, {best_sens['ppv_5pct']*100:.1f}% PPV")

# Save results
print(f"\n{'='*60}")
print("SAVING RESULTS")
print(f"{'='*60}")

import json
optimal_configs = {
    'balanced': {'config': best_balanced['config'], 'metrics': {k: v for k, v in best_balanced.items() if k != 'config'}},
    'beats_news': {'config': best_news['config'], 'metrics': {k: v for k, v in best_news.items() if k != 'config'}} if news_beaters else None,
    'high_ppv': {'config': best_qsofa['config'], 'metrics': {k: v for k, v in best_qsofa.items() if k != 'config'}} if qsofa_ppv_beaters else None,
    'max_sensitivity': {'config': best_sens['config'], 'metrics': {k: v for k, v in best_sens.items() if k != 'config'}},
    'max_ppv': {'config': best_ppv['config'], 'metrics': {k: v for k, v in best_ppv.items() if k != 'config'}},
}

with open('optimal_configs_final.json', 'w') as f:
    json.dump({k: v for k, v in optimal_configs.items() if v is not None}, f, indent=2)
print("Saved to optimal_configs_final.json")

# Generate code
print(f"\n{'='*60}")
print("CODE FOR main.py")
print(f"{'='*60}")

if qsofa_ppv_beaters:
    c = best_qsofa['config']
    print(f"""
# HIGH_CONFIDENCE_MODE - Beats qSOFA PPV
HIGH_CONFIDENCE_CONFIG = {{
    "min_domains": {c['min_domains']},
    "require_critical": {c['require_critical']},
    "alert_threshold": {c['alert_threshold']},
    "critical_bonus": {c['critical_bonus']},
    "domain_bonus_3": {c['domain_bonus_3']},
    "trajectory_threshold": {c['trajectory_threshold']},
    # Expected: {best_qsofa['sensitivity']*100:.1f}% sens, {best_qsofa['specificity']*100:.1f}% spec, {best_qsofa['ppv_5pct']*100:.1f}% PPV
}}
""")

c = best_balanced['config']
print(f"""
# BALANCED_MODE - Best overall
BALANCED_CONFIG = {{
    "min_domains": {c['min_domains']},
    "require_critical": {c['require_critical']},
    "alert_threshold": {c['alert_threshold']},
    "critical_bonus": {c['critical_bonus']},
    "domain_bonus_3": {c['domain_bonus_3']},
    "trajectory_threshold": {c['trajectory_threshold']},
    # Expected: {best_balanced['sensitivity']*100:.1f}% sens, {best_balanced['specificity']*100:.1f}% spec, {best_balanced['ppv_5pct']*100:.1f}% PPV
}}
""")
