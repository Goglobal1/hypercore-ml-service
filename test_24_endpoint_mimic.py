"""
HyperCore 24-Endpoint System Validation on MIMIC-IV
"""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from datetime import datetime

# Import new 24-endpoint components
from app.core.endpoints.endpoint_definitions import ENDPOINT_DEFINITIONS, EndpointScorer, ALL_ENDPOINTS
from app.core.cross_loop_engine_v2 import CrossLoopEngineV2
from app.core.handler_metrics import calculate_handler_metrics
from app.core.data_ingest.universal_ingest import UniversalDataIngest, ingest_data

print('='*60)
print('HyperCore 24-Endpoint System Validation on MIMIC-IV')
print('='*60)
print()

# Load MIMIC data
df = pd.read_csv('outputs/mimic_validation_dataset.csv')
print(f'Loaded {len(df)} records from MIMIC-IV')
print(f'Positive cases: {df["event_in_12h"].sum()} ({df["event_in_12h"].mean()*100:.1f}%)')
print()

# Initialize components
scorer = EndpointScorer()
cross_loop = CrossLoopEngineV2()
ingestor = UniversalDataIngest()

print('Testing Universal Data Ingest...')
# Test with sample row
sample = df.iloc[0].dropna().to_dict()
result = ingestor.ingest(sample, 'dict')
print(f'  Normalized fields: {len(result.normalized)}')
print(f'  Endpoint mappings: {list(result.endpoint_mapping.keys())}')
print(f'  Data completeness: {result.data_completeness:.1%}')
print()

# Run 24-endpoint analysis on all patients
print('Running 24-Endpoint Analysis on all patients...')
y_true = []
patient_ids = []
timestamps = []

modes = ['screening', 'balanced', 'high_confidence']
results_by_mode = {mode: {'y_pred': [], 'y_scores': []} for mode in modes}

for idx, row in df.iterrows():
    # Prepare patient data
    patient_data = row.dropna().to_dict()

    y_true.append(int(row['event_in_12h']))
    patient_ids.append(row.get('patient_id', idx))
    timestamps.append(row.get('prediction_time', None))

    for mode in modes:
        analysis = cross_loop.analyze_patient(patient_data, mode=mode)

        # Use convergence score as risk score
        score = analysis.convergence_score

        # Mode-specific prediction thresholds
        if mode == 'screening':
            # Sensitive: 2+ endpoints OR any pathway OR elevated urgency
            pred = 1 if (analysis.n_endpoints_alerting >= 2 or
                        len(analysis.detected_pathways) > 0 or
                        analysis.urgency in ['moderate', 'high', 'critical']) else 0
        elif mode == 'balanced':
            # Balanced: 3+ endpoints OR high urgency pathway
            high_urgency_pathways = [p for p in analysis.detected_pathways if p.get('urgency') in ['high', 'critical']]
            pred = 1 if (analysis.n_endpoints_alerting >= 3 or
                        len(high_urgency_pathways) > 0 or
                        analysis.urgency in ['high', 'critical']) else 0
        else:  # high_confidence
            # Specific: 4+ endpoints OR critical urgency OR multi-system with critical pathway
            critical_pathways = [p for p in analysis.detected_pathways if p.get('urgency') == 'critical']
            pred = 1 if (analysis.n_endpoints_alerting >= 4 or
                        len(critical_pathways) > 0 or
                        analysis.urgency == 'critical' or
                        (analysis.multi_system_failure and analysis.convergence_score > 0.8)) else 0

        results_by_mode[mode]['y_pred'].append(pred)
        results_by_mode[mode]['y_scores'].append(score)

print(f'Analyzed {len(y_true)} patient records')
print()

# Calculate Handler metrics for each mode
print('='*60)
print('Handler Metrics by Mode')
print('='*60)

for mode in modes:
    y_pred_mode = results_by_mode[mode]['y_pred']
    y_scores_mode = results_by_mode[mode]['y_scores']

    metrics = calculate_handler_metrics(
        y_true=y_true,
        y_pred=y_pred_mode,
        y_scores=y_scores_mode,
        patient_ids=patient_ids,
        mode=mode
    )

    print(f'\n{mode.upper()} MODE:')
    print(f'  Sensitivity: {metrics["sensitivity"]*100:.1f}%')
    print(f'  Specificity: {metrics["specificity"]*100:.1f}%')
    print(f'  PPV: {metrics["ppv"]*100:.1f}%')
    print(f'  PPV@5%: {metrics["ppv_at_5_percent"]*100:.1f}%')
    print(f'  ROC-AUC: {metrics["roc_auc"]:.3f}')
    print(f'  PR-AUC: {metrics["pr_auc"]:.3f}')
    alert_burden_pct = metrics["alert_burden"]["alerts_per_patient"] * 100
    print(f'  Alert Burden: {alert_burden_pct:.1f}%')

    if mode == 'high_confidence':
        hc_metrics = metrics

print()
print('='*60)
print('Comparison vs Epic Deterioration Index')
print('='*60)

# Use baselines already calculated in metrics
baselines = hc_metrics.get('vs_baselines', {})
print()
print('HyperCore HIGH_CONFIDENCE vs Baselines:')
for baseline_name, baseline in baselines.items():
    print(f'  vs {baseline_name}:')
    print(f'    Sens: {baseline["sensitivity"]*100:.0f}% vs our {hc_metrics["sensitivity"]*100:.0f}%')
    print(f'    Spec: {baseline["specificity"]*100:.0f}% vs our {hc_metrics["specificity"]*100:.0f}%')
    print(f'    PPV@5%: {baseline["ppv_at_5_percent"]*100:.1f}% vs our {hc_metrics["ppv_at_5_percent"]*100:.1f}%')
    print(f'    Advantage: {baseline.get("hypercore_advantage", "N/A")}')

# Save detailed results
print()
print('='*60)
print('Detailed 24-Endpoint Analysis Sample')
print('='*60)

# Analyze a positive case in detail
positive_cases = df[df['event_in_12h'] == 1]
if len(positive_cases) > 0:
    sample_patient = positive_cases.iloc[0].dropna().to_dict()
    print(f'\nSample positive case (patient {sample_patient.get("patient_id", "unknown")}):')

    analysis = cross_loop.analyze_patient(sample_patient, mode='balanced')

    print(f'  Endpoints analyzed: {analysis.endpoints_analyzed}')
    print(f'  Endpoints with data: {analysis.endpoints_with_data}')
    print(f'  Endpoints alerting: {analysis.endpoints_alerting}')
    print(f'  Convergence score: {analysis.convergence_score:.2f}')
    print(f'  Convergence type: {analysis.convergence_type}')
    print(f'  Multi-system failure: {analysis.multi_system_failure}')
    print(f'  Detected pathways: {[p["pathway_name"] for p in analysis.detected_pathways[:3]]}')
    print(f'  Urgency: {analysis.urgency}')
    print(f'  Data completeness: {analysis.data_completeness:.1%}')

    if analysis.unknown_pattern:
        print(f'  Unknown pattern detected: {analysis.unknown_pattern.get("unexplained_endpoints", [])}')

    print(f'\n  Cross-domain insights:')
    for insight in analysis.cross_domain_insights[:5]:
        print(f'    - {insight}')

    print(f'\n  Recommended actions:')
    for action in analysis.recommended_actions[:3]:
        print(f'    - {action}')

print()
print('='*60)
print('VALIDATION COMPLETE')
print('='*60)
