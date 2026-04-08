"""
Longitudinal Analysis Module
============================

Analyzes patient data across multiple visits to detect trends,
trajectories, and early warning signals.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# State ordering for trajectory comparison
STATE_ORDER = {'S0': 0, 'S1': 1, 'S2': 2, 'S3': 3}

# Biomarkers to track for trend analysis
KEY_BIOMARKERS = [
    'creatinine', 'glucose', 'potassium', 'sodium', 'hemoglobin',
    'wbc', 'platelets', 'bilirubin', 'alt', 'ast', 'albumin',
    'bun', 'lactate', 'troponin', 'procalcitonin'
]


def calculate_longitudinal_trends(visit_results: List[Dict]) -> Dict[str, Any]:
    """
    Calculate trends across multiple visits for the same patient.

    Args:
        visit_results: List of analysis results for each visit, ordered by date

    Returns:
        Dict with trajectory info, key changes, and early warning flag
    """
    trends = {
        'trajectory': 'stable',
        'key_changes': [],
        'early_warning': False,
        'state_progression': [],
        'risk_progression': []
    }

    if len(visit_results) < 2:
        return trends

    # Track state progression
    for vr in visit_results:
        state = vr.get('clinical_state', 'S0')
        risk = vr.get('risk_score', 0)
        trends['state_progression'].append(state)
        trends['risk_progression'].append(risk)

    # Compare first and last visits
    first = visit_results[0]
    last = visit_results[-1]

    # State trajectory
    first_state = STATE_ORDER.get(first.get('clinical_state', 'S0'), 0)
    last_state = STATE_ORDER.get(last.get('clinical_state', 'S0'), 0)

    if last_state > first_state:
        trends['trajectory'] = 'worsening'
        trends['early_warning'] = True
    elif last_state < first_state:
        trends['trajectory'] = 'improving'
    else:
        # Check if risk score changed significantly
        first_risk = first.get('risk_score', 0) or 0
        last_risk = last.get('risk_score', 0) or 0
        if last_risk > first_risk + 0.2:
            trends['trajectory'] = 'worsening'
            trends['early_warning'] = True
        elif last_risk < first_risk - 0.2:
            trends['trajectory'] = 'improving'
        else:
            trends['trajectory'] = 'stable'

    # Identify key changes in biomarkers
    for marker in KEY_BIOMARKERS:
        first_val = first.get(marker)
        last_val = last.get(marker)

        if first_val is not None and last_val is not None:
            try:
                first_val = float(first_val)
                last_val = float(last_val)
                if first_val != 0:
                    pct_change = ((last_val - first_val) / first_val) * 100
                    if abs(pct_change) > 20:  # Significant change
                        change_info = {
                            'marker': marker,
                            'first_value': round(first_val, 2),
                            'last_value': round(last_val, 2),
                            'percent_change': round(pct_change, 1),
                            'direction': 'increased' if pct_change > 0 else 'decreased',
                            'visits_apart': len(visit_results) - 1
                        }

                        # Add clinical context
                        change_info['clinical_note'] = _get_clinical_note(marker, pct_change, last_val)

                        trends['key_changes'].append(change_info)
            except (ValueError, TypeError):
                continue

    # Sort changes by absolute percent change
    trends['key_changes'].sort(key=lambda x: abs(x.get('percent_change', 0)), reverse=True)

    return trends


def _get_clinical_note(marker: str, pct_change: float, current_value: float) -> str:
    """Generate clinical context note for a biomarker change."""
    notes = {
        'creatinine': {
            'up': 'Kidney function may be declining',
            'down': 'Kidney function improving'
        },
        'glucose': {
            'up': 'Trending toward hyperglycemia' if current_value > 100 else 'Glucose elevated',
            'down': 'Glucose improving' if current_value >= 70 else 'Risk of hypoglycemia'
        },
        'potassium': {
            'up': 'Risk of hyperkalemia' if current_value > 5.0 else 'Potassium rising',
            'down': 'Risk of hypokalemia' if current_value < 3.5 else 'Potassium decreasing'
        },
        'hemoglobin': {
            'up': 'Hemoglobin improving',
            'down': 'Progressive anemia'
        },
        'wbc': {
            'up': 'Possible infection or inflammation',
            'down': 'Resolving infection' if current_value > 4.0 else 'Risk of immunosuppression'
        },
        'platelets': {
            'up': 'Platelet recovery',
            'down': 'Risk of thrombocytopenia'
        },
        'bilirubin': {
            'up': 'Liver function declining',
            'down': 'Liver function improving'
        },
        'lactate': {
            'up': 'Tissue hypoperfusion concern',
            'down': 'Perfusion improving'
        },
        'troponin': {
            'up': 'Cardiac stress indicator',
            'down': 'Cardiac markers improving'
        }
    }

    direction = 'up' if pct_change > 0 else 'down'
    marker_notes = notes.get(marker, {'up': f'{marker} increasing', 'down': f'{marker} decreasing'})
    return marker_notes.get(direction, f'{marker} changed')


def compare_visits(current: Dict, previous: Dict) -> Dict[str, Any]:
    """
    Compare two visits and return changes.

    Args:
        current: Current visit analysis result
        previous: Previous visit analysis result

    Returns:
        Dict with comparison details
    """
    comparison = {
        'changes': {},
        'state_change': None,
        'risk_change': None,
        'new_conditions': [],
        'resolved_conditions': []
    }

    # State change
    current_state = current.get('clinical_state', 'S0')
    previous_state = previous.get('clinical_state', 'S0')
    if current_state != previous_state:
        comparison['state_change'] = {
            'from': previous_state,
            'to': current_state,
            'direction': 'worsening' if STATE_ORDER.get(current_state, 0) > STATE_ORDER.get(previous_state, 0) else 'improving'
        }

    # Risk change
    current_risk = current.get('risk_score', 0) or 0
    previous_risk = previous.get('risk_score', 0) or 0
    comparison['risk_change'] = {
        'from': round(previous_risk, 2),
        'to': round(current_risk, 2),
        'delta': round(current_risk - previous_risk, 2)
    }

    # Biomarker changes
    for marker in KEY_BIOMARKERS:
        current_val = current.get(marker)
        previous_val = previous.get(marker)

        if current_val is not None and previous_val is not None:
            try:
                current_val = float(current_val)
                previous_val = float(previous_val)
                if previous_val != 0:
                    pct_change = ((current_val - previous_val) / previous_val) * 100
                    comparison['changes'][marker] = {
                        'previous': round(previous_val, 2),
                        'current': round(current_val, 2),
                        'change': f"{'+' if pct_change > 0 else ''}{round(pct_change, 1)}%"
                    }
            except (ValueError, TypeError):
                continue

    # Condition changes
    current_conditions = {c.get('disease', '') for c in current.get('conditions', [])}
    previous_conditions = {c.get('disease', '') for c in previous.get('conditions', [])}

    comparison['new_conditions'] = list(current_conditions - previous_conditions)
    comparison['resolved_conditions'] = list(previous_conditions - current_conditions)

    return comparison


def analyze_with_longitudinal_context(
    patient_results: List[Dict],
    original_patient_col: str = 'original_patient_id'
) -> Dict[str, Any]:
    """
    Analyze patients with awareness of their longitudinal history.

    Args:
        patient_results: List of individual patient/visit analysis results
        original_patient_col: Column name containing original patient ID

    Returns:
        Enhanced results including per-visit analysis, cross-visit trends,
        and patient-level summaries
    """
    results = {
        'visits': patient_results,
        'patient_summaries': {},
        'longitudinal_insights': [],
        'aggregate_trajectory': {
            'improving': 0,
            'stable': 0,
            'worsening': 0,
            'single_visit': 0
        }
    }

    # Group results by original patient
    patient_visits = {}
    for pr in patient_results:
        pid = pr.get(original_patient_col) or pr.get('patient_id', 'unknown')
        if pid not in patient_visits:
            patient_visits[pid] = []
        patient_visits[pid].append(pr)

    # Process each patient
    for patient_id, visits in patient_visits.items():
        # Sort by visit number or date
        visits_sorted = sorted(visits, key=lambda x: (
            x.get('visit_number', 0),
            str(x.get('date', ''))
        ))

        # Calculate cross-visit trends
        if len(visits_sorted) > 1:
            trends = calculate_longitudinal_trends(visits_sorted)

            # Add compared_to_previous for each visit
            for i, visit in enumerate(visits_sorted):
                if i > 0:
                    visit['compared_to_previous'] = compare_visits(visit, visits_sorted[i-1])

            results['longitudinal_insights'].append({
                'patient_id': patient_id,
                'visit_count': len(visits_sorted),
                'date_range': f"{visits_sorted[0].get('date', 'N/A')} to {visits_sorted[-1].get('date', 'N/A')}",
                'trajectory': trends['trajectory'],
                'current_state': visits_sorted[-1].get('clinical_state', 'S0'),
                'worst_state': max(
                    (v.get('clinical_state', 'S0') for v in visits_sorted),
                    key=lambda s: STATE_ORDER.get(s, 0)
                ),
                'key_changes': trends['key_changes'][:5],  # Top 5 changes
                'early_warning': trends['early_warning'],
                'state_progression': trends['state_progression'],
                'recommendation': _get_trajectory_recommendation(trends)
            })

            results['aggregate_trajectory'][trends['trajectory']] += 1
        else:
            results['aggregate_trajectory']['single_visit'] += 1

        # Patient-level summary
        results['patient_summaries'][patient_id] = {
            'total_visits': len(visits_sorted),
            'date_range': f"{visits_sorted[0].get('date', 'N/A')} to {visits_sorted[-1].get('date', 'N/A')}" if len(visits_sorted) > 1 else visits_sorted[0].get('date', 'N/A'),
            'worst_state': max(
                (v.get('clinical_state', 'S0') for v in visits_sorted),
                key=lambda s: STATE_ORDER.get(s, 0)
            ),
            'current_state': visits_sorted[-1].get('clinical_state', 'S0'),
            'trajectory': (
                calculate_longitudinal_trends(visits_sorted)['trajectory']
                if len(visits_sorted) > 1 else 'single_visit'
            )
        }

    return results


def _get_trajectory_recommendation(trends: Dict) -> str:
    """Generate a recommendation based on trajectory analysis."""
    if trends['early_warning']:
        if trends['trajectory'] == 'worsening':
            changes = trends.get('key_changes', [])
            if changes:
                top_marker = changes[0]['marker']
                return f"Urgent: Patient deteriorating. Monitor {top_marker} closely."
            return "Urgent: Patient condition worsening. Schedule immediate review."
        return "Early warning: Consider proactive intervention."

    if trends['trajectory'] == 'improving':
        return "Positive trend: Continue current treatment plan."

    return "Stable: Continue routine monitoring."


def generate_longitudinal_summary(results: Dict) -> Dict[str, Any]:
    """
    Generate a high-level summary of longitudinal patterns across all patients.

    Args:
        results: Output from analyze_with_longitudinal_context

    Returns:
        Summary dict with aggregate statistics and insights
    """
    insights = results.get('longitudinal_insights', [])
    summaries = results.get('patient_summaries', {})
    trajectory = results.get('aggregate_trajectory', {})

    total_patients = len(summaries)
    patients_with_multiple = sum(1 for s in summaries.values() if s['total_visits'] > 1)

    early_warnings = sum(1 for i in insights if i.get('early_warning'))
    worsening = trajectory.get('worsening', 0)

    # Find most common concerning changes
    all_changes = []
    for insight in insights:
        all_changes.extend(insight.get('key_changes', []))

    marker_concerns = {}
    for change in all_changes:
        marker = change['marker']
        if change['direction'] == 'increased' and change['percent_change'] > 0:
            # Only track increases that are typically concerning
            if marker in ['creatinine', 'glucose', 'bilirubin', 'lactate', 'troponin', 'wbc']:
                marker_concerns[marker] = marker_concerns.get(marker, 0) + 1
        elif change['direction'] == 'decreased':
            # Track decreases that are concerning
            if marker in ['hemoglobin', 'platelets', 'albumin']:
                marker_concerns[marker] = marker_concerns.get(marker, 0) + 1

    top_concerns = sorted(marker_concerns.items(), key=lambda x: -x[1])[:5]

    return {
        'total_patients': total_patients,
        'patients_with_longitudinal_data': patients_with_multiple,
        'trajectory_distribution': trajectory,
        'early_warnings_count': early_warnings,
        'patients_worsening': worsening,
        'top_biomarker_concerns': [
            {'marker': m, 'patient_count': c} for m, c in top_concerns
        ],
        'action_required': early_warnings > 0 or worsening > 0,
        'summary_text': _generate_summary_text(total_patients, worsening, early_warnings, top_concerns)
    }


def _generate_summary_text(total: int, worsening: int, warnings: int, concerns: List) -> str:
    """Generate human-readable summary text."""
    parts = []

    if worsening > 0:
        pct = round(worsening / total * 100, 1) if total > 0 else 0
        parts.append(f"{worsening} patients ({pct}%) showing worsening trajectory")

    if warnings > 0:
        parts.append(f"{warnings} early warning signals detected")

    if concerns:
        top = concerns[0]
        parts.append(f"Most common concern: {top[0]} ({top[1]} patients)")

    if not parts:
        return "Patient population stable across visits."

    return ". ".join(parts) + "."
