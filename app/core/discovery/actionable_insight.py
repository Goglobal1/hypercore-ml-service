"""
Actionable Insight Layer
========================
Integrates CSE, Utility Engine, and generates Handler-aligned actionable fields.

This layer is called AFTER analysis to add:
1. Clinical State (CSE) - S0/S1/S2/S3 state mapping
2. Utility Score - FIRE/SUPPRESS/DELAY decision
3. Immediate Actions - Verb + target + reason
4. Abnormal Values - With reference ranges and trends
5. Convergence Explanation - Clinical significance
6. Time to Harm - Intervention window
7. Conditions - Disease matches with evidence
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class ClinicalState(Enum):
    S0_STABLE = "S0"
    S1_WATCH = "S1"
    S2_ESCALATING = "S2"
    S3_CRITICAL = "S3"


class EmissionDecision(Enum):
    FIRE = "fire"
    SUPPRESS = "suppress"
    DELAY = "delay"
    DOWNGRADE = "downgrade"


@dataclass
class ActionableInsight:
    """Complete actionable insight for a patient analysis."""
    # CSE Fields
    clinical_state: str
    state_label: str
    previous_state: Optional[str]
    state_changed: bool

    # Utility Fields
    utility_score: float
    decision: str
    utility_components: Dict[str, float]

    # Actionable Fields
    immediate_actions: List[Dict]
    abnormal_values: List[Dict]
    convergence: Dict[str, Any]
    time_to_harm: Dict[str, Any]
    conditions: List[Dict]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActionableInsightGenerator:
    """
    Generates actionable insights from analysis results.
    Wires together CSE, Utility Engine, and insight generation.
    """

    # State thresholds (default)
    STATE_THRESHOLDS = {
        'S0_upper': 0.30,
        'S1_upper': 0.55,
        'S2_upper': 0.80,
        'S3_upper': 1.00
    }

    # Utility weights
    UTILITY_WEIGHTS = {
        'information_gain': 0.30,
        'urgency_factor': 0.25,
        'actionability': 0.25,
        'redundancy_penalty': 0.15,
        'interruption_cost': 0.05
    }

    # Decision thresholds
    DECISION_THRESHOLDS = {
        'fire': 0.70,
        'conditional': 0.50,
        'delay': 0.30,
        'suppress': 0.0
    }

    def __init__(self):
        self.previous_states: Dict[str, str] = {}  # patient_id -> previous state

    def generate(
        self,
        patient_id: str,
        analysis_result: Dict[str, Any],
        risk_score: Optional[float] = None
    ) -> ActionableInsight:
        """
        Generate complete actionable insight from analysis result.

        Args:
            patient_id: Patient identifier
            analysis_result: Output from Discovery Engine or other analysis
            risk_score: Optional explicit risk score (otherwise derived)

        Returns:
            ActionableInsight with all Handler-aligned fields
        """
        # Extract or derive risk score
        if risk_score is None:
            risk_score = self._derive_risk_score(analysis_result)

        # 1. Evaluate Clinical State
        clinical_state, state_label = self._evaluate_clinical_state(risk_score)
        previous_state = self.previous_states.get(patient_id)
        state_changed = previous_state is not None and previous_state != clinical_state
        self.previous_states[patient_id] = clinical_state

        # 2. Calculate Utility Score
        utility_score, decision, utility_components = self._calculate_utility(
            analysis_result, clinical_state, state_changed
        )

        # 3. Generate Immediate Actions
        immediate_actions = self._generate_immediate_actions(analysis_result, clinical_state)

        # 4. Extract Abnormal Values
        abnormal_values = self._extract_abnormal_values(analysis_result)

        # 5. Build Convergence Explanation
        convergence = self._build_convergence_explanation(analysis_result)

        # 6. Estimate Time to Harm
        time_to_harm = self._estimate_time_to_harm(analysis_result, clinical_state)

        # 7. Build Conditions List
        conditions = self._build_conditions(analysis_result)

        return ActionableInsight(
            clinical_state=clinical_state,
            state_label=state_label,
            previous_state=previous_state,
            state_changed=state_changed,
            utility_score=utility_score,
            decision=decision,
            utility_components=utility_components,
            immediate_actions=immediate_actions,
            abnormal_values=abnormal_values,
            convergence=convergence,
            time_to_harm=time_to_harm,
            conditions=conditions
        )

    def _derive_risk_score(self, analysis_result: Dict) -> float:
        """Derive risk score from analysis result."""
        # Try various sources
        if 'risk_score' in analysis_result:
            return float(analysis_result['risk_score'])

        summary = analysis_result.get('summary', {})
        if 'overall_risk' in summary:
            risk_map = {'critical': 0.9, 'high': 0.7, 'moderate': 0.5, 'watch': 0.35, 'low': 0.15}
            return risk_map.get(summary['overall_risk'], 0.5)

        # From convergence
        conv = analysis_result.get('convergence', {})
        if 'convergence_score' in conv:
            return min(1.0, conv['convergence_score'] / 100)

        # From aggregate
        agg = analysis_result.get('aggregate', {})
        if 'risk_percentages' in agg:
            pct = agg['risk_percentages']
            return (pct.get('critical', 0) * 0.9 + pct.get('high', 0) * 0.7 +
                    pct.get('moderate', 0) * 0.5 + pct.get('low', 0) * 0.2) / 100

        return 0.5  # Default

    def _evaluate_clinical_state(self, risk_score: float) -> Tuple[str, str]:
        """Map risk score to clinical state."""
        if risk_score >= self.STATE_THRESHOLDS['S2_upper']:
            return 'S3', 'CRITICAL'
        elif risk_score >= self.STATE_THRESHOLDS['S1_upper']:
            return 'S2', 'ESCALATING'
        elif risk_score >= self.STATE_THRESHOLDS['S0_upper']:
            return 'S1', 'WATCH'
        else:
            return 'S0', 'STABLE'

    def _calculate_utility(
        self,
        analysis_result: Dict,
        clinical_state: str,
        state_changed: bool
    ) -> Tuple[float, str, Dict[str, float]]:
        """Calculate utility score and emission decision."""
        # Information Gain - higher if state changed or critical
        info_gain = 0.5
        if state_changed:
            info_gain = 0.9
        elif clinical_state in ['S2', 'S3']:
            info_gain = 0.8
        elif clinical_state == 'S1':
            info_gain = 0.6

        # Urgency Factor - based on state
        urgency_map = {'S3': 1.0, 'S2': 0.8, 'S1': 0.5, 'S0': 0.2}
        urgency = urgency_map.get(clinical_state, 0.5)

        # Actionability - based on recommendations
        recommendations = analysis_result.get('recommendations', [])
        actionability = min(1.0, 0.3 + len(recommendations) * 0.1)

        # Redundancy Penalty - lower if novel
        redundancy = 0.1 if state_changed else 0.4

        # Interruption Cost - based on state (critical = low cost to interrupt)
        interruption_map = {'S3': 0.1, 'S2': 0.2, 'S1': 0.4, 'S0': 0.6}
        interruption = interruption_map.get(clinical_state, 0.3)

        components = {
            'information_gain': info_gain,
            'urgency_factor': urgency,
            'actionability': actionability,
            'redundancy_penalty': redundancy,
            'interruption_cost': interruption
        }

        # Calculate weighted utility
        utility = (
            self.UTILITY_WEIGHTS['information_gain'] * info_gain +
            self.UTILITY_WEIGHTS['urgency_factor'] * urgency +
            self.UTILITY_WEIGHTS['actionability'] * actionability -
            self.UTILITY_WEIGHTS['redundancy_penalty'] * redundancy -
            self.UTILITY_WEIGHTS['interruption_cost'] * interruption
        )
        utility = max(0, min(1, utility))  # Clamp to [0,1]

        # Decision
        if utility >= self.DECISION_THRESHOLDS['fire']:
            decision = 'FIRE'
        elif utility >= self.DECISION_THRESHOLDS['conditional']:
            decision = 'CONDITIONAL'
        elif utility >= self.DECISION_THRESHOLDS['delay']:
            decision = 'DELAY'
        else:
            decision = 'SUPPRESS'

        return round(utility, 3), decision, {k: round(v, 3) for k, v in components.items()}

    def _generate_immediate_actions(
        self,
        analysis_result: Dict,
        clinical_state: str
    ) -> List[Dict]:
        """Generate immediate actions from analysis."""
        actions = []
        priority = 1

        # From abnormal endpoint results
        endpoint_results = analysis_result.get('endpoint_results', {})
        for endpoint, data in endpoint_results.items():
            if isinstance(data, dict):
                risk_level = data.get('risk_level', 'normal')
                abnormals = data.get('abnormal_values', [])

                if risk_level == 'critical' or len(abnormals) > 0:
                    for abnormal in abnormals[:2]:  # Top 2 per endpoint
                        col = abnormal.get('column', endpoint)
                        value = abnormal.get('value', 'N/A')
                        status = abnormal.get('status', 'abnormal')

                        verb = 'CHECK' if status == 'high' else 'MONITOR' if status == 'low' else 'REVIEW'
                        urgency = 'immediate' if clinical_state in ['S2', 'S3'] else 'soon'

                        actions.append({
                            'verb': verb,
                            'target': col,
                            'reason': f"Current {value}, status: {status}",
                            'priority': priority,
                            'urgency': urgency,
                            'endpoint': endpoint
                        })
                        priority += 1

        # From recommendations
        recommendations = analysis_result.get('recommendations', [])
        for rec in recommendations[:3]:  # Top 3
            actions.append({
                'verb': 'ACTION',
                'target': rec.get('category', 'general'),
                'reason': rec.get('action', rec.get('reason', '')),
                'priority': priority,
                'urgency': rec.get('urgency', 'moderate')
            })
            priority += 1

        # State-based actions
        if clinical_state == 'S3':
            actions.insert(0, {
                'verb': 'ALERT',
                'target': 'clinical_team',
                'reason': 'Critical state detected - immediate clinical review required',
                'priority': 0,
                'urgency': 'immediate'
            })
        elif clinical_state == 'S2':
            actions.insert(0, {
                'verb': 'ESCALATE',
                'target': 'attending_physician',
                'reason': 'Escalating condition - enhanced monitoring recommended',
                'priority': 0,
                'urgency': 'soon'
            })

        return actions[:10]  # Max 10 actions

    def _extract_abnormal_values(self, analysis_result: Dict) -> List[Dict]:
        """Extract abnormal values with reference ranges."""
        abnormals = []

        # From endpoint results
        endpoint_results = analysis_result.get('endpoint_results', {})
        for endpoint, data in endpoint_results.items():
            if isinstance(data, dict):
                for abnormal in data.get('abnormal_values', []):
                    abnormals.append({
                        'marker': abnormal.get('column', 'unknown'),
                        'current_value': abnormal.get('value'),
                        'unit': abnormal.get('unit', ''),
                        'reference_low': abnormal.get('reference_range', [None, None])[0] if isinstance(abnormal.get('reference_range'), (list, tuple)) else None,
                        'reference_high': abnormal.get('reference_range', [None, None])[1] if isinstance(abnormal.get('reference_range'), (list, tuple)) else None,
                        'status': abnormal.get('status', 'abnormal'),
                        'direction': 'up' if abnormal.get('status') == 'high' else 'down' if abnormal.get('status') == 'low' else 'unknown',
                        'percent_change': abnormal.get('percent_change'),
                        'trend': abnormal.get('trend'),
                        'endpoint': endpoint
                    })

        # From anomalies
        for anomaly in analysis_result.get('anomalies', [])[:10]:
            if anomaly.get('anomaly_type') == 'rapid_change':
                for affected in anomaly.get('affected_values', []):
                    abnormals.append({
                        'marker': affected.get('column', 'unknown'),
                        'current_value': affected.get('to'),
                        'unit': '',
                        'reference_low': None,
                        'reference_high': None,
                        'status': 'rapid_change',
                        'direction': 'up' if affected.get('change', 0) > 0 else 'down',
                        'percent_change': round(affected.get('pct_change', 0) * 100, 1),
                        'trend': 'rapid_change'
                    })

        return abnormals[:20]  # Max 20

    def _build_convergence_explanation(self, analysis_result: Dict) -> Dict[str, Any]:
        """Build convergence with clinical explanation."""
        conv = analysis_result.get('convergence', {})
        systems = conv.get('systems_involved', [])
        conv_type = conv.get('convergence_type', 'none')
        score = conv.get('convergence_score', 0)

        # Generate clinical explanation
        if conv_type == 'none' or not systems:
            explanation = "No significant multi-system involvement detected."
            clinical_significance = "Single-system monitoring appropriate."
        elif len(systems) == 1:
            explanation = f"Primary involvement of {systems[0]} system."
            clinical_significance = f"Focus monitoring on {systems[0]} markers."
        elif len(systems) == 2:
            explanation = f"Convergent deterioration in {systems[0]} and {systems[1]} systems suggests interconnected pathophysiology."
            clinical_significance = f"Requires coordinated {systems[0]}-{systems[1]} intervention strategy."
        else:
            explanation = f"Multi-organ involvement across {', '.join(systems[:3])} systems indicates systemic deterioration pattern."
            clinical_significance = "Requires multi-disciplinary team coordination and aggressive intervention."

        # Add severity-specific guidance
        if conv_type == 'critical' or score >= 80:
            clinical_significance += " CRITICAL: Immediate escalation recommended."
        elif conv_type == 'severe' or score >= 60:
            clinical_significance += " Consider ICU-level monitoring."

        return {
            'convergence_type': conv_type,
            'convergence_score': score,
            'systems_involved': systems,
            'explanation': explanation,
            'clinical_significance': clinical_significance,
            'system_count': len(systems),
            'velocity': conv.get('velocity', 'stable')
        }

    def _estimate_time_to_harm(
        self,
        analysis_result: Dict,
        clinical_state: str
    ) -> Dict[str, Any]:
        """Estimate time to harm based on state and patterns."""
        conv = analysis_result.get('convergence', {})
        existing_tth = conv.get('estimated_time_to_harm')

        # State-based default ranges
        state_tth = {
            'S3': {'min': 2, 'max': 12},
            'S2': {'min': 12, 'max': 48},
            'S1': {'min': 48, 'max': 168},
            'S0': {'min': 168, 'max': 336}
        }

        base_tth = state_tth.get(clinical_state, {'min': 48, 'max': 168})

        # Adjust for convergence
        if conv.get('convergence_type') in ['critical', 'severe']:
            base_tth['min'] = max(2, base_tth['min'] // 2)
            base_tth['max'] = base_tth['max'] // 2

        # Build consequences
        if_untreated = []
        summary = analysis_result.get('summary', {})

        if clinical_state == 'S3':
            if_untreated = ['Organ failure risk', 'ICU admission likely', 'Mortality risk elevated']
        elif clinical_state == 'S2':
            if_untreated = ['Condition progression', 'May require intensive intervention', 'Risk of escalation to critical']
        elif clinical_state == 'S1':
            if_untreated = ['Close monitoring required', 'May progress if unaddressed']

        # Add disease-specific consequences
        for disease in analysis_result.get('identified_diseases', [])[:2]:
            disease_name = disease.get('disease_name', disease.get('disease', ''))
            if disease_name:
                if_untreated.append(f"{disease_name} progression")

        # Intervention window
        if base_tth['min'] <= 6:
            intervention_window = "Immediate action required (within 6 hours)"
        elif base_tth['min'] <= 24:
            intervention_window = "Action within 24 hours recommended"
        elif base_tth['min'] <= 48:
            intervention_window = "Address within 48 hours for optimal outcome"
        else:
            intervention_window = "Monitor and reassess within 72 hours"

        return {
            'min_hours': base_tth['min'],
            'max_hours': base_tth['max'],
            'if_untreated': if_untreated[:5],
            'intervention_window': intervention_window,
            'confidence': 'high' if clinical_state in ['S2', 'S3'] else 'moderate'
        }

    def _build_conditions(self, analysis_result: Dict) -> List[Dict]:
        """Build conditions list with evidence."""
        conditions = []

        for disease in analysis_result.get('identified_diseases', []):
            disease_name = disease.get('disease_name', disease.get('disease', 'Unknown'))
            confidence = disease.get('confidence', disease.get('match_confidence', 'moderate'))

            # Map confidence to numeric
            conf_map = {'high': 0.85, 'moderate': 0.65, 'low': 0.45}
            conf_score = conf_map.get(confidence, 0.65) if isinstance(confidence, str) else confidence

            # Build evidence
            evidence = []
            matched = disease.get('matched_indicators', [])
            if matched:
                evidence.extend([f"{m} detected" for m in matched[:3]])

            endpoints = disease.get('affected_endpoints', [])
            if endpoints:
                evidence.append(f"Affects: {', '.join(endpoints[:3])}")

            conditions.append({
                'disease': disease_name,
                'confidence': conf_score,
                'confidence_label': confidence if isinstance(confidence, str) else 'moderate',
                'stage': disease.get('stage', disease.get('severity', '')),
                'evidence': evidence,
                'icd10_codes': disease.get('icd10_codes', [])
            })

        return conditions[:10]  # Max 10


# Singleton
_insight_generator = None


def get_insight_generator() -> ActionableInsightGenerator:
    """Get singleton insight generator."""
    global _insight_generator
    if _insight_generator is None:
        _insight_generator = ActionableInsightGenerator()
    return _insight_generator


def enrich_with_actionable_insights(
    patient_id: str,
    analysis_result: Dict[str, Any],
    risk_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Enrich analysis result with actionable insights.

    This is the main entry point for adding CSE, Utility, and Actionable fields
    to any analysis result.
    """
    generator = get_insight_generator()
    insight = generator.generate(patient_id, analysis_result, risk_score)

    # Merge insight into analysis result
    enriched = analysis_result.copy()
    enriched.update({
        'clinical_state': insight.clinical_state,
        'state_label': insight.state_label,
        'previous_state': insight.previous_state,
        'state_changed': insight.state_changed,
        'utility_score': insight.utility_score,
        'decision': insight.decision,
        'utility_components': insight.utility_components,
        'immediate_actions': insight.immediate_actions,
        'abnormal_values': insight.abnormal_values,
        'convergence': insight.convergence,
        'time_to_harm': insight.time_to_harm,
        'conditions': insight.conditions
    })

    return enriched
