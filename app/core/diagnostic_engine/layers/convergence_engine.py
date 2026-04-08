"""
Layer 6: Cross-Axis Reasoning (Convergence)
Detects multi-system interactions and cascades.
"""

from typing import Dict, List, Any


class ConvergenceEngine:
    """
    Layer 6: Cross-axis reasoning for multi-system patterns.
    """

    def __init__(self):
        # Known interaction patterns between organ systems
        self.known_interactions = [
            {
                'axes': ['metabolic', 'inflammatory'],
                'pattern': 'metabolic_inflammation',
                'description': 'Inflammatory-metabolic coupling (insulin resistance driver)',
                'clinical_significance': 'high'
            },
            {
                'axes': ['renal', 'cardiac'],
                'pattern': 'cardiorenal_syndrome',
                'description': 'Cardiorenal syndrome (heart-kidney interaction)',
                'clinical_significance': 'high'
            },
            {
                'axes': ['hepatic', 'renal'],
                'pattern': 'hepatorenal_syndrome',
                'description': 'Hepatorenal syndrome (liver-kidney interaction)',
                'clinical_significance': 'critical'
            },
            {
                'axes': ['hepatic', 'metabolic'],
                'pattern': 'hepatic_metabolic',
                'description': 'Hepatic-metabolic dysfunction (fatty liver, metabolic syndrome)',
                'clinical_significance': 'moderate'
            },
            {
                'axes': ['inflammatory', 'hematologic'],
                'pattern': 'inflammatory_hematologic',
                'description': 'Systemic inflammatory response with hematologic involvement',
                'clinical_significance': 'high'
            },
            {
                'axes': ['respiratory', 'cardiac'],
                'pattern': 'cardiopulmonary',
                'description': 'Cardiopulmonary interaction',
                'clinical_significance': 'high'
            },
            {
                'axes': ['perfusion', 'renal'],
                'pattern': 'shock_aki',
                'description': 'Hypoperfusion-induced acute kidney injury',
                'clinical_significance': 'critical'
            },
            {
                'axes': ['inflammatory', 'perfusion'],
                'pattern': 'sepsis_cascade',
                'description': 'Sepsis-related multi-organ involvement',
                'clinical_significance': 'critical'
            },
            {
                'axes': ['endocrine', 'metabolic'],
                'pattern': 'endocrine_metabolic',
                'description': 'Endocrine-metabolic dysregulation',
                'clinical_significance': 'moderate'
            },
            {
                'axes': ['coagulation', 'hepatic'],
                'pattern': 'hepatic_coagulopathy',
                'description': 'Liver-related coagulation dysfunction',
                'clinical_significance': 'high'
            }
        ]

    def analyze_convergence(self, axis_scores: Dict, diseases: List[Dict], anomalies: List[Dict]) -> Dict:
        """
        Analyze interactions between organ systems.
        """

        # Find disturbed axes
        disturbed_axes = [
            axis_name for axis_name, score in axis_scores.items()
            if score.get('status') in ['borderline', 'abnormal', 'critical']
        ]

        if len(disturbed_axes) < 2:
            return {
                'type': 'none',
                'score': 0.0,
                'systems_involved': disturbed_axes,
                'system_count': len(disturbed_axes),
                'interactions': [],
                'explanation': 'No multi-system involvement detected',
                'urgency': 'routine',
                'clinical_significance': 'Low - isolated system involvement'
            }

        # Calculate convergence severity
        severity_scores = [
            axis_scores[axis].get('score', 0) for axis in disturbed_axes
        ]
        avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0

        # Identify interaction patterns
        interactions = self._identify_interactions(disturbed_axes, axis_scores)

        # Determine convergence type
        critical_axes = sum(1 for axis in disturbed_axes if axis_scores[axis].get('status') == 'critical')

        if critical_axes >= 2 or len(disturbed_axes) >= 5:
            conv_type = 'critical'
            urgency = 'immediate'
        elif critical_axes >= 1 or len(disturbed_axes) >= 4 or avg_severity >= 0.6:
            conv_type = 'severe'
            urgency = 'urgent'
        elif len(disturbed_axes) >= 3 or avg_severity >= 0.4:
            conv_type = 'moderate'
            urgency = 'high'
        else:
            conv_type = 'mild'
            urgency = 'moderate'

        # Calculate convergence score (0-100)
        convergence_score = avg_severity * 100
        if len(interactions) > 0:
            convergence_score += len(interactions) * 10
        convergence_score = min(convergence_score, 100)

        # Determine velocity based on temporal data
        velocity = self._assess_velocity(axis_scores)

        return {
            'type': conv_type,
            'convergence_type': conv_type,
            'score': avg_severity,
            'convergence_score': convergence_score,
            'systems_involved': disturbed_axes,
            'system_count': len(disturbed_axes),
            'interactions': interactions,
            'explanation': self._generate_explanation(disturbed_axes, interactions),
            'clinical_significance': self._get_clinical_significance(conv_type, interactions),
            'urgency': urgency,
            'velocity': velocity,
            'cascade_risk': self._assess_cascade_risk(disturbed_axes, interactions)
        }

    def _identify_interactions(self, axes: List[str], scores: Dict) -> List[Dict]:
        """Identify known interaction patterns between axes."""

        detected = []
        for interaction in self.known_interactions:
            required_axes = interaction['axes']
            if all(axis in axes for axis in required_axes):
                # Check severity of involved axes
                involved_scores = [scores[axis].get('score', 0) for axis in required_axes]
                interaction_severity = sum(involved_scores) / len(involved_scores)

                detected.append({
                    'pattern': interaction['pattern'],
                    'axes': required_axes,
                    'description': interaction['description'],
                    'clinical_significance': interaction['clinical_significance'],
                    'severity': interaction_severity
                })

        # Sort by severity
        detected.sort(key=lambda x: x['severity'], reverse=True)

        return detected

    def _generate_explanation(self, axes: List[str], interactions: List[Dict]) -> str:
        """Generate human-readable explanation."""

        if len(interactions) > 0:
            patterns = [i['description'] for i in interactions[:3]]
            return f"Multi-system involvement: {', '.join(axes)}. Detected patterns: {'; '.join(patterns)}"
        else:
            return f"Concurrent abnormalities in {', '.join(axes)} systems without recognized interaction pattern."

    def _get_clinical_significance(self, conv_type: str, interactions: List[Dict]) -> str:
        """Determine clinical significance."""

        if conv_type == 'critical':
            return 'Critical - immediate intervention required'
        elif conv_type == 'severe':
            return 'Severe - urgent clinical review needed'
        elif conv_type == 'moderate':
            if len(interactions) > 0:
                return f"Moderate - {interactions[0]['description']}"
            return 'Moderate - coordinated multi-system management recommended'
        else:
            return 'Mild - monitor for progression'

    def _assess_velocity(self, axis_scores: Dict) -> str:
        """Assess how fast the convergence is developing."""

        worsening_count = 0
        stable_count = 0

        for axis_name, score in axis_scores.items():
            trajectory = score.get('trajectory', 'unknown')
            if trajectory == 'worsening':
                worsening_count += 1
            elif trajectory == 'stable':
                stable_count += 1

        if worsening_count >= 2:
            return 'rapid'
        elif worsening_count == 1:
            return 'moderate'
        else:
            return 'slow'

    def _assess_cascade_risk(self, axes: List[str], interactions: List[Dict]) -> Dict:
        """Assess risk of cascade failure."""

        # High-risk combinations
        high_risk_combos = [
            {'cardiac', 'renal'},
            {'hepatic', 'renal'},
            {'perfusion', 'renal'},
            {'inflammatory', 'perfusion'}
        ]

        risk_level = 'low'
        risk_factors = []

        axes_set = set(axes)
        for combo in high_risk_combos:
            if combo.issubset(axes_set):
                risk_level = 'high'
                risk_factors.append(f"{'-'.join(combo)} interaction")

        if len(axes) >= 4:
            risk_level = 'high'
            risk_factors.append('multiple_system_involvement')

        critical_interactions = [i for i in interactions if i.get('clinical_significance') == 'critical']
        if critical_interactions:
            risk_level = 'critical'
            risk_factors.extend([i['pattern'] for i in critical_interactions])

        return {
            'level': risk_level,
            'factors': risk_factors,
            'description': f"Cascade risk: {risk_level}" + (f" due to {', '.join(risk_factors)}" if risk_factors else "")
        }
