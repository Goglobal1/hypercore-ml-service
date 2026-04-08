"""
Layer 8: Recommendation / Next-Step
Generates actionable recommendations based on findings.
"""

from typing import Dict, List, Any


class RecommendationEngine:
    """
    Layer 8: Generate actionable recommendations.
    """

    def generate_recommendations(self, result: Dict) -> List[Dict]:
        """
        Generate prioritized recommendations.
        """

        recommendations = []

        # From convergence (highest priority if critical)
        convergence = result.get('convergence', {})
        if convergence.get('type') in ['severe', 'critical']:
            rec = {
                'priority': 'immediate',
                'priority_order': 0,
                'type': 'escalation',
                'category': 'convergence',
                'condition': 'Multi-system involvement',
                'action': 'Immediate clinical review required',
                'reason': convergence.get('explanation', 'Multiple systems affected'),
                'urgency': convergence.get('urgency', 'urgent'),
                'tests': [],
                'rationale': f"Convergence type: {convergence.get('type')}, {len(convergence.get('systems_involved', []))} systems"
            }
            recommendations.append(rec)

        # From diseases
        for disease in result.get('diseases', []):
            priority = self._calculate_priority(disease)
            urgency = self._determine_urgency(disease)

            rec = {
                'priority': priority,
                'priority_order': {'immediate': 0, 'high': 1, 'moderate': 2, 'low': 3}.get(priority, 4),
                'type': 'followup',
                'category': 'disease',
                'condition': disease.get('disease_name', 'Unknown'),
                'action': f"Evaluate for {disease.get('disease_name', 'condition')}",
                'reason': f"Confidence: {disease.get('confidence', 0):.0%}",
                'urgency': urgency,
                'tests': disease.get('recommended_followup', []),
                'rationale': "; ".join(disease.get('evidence', [])[:3]),
                'icd10': disease.get('icd10'),
                'missing_data': disease.get('missing_data', [])
            }
            recommendations.append(rec)

        # From anomalies
        for anomaly in result.get('anomalies', []):
            priority = 'high' if anomaly.get('severity', 0) >= 0.7 else 'moderate'

            rec = {
                'priority': priority,
                'priority_order': {'immediate': 0, 'high': 1, 'moderate': 2, 'low': 3}.get(priority, 4),
                'type': 'investigation',
                'category': 'anomaly',
                'condition': f"Unexplained {anomaly.get('organ_system', 'system')} abnormality",
                'action': 'Investigate unexplained abnormality',
                'reason': anomaly.get('description', 'Novel pattern detected'),
                'urgency': 'moderate',
                'tests': anomaly.get('recommended_investigation', []),
                'rationale': f"Novelty score: {anomaly.get('novelty_score', 0):.0%}"
            }
            recommendations.append(rec)

        # Sort by priority
        recommendations.sort(key=lambda x: x.get('priority_order', 4))

        # Add sequence numbers
        for i, rec in enumerate(recommendations):
            rec['sequence'] = i + 1

        return recommendations

    def _calculate_priority(self, disease: Dict) -> str:
        """Calculate recommendation priority."""

        confidence = disease.get('confidence', 0)
        severity = disease.get('severity')
        category = disease.get('category', '')

        # Critical conditions
        if category in ['cardiac', 'sepsis', 'respiratory']:
            if confidence >= 0.7:
                return 'immediate'
            elif confidence >= 0.5:
                return 'high'

        # Severity-based
        if severity in ['severe', 'critical']:
            return 'high'

        # Confidence-based
        if confidence >= 0.7:
            return 'high'
        elif confidence >= 0.5:
            return 'moderate'
        else:
            return 'low'

    def _determine_urgency(self, disease: Dict) -> str:
        """Determine urgency level."""

        category = disease.get('category', '')
        severity = disease.get('severity')
        confidence = disease.get('confidence', 0)

        # Acute conditions
        acute_categories = ['cardiac', 'sepsis', 'respiratory', 'renal']
        if category in acute_categories and confidence >= 0.6:
            if severity in ['severe', 'critical']:
                return 'stat'
            return 'urgent'

        # Moderate urgency
        if confidence >= 0.6:
            return 'soon'

        return 'routine'

    def get_immediate_actions(self, recommendations: List[Dict]) -> List[Dict]:
        """Extract only immediate priority actions."""
        return [r for r in recommendations if r.get('priority') == 'immediate']

    def format_for_display(self, recommendations: List[Dict]) -> List[Dict]:
        """Format recommendations for clinical display."""
        formatted = []

        for rec in recommendations:
            formatted.append({
                'priority': rec.get('priority', 'moderate').upper(),
                'action': rec.get('action', ''),
                'reason': rec.get('reason', ''),
                'tests': rec.get('tests', []),
                'urgency': rec.get('urgency', 'routine')
            })

        return formatted
