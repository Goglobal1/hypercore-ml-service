"""
Layer 7: Confidence + Explainability
Generates explanations and confidence assessments.
"""

from typing import Dict, List, Any


class ExplainabilityEngine:
    """
    Layer 7: Generate explanations and confidence assessments.
    """

    def generate_explanation(self, result: Dict) -> Dict:
        """
        Generate comprehensive explanation for all findings.
        """

        explanation = {
            'summary': self._generate_summary(result),
            'confidence_assessment': self._assess_confidence(result),
            'evidence_hierarchy': self._build_evidence_hierarchy(result),
            'missing_data_impact': self._assess_missing_data(result),
            'differential_considerations': self._generate_differential(result),
            'limitations': self._note_limitations(result)
        }

        return explanation

    def _generate_summary(self, result: Dict) -> str:
        """Generate overall summary."""

        diseases = result.get('diseases', [])
        anomalies = result.get('anomalies', [])
        convergence = result.get('convergence', {})

        parts = []

        # Disease summary
        if diseases:
            high_conf = [d for d in diseases if d.get('confidence', 0) >= 0.7]
            moderate_conf = [d for d in diseases if 0.4 <= d.get('confidence', 0) < 0.7]

            if high_conf:
                names = [d['disease_name'] for d in high_conf]
                parts.append(f"High-confidence findings: {', '.join(names)}")

            if moderate_conf:
                names = [d['disease_name'] for d in moderate_conf]
                parts.append(f"Possible conditions: {', '.join(names)}")

        # Anomaly summary
        if anomalies:
            total_unexplained = sum(a.get('marker_count', 0) for a in anomalies)
            systems = list(set(a.get('organ_system') for a in anomalies))
            parts.append(f"{total_unexplained} unexplained abnormalities in {len(systems)} system(s)")

        # Convergence summary
        if convergence.get('type') not in ['none', None]:
            parts.append(f"Multi-system involvement: {convergence.get('type', 'moderate')}")

        if not parts:
            return "No significant abnormalities detected"

        return ". ".join(parts)

    def _assess_confidence(self, result: Dict) -> Dict:
        """
        Categorize confidence levels.
        """
        diseases = result.get('diseases', [])

        return {
            'confirmed_patterns': [d for d in diseases if d.get('confidence', 0) >= 0.8],
            'probable_patterns': [d for d in diseases if 0.6 <= d.get('confidence', 0) < 0.8],
            'possible_patterns': [d for d in diseases if 0.4 <= d.get('confidence', 0) < 0.6],
            'weak_signals': [d for d in diseases if 0.2 <= d.get('confidence', 0) < 0.4],
            'unclassified_abnormalities': result.get('anomalies', [])
        }

    def _build_evidence_hierarchy(self, result: Dict) -> Dict:
        """
        Separate measured vs inferred vs predicted evidence.
        """
        measured = []
        derived = []
        inferred = []

        # From axis scores
        for axis_name, axis_data in result.get('axis_scores', {}).items():
            for marker in axis_data.get('abnormal_markers', []):
                measured.append({
                    'type': 'measured',
                    'marker': marker.get('marker'),
                    'value': marker.get('value'),
                    'status': marker.get('status'),
                    'system': axis_name
                })

        # From diseases (inferred from patterns)
        for disease in result.get('diseases', []):
            inferred.append({
                'type': 'inferred',
                'finding': disease.get('disease_name'),
                'confidence': disease.get('confidence'),
                'evidence_count': len(disease.get('evidence', []))
            })

        return {
            'directly_measured': measured,
            'derived_calculated': derived,
            'inferred_from_patterns': inferred
        }

    def _assess_missing_data(self, result: Dict) -> Dict:
        """
        Assess impact of missing data on conclusions.
        """
        total_missing = 0
        critical_missing = []

        for axis_name, axis_data in result.get('axis_scores', {}).items():
            missing = axis_data.get('missing_markers', [])
            total_missing += len(missing)

            # Primary markers are critical
            if axis_data.get('confidence', 1.0) < 0.5:
                critical_missing.append({
                    'system': axis_name,
                    'missing': missing,
                    'impact': 'reduced_confidence'
                })

        impact = 'low'
        if total_missing > 10:
            impact = 'high'
        elif total_missing > 5:
            impact = 'moderate'

        return {
            'total_missing': total_missing,
            'critical_missing': critical_missing,
            'impact_level': impact,
            'recommendation': 'Consider additional testing' if impact != 'low' else 'Data sufficient for analysis'
        }

    def _generate_differential(self, result: Dict) -> List[str]:
        """
        Generate differential diagnosis considerations.
        """
        differentials = []

        diseases = result.get('diseases', [])
        anomalies = result.get('anomalies', [])

        # Add related conditions from detected diseases
        for disease in diseases:
            if disease.get('confidence', 0) >= 0.5:
                name = disease.get('disease_name', '')
                differentials.append(f"Consider {name} and related conditions")

        # Add considerations from anomalies
        for anomaly in anomalies:
            similar = anomaly.get('similar_conditions', [])
            if similar:
                differentials.append(f"Unexplained {anomaly.get('organ_system')} findings - consider {', '.join(similar)}")

        return differentials[:5]  # Limit to top 5

    def _note_limitations(self, result: Dict) -> List[str]:
        """
        Note limitations of the analysis.
        """
        limitations = []

        # Check data completeness
        for axis_name, axis_data in result.get('axis_scores', {}).items():
            if axis_data.get('confidence', 1.0) < 0.3:
                limitations.append(f"Limited data for {axis_name} axis - interpret with caution")

        # Note if no temporal data
        has_temporal = any(
            axis_data.get('trajectory') not in ['unknown', None]
            for axis_data in result.get('axis_scores', {}).values()
        )
        if not has_temporal:
            limitations.append("Single time point analysis - trends not available")

        # Standard disclaimer
        limitations.append("Clinical correlation required - this is a decision support tool")

        return limitations
