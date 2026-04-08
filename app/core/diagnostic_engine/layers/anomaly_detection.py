"""
Layer 5: Unknown/Anomaly Detection
Detects abnormal patterns that don't fit any known disease class.
"""

from typing import Dict, List, Any


class AnomalyDetector:
    """
    Layer 5: Detect unknown/novel patterns.

    This uses different logic than known disease detection:
    - Finds unexplained abnormalities
    - Groups them by organ system
    - Calculates novelty scores
    - Suggests investigations
    """

    def __init__(self, known_disease_markers: List[str] = None):
        self.known_disease_markers = known_disease_markers or []

        # Investigation suggestions by system
        self.investigation_suggestions = {
            'metabolic': ['hba1c', 'fasting_insulin', 'lipid_panel', 'ogtt'],
            'renal': ['urine_albumin', 'cystatin_c', 'renal_ultrasound', 'nephrology_consult'],
            'hepatic': ['hepatitis_panel', 'liver_ultrasound', 'ggt', 'mrcp'],
            'inflammatory': ['il6', 'tnf_alpha', 'autoimmune_panel', 'infectious_workup'],
            'cardiac': ['echocardiogram', 'stress_test', 'coronary_ct', 'cardiology_consult'],
            'hematologic': ['peripheral_smear', 'reticulocytes', 'iron_studies', 'hematology_consult'],
            'endocrine': ['thyroid_antibodies', 'cortisol_stimulation', 'endocrine_panel'],
            'coagulation': ['mixing_studies', 'factor_levels', 'hematology_consult'],
            'respiratory': ['pulmonary_function', 'chest_ct', 'pulmonology_consult'],
            'perfusion': ['lactate_clearance', 'echocardiogram', 'volume_assessment'],
            'vitals': ['continuous_monitoring', 'comprehensive_workup']
        }

    def detect_anomalies(self, features: Dict, axis_scores: Dict, known_matches: List[Dict]) -> List[Dict]:
        """
        Find abnormal patterns that don't match known diseases.
        """

        anomalies = []

        # Get all abnormal markers from axis scoring
        abnormal_markers = self._get_all_abnormal_markers(axis_scores)

        # Get markers already explained by known disease matches
        explained_markers = set()
        for match in known_matches:
            for evidence in match.get('evidence', []):
                # Extract marker name from evidence string (format: "marker: value (condition)")
                if ':' in evidence:
                    marker = evidence.split(':')[0].strip()
                    explained_markers.add(marker)

        # Find unexplained abnormalities
        unexplained = [m for m in abnormal_markers if m['marker'] not in explained_markers]

        if len(unexplained) == 0:
            return anomalies

        # Group unexplained abnormalities by organ system
        by_system = {}
        for marker_info in unexplained:
            system = marker_info.get('organ_system', 'unknown')
            if system not in by_system:
                by_system[system] = []
            by_system[system].append(marker_info)

        # Create anomaly findings
        for system, markers in by_system.items():
            if len(markers) == 0:
                continue

            anomaly = {
                'type': 'unknown_abnormality',
                'classification': self._classify_anomaly(markers),
                'organ_system': system,
                'affected_markers': markers,
                'marker_count': len(markers),
                'severity': max(m.get('score', 0) for m in markers),
                'novelty_score': self._calculate_novelty(markers, known_matches),
                'similar_conditions': self._find_similar_conditions(markers, system),
                'recommended_investigation': self._suggest_investigation(system, markers),
                'evidence': [f"{m['marker']}: {m['value']} ({m['status']})" for m in markers],
                'description': self._generate_description(system, markers)
            }

            anomalies.append(anomaly)

        return anomalies

    def _classify_anomaly(self, markers: List[Dict]) -> str:
        """
        Classify the type of anomaly.
        """
        if len(markers) == 1:
            return 'isolated_abnormality'
        elif len(markers) == 2:
            return 'paired_abnormality'
        elif len(markers) <= 3:
            return 'abnormal_unclassified_state'
        else:
            return 'novel_syndrome_pattern'

    def _calculate_novelty(self, markers: List[Dict], known_matches: List[Dict]) -> float:
        """
        Calculate how novel/unusual this pattern is.
        Higher = more unusual, less like known diseases.
        """
        # If no known matches, higher novelty
        if len(known_matches) == 0:
            return 0.8

        # Base novelty on count of unexplained markers
        base_novelty = min(len(markers) * 0.15, 0.7)

        # Increase novelty for severe abnormalities
        severe_count = sum(1 for m in markers if m.get('score', 0) >= 0.7)
        severity_bonus = severe_count * 0.1

        return min(base_novelty + severity_bonus, 0.95)

    def _find_similar_conditions(self, markers: List[Dict], system: str) -> List[str]:
        """
        Find known conditions that might partially match this pattern.
        """
        similar = []

        marker_names = [m['marker'] for m in markers]

        # Common patterns by system
        patterns = {
            'metabolic': {
                'markers': ['glucose', 'hba1c', 'triglycerides'],
                'condition': 'metabolic_syndrome'
            },
            'renal': {
                'markers': ['creatinine', 'bun', 'potassium'],
                'condition': 'renal_dysfunction'
            },
            'hepatic': {
                'markers': ['ast', 'alt', 'bilirubin'],
                'condition': 'hepatic_dysfunction'
            },
            'inflammatory': {
                'markers': ['crp', 'wbc', 'procalcitonin'],
                'condition': 'systemic_inflammation'
            }
        }

        if system in patterns:
            pattern = patterns[system]
            overlap = len(set(marker_names) & set(pattern['markers']))
            if overlap >= 1:
                similar.append(f"partially_matches_{pattern['condition']}")

        return similar

    def _suggest_investigation(self, system: str, markers: List[Dict]) -> List[str]:
        """
        Suggest additional tests to clarify the anomaly.
        """
        base_suggestions = self.investigation_suggestions.get(
            system,
            ['comprehensive_metabolic_panel', 'specialist_referral']
        )

        # Add specific suggestions based on severity
        suggestions = base_suggestions.copy()

        max_severity = max(m.get('score', 0) for m in markers)
        if max_severity >= 0.7:
            suggestions.insert(0, 'urgent_specialist_review')

        return suggestions[:5]  # Limit to 5 suggestions

    def _generate_description(self, system: str, markers: List[Dict]) -> str:
        """Generate human-readable description of the anomaly."""
        marker_names = [m['marker'] for m in markers]

        if len(markers) == 1:
            return f"Isolated {markers[0]['status']} {markers[0]['marker']} in {system} system"
        else:
            return f"Unexplained {system} abnormalities involving {', '.join(marker_names)}"

    def _get_all_abnormal_markers(self, axis_scores: Dict) -> List[Dict]:
        """Extract all abnormal markers from axis scores."""
        abnormal = []

        for axis_name, axis_data in axis_scores.items():
            for marker_info in axis_data.get('abnormal_markers', []):
                # Add organ system info
                marker_copy = marker_info.copy()
                marker_copy['organ_system'] = axis_name
                abnormal.append(marker_copy)

        return abnormal

    def get_total_unexplained(self, anomalies: List[Dict]) -> int:
        """Get total count of unexplained abnormalities."""
        return sum(a.get('marker_count', 0) for a in anomalies)
