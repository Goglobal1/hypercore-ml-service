"""
Layer 4: Known Disease Classification
Multi-label pattern matching against a disease ontology.
"""

from typing import Dict, List, Any, Optional
import json
import os


class DiseaseClassifier:
    """
    Layer 4: Multi-label disease classification.

    This is NOT a single-label classifier.
    It can output multiple concurrent conditions.
    """

    def __init__(self, disease_ontology: Dict = None, reference_ranges: Dict = None):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        if disease_ontology is None:
            with open(os.path.join(data_dir, 'disease_ontology.json'), 'r') as f:
                disease_ontology = json.load(f)

        if reference_ranges is None:
            with open(os.path.join(data_dir, 'reference_ranges.json'), 'r') as f:
                reference_ranges = json.load(f)

        self.diseases = disease_ontology.get('diseases', {})
        self.ranges = reference_ranges.get('ranges', {})

    def classify(self, features: Dict, axis_scores: Dict) -> List[Dict]:
        """
        Classify patient against all known disease patterns.
        Returns list of detected conditions with confidence.
        """

        detected = []

        for disease_id, disease_config in self.diseases.items():
            result = self._evaluate_disease(disease_id, disease_config, features, axis_scores)

            if result['detected']:
                detected.append(result)

        # Sort by confidence
        detected.sort(key=lambda x: x['confidence'], reverse=True)

        return detected

    def _evaluate_disease(self, disease_id: str, config: Dict, features: Dict, axis_scores: Dict) -> Dict:
        """
        Evaluate a single disease pattern against patient data.
        """

        result = {
            'disease_id': disease_id,
            'disease_name': config.get('name', disease_id),
            'icd10': config.get('icd10'),
            'category': config.get('category'),
            'detected': False,
            'confidence': 0.0,
            'confidence_label': 'none',
            'severity': None,
            'stage': None,
            'evidence': [],
            'missing_data': [],
            'exclusions_triggered': [],
            'organ_systems': config.get('organ_systems', []),
            'recommended_followup': config.get('recommended_followup', [])
        }

        # Get raw features
        raw_features = features.get('raw_features', features)
        temporal = features.get('temporal_features', {})

        # Check exclusions first
        for exclusion in config.get('exclusions', []):
            if self._check_condition(exclusion, raw_features):
                result['exclusions_triggered'].append(exclusion)
                return result  # Excluded, don't detect

        # Check primary patterns
        primary_matches = 0
        primary_patterns = config.get('patterns', {}).get('primary', [])

        for pattern in primary_patterns:
            marker = pattern.get('marker')
            if marker not in raw_features:
                result['missing_data'].append(marker)
                continue

            if self._check_condition(pattern, raw_features):
                primary_matches += 1
                value = self._get_value(raw_features, marker)
                condition_str = pattern.get('condition', 'pattern')
                if 'value' in pattern:
                    condition_str = f"{condition_str} {pattern['value']}"
                result['evidence'].append(f"{marker}: {value} ({condition_str})")

        # Check supportive patterns
        supportive_matches = 0
        supportive_patterns = config.get('patterns', {}).get('supportive', [])

        for pattern in supportive_patterns:
            marker = pattern.get('marker')
            if marker not in raw_features:
                continue

            if self._check_condition(pattern, raw_features):
                supportive_matches += 1
                value = self._get_value(raw_features, marker)
                result['evidence'].append(f"{marker}: {value} (supportive)")

        # Determine if detected
        require_any = config.get('require_any_primary', False)
        require_all = config.get('require_all_primary', False)

        if require_all:
            # All primary patterns must match
            if len(primary_patterns) > 0 and primary_matches == len(primary_patterns):
                result['detected'] = True
        elif require_any:
            # At least one primary pattern must match
            if primary_matches > 0:
                result['detected'] = True
        else:
            # Default: at least one primary match
            if primary_matches > 0:
                result['detected'] = True

        if not result['detected']:
            return result

        # Calculate confidence
        conf_factors = config.get('confidence_factors', {})
        confidence = conf_factors.get('base', 0.3)
        confidence += primary_matches * conf_factors.get('per_primary_match', 0.2)
        confidence += supportive_matches * conf_factors.get('per_supportive_match', 0.05)

        # Both primary bonus
        if primary_matches >= 2 and 'both_primary_bonus' in conf_factors:
            confidence += conf_factors['both_primary_bonus']

        # Temporal bonus if worsening
        for marker in [p.get('marker') for p in primary_patterns]:
            if marker in temporal:
                if temporal[marker].get('direction') == 'increasing':
                    confidence += conf_factors.get('temporal_bonus', 0.05)

        # Scale with count if specified
        if config.get('confidence_scales_with_count', False):
            total_matches = primary_matches + supportive_matches
            confidence = min(confidence * (1 + total_matches * 0.1), conf_factors.get('max', 0.95))

        result['confidence'] = min(confidence, conf_factors.get('max', 0.95))

        # Set confidence label
        if result['confidence'] >= 0.8:
            result['confidence_label'] = 'high'
        elif result['confidence'] >= 0.6:
            result['confidence_label'] = 'moderate'
        elif result['confidence'] >= 0.4:
            result['confidence_label'] = 'low'
        else:
            result['confidence_label'] = 'weak'

        # Determine severity
        severity_logic = config.get('severity_logic', {})
        result['severity'] = self._determine_severity(severity_logic, raw_features)

        # Determine stage if applicable
        stages = config.get('stages', {})
        result['stage'] = self._determine_stage(stages, raw_features)

        return result

    def _check_condition(self, pattern: Dict, features: Dict) -> bool:
        """Check if a pattern condition is met."""
        marker = pattern.get('marker')
        value = self._get_value(features, marker)

        if value is None:
            return False

        condition = pattern.get('condition')

        if condition == 'borderline':
            return self._is_in_range(marker, value, 'borderline')
        elif condition == 'abnormal':
            return self._is_abnormal(marker, value)
        elif condition == 'elevated':
            ref = self.ranges.get(marker, {}).get('normal', {})
            return value > ref.get('high', float('inf'))
        elif condition == 'low':
            ref = self.ranges.get(marker, {}).get('normal', {})
            return value < ref.get('low', 0)
        elif condition == '>=':
            return value >= pattern.get('value', 0)
        elif condition == '<=':
            return value <= pattern.get('value', float('inf'))
        elif condition == '<':
            return value < pattern.get('value', float('inf'))
        elif condition == '>':
            return value > pattern.get('value', 0)
        elif 'range' in pattern:
            return pattern['range'][0] <= value <= pattern['range'][1]

        return False

    def _get_value(self, features: Dict, marker: str) -> Optional[float]:
        """Extract numeric value from features."""
        if marker not in features:
            return None

        val = features[marker]
        if isinstance(val, dict):
            return val.get('value')
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _is_in_range(self, marker: str, value: float, range_name: str) -> bool:
        """Check if value is in specified range."""
        if marker not in self.ranges:
            return False

        ref = self.ranges[marker].get(range_name)
        if not ref:
            return False

        return ref['low'] <= value <= ref['high']

    def _is_abnormal(self, marker: str, value: float) -> bool:
        """Check if value is abnormal (outside normal range)."""
        if marker not in self.ranges:
            return False

        ref = self.ranges[marker]

        # Check normal range
        if 'normal' in ref:
            if value < ref['normal']['low'] or value > ref['normal']['high']:
                return True

        return False

    def _determine_severity(self, severity_logic: Dict, features: Dict) -> Optional[str]:
        """Determine severity based on severity_logic rules."""
        for severity_name, criteria in severity_logic.items():
            if self._check_severity_criteria(criteria, features):
                return severity_name
        return None

    def _check_severity_criteria(self, criteria: Dict, features: Dict) -> bool:
        """Check if severity criteria are met."""
        for marker_key, range_val in criteria.items():
            # Handle marker_range format (e.g., glucose_range)
            if marker_key.endswith('_range'):
                marker = marker_key.replace('_range', '')
                value = self._get_value(features, marker)
                if value is None:
                    return False
                if not (range_val[0] <= value <= range_val[1]):
                    return False
            else:
                # Direct marker check
                value = self._get_value(features, marker_key)
                if value is None:
                    return False
                if isinstance(range_val, list):
                    if not (range_val[0] <= value <= range_val[1]):
                        return False

        return True

    def _determine_stage(self, stages: Dict, features: Dict) -> Optional[str]:
        """Determine disease stage if staging criteria exist."""
        for stage_name, criteria in stages.items():
            # Check egfr_range for CKD staging
            if 'egfr_range' in criteria:
                egfr = self._get_value(features, 'egfr')
                if egfr is not None:
                    if criteria['egfr_range'][0] <= egfr <= criteria['egfr_range'][1]:
                        return stage_name

            # Check creatinine_increase for AKI staging
            if 'creatinine_increase' in criteria:
                # Would need baseline creatinine for this
                pass

        return None
