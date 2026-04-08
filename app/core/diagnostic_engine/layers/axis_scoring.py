"""
Layer 3: Biologic Axis Scoring
Score the health/disturbance of each organ system based on its biomarkers.
"""

from typing import Dict, Any, Tuple, Optional
import json
import os


class AxisScorer:
    """
    Layer 3: Score each biologic axis based on its biomarkers.
    """

    def __init__(self, axes_config: Dict = None, reference_ranges: Dict = None):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

        if axes_config is None:
            with open(os.path.join(data_dir, 'biologic_axes.json'), 'r') as f:
                axes_config = json.load(f)

        if reference_ranges is None:
            with open(os.path.join(data_dir, 'reference_ranges.json'), 'r') as f:
                reference_ranges = json.load(f)

        self.axes = axes_config.get('axes', {})
        self.ranges = reference_ranges.get('ranges', {})

    def score_all_axes(self, features: Dict) -> Dict:
        """
        Score all biologic axes.
        Returns dict with axis scores, abnormalities, and trajectories.
        """
        axis_scores = {}

        for axis_name, axis_config in self.axes.items():
            axis_scores[axis_name] = self._score_axis(axis_name, axis_config, features)

        return axis_scores

    def _score_axis(self, axis_name: str, config: Dict, features: Dict) -> Dict:
        """Score a single axis."""

        result = {
            'name': config.get('name', axis_name),
            'description': config.get('description', ''),
            'score': 0.0,  # 0 = healthy, 1 = severely disturbed
            'status': 'normal',  # normal, borderline, abnormal, critical
            'confidence': 0.0,
            'available_markers': 0,
            'total_markers': 0,
            'abnormal_markers': [],
            'normal_markers': [],
            'missing_markers': [],
            'trajectory': 'unknown',
            'evidence': []
        }

        biomarkers = config.get('biomarkers', {})
        all_markers = (
            biomarkers.get('primary', []) +
            biomarkers.get('secondary', []) +
            biomarkers.get('supportive', [])
        )
        result['total_markers'] = len(all_markers)

        weighted_score = 0.0
        total_weight = 0.0

        # Get raw features - handle both nested and flat structures
        raw_features = features.get('raw_features', features)
        temporal = features.get('temporal_features', {})

        for marker in all_markers:
            # Determine weight (primary markers weighted higher)
            if marker in biomarkers.get('primary', []):
                weight = 1.0
            elif marker in biomarkers.get('secondary', []):
                weight = 0.5
            else:
                weight = 0.25

            # Check if marker is available
            if marker not in raw_features:
                result['missing_markers'].append(marker)
                continue

            result['available_markers'] += 1

            # Get value
            marker_data = raw_features[marker]
            if isinstance(marker_data, dict):
                value = marker_data.get('value')
            else:
                value = marker_data

            if value is None:
                continue

            # Score this marker
            marker_score, marker_status = self._score_marker(marker, value)

            if marker_status in ['borderline', 'abnormal', 'critical', 'borderline_low',
                                 'borderline_high', 'abnormal_low', 'abnormal_high',
                                 'critical_low', 'critical_high', 'low', 'high']:
                result['abnormal_markers'].append({
                    'marker': marker,
                    'value': value,
                    'status': marker_status,
                    'score': marker_score,
                    'reference': self._get_normal_range(marker)
                })
                result['evidence'].append(f"{marker}: {value} ({marker_status})")
            else:
                result['normal_markers'].append(marker)

            weighted_score += marker_score * weight
            total_weight += weight

        # Calculate final axis score
        if total_weight > 0:
            result['score'] = weighted_score / total_weight

        # Determine overall status
        if result['score'] >= 0.75:
            result['status'] = 'critical'
        elif result['score'] >= 0.5:
            result['status'] = 'abnormal'
        elif result['score'] >= 0.25:
            result['status'] = 'borderline'
        else:
            result['status'] = 'normal'

        # Confidence based on data availability
        primary_markers = biomarkers.get('primary', [])
        if len(primary_markers) > 0:
            available_primary = sum(1 for m in primary_markers if m in raw_features)
            result['confidence'] = available_primary / len(primary_markers)
        else:
            result['confidence'] = min(result['available_markers'] / max(result['total_markers'], 1), 1.0)

        # Add trajectory info if available
        marker_trajectories = []
        for marker in all_markers:
            if marker in temporal:
                marker_trajectories.append(temporal[marker].get('trajectory', 'unknown'))

        if marker_trajectories:
            increasing = marker_trajectories.count('consistently_increasing')
            decreasing = marker_trajectories.count('consistently_decreasing')
            if increasing > decreasing:
                result['trajectory'] = 'worsening' if axis_name in ['inflammatory', 'cardiac'] else 'improving'
            elif decreasing > increasing:
                result['trajectory'] = 'improving' if axis_name in ['inflammatory', 'cardiac'] else 'worsening'
            else:
                result['trajectory'] = 'stable'

        return result

    def _score_marker(self, marker: str, value: float) -> Tuple[float, str]:
        """
        Score a single marker.
        Returns (score 0-1, status string).
        """
        if marker not in self.ranges:
            return (0.0, 'unknown')

        ref = self.ranges[marker]

        # Check critical ranges first (both high and low)
        if 'critical' in ref:
            if ref['critical']['low'] <= value <= ref['critical']['high']:
                return (1.0, 'critical')

        if 'critical_high' in ref:
            if value >= ref['critical_high']['low']:
                return (1.0, 'critical_high')

        if 'critical_low' in ref:
            if value <= ref['critical_low']['high']:
                return (1.0, 'critical_low')

        # Check abnormal ranges
        if 'abnormal' in ref:
            if ref['abnormal']['low'] <= value <= ref['abnormal']['high']:
                return (0.7, 'abnormal')

        if 'abnormal_high' in ref:
            if ref['abnormal_high']['low'] <= value <= ref['abnormal_high']['high']:
                return (0.7, 'abnormal_high')

        if 'abnormal_low' in ref:
            if ref['abnormal_low']['low'] <= value <= ref['abnormal_low']['high']:
                return (0.7, 'abnormal_low')

        # Check borderline ranges
        if 'borderline' in ref:
            if ref['borderline']['low'] <= value <= ref['borderline']['high']:
                return (0.4, 'borderline')

        if 'borderline_high' in ref:
            if ref['borderline_high']['low'] <= value <= ref['borderline_high']['high']:
                return (0.4, 'borderline_high')

        if 'borderline_low' in ref:
            if ref['borderline_low']['low'] <= value <= ref['borderline_low']['high']:
                return (0.4, 'borderline_low')

        # Check normal
        if 'normal' in ref:
            if ref['normal']['low'] <= value <= ref['normal']['high']:
                return (0.0, 'normal')

        # Default: slightly abnormal if outside normal but not in defined ranges
        if 'normal' in ref:
            if value < ref['normal']['low']:
                return (0.3, 'low')
            if value > ref['normal']['high']:
                return (0.3, 'high')

        return (0.0, 'normal')

    def _get_normal_range(self, marker: str) -> str:
        """Get the normal range as a string."""
        if marker not in self.ranges:
            return 'unknown'

        ref = self.ranges[marker]
        if 'normal' in ref:
            return f"{ref['normal']['low']}-{ref['normal']['high']}"

        return 'unknown'

    def get_disturbed_axes(self, axis_scores: Dict, min_status: str = 'borderline') -> list:
        """Get list of axes with at least the specified disturbance level."""
        status_levels = {'normal': 0, 'borderline': 1, 'abnormal': 2, 'critical': 3}
        min_level = status_levels.get(min_status, 1)

        disturbed = []
        for axis_name, score in axis_scores.items():
            axis_level = status_levels.get(score.get('status', 'normal'), 0)
            if axis_level >= min_level:
                disturbed.append(axis_name)

        return disturbed
