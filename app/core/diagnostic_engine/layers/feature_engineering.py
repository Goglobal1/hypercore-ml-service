"""
Layer 2: Feature Engineering + Temporal
Transforms single values into rich features including trends and trajectories.
"""

from typing import Dict, List, Any, Optional


class FeatureEngineer:
    """
    Layer 2: Create derived features including temporal patterns.
    """

    def engineer_features(self, current: Dict, history: List[Dict] = None) -> Dict:
        """
        Takes normalized current data and optional history.
        Returns enriched feature set.
        """
        features = current.get('features', {}).copy()

        engineered = {
            'raw_features': features,
            'derived_features': {},
            'temporal_features': {},
            'interaction_features': {}
        }

        # Derived features (ratios, calculations)
        engineered['derived_features'] = self._calculate_derived(features)

        # Temporal features (if history available)
        if history and len(history) > 0:
            engineered['temporal_features'] = self._calculate_temporal(features, history)

        # Interaction features (cross-marker relationships)
        engineered['interaction_features'] = self._calculate_interactions(features)

        return engineered

    def _calculate_derived(self, features: Dict) -> Dict:
        """Calculate derived features like BUN/Cr ratio, eGFR, anion gap, etc."""
        derived = {}

        # BUN/Creatinine ratio
        bun = self._get_value(features, 'bun')
        cr = self._get_value(features, 'creatinine')
        if bun is not None and cr is not None and cr > 0:
            ratio = bun / cr
            derived['bun_cr_ratio'] = {
                'value': ratio,
                'source': 'derived',
                'interpretation': self._interpret_bun_cr_ratio(ratio)
            }

        # Anion gap
        na = self._get_value(features, 'sodium')
        cl = self._get_value(features, 'chloride')
        bicarb = self._get_value(features, 'bicarbonate')
        if all(v is not None for v in [na, cl, bicarb]):
            anion_gap = na - cl - bicarb
            derived['anion_gap'] = {
                'value': anion_gap,
                'source': 'derived',
                'interpretation': 'elevated' if anion_gap > 12 else 'normal'
            }

        # Corrected calcium (for albumin)
        calcium = self._get_value(features, 'calcium')
        albumin = self._get_value(features, 'albumin')
        if calcium is not None and albumin is not None:
            # Corrected Ca = measured Ca + 0.8 * (4 - albumin)
            corrected_ca = calcium + 0.8 * (4 - albumin)
            derived['corrected_calcium'] = {
                'value': corrected_ca,
                'source': 'derived'
            }

        # AST/ALT ratio (potential alcoholic liver disease marker)
        ast = self._get_value(features, 'ast')
        alt = self._get_value(features, 'alt')
        if ast is not None and alt is not None and alt > 0:
            ratio = ast / alt
            derived['ast_alt_ratio'] = {
                'value': ratio,
                'source': 'derived',
                'interpretation': 'suggests_alcoholic' if ratio > 2 else 'non_alcoholic_pattern'
            }

        # NLR (Neutrophil-Lymphocyte Ratio) - inflammation marker
        neut = self._get_value(features, 'neutrophils')
        lymph = self._get_value(features, 'lymphocytes')
        if neut is not None and lymph is not None and lymph > 0:
            nlr = neut / lymph
            derived['nlr'] = {
                'value': nlr,
                'source': 'derived',
                'interpretation': 'elevated_inflammation' if nlr > 3 else 'normal'
            }

        # Mean Arterial Pressure (MAP)
        sbp = self._get_value(features, 'sbp')
        dbp = self._get_value(features, 'dbp')
        if sbp is not None and dbp is not None:
            map_val = dbp + (sbp - dbp) / 3
            derived['map'] = {
                'value': map_val,
                'source': 'derived',
                'interpretation': 'hypotensive' if map_val < 65 else 'normal'
            }

        # Shock Index (HR/SBP)
        hr = self._get_value(features, 'heart_rate')
        if hr is not None and sbp is not None and sbp > 0:
            shock_index = hr / sbp
            derived['shock_index'] = {
                'value': shock_index,
                'source': 'derived',
                'interpretation': 'concerning' if shock_index > 0.9 else 'normal'
            }

        return derived

    def _calculate_temporal(self, current: Dict, history: List[Dict]) -> Dict:
        """Calculate trends, velocities, trajectories."""
        temporal = {}

        for marker, current_data in current.items():
            if not isinstance(current_data, dict) or 'value' not in current_data:
                continue

            current_val = current_data['value']
            if current_val is None:
                continue

            # Find this marker in history
            historical_values = []
            for h in history:
                h_features = h.get('features', h)
                if marker in h_features:
                    h_data = h_features[marker]
                    if isinstance(h_data, dict) and 'value' in h_data:
                        historical_values.append({
                            'value': h_data['value'],
                            'timestamp': h.get('timestamp')
                        })
                    elif isinstance(h_data, (int, float)):
                        historical_values.append({
                            'value': h_data,
                            'timestamp': h.get('timestamp')
                        })

            if len(historical_values) == 0:
                continue

            prev_val = historical_values[-1]['value']
            if prev_val is None or prev_val == 0:
                continue

            temporal[marker] = {
                'current': current_val,
                'previous': prev_val,
                'absolute_change': current_val - prev_val,
                'percent_change': ((current_val - prev_val) / abs(prev_val) * 100),
                'direction': self._get_direction(current_val, prev_val),
                'trajectory': self._calculate_trajectory(historical_values + [{'value': current_val}]),
                'velocity': current_val - prev_val,  # Simplified: change per interval
                'data_points': len(historical_values) + 1
            }

        return temporal

    def _calculate_interactions(self, features: Dict) -> Dict:
        """Calculate cross-marker interaction features."""
        interactions = {}

        # Cardiorenal interaction
        cr = self._get_value(features, 'creatinine')
        bnp = self._get_value(features, 'bnp')
        if cr is not None and bnp is not None:
            if cr > 1.3 and bnp > 100:
                interactions['cardiorenal_syndrome_risk'] = {
                    'present': True,
                    'evidence': ['elevated_creatinine', 'elevated_bnp'],
                    'source': 'interaction'
                }

        # Hepatorenal interaction
        bili = self._get_value(features, 'bilirubin')
        if cr is not None and bili is not None:
            if cr > 1.5 and bili > 2.0:
                interactions['hepatorenal_syndrome_risk'] = {
                    'present': True,
                    'evidence': ['elevated_creatinine', 'elevated_bilirubin'],
                    'source': 'interaction'
                }

        # Inflammatory-metabolic coupling
        crp = self._get_value(features, 'crp')
        glucose = self._get_value(features, 'glucose')
        if crp is not None and glucose is not None:
            if crp > 3.0 and glucose > 100:
                interactions['inflammatory_metabolic_coupling'] = {
                    'present': True,
                    'evidence': ['elevated_crp', 'elevated_glucose'],
                    'source': 'interaction'
                }

        return interactions

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

    def _get_direction(self, current: float, previous: float) -> str:
        """Determine direction of change."""
        if current > previous * 1.05:
            return 'increasing'
        elif current < previous * 0.95:
            return 'decreasing'
        else:
            return 'stable'

    def _calculate_trajectory(self, values: List[Dict]) -> str:
        """Determine overall trajectory: stable, improving, worsening, volatile."""
        if len(values) < 2:
            return 'insufficient_data'

        vals = [v['value'] for v in values if v.get('value') is not None]
        if len(vals) < 2:
            return 'insufficient_data'

        changes = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]

        positive = sum(1 for c in changes if c > 0)
        negative = sum(1 for c in changes if c < 0)

        if positive > negative * 2:
            return 'consistently_increasing'
        elif negative > positive * 2:
            return 'consistently_decreasing'
        elif abs(positive - negative) <= 1:
            return 'stable'
        else:
            return 'volatile'

    def _interpret_bun_cr_ratio(self, ratio: float) -> str:
        """Interpret BUN/Creatinine ratio."""
        if ratio > 20:
            return 'prerenal_azotemia_or_gi_bleed'
        elif ratio < 10:
            return 'intrinsic_renal_disease_or_low_protein'
        else:
            return 'normal'
