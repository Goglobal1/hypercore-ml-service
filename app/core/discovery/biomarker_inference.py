"""
Biomarker Inference Engine
Identifies unknown biomarkers by their value distributions
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class BiomarkerProfile:
    """Statistical profile of a known biomarker."""
    name: str
    typical_min: float
    typical_max: float
    critical_low: Optional[float]
    critical_high: Optional[float]
    mean: float
    std: float
    unit: str
    endpoints: List[str]


# Known biomarker profiles based on clinical data
BIOMARKER_PROFILES = {
    # Metabolic
    'glucose': BiomarkerProfile(
        name='glucose', typical_min=70, typical_max=400,
        critical_low=50, critical_high=500,
        mean=100, std=50, unit='mg/dL',
        endpoints=['metabolic', 'endocrine']
    ),
    'hba1c': BiomarkerProfile(
        name='hba1c', typical_min=4.0, typical_max=14.0,
        critical_low=None, critical_high=10.0,
        mean=5.7, std=1.5, unit='%',
        endpoints=['metabolic', 'endocrine']
    ),

    # Renal
    'creatinine': BiomarkerProfile(
        name='creatinine', typical_min=0.5, typical_max=10.0,
        critical_low=None, critical_high=4.0,
        mean=1.0, std=0.5, unit='mg/dL',
        endpoints=['renal']
    ),
    'egfr': BiomarkerProfile(
        name='egfr', typical_min=5, typical_max=120,
        critical_low=15, critical_high=None,
        mean=90, std=25, unit='mL/min',
        endpoints=['renal']
    ),
    'bun': BiomarkerProfile(
        name='bun', typical_min=5, typical_max=100,
        critical_low=None, critical_high=50,
        mean=15, std=8, unit='mg/dL',
        endpoints=['renal']
    ),

    # Cardiac
    'troponin': BiomarkerProfile(
        name='troponin', typical_min=0.001, typical_max=50.0,
        critical_low=None, critical_high=0.04,
        mean=0.01, std=0.02, unit='ng/mL',
        endpoints=['cardiac']
    ),
    'bnp': BiomarkerProfile(
        name='bnp', typical_min=0, typical_max=5000,
        critical_low=None, critical_high=400,
        mean=50, std=100, unit='pg/mL',
        endpoints=['cardiac']
    ),

    # Hepatic
    'alt': BiomarkerProfile(
        name='alt', typical_min=5, typical_max=1000,
        critical_low=None, critical_high=200,
        mean=30, std=25, unit='U/L',
        endpoints=['hepatic']
    ),
    'ast': BiomarkerProfile(
        name='ast', typical_min=5, typical_max=1000,
        critical_low=None, critical_high=200,
        mean=25, std=20, unit='U/L',
        endpoints=['hepatic']
    ),
    'bilirubin': BiomarkerProfile(
        name='bilirubin', typical_min=0.1, typical_max=30,
        critical_low=None, critical_high=3.0,
        mean=0.7, std=0.5, unit='mg/dL',
        endpoints=['hepatic']
    ),
    'albumin': BiomarkerProfile(
        name='albumin', typical_min=1.5, typical_max=5.5,
        critical_low=2.5, critical_high=None,
        mean=4.0, std=0.5, unit='g/dL',
        endpoints=['hepatic', 'nutritional']
    ),

    # Inflammatory/Sepsis
    'wbc': BiomarkerProfile(
        name='wbc', typical_min=1, typical_max=50,
        critical_low=2, critical_high=20,
        mean=7.5, std=3, unit='K/uL',
        endpoints=['immune', 'sepsis', 'hematologic', 'infection']
    ),
    'crp': BiomarkerProfile(
        name='crp', typical_min=0, typical_max=300,
        critical_low=None, critical_high=50,
        mean=3, std=10, unit='mg/L',
        endpoints=['inflammatory', 'sepsis', 'infection']
    ),
    'procalcitonin': BiomarkerProfile(
        name='procalcitonin', typical_min=0, typical_max=100,
        critical_low=None, critical_high=2.0,
        mean=0.1, std=0.5, unit='ng/mL',
        endpoints=['sepsis', 'infection']
    ),
    'lactate': BiomarkerProfile(
        name='lactate', typical_min=0.3, typical_max=15,
        critical_low=None, critical_high=4.0,
        mean=1.2, std=0.8, unit='mmol/L',
        endpoints=['sepsis', 'perfusion', 'acid_base']
    ),

    # Hematologic
    'hemoglobin': BiomarkerProfile(
        name='hemoglobin', typical_min=4, typical_max=20,
        critical_low=7, critical_high=18,
        mean=14, std=2, unit='g/dL',
        endpoints=['hematologic']
    ),
    'hematocrit': BiomarkerProfile(
        name='hematocrit', typical_min=15, typical_max=60,
        critical_low=25, critical_high=55,
        mean=42, std=5, unit='%',
        endpoints=['hematologic']
    ),
    'platelets': BiomarkerProfile(
        name='platelets', typical_min=10, typical_max=1000,
        critical_low=50, critical_high=500,
        mean=250, std=80, unit='K/uL',
        endpoints=['hematologic', 'coagulation']
    ),

    # Electrolytes
    'sodium': BiomarkerProfile(
        name='sodium', typical_min=120, typical_max=160,
        critical_low=125, critical_high=150,
        mean=140, std=3, unit='mEq/L',
        endpoints=['fluid_balance', 'renal']
    ),
    'potassium': BiomarkerProfile(
        name='potassium', typical_min=2.5, typical_max=7.0,
        critical_low=3.0, critical_high=6.0,
        mean=4.2, std=0.5, unit='mEq/L',
        endpoints=['fluid_balance', 'renal', 'cardiac']
    ),

    # Vitals
    'heart_rate': BiomarkerProfile(
        name='heart_rate', typical_min=30, typical_max=220,
        critical_low=40, critical_high=150,
        mean=75, std=15, unit='bpm',
        endpoints=['vitals', 'cardiac']
    ),
    'temperature': BiomarkerProfile(
        name='temperature', typical_min=94, typical_max=106,
        critical_low=95, critical_high=103,
        mean=98.6, std=0.8, unit='F',
        endpoints=['vitals', 'sepsis', 'infection']
    ),
    'bp_systolic': BiomarkerProfile(
        name='bp_systolic', typical_min=60, typical_max=220,
        critical_low=90, critical_high=180,
        mean=120, std=20, unit='mmHg',
        endpoints=['vitals', 'cardiac', 'perfusion']
    ),
    'spo2': BiomarkerProfile(
        name='spo2', typical_min=70, typical_max=100,
        critical_low=88, critical_high=None,
        mean=97, std=2, unit='%',
        endpoints=['respiratory', 'vitals']
    ),
    'respiratory_rate': BiomarkerProfile(
        name='respiratory_rate', typical_min=8, typical_max=40,
        critical_low=10, critical_high=30,
        mean=16, std=4, unit='breaths/min',
        endpoints=['respiratory', 'vitals']
    ),

    # Coagulation
    'inr': BiomarkerProfile(
        name='inr', typical_min=0.8, typical_max=10,
        critical_low=None, critical_high=4.0,
        mean=1.0, std=0.2, unit='ratio',
        endpoints=['coagulation', 'hepatic']
    ),
    'd_dimer': BiomarkerProfile(
        name='d_dimer', typical_min=0, typical_max=20,
        critical_low=None, critical_high=2.0,
        mean=0.3, std=0.5, unit='mg/L',
        endpoints=['coagulation']
    ),

    # Thyroid
    'tsh': BiomarkerProfile(
        name='tsh', typical_min=0.1, typical_max=20,
        critical_low=0.1, critical_high=10,
        mean=2.0, std=1.5, unit='mIU/L',
        endpoints=['endocrine']
    ),

    # Lipids
    'triglycerides': BiomarkerProfile(
        name='triglycerides', typical_min=30, typical_max=1000,
        critical_low=None, critical_high=500,
        mean=150, std=80, unit='mg/dL',
        endpoints=['metabolic']
    ),
    'cholesterol': BiomarkerProfile(
        name='cholesterol', typical_min=100, typical_max=400,
        critical_low=None, critical_high=240,
        mean=200, std=40, unit='mg/dL',
        endpoints=['metabolic']
    ),

    # Iron
    'ferritin': BiomarkerProfile(
        name='ferritin', typical_min=10, typical_max=1000,
        critical_low=12, critical_high=500,
        mean=100, std=100, unit='ng/mL',
        endpoints=['hematologic', 'inflammatory']
    ),

    # Other
    'bmi': BiomarkerProfile(
        name='bmi', typical_min=15, typical_max=60,
        critical_low=16, critical_high=40,
        mean=25, std=5, unit='kg/m2',
        endpoints=['nutritional', 'metabolic']
    ),
    'age': BiomarkerProfile(
        name='age', typical_min=0, typical_max=120,
        critical_low=None, critical_high=None,
        mean=50, std=25, unit='years',
        endpoints=[]
    ),
}


class BiomarkerInferenceEngine:
    """
    Infers what biomarker a column represents based on its values.
    """

    def __init__(self):
        self.profiles = BIOMARKER_PROFILES

    def infer_column(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """
        Infer what biomarker a column represents.

        Returns:
        {
            'inferred_biomarker': 'creatinine',
            'confidence': 0.87,
            'endpoints': ['renal'],
            'method': 'statistical_inference',
            'alternatives': [
                {'biomarker': 'bun', 'confidence': 0.45},
                ...
            ]
        }
        """
        # First, try name matching
        name_match = self._match_by_name(column_name)
        if name_match and name_match['confidence'] > 0.8:
            return name_match

        # Then, try value pattern matching
        numeric_values = self._extract_numeric(values)
        if len(numeric_values) < 1:
            return {
                'inferred_biomarker': None,
                'confidence': 0,
                'endpoints': [],
                'method': 'no_data',
                'reason': 'Not enough numeric values to infer'
            }

        # Score each biomarker profile
        scores = []
        for biomarker_name, profile in self.profiles.items():
            score = self._calculate_match_score(numeric_values, profile)
            scores.append({
                'biomarker': biomarker_name,
                'confidence': score,
                'endpoints': profile.endpoints
            })

        # Sort by confidence
        scores.sort(key=lambda x: x['confidence'], reverse=True)

        if scores and scores[0]['confidence'] > 0.3:
            best = scores[0]
            return {
                'inferred_biomarker': best['biomarker'],
                'confidence': best['confidence'],
                'endpoints': best['endpoints'],
                'method': 'statistical_inference',
                'alternatives': scores[1:4]  # Top 3 alternatives
            }

        return {
            'inferred_biomarker': None,
            'confidence': 0,
            'endpoints': [],
            'method': 'no_match',
            'reason': 'Values do not match any known biomarker pattern'
        }

    def _match_by_name(self, column_name: str) -> Optional[Dict]:
        """Try to match column name to known biomarkers."""
        normalized = column_name.lower().replace(' ', '_').replace('-', '_')

        # Direct match
        if normalized in self.profiles:
            profile = self.profiles[normalized]
            return {
                'inferred_biomarker': normalized,
                'confidence': 1.0,
                'endpoints': profile.endpoints,
                'method': 'name_match'
            }

        # Partial match
        for biomarker_name, profile in self.profiles.items():
            if biomarker_name in normalized or normalized in biomarker_name:
                return {
                    'inferred_biomarker': biomarker_name,
                    'confidence': 0.85,
                    'endpoints': profile.endpoints,
                    'method': 'partial_name_match'
                }

        return None

    def _extract_numeric(self, values: List[Any]) -> List[float]:
        """Extract numeric values from a list."""
        numeric = []
        for v in values:
            if v is None:
                continue
            try:
                if isinstance(v, (int, float)):
                    if not np.isnan(v):
                        numeric.append(float(v))
                elif isinstance(v, str):
                    # Try to parse
                    v_clean = v.strip().replace(',', '')
                    numeric.append(float(v_clean))
            except (ValueError, TypeError):
                continue
        return numeric

    def _calculate_match_score(self, values: List[float], profile: BiomarkerProfile) -> float:
        """
        Calculate how well values match a biomarker profile.
        Returns confidence score 0-1.
        """
        if not values:
            return 0

        arr = np.array(values)
        val_min = np.min(arr)
        val_max = np.max(arr)
        val_mean = np.mean(arr)
        val_std = np.std(arr) if len(arr) > 1 else 0

        # Check if values fall within typical range
        range_score = 0
        if val_min >= profile.typical_min * 0.5 and val_max <= profile.typical_max * 1.5:
            # Values are within extended range
            range_score = 0.5
            if val_min >= profile.typical_min and val_max <= profile.typical_max:
                # Values are within typical range
                range_score = 1.0

        # Check if mean is close to expected
        mean_diff = abs(val_mean - profile.mean) / (profile.mean + 0.001)
        mean_score = max(0, 1 - mean_diff)

        # Check if std is similar (if we have enough values)
        std_score = 0.5  # Default
        if len(values) >= 5 and profile.std > 0:
            std_diff = abs(val_std - profile.std) / (profile.std + 0.001)
            std_score = max(0, 1 - std_diff * 0.5)

        # Weighted combination
        total_score = (range_score * 0.5) + (mean_score * 0.35) + (std_score * 0.15)

        return min(1.0, total_score)

    def infer_all_columns(self, df) -> Dict[str, Dict]:
        """
        Infer biomarkers for all columns in a DataFrame.
        """
        results = {}
        for column in df.columns:
            if column.lower() in ['patient_id', 'id', 'mrn', 'sample_id', 'label']:
                continue
            values = df[column].tolist()
            results[column] = self.infer_column(column, values)
        return results


# Singleton
_inference_engine = None


def get_inference_engine() -> BiomarkerInferenceEngine:
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = BiomarkerInferenceEngine()
    return _inference_engine
