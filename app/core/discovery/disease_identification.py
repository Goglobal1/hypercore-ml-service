"""
Disease Identification Layer
Matches patterns to known diseases and flags unknowns
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class IdentificationConfidence(Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class DiseaseMatch:
    disease_name: str
    icd10_codes: List[str]
    confidence: IdentificationConfidence
    confidence_score: float
    matching_indicators: List[str]
    missing_indicators: List[str]
    description: str

    def to_dict(self) -> Dict:
        return {
            'disease_name': self.disease_name,
            'icd10_codes': self.icd10_codes,
            'confidence': self.confidence.value,
            'confidence_score': self.confidence_score,
            'matching_indicators': self.matching_indicators,
            'missing_indicators': self.missing_indicators,
            'description': self.description
        }


@dataclass
class UnknownPattern:
    pattern_id: str
    involved_systems: List[str]
    abnormal_values: List[Dict]
    description: str
    recommendation: str

    def to_dict(self) -> Dict:
        return {
            'pattern_id': self.pattern_id,
            'involved_systems': self.involved_systems,
            'abnormal_values': self.abnormal_values,
            'description': self.description,
            'recommendation': self.recommendation
        }


DISEASE_SIGNATURES = {
    'type_2_diabetes': {
        'icd10': ['E11'],
        'required_indicators': {'metabolic': ['glucose', 'hba1c']},
        'supporting_indicators': {
            'metabolic': ['insulin', 'c_peptide', 'triglycerides'],
            'renal': ['creatinine', 'egfr']
        },
        'thresholds': {'glucose': {'high': 126}, 'hba1c': {'high': 6.5}},
        'description': 'Type 2 Diabetes Mellitus'
    },
    'diabetic_ketoacidosis': {
        'icd10': ['E10.1', 'E11.1'],
        'required_indicators': {'metabolic': ['glucose'], 'acid_base': ['ph', 'anion_gap']},
        'supporting_indicators': {'metabolic': ['ketones'], 'fluid_balance': ['sodium', 'potassium']},
        'thresholds': {'glucose': {'high': 250}, 'ph': {'low': 7.3}, 'anion_gap': {'high': 12}},
        'description': 'Diabetic Ketoacidosis'
    },
    'acute_mi': {
        'icd10': ['I21'],
        'required_indicators': {'cardiac': ['troponin']},
        'supporting_indicators': {'cardiac': ['ck_mb', 'bnp'], 'vitals': ['heart_rate', 'bp_systolic']},
        'thresholds': {'troponin': {'high': 0.04}},
        'description': 'Acute Myocardial Infarction'
    },
    'heart_failure': {
        'icd10': ['I50'],
        'required_indicators': {'cardiac': ['bnp']},
        'supporting_indicators': {'cardiac': ['ejection_fraction'], 'respiratory': ['spo2'], 'renal': ['creatinine']},
        'thresholds': {'bnp': {'high': 100}},
        'description': 'Heart Failure'
    },
    'acute_kidney_injury': {
        'icd10': ['N17'],
        'required_indicators': {'renal': ['creatinine']},
        'supporting_indicators': {'renal': ['bun', 'potassium', 'urine_output'], 'fluid_balance': ['sodium']},
        # CRITICAL: Creatinine must be > 1.3 (above normal) to indicate AKI
        'thresholds': {'creatinine': {'high': 1.3}, 'bun': {'high': 20}},
        'description': 'Acute Kidney Injury'
    },
    'chronic_kidney_disease': {
        'icd10': ['N18'],
        # CRITICAL: CKD requires eGFR < 60 - creatinine alone is NOT sufficient
        'required_indicators': {'renal': ['egfr']},
        'supporting_indicators': {'renal': ['creatinine', 'bun', 'potassium'], 'hematologic': ['hemoglobin']},
        # eGFR must be < 60 for CKD diagnosis
        'thresholds': {'egfr': {'low': 60}, 'creatinine': {'high': 1.3}},
        'description': 'Chronic Kidney Disease'
    },
    'acute_liver_failure': {
        'icd10': ['K72.0'],
        'required_indicators': {'hepatic': ['alt', 'ast', 'inr']},
        'supporting_indicators': {'hepatic': ['bilirubin', 'albumin'], 'coagulation': ['pt']},
        'thresholds': {'alt': {'high': 200}, 'ast': {'high': 200}, 'inr': {'high': 1.5}},
        'description': 'Acute Liver Failure'
    },
    'sepsis': {
        'icd10': ['A41'],
        'required_indicators': {'sepsis': ['wbc', 'temperature'], 'vitals': ['heart_rate', 'respiratory_rate']},
        'supporting_indicators': {'sepsis': ['procalcitonin', 'lactate'], 'perfusion': ['bp_systolic']},
        'thresholds': {'wbc': {'high': 12, 'low': 4}, 'temperature': {'high': 100.4, 'low': 96.8}},
        'description': 'Sepsis'
    },
    'septic_shock': {
        'icd10': ['R65.21'],
        'required_indicators': {'sepsis': ['lactate'], 'perfusion': ['bp_systolic', 'map']},
        'supporting_indicators': {'sepsis': ['procalcitonin', 'wbc'], 'renal': ['creatinine']},
        'thresholds': {'lactate': {'high': 2}, 'map': {'low': 65}},
        'description': 'Septic Shock'
    },
    'acute_respiratory_failure': {
        'icd10': ['J96.0'],
        'required_indicators': {'respiratory': ['spo2', 'pao2']},
        'supporting_indicators': {'respiratory': ['paco2', 'respiratory_rate'], 'acid_base': ['ph']},
        'thresholds': {'spo2': {'low': 90}, 'pao2': {'low': 60}},
        'description': 'Acute Respiratory Failure'
    },
    'anemia': {
        'icd10': ['D64.9'],
        'required_indicators': {'hematologic': ['hemoglobin', 'hematocrit']},
        'supporting_indicators': {'hematologic': ['mcv', 'mch', 'rbc', 'ferritin', 'iron']},
        'thresholds': {'hemoglobin': {'low': 12}, 'hematocrit': {'low': 36}},
        'description': 'Anemia'
    },
    'thrombocytopenia': {
        'icd10': ['D69.6'],
        'required_indicators': {'hematologic': ['platelets']},
        'supporting_indicators': {'coagulation': ['pt', 'ptt']},
        'thresholds': {'platelets': {'low': 150}},
        'description': 'Thrombocytopenia'
    },
    'hypothyroidism': {
        'icd10': ['E03'],
        'required_indicators': {'endocrine': ['tsh']},
        'supporting_indicators': {'endocrine': ['t4', 't3']},
        'thresholds': {'tsh': {'high': 4.0}},
        'description': 'Hypothyroidism'
    },
    'hyperthyroidism': {
        'icd10': ['E05'],
        'required_indicators': {'endocrine': ['tsh']},
        'supporting_indicators': {'endocrine': ['t4', 't3'], 'cardiac': ['heart_rate']},
        'thresholds': {'tsh': {'low': 0.4}},
        'description': 'Hyperthyroidism'
    },
    'hyponatremia': {
        'icd10': ['E87.1'],
        'required_indicators': {'fluid_balance': ['sodium']},
        'supporting_indicators': {'fluid_balance': ['osmolality'], 'neurologic': ['mental_status']},
        'thresholds': {'sodium': {'low': 135}},
        'description': 'Hyponatremia'
    },
    'hyperkalemia': {
        'icd10': ['E87.5'],
        'required_indicators': {'fluid_balance': ['potassium']},
        'supporting_indicators': {'renal': ['creatinine', 'egfr'], 'cardiac': ['ecg']},
        'thresholds': {'potassium': {'high': 5.0}},
        'description': 'Hyperkalemia'
    }
}


class DiseaseIdentifier:
    """
    Identifies known diseases and flags unknown patterns.
    """

    def __init__(self):
        self.signatures = DISEASE_SIGNATURES

    def identify(
        self,
        endpoint_results: Dict[str, Any],
        endpoint_data: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Identify diseases and unknown patterns.
        """
        matched_diseases = []
        unknown_patterns = []

        for disease_name, signature in self.signatures.items():
            match_result = self._check_disease_match(
                disease_name, signature, endpoint_results, endpoint_data
            )
            if match_result:
                matched_diseases.append(match_result)

        matched_diseases.sort(key=lambda x: x.confidence_score, reverse=True)

        unknown = self._identify_unknown_patterns(endpoint_results, matched_diseases)
        unknown_patterns.extend(unknown)

        return {
            'identified_diseases': matched_diseases,
            'unknown_patterns': unknown_patterns,
            'total_diseases': len(matched_diseases),
            'total_unknown': len(unknown_patterns)
        }

    def _check_disease_match(
        self,
        disease_name: str,
        signature: Dict,
        endpoint_results: Dict,
        endpoint_data: Dict
    ) -> Optional[DiseaseMatch]:
        """Check if data matches a disease signature."""
        required = signature.get('required_indicators', {})
        supporting = signature.get('supporting_indicators', {})
        thresholds = signature.get('thresholds', {})

        matching_indicators = []
        missing_indicators = []

        required_met = 0
        required_total = 0

        for endpoint, markers in required.items():
            for marker in markers:
                required_total += 1
                if self._marker_present_and_abnormal(endpoint, marker, endpoint_data, thresholds):
                    matching_indicators.append(f"{endpoint}:{marker}")
                    required_met += 1
                else:
                    missing_indicators.append(f"{endpoint}:{marker}")

        if required_total > 0 and required_met == 0:
            return None

        supporting_met = 0
        supporting_total = 0

        for endpoint, markers in supporting.items():
            for marker in markers:
                supporting_total += 1
                if self._marker_present_and_abnormal(endpoint, marker, endpoint_data, thresholds):
                    matching_indicators.append(f"{endpoint}:{marker}")
                    supporting_met += 1

        if required_total > 0:
            required_confidence = required_met / required_total
        else:
            required_confidence = 0.5

        if supporting_total > 0:
            supporting_confidence = supporting_met / supporting_total
        else:
            supporting_confidence = 0

        overall_confidence = required_confidence * 0.7 + supporting_confidence * 0.3

        if overall_confidence >= 0.7:
            confidence_level = IdentificationConfidence.HIGH
        elif overall_confidence >= 0.4:
            confidence_level = IdentificationConfidence.MODERATE
        elif overall_confidence >= 0.2:
            confidence_level = IdentificationConfidence.LOW
        else:
            return None

        return DiseaseMatch(
            disease_name=signature.get('description', disease_name),
            icd10_codes=signature.get('icd10', []),
            confidence=confidence_level,
            confidence_score=overall_confidence,
            matching_indicators=matching_indicators,
            missing_indicators=missing_indicators,
            description=f"Matches {len(matching_indicators)} indicators for {signature.get('description', disease_name)}"
        )

    def _marker_present_and_abnormal(
        self,
        endpoint: str,
        marker: str,
        endpoint_data: Dict,
        thresholds: Dict
    ) -> bool:
        """Check if a marker is present and abnormal."""
        if endpoint not in endpoint_data:
            return False

        data = endpoint_data[endpoint]

        for column, values in data.items():
            if marker in column.lower() or column.lower() in marker:
                if marker in thresholds:
                    threshold = thresholds[marker]
                    for value in values:
                        try:
                            v = float(value)
                            if 'high' in threshold and v > threshold['high']:
                                return True
                            if 'low' in threshold and v < threshold['low']:
                                return True
                        except:
                            pass
                else:
                    # CRITICAL FIX: If no threshold defined, use standard reference ranges
                    # Do NOT assume abnormal just because marker is present
                    from .disease_detection import REFERENCE_RANGES, is_abnormal
                    if marker in REFERENCE_RANGES:
                        for value in values:
                            try:
                                v = float(value)
                                if is_abnormal(v, marker):
                                    return True
                            except:
                                pass
                    # If no reference range, marker presence alone is NOT abnormal
                    return False

        return False

    def _identify_unknown_patterns(
        self,
        endpoint_results: Dict,
        matched_diseases: List[DiseaseMatch]
    ) -> List[UnknownPattern]:
        """Identify abnormal patterns that don't match known diseases."""
        unknown = []

        explained_indicators = set()
        for disease in matched_diseases:
            explained_indicators.update(disease.matching_indicators)

        unexplained = []
        for endpoint, result in endpoint_results.items():
            if hasattr(result, 'abnormal_values'):
                abnormal = result.abnormal_values
            else:
                abnormal = result.get('abnormal_values', [])

            for abn in abnormal:
                indicator = f"{endpoint}:{abn.get('column', 'unknown')}"
                if indicator not in explained_indicators:
                    unexplained.append({
                        'endpoint': endpoint,
                        'indicator': indicator,
                        'details': abn
                    })

        if unexplained:
            endpoints_involved = list(set(u['endpoint'] for u in unexplained))

            unknown.append(UnknownPattern(
                pattern_id=f"UNK-{len(endpoints_involved)}-{len(unexplained)}",
                involved_systems=endpoints_involved,
                abnormal_values=[u['details'] for u in unexplained],
                description=f"Unexplained abnormalities in {len(endpoints_involved)} systems",
                recommendation="Investigate: These abnormalities don't match known disease patterns"
            ))

        return unknown
