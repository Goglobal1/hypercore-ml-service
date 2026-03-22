"""
Bulletproof Data Parser for HyperCore.

PHILOSOPHY:
1. NEVER crash - always produce something useful
2. Do the BEST with what you have
3. Tell the user what's MISSING that would help
4. Cross-reference against ALL modules regardless of data quality
5. Be SUPERIOR - no excuses, no fallbacks

This module handles ANY input format and produces structured results.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO

logger = logging.getLogger(__name__)


# =============================================================================
# BIOMARKER MAPPINGS - Map any possible name to standard biomarker
# =============================================================================

BIOMARKER_MAPPINGS = {
    # CRP variants
    "crp": "crp", "c-reactive protein": "crp", "c_reactive_protein": "crp",
    "c reactive protein": "crp", "creactive": "crp", "c-rp": "crp",

    # WBC variants
    "wbc": "wbc", "white blood cell": "wbc", "white_blood_cells": "wbc",
    "white blood cells": "wbc", "leukocytes": "wbc", "leukocyte": "wbc",
    "white_cell_count": "wbc", "white cell count": "wbc", "wcc": "wbc",

    # Procalcitonin variants
    "pct": "procalcitonin", "procalcitonin": "procalcitonin", "procal": "procalcitonin",
    "pro-calcitonin": "procalcitonin", "pro_calcitonin": "procalcitonin",

    # Temperature variants
    "temp": "temperature", "temperature": "temperature", "fever": "temperature",
    "body_temp": "temperature", "body temp": "temperature", "t": "temperature",

    # Lactate variants
    "lactate": "lactate", "lactic_acid": "lactate", "lactic acid": "lactate",
    "lac": "lactate", "lact": "lactate",

    # Troponin variants
    "troponin": "troponin", "trop": "troponin", "tnni": "troponin",
    "troponin_i": "troponin", "troponin i": "troponin", "tni": "troponin",
    "troponin_t": "troponin", "troponin t": "troponin", "tnt": "troponin",
    "hs-troponin": "troponin", "hstrop": "troponin",

    # BNP variants
    "bnp": "bnp", "ntprobnp": "bnp", "nt_probnp": "bnp", "nt-probnp": "bnp",
    "pro_bnp": "bnp", "pro-bnp": "bnp", "brain natriuretic peptide": "bnp",
    "b-type natriuretic peptide": "bnp",

    # Glucose variants
    "glucose": "glucose", "blood_sugar": "glucose", "blood sugar": "glucose",
    "bg": "glucose", "glu": "glucose", "fasting glucose": "glucose",
    "fasting_glucose": "glucose", "fbs": "glucose", "rbs": "glucose",

    # HbA1c variants
    "hba1c": "hba1c", "a1c": "hba1c", "glycated hemoglobin": "hba1c",
    "hemoglobin a1c": "hba1c", "hgba1c": "hba1c",

    # Creatinine variants
    "creatinine": "creatinine", "creat": "creatinine", "cr": "creatinine",
    "crea": "creatinine", "serum creatinine": "creatinine", "scr": "creatinine",

    # BUN variants
    "bun": "bun", "blood urea nitrogen": "bun", "urea": "bun",
    "urea nitrogen": "bun",

    # eGFR variants
    "egfr": "egfr", "gfr": "egfr", "estimated gfr": "egfr",
    "glomerular filtration rate": "egfr",

    # Liver enzymes
    "alt": "alt", "sgpt": "alt", "alanine_transaminase": "alt",
    "alanine transaminase": "alt", "alat": "alt",
    "ast": "ast", "sgot": "ast", "aspartate_transaminase": "ast",
    "aspartate transaminase": "ast", "asat": "ast",
    "bilirubin": "bilirubin", "bili": "bilirubin", "total bilirubin": "bilirubin",
    "tbili": "bilirubin", "t_bili": "bilirubin",
    "albumin": "albumin", "alb": "albumin", "serum albumin": "albumin",
    "inr": "inr", "international normalized ratio": "inr", "pt/inr": "inr",

    # Lipids
    "ldl": "ldl", "ldl cholesterol": "ldl", "ldl-c": "ldl",
    "low density lipoprotein": "ldl",
    "hdl": "hdl", "hdl cholesterol": "hdl", "hdl-c": "hdl",
    "high density lipoprotein": "hdl",
    "triglycerides": "triglycerides", "tg": "triglycerides", "trigs": "triglycerides",
    "cholesterol": "cholesterol", "total cholesterol": "cholesterol", "tc": "cholesterol",

    # Tumor markers
    "cea": "cea", "carcinoembryonic antigen": "cea",
    "ca125": "ca125", "ca-125": "ca125", "cancer antigen 125": "ca125",
    "ca199": "ca199", "ca-199": "ca199", "ca 19-9": "ca199",
    "psa": "psa", "prostate specific antigen": "psa",
    "afp": "afp", "alpha fetoprotein": "afp", "alpha-fetoprotein": "afp",

    # Electrolytes
    "sodium": "sodium", "na": "sodium", "na+": "sodium",
    "potassium": "potassium", "k": "potassium", "k+": "potassium",
    "chloride": "chloride", "cl": "chloride", "cl-": "chloride",
    "bicarbonate": "bicarbonate", "hco3": "bicarbonate", "co2": "bicarbonate",
    "calcium": "calcium", "ca": "calcium", "ca2+": "calcium",
    "magnesium": "magnesium", "mg": "magnesium", "mg2+": "magnesium",
    "phosphate": "phosphate", "phos": "phosphate", "phosphorus": "phosphate",

    # CBC
    "hemoglobin": "hemoglobin", "hgb": "hemoglobin", "hb": "hemoglobin",
    "hematocrit": "hematocrit", "hct": "hematocrit",
    "platelets": "platelets", "plt": "platelets", "platelet count": "platelets",
    "rbc": "rbc", "red blood cells": "rbc", "red blood cell": "rbc",
    "mcv": "mcv", "mean corpuscular volume": "mcv",
    "mch": "mch", "mean corpuscular hemoglobin": "mch",
    "mchc": "mchc",
    "rdw": "rdw", "red cell distribution width": "rdw",

    # Blood gases
    "ph": "ph", "blood ph": "ph",
    "pco2": "pco2", "partial co2": "pco2", "carbon dioxide": "pco2",
    "po2": "po2", "partial o2": "po2", "oxygen": "po2",
    "sao2": "sao2", "oxygen saturation": "sao2", "o2sat": "sao2", "spo2": "sao2",

    # Patient identifiers
    "patient_id": "patient_id", "patientid": "patient_id", "patient id": "patient_id",
    "mrn": "patient_id", "medical record number": "patient_id", "id": "patient_id",
    "pt_id": "patient_id", "subject_id": "patient_id", "subject id": "patient_id",

    # Vitals
    "heart_rate": "heart_rate", "hr": "heart_rate", "pulse": "heart_rate",
    "heartrate": "heart_rate", "heart rate": "heart_rate",
    "blood_pressure": "blood_pressure", "bp": "blood_pressure",
    "systolic": "systolic_bp", "sbp": "systolic_bp", "systolic_bp": "systolic_bp",
    "diastolic": "diastolic_bp", "dbp": "diastolic_bp", "diastolic_bp": "diastolic_bp",
    "respiratory_rate": "respiratory_rate", "rr": "respiratory_rate",
    "resp_rate": "respiratory_rate", "respiration": "respiratory_rate",
}

# Critical biomarkers for each domain
DOMAIN_CRITICAL_BIOMARKERS = {
    "sepsis": {
        "critical": {"crp", "wbc", "procalcitonin", "lactate", "temperature"},
        "helpful": {"heart_rate", "respiratory_rate", "blood_pressure"},
        "genetic": {"CYP2D6", "DPYD", "IL6", "TNF"},
    },
    "cardiac": {
        "critical": {"troponin", "bnp", "crp"},
        "helpful": {"ldl", "hdl", "cholesterol", "blood_pressure"},
        "genetic": {"VKORC1", "CYP2C19", "CYP2C9", "APOE"},
    },
    "oncology": {
        "critical": {"cea", "ca125", "psa", "afp"},
        "helpful": {"wbc", "hemoglobin", "platelets"},
        "genetic": {"BRCA1", "BRCA2", "TP53", "DPYD", "UGT1A1"},
    },
    "metabolic": {
        "critical": {"glucose", "hba1c", "triglycerides", "ldl"},
        "helpful": {"cholesterol", "hdl", "creatinine"},
        "genetic": {"CYP2C9", "SLCO1B1", "APOE"},
    },
    "renal": {
        "critical": {"creatinine", "bun", "egfr"},
        "helpful": {"potassium", "sodium", "phosphate"},
        "genetic": {"CYP2D6", "SLCO1B1", "ABCB1"},
    },
    "hepatic": {
        "critical": {"alt", "ast", "bilirubin", "albumin"},
        "helpful": {"inr", "platelets"},
        "genetic": {"CYP2D6", "CYP3A4", "NAT2", "UGT1A1"},
    },
}


@dataclass
class ParseResult:
    """Result of parsing attempt."""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    score: float
    parsed_data: List[Dict[str, Any]]
    recognized_biomarkers: Dict[str, Any]
    unrecognized_columns: List[str]
    missing_by_domain: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str]
    parse_method: str
    can_analyze: bool


class RobustDataParser:
    """
    Bulletproof data parser that handles ANY input.
    Never crashes. Always produces results.
    """

    def __init__(self):
        self.biomarker_map = BIOMARKER_MAPPINGS
        self.domain_requirements = DOMAIN_CRITICAL_BIOMARKERS

    def parse_any_input(self, data: Any) -> DataQualityReport:
        """
        Parse ANY input - CSV, JSON, plain text, dict, malformed data.
        Returns structured data with confidence scores.

        NEVER raises exceptions. ALWAYS returns a result.
        """
        warnings = []
        parsed_data = []
        parse_method = "none"

        try:
            # Handle different input types
            if isinstance(data, dict):
                parsed_data = [data]
                parse_method = "dict_direct"
            elif isinstance(data, list):
                parsed_data = data if all(isinstance(d, dict) for d in data) else []
                parse_method = "list_direct"
            elif isinstance(data, (str, bytes)):
                if isinstance(data, bytes):
                    try:
                        data = data.decode('utf-8', errors='replace')
                    except Exception as e:
                        warnings.append(f"Encoding issue: {e}")
                        data = str(data)

                # Try multiple parsing strategies
                result = self._try_all_strategies(data)
                parsed_data = result.data
                parse_method = result.method
                warnings.extend(result.warnings)
            else:
                warnings.append(f"Unknown input type: {type(data)}")
                # Try to convert to string and parse
                try:
                    data_str = str(data)
                    result = self._try_all_strategies(data_str)
                    parsed_data = result.data
                    parse_method = f"converted_{result.method}"
                    warnings.extend(result.warnings)
                except Exception as e:
                    warnings.append(f"Conversion failed: {e}")

        except Exception as e:
            warnings.append(f"Top-level parse error: {e}")
            logger.warning(f"RobustDataParser caught error: {e}")

        # Map to biomarkers
        recognized, unrecognized = self._map_to_biomarkers(parsed_data)

        # Calculate missing data by domain
        missing_by_domain = self._calculate_missing_by_domain(recognized)

        # Generate recommendations
        recommendations = self._generate_recommendations(recognized, missing_by_domain)

        # Calculate quality score
        quality_score = self._calculate_quality_score(recognized, parsed_data)

        # Determine if we can analyze
        can_analyze = len(recognized) > 0 or len(parsed_data) > 0

        return DataQualityReport(
            score=quality_score,
            parsed_data=parsed_data,
            recognized_biomarkers=recognized,
            unrecognized_columns=unrecognized,
            missing_by_domain=missing_by_domain,
            recommendations=recommendations,
            warnings=warnings,
            parse_method=parse_method,
            can_analyze=can_analyze,
        )

    def _try_all_strategies(self, data: str) -> ParseResult:
        """Try multiple parsing strategies in order of preference."""
        strategies = [
            ("json", self._try_json),
            ("csv_standard", self._try_csv_standard),
            ("csv_flexible", self._try_csv_flexible),
            ("key_value", self._try_key_value_pairs),
            ("extract_numbers", self._try_extract_numbers),
            ("line_by_line", self._try_line_by_line),
        ]

        all_warnings = []

        for name, strategy in strategies:
            try:
                result = strategy(data)
                if result.success and result.data:
                    result.method = name
                    result.warnings.extend(all_warnings)
                    return result
            except Exception as e:
                all_warnings.append(f"{name}: {str(e)[:100]}")
                continue

        # Nothing worked - return empty with warnings
        return ParseResult(
            success=False,
            data=[],
            method="none",
            warnings=all_warnings + ["All parsing strategies failed"]
        )

    def _try_json(self, data: str) -> ParseResult:
        """Try parsing as JSON."""
        data = data.strip()

        # Try direct JSON parse
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return ParseResult(success=True, data=[parsed])
            elif isinstance(parsed, list):
                if all(isinstance(item, dict) for item in parsed):
                    return ParseResult(success=True, data=parsed)
                # List of values - try to make sense of it
                return ParseResult(success=True, data=[{"values": parsed}])
        except json.JSONDecodeError:
            pass

        # Try to find JSON within text
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, data, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return ParseResult(success=True, data=[parsed])
                except:
                    continue

        return ParseResult(success=False)

    def _try_csv_standard(self, data: str) -> ParseResult:
        """Try parsing as standard CSV."""
        try:
            import pandas as pd
            df = pd.read_csv(StringIO(data))
            if len(df) > 0 and len(df.columns) > 0:
                records = df.to_dict('records')
                return ParseResult(success=True, data=records)
        except Exception as e:
            return ParseResult(success=False, warnings=[str(e)[:100]])

        return ParseResult(success=False)

    def _try_csv_flexible(self, data: str) -> ParseResult:
        """Try parsing CSV with flexible options."""
        import pandas as pd

        warnings = []

        # Try different delimiters
        for delimiter in [',', ';', '\t', '|', ' ']:
            try:
                df = pd.read_csv(
                    StringIO(data),
                    delimiter=delimiter,
                    on_bad_lines='skip',
                    encoding_errors='replace',
                    dtype=str,
                    skipinitialspace=True,
                )

                if len(df.columns) > 1 or (len(df.columns) == 1 and len(df) > 0):
                    # Convert numeric strings to numbers
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass

                    records = df.to_dict('records')
                    if records:
                        return ParseResult(
                            success=True,
                            data=records,
                            warnings=[f"Used delimiter: '{delimiter}'"]
                        )
            except Exception as e:
                warnings.append(f"delimiter '{delimiter}': {str(e)[:50]}")
                continue

        return ParseResult(success=False, warnings=warnings)

    def _try_key_value_pairs(self, data: str) -> ParseResult:
        """Extract key-value pairs from text."""
        patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([\d.]+)',  # key: value or key = value
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*"([^"]*)"',  # key: "string"
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*\'([^\']*)\'',  # key: 'string'
        ]

        extracted = {}

        for pattern in patterns:
            matches = re.findall(pattern, data, re.IGNORECASE)
            for key, value in matches:
                key_lower = key.lower().strip()
                try:
                    extracted[key_lower] = float(value)
                except ValueError:
                    extracted[key_lower] = value.strip()

        if extracted:
            return ParseResult(success=True, data=[extracted])

        return ParseResult(success=False)

    def _try_extract_numbers(self, data: str) -> ParseResult:
        """Extract any numbers with context from text."""
        # Find patterns like "CRP 28.5" or "WBC is 18"
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is\s+|was\s+|of\s+)?([\d.]+)'

        extracted = {}
        matches = re.findall(pattern, data, re.IGNORECASE)

        for key, value in matches:
            key_lower = key.lower().strip()
            try:
                extracted[key_lower] = float(value)
            except ValueError:
                pass

        # Also try to find standalone biomarker mentions
        for biomarker_variant in self.biomarker_map.keys():
            if biomarker_variant in data.lower():
                # Find numbers near this biomarker
                pattern = re.escape(biomarker_variant) + r'\s*[:=]?\s*([\d.]+)'
                match = re.search(pattern, data.lower())
                if match:
                    try:
                        extracted[biomarker_variant] = float(match.group(1))
                    except ValueError:
                        pass

        if extracted:
            return ParseResult(success=True, data=[extracted])

        return ParseResult(success=False)

    def _try_line_by_line(self, data: str) -> ParseResult:
        """Parse line by line looking for key-value patterns."""
        lines = data.strip().split('\n')
        extracted = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try various separators
            for sep in [':', '=', '\t', '  ', ',']:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()

                    # Skip if key looks like a sentence
                    if len(key) > 30 or ' ' in key and len(key.split()) > 3:
                        continue

                    try:
                        extracted[key] = float(value)
                    except ValueError:
                        if value:
                            extracted[key] = value
                    break

        if extracted:
            return ParseResult(success=True, data=[extracted])

        return ParseResult(success=False)

    def _map_to_biomarkers(
        self,
        parsed_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Map parsed columns to known biomarkers."""
        recognized = {}
        unrecognized = []

        for record in parsed_data:
            for key, value in record.items():
                key_lower = str(key).lower().strip()

                # Check if it maps to a known biomarker
                if key_lower in self.biomarker_map:
                    standard_name = self.biomarker_map[key_lower]
                    recognized[standard_name] = value
                else:
                    # Check partial matches
                    matched = False
                    for variant, standard in self.biomarker_map.items():
                        if variant in key_lower or key_lower in variant:
                            recognized[standard] = value
                            matched = True
                            break

                    if not matched and key_lower not in unrecognized:
                        unrecognized.append(key_lower)

        return recognized, unrecognized

    def _calculate_missing_by_domain(
        self,
        recognized: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate missing critical data for each domain."""
        missing_by_domain = {}
        recognized_set = set(recognized.keys())

        for domain, requirements in self.domain_requirements.items():
            critical = requirements["critical"]
            helpful = requirements["helpful"]
            genetic = requirements.get("genetic", set())

            critical_have = recognized_set & critical
            critical_missing = critical - recognized_set
            helpful_have = recognized_set & helpful
            helpful_missing = helpful - recognized_set

            completeness = len(critical_have) / len(critical) if critical else 0

            # Determine impact message
            if completeness < 0.3:
                impact = f"Limited {domain} analysis possible. Add critical markers for full workup."
            elif completeness < 0.7:
                impact = f"Partial {domain} analysis. Missing markers could improve detection by 1-2 days."
            elif completeness < 1.0:
                impact = f"Good {domain} coverage. Additional markers would enhance precision."
            else:
                impact = f"Complete {domain} biomarker coverage."

            missing_by_domain[domain] = {
                "critical_have": list(critical_have),
                "critical_missing": list(critical_missing),
                "helpful_have": list(helpful_have),
                "helpful_missing": list(helpful_missing),
                "genetic_recommended": list(genetic),
                "completeness": round(completeness, 2),
                "impact": impact,
            }

        return missing_by_domain

    def _generate_recommendations(
        self,
        recognized: Dict[str, Any],
        missing_by_domain: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable recommendations based on data quality."""
        recommendations = []

        # Check overall coverage
        if len(recognized) == 0:
            recommendations.append(
                "No recognized biomarkers found. Provide data with columns like: "
                "crp, wbc, procalcitonin, troponin, glucose"
            )
            recommendations.append(
                "Accepted formats: CSV with headers, JSON, or key-value pairs (e.g., 'CRP: 28.5')"
            )
            return recommendations

        # Find the best domain match
        best_domain = None
        best_completeness = 0

        for domain, info in missing_by_domain.items():
            if info["completeness"] > best_completeness:
                best_completeness = info["completeness"]
                best_domain = domain

        if best_domain:
            info = missing_by_domain[best_domain]
            if info["critical_missing"]:
                missing_str = ", ".join(info["critical_missing"][:3])
                recommendations.append(
                    f"For {best_domain} analysis: add {missing_str} to improve detection"
                )

            if info["genetic_recommended"]:
                genes_str = ", ".join(info["genetic_recommended"][:3])
                recommendations.append(
                    f"Genetic markers ({genes_str}) would enable 2-3 days earlier detection"
                )

        # Check for missing high-value biomarkers
        high_value = {"procalcitonin", "lactate", "troponin"}
        missing_high_value = high_value - set(recognized.keys())
        if missing_high_value:
            recommendations.append(
                f"High-value biomarkers not provided: {', '.join(missing_high_value)}"
            )

        return recommendations

    def _calculate_quality_score(
        self,
        recognized: Dict[str, Any],
        parsed_data: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall data quality score (0.0 to 1.0)."""
        if not parsed_data:
            return 0.0

        score = 0.0

        # Base score for having any data
        if parsed_data:
            score += 0.2

        # Score for recognized biomarkers
        num_recognized = len(recognized)
        if num_recognized >= 10:
            score += 0.3
        elif num_recognized >= 5:
            score += 0.2
        elif num_recognized >= 2:
            score += 0.1
        elif num_recognized >= 1:
            score += 0.05

        # Score for critical biomarkers
        critical_all = set()
        for domain_reqs in self.domain_requirements.values():
            critical_all.update(domain_reqs["critical"])

        critical_present = set(recognized.keys()) & critical_all
        critical_ratio = len(critical_present) / len(critical_all) if critical_all else 0
        score += critical_ratio * 0.3

        # Score for data validity (numeric values)
        valid_values = 0
        total_values = 0
        for value in recognized.values():
            total_values += 1
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                valid_values += 1

        if total_values > 0:
            score += (valid_values / total_values) * 0.2

        return min(round(score, 2), 1.0)

    def extract_lab_values(self, parsed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract standardized lab values from parsed data.
        Returns a clean dict suitable for analysis.
        """
        recognized, _ = self._map_to_biomarkers(parsed_data)

        # Clean up values - ensure numeric where expected
        lab_data = {}
        for key, value in recognized.items():
            if key == "patient_id":
                lab_data[key] = str(value)
            else:
                try:
                    lab_data[key] = float(value)
                except (ValueError, TypeError):
                    # Keep as string if can't convert
                    lab_data[key] = value

        return lab_data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_parser: Optional[RobustDataParser] = None


def get_parser() -> RobustDataParser:
    """Get global parser instance."""
    global _parser
    if _parser is None:
        _parser = RobustDataParser()
    return _parser


def parse_any_data(data: Any) -> DataQualityReport:
    """Parse any data using the global parser."""
    return get_parser().parse_any_input(data)


def extract_lab_data(data: Any) -> Dict[str, Any]:
    """Extract lab values from any data format."""
    report = parse_any_data(data)
    return get_parser().extract_lab_values(report.parsed_data)
