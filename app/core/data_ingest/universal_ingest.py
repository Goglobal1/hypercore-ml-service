"""
Universal Data Ingest
=====================

Accepts ANY file format, auto-detects structure, maps to HyperCore endpoints.
Supports: CSV, JSON, HL7, FHIR, and generic text formats.
"""

import io
import re
import json
import csv
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from app.core.endpoints.endpoint_definitions import ENDPOINT_DEFINITIONS


# Comprehensive biomarker synonym mapping
BIOMARKER_SYNONYMS = {
    # Vitals
    "heart_rate": ["hr", "pulse", "bpm", "heart_rate_bpm", "heartrate", "pulse_rate"],
    "respiratory_rate": ["rr", "resp_rate", "breaths_per_min", "respiratoryrate", "resp"],
    "sbp": ["systolic", "systolic_bp", "sys_bp", "systolic_blood_pressure", "systolicbp"],
    "dbp": ["diastolic", "diastolic_bp", "dia_bp", "diastolic_blood_pressure", "diastolicbp"],
    "spo2": ["oxygen_saturation", "o2_sat", "sao2", "oxygen", "o2sat", "saturation"],
    "temperature": ["temp", "body_temp", "body_temperature", "fever"],
    "map": ["mean_arterial_pressure", "mean_bp", "meanarterialpressure"],

    # Metabolic
    "glucose": ["glu", "blood_glucose", "fasting_glucose", "fbg", "blood_sugar", "sugar", "bg"],
    "lactate": ["lac", "lactic_acid", "serum_lactate"],
    "bicarbonate": ["bicarb", "hco3", "co2", "serum_bicarbonate"],
    "potassium": ["k", "k+", "serum_potassium"],
    "sodium": ["na", "na+", "serum_sodium"],
    "chloride": ["cl", "cl-", "serum_chloride"],
    "anion_gap": ["ag", "aniongap"],

    # Renal
    "creatinine": ["cr", "creat", "serum_creatinine", "scr", "crea"],
    "bun": ["blood_urea_nitrogen", "urea_nitrogen", "urea"],
    "egfr": ["gfr", "estimated_gfr", "glomerular_filtration_rate"],

    # Cardiac
    "troponin": ["trop", "troponin_i", "troponin_t", "tni", "tnt", "hs_troponin", "hstnt"],
    "bnp": ["b_natriuretic_peptide", "brain_natriuretic_peptide"],
    "pro_bnp": ["nt_pro_bnp", "ntprobnp", "nt_probnp", "proBNP"],

    # Hematologic
    "wbc": ["white_blood_cells", "leukocytes", "white_count", "wcc"],
    "rbc": ["red_blood_cells", "erythrocytes", "red_count"],
    "hemoglobin": ["hgb", "hb", "haemoglobin"],
    "hematocrit": ["hct", "packed_cell_volume", "pcv"],
    "platelets": ["plt", "platelet_count", "thrombocytes"],

    # Inflammatory
    "crp": ["c_reactive_protein", "creactive_protein", "c_rp"],
    "procalcitonin": ["pct", "procal"],
    "esr": ["sed_rate", "sedimentation_rate", "erythrocyte_sedimentation_rate"],

    # Hepatic
    "alt": ["alanine_aminotransferase", "sgpt", "alat"],
    "ast": ["aspartate_aminotransferase", "sgot", "asat"],
    "alp": ["alkaline_phosphatase", "alk_phos", "alkphos"],
    "bilirubin": ["bili", "total_bilirubin", "tbili"],
    "albumin": ["alb", "serum_albumin"],

    # Coagulation
    "inr": ["international_normalized_ratio"],
    "pt": ["prothrombin_time", "pro_time"],
    "ptt": ["partial_thromboplastin_time", "aptt", "activated_ptt"],
    "d_dimer": ["ddimer", "d-dimer", "fibrin_degradation"],
    "fibrinogen": ["fib", "factor_i"],

    # Lipids
    "total_cholesterol": ["cholesterol", "tc", "total_chol"],
    "ldl": ["ldl_cholesterol", "ldl_c", "low_density_lipoprotein"],
    "hdl": ["hdl_cholesterol", "hdl_c", "high_density_lipoprotein"],
    "triglycerides": ["tg", "trigs", "triglyceride"],

    # Endocrine
    "hba1c": ["a1c", "glycated_hemoglobin", "glycohemoglobin", "hemoglobin_a1c"],
    "tsh": ["thyroid_stimulating_hormone", "thyrotropin"],
    "insulin": ["serum_insulin", "fasting_insulin"],
}

# Build reverse lookup
SYNONYM_TO_BIOMARKER = {}
for biomarker, synonyms in BIOMARKER_SYNONYMS.items():
    SYNONYM_TO_BIOMARKER[biomarker.lower()] = biomarker
    for syn in synonyms:
        SYNONYM_TO_BIOMARKER[syn.lower()] = biomarker


@dataclass
class IngestResult:
    """Result from data ingestion."""
    raw_data: Dict[str, Any]
    normalized: Dict[str, Any]
    endpoint_mapping: Dict[str, List[str]]
    data_completeness: float
    unmapped_fields: List[str]
    file_type: str
    n_records: int
    warnings: List[str]


class UniversalDataIngest:
    """
    Universal data ingestion for any file format.
    Auto-detects structure and maps to HyperCore biomarkers.
    """

    def __init__(self):
        self.endpoints = ENDPOINT_DEFINITIONS
        self.synonyms = BIOMARKER_SYNONYMS

    def ingest(
        self,
        file_content: Union[str, bytes, dict],
        file_type: Optional[str] = None
    ) -> IngestResult:
        """
        Accept any data format and normalize for HyperCore processing.

        Args:
            file_content: Raw file content (string, bytes, or dict)
            file_type: Optional file type hint (csv, json, hl7, fhir)

        Returns:
            IngestResult with normalized data and endpoint mapping
        """
        warnings = []

        # Auto-detect file type if not provided
        if file_type is None:
            file_type = self._detect_file_type(file_content)

        # Parse based on type
        if file_type == "csv":
            data, parse_warnings = self._parse_csv(file_content)
        elif file_type == "json":
            data, parse_warnings = self._parse_json(file_content)
        elif file_type == "hl7":
            data, parse_warnings = self._parse_hl7(file_content)
        elif file_type == "fhir":
            data, parse_warnings = self._parse_fhir(file_content)
        elif file_type == "dict":
            data = file_content if isinstance(file_content, dict) else {}
            parse_warnings = []
        else:
            data, parse_warnings = self._parse_generic(file_content)

        warnings.extend(parse_warnings)

        # Normalize to HyperCore format
        normalized, unmapped = self._normalize_to_hypercore(data)

        # Map to endpoints
        endpoint_mapping = self._map_to_endpoints(normalized)

        # Calculate completeness
        completeness = self._calculate_completeness(endpoint_mapping)

        return IngestResult(
            raw_data=data,
            normalized=normalized,
            endpoint_mapping=endpoint_mapping,
            data_completeness=completeness,
            unmapped_fields=unmapped,
            file_type=file_type,
            n_records=len(data) if isinstance(data, list) else 1,
            warnings=warnings,
        )

    def _detect_file_type(self, content: Union[str, bytes, dict]) -> str:
        """Auto-detect file type from content."""
        if isinstance(content, dict):
            return "dict"

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        content_str = str(content).strip()

        # Check for JSON
        if content_str.startswith("{") or content_str.startswith("["):
            try:
                json.loads(content_str)
                return "json"
            except json.JSONDecodeError:
                pass

        # Check for HL7
        if content_str.startswith("MSH|"):
            return "hl7"

        # Check for CSV (has commas and newlines)
        if "," in content_str and "\n" in content_str:
            return "csv"

        # Check for FHIR (JSON with resourceType)
        if "resourceType" in content_str:
            return "fhir"

        return "generic"

    def _parse_csv(self, content: Union[str, bytes]) -> Tuple[List[Dict], List[str]]:
        """Parse CSV content."""
        warnings = []

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        try:
            reader = csv.DictReader(io.StringIO(content))
            records = list(reader)
            return records, warnings
        except Exception as e:
            warnings.append(f"CSV parsing error: {str(e)}")
            return [], warnings

    def _parse_json(self, content: Union[str, bytes, dict]) -> Tuple[Any, List[str]]:
        """Parse JSON content."""
        warnings = []

        if isinstance(content, dict):
            return content, warnings

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        try:
            data = json.loads(content)
            return data, warnings
        except json.JSONDecodeError as e:
            warnings.append(f"JSON parsing error: {str(e)}")
            return {}, warnings

    def _parse_hl7(self, content: Union[str, bytes]) -> Tuple[Dict, List[str]]:
        """Parse HL7 message."""
        warnings = []

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        try:
            # Basic HL7 parsing
            segments = content.strip().split("\r")
            if not segments:
                segments = content.strip().split("\n")

            data = {"segments": {}}

            for segment in segments:
                if "|" in segment:
                    fields = segment.split("|")
                    segment_type = fields[0]
                    data["segments"][segment_type] = fields[1:]

            # Extract OBX (observation) segments
            observations = {}
            for key, value in data.get("segments", {}).items():
                if key.startswith("OBX"):
                    # OBX format: OBX|1|NM|glucose||100|mg/dL
                    if len(value) >= 5:
                        obs_name = value[2] if len(value) > 2 else ""
                        obs_value = value[4] if len(value) > 4 else ""
                        observations[obs_name] = obs_value

            data["observations"] = observations
            return data, warnings

        except Exception as e:
            warnings.append(f"HL7 parsing error: {str(e)}")
            return {}, warnings

    def _parse_fhir(self, content: Union[str, bytes, dict]) -> Tuple[Dict, List[str]]:
        """Parse FHIR resource."""
        warnings = []

        # First parse as JSON
        data, json_warnings = self._parse_json(content)
        warnings.extend(json_warnings)

        if not data:
            return {}, warnings

        # Extract observations from FHIR format
        observations = {}

        resource_type = data.get("resourceType", "")

        if resource_type == "Bundle":
            for entry in data.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Observation":
                    code = resource.get("code", {}).get("coding", [{}])[0].get("code", "")
                    display = resource.get("code", {}).get("coding", [{}])[0].get("display", "")
                    value = resource.get("valueQuantity", {}).get("value")
                    if value is not None:
                        observations[display or code] = value

        elif resource_type == "Observation":
            code = data.get("code", {}).get("coding", [{}])[0].get("code", "")
            display = data.get("code", {}).get("coding", [{}])[0].get("display", "")
            value = data.get("valueQuantity", {}).get("value")
            if value is not None:
                observations[display or code] = value

        data["observations"] = observations
        return data, warnings

    def _parse_generic(self, content: Union[str, bytes]) -> Tuple[Dict, List[str]]:
        """Parse generic text content."""
        warnings = []

        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        # Try to extract key-value pairs
        data = {}
        patterns = [
            r"(\w+)\s*[=:]\s*([\d.]+)",  # key=value or key: value
            r"(\w+)\s+([\d.]+)\s*\w*",    # key value unit
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for key, value in matches:
                try:
                    data[key.lower()] = float(value)
                except ValueError:
                    data[key.lower()] = value

        if not data:
            warnings.append("Could not extract structured data from generic format")

        return data, warnings

    def _normalize_to_hypercore(
        self,
        data: Union[Dict, List]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Normalize data to HyperCore biomarker names.

        Returns:
            Tuple of (normalized_data, unmapped_fields)
        """
        if isinstance(data, list):
            # If list of records, normalize each
            if not data:
                return {}, []
            data = data[0] if len(data) == 1 else data[-1]  # Use last/most recent

        normalized = {}
        unmapped = []

        for key, value in data.items():
            if key in ["observations", "segments"]:
                # Handle nested data from HL7/FHIR
                if isinstance(value, dict):
                    for nested_key, nested_value in value.items():
                        mapped = self._map_column_name(nested_key)
                        if mapped:
                            normalized[mapped] = self._parse_value(nested_value)
                        else:
                            unmapped.append(nested_key)
                continue

            mapped = self._map_column_name(key)
            if mapped:
                normalized[mapped] = self._parse_value(value)
            else:
                # Keep original for non-biomarker fields
                if key.lower() not in ["patient_id", "timestamp", "outcome", "event_in_12h"]:
                    unmapped.append(key)
                else:
                    normalized[key.lower()] = value

        return normalized, unmapped

    def _map_column_name(self, column: str) -> Optional[str]:
        """Map column name to standard biomarker name."""
        column_lower = column.lower().strip()

        # Remove common prefixes/suffixes
        column_clean = re.sub(r"^(serum_|blood_|plasma_)", "", column_lower)
        column_clean = re.sub(r"(_level|_value|_result)$", "", column_clean)

        # Direct lookup
        if column_clean in SYNONYM_TO_BIOMARKER:
            return SYNONYM_TO_BIOMARKER[column_clean]

        # Fuzzy match
        for synonym, biomarker in SYNONYM_TO_BIOMARKER.items():
            if synonym in column_clean or column_clean in synonym:
                return biomarker

        return None

    def _parse_value(self, value: Any) -> Optional[float]:
        """Parse value to float."""
        if value is None or value == "":
            return None

        if isinstance(value, (int, float)):
            if np.isnan(value) if isinstance(value, float) else False:
                return None
            return float(value)

        # Handle string values
        if isinstance(value, str):
            # Remove units and extract number
            match = re.search(r"([\d.]+)", value)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass

        return None

    def _map_to_endpoints(self, normalized: Dict[str, Any]) -> Dict[str, List[str]]:
        """Map normalized biomarkers to endpoints."""
        mapping = {}

        for endpoint_name, endpoint_def in self.endpoints.items():
            endpoint_biomarkers = []

            for biomarker in endpoint_def.get("biomarkers", {}).keys():
                if biomarker in normalized and normalized[biomarker] is not None:
                    endpoint_biomarkers.append(biomarker)

            if endpoint_biomarkers:
                mapping[endpoint_name] = endpoint_biomarkers

        return mapping

    def _calculate_completeness(self, endpoint_mapping: Dict[str, List[str]]) -> float:
        """Calculate data completeness across endpoints."""
        total_biomarkers = sum(
            len(ep.get("biomarkers", {}))
            for ep in self.endpoints.values()
        )

        available_biomarkers = sum(
            len(biomarkers)
            for biomarkers in endpoint_mapping.values()
        )

        return available_biomarkers / total_biomarkers if total_biomarkers > 0 else 0


def ingest_data(
    content: Union[str, bytes, dict],
    file_type: Optional[str] = None
) -> IngestResult:
    """Convenience function for data ingestion."""
    ingestor = UniversalDataIngest()
    return ingestor.ingest(content, file_type)
