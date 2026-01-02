# smart_formatter.py
# Location: app/core/smart_formatter.py
"""
Universal data formatter that accepts flexible inputs
and transforms them to endpoint-specific formats.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import json
import re
from io import StringIO
from datetime import datetime

try:
    from .field_mappings import FIELD_ALIASES, ENDPOINT_SCHEMAS, DATA_TYPE_PATTERNS
except ImportError:
    from field_mappings import FIELD_ALIASES, ENDPOINT_SCHEMAS, DATA_TYPE_PATTERNS


class SmartFormatter:
    """
    Universal data formatter that accepts ANY reasonable input format
    and transforms it to endpoint-specific formats.
    """

    FORMAT_CSV_STRING = "csv_string"
    FORMAT_JSON_ARRAY = "json_array"
    FORMAT_JSON_OBJECT = "json_object"
    FORMAT_DATAFRAME = "dataframe"
    FORMAT_FHIR = "fhir"
    FORMAT_HL7 = "hl7"
    FORMAT_RAW_STRING = "raw_string"

    def __init__(self):
        self.field_aliases = FIELD_ALIASES
        self.endpoint_schemas = ENDPOINT_SCHEMAS
        self.data_type_patterns = DATA_TYPE_PATTERNS
        self._warnings = []

    def normalize(self, data: Union[Dict, str, pd.DataFrame, List], target_endpoint: str, strict: bool = False) -> Dict[str, Any]:
        """Main entry point. Accepts any format, returns endpoint-ready data."""
        self._warnings = []

        try:
            input_format = self._detect_format(data)
            schema = self.endpoint_schemas.get(target_endpoint, {})

            if schema.get("custom_format"):
                return self._handle_custom_format(data, target_endpoint, schema)

            df, metadata = self._to_dataframe(data, input_format)
            df = self._normalize_columns(df)
            output = self._transform_for_endpoint(df, metadata, target_endpoint)

            if self._warnings:
                output["_formatter_warnings"] = self._warnings

            return output

        except Exception as e:
            if strict:
                raise
            return {"error": str(e), "error_type": "formatting_error", "original_data_type": str(type(data)), "target_endpoint": target_endpoint}

    def detect_data_type(self, data: Any) -> str:
        """Detect what type of data this is for routing."""
        input_format = self._detect_format(data)
        try:
            df, metadata = self._to_dataframe(data, input_format)
            df = self._normalize_columns(df)
            columns = set(df.columns)
        except:
            columns = set()

        if isinstance(data, dict):
            keys = set(data.keys())
            if keys & {"immune", "metabolic", "microbiome"}:
                return "multi_omic"
            if "medications" in keys:
                return "medication_list"
            if "fluview_json" in keys:
                return "surveillance"

        for data_type, pattern in self.data_type_patterns.items():
            required = set(pattern.get("required_columns", []))
            if required and required.issubset(columns):
                return data_type

        return "unknown"

    def get_applicable_endpoints(self, data_type: str) -> List[str]:
        """Get list of endpoints applicable for this data type."""
        pattern = self.data_type_patterns.get(data_type, {})
        return pattern.get("applicable_endpoints", ["analyze"])

    def _detect_format(self, data: Any) -> str:
        """Auto-detect the input format."""
        if isinstance(data, pd.DataFrame):
            return self.FORMAT_DATAFRAME

        if isinstance(data, str):
            data = data.strip()
            if data.startswith('{') or data.startswith('['):
                try:
                    json.loads(data)
                    return self.FORMAT_JSON_OBJECT
                except:
                    pass
            if "," in data and "\n" in data:
                return self.FORMAT_CSV_STRING
            return self.FORMAT_RAW_STRING

        if isinstance(data, dict):
            if "resourceType" in data:
                return self.FORMAT_FHIR
            if "MSH" in str(data.get("segments", "")):
                return self.FORMAT_HL7
            if "csv" in data and isinstance(data.get("csv"), str):
                return self.FORMAT_CSV_STRING
            if "data" in data and isinstance(data.get("data"), list):
                return self.FORMAT_JSON_ARRAY
            return self.FORMAT_JSON_OBJECT

        if isinstance(data, list):
            return self.FORMAT_JSON_ARRAY

        raise ValueError(f"Unsupported data format: {type(data)}")

    def _to_dataframe(self, data: Any, format_type: str) -> Tuple[pd.DataFrame, Dict]:
        """Convert any format to DataFrame + metadata."""
        metadata = {}

        if format_type == self.FORMAT_DATAFRAME:
            return data.copy(), metadata

        if format_type == self.FORMAT_CSV_STRING:
            if isinstance(data, dict):
                csv_str = data.get("csv", "")
                metadata = {k: v for k, v in data.items() if k != "csv"}
            else:
                csv_str = data
            df = pd.read_csv(StringIO(csv_str))
            return df, metadata

        if format_type == self.FORMAT_JSON_ARRAY:
            if isinstance(data, dict):
                records = data.get("data", data.get("records", []))
                metadata = {k: v for k, v in data.items() if k not in ["data", "records"]}
            else:
                records = data
            df = pd.DataFrame(records)
            return df, metadata

        if format_type == self.FORMAT_JSON_OBJECT:
            if isinstance(data, dict):
                if "baseline_profile" in data:
                    profile = data["baseline_profile"]
                    df = pd.DataFrame([profile])
                    metadata = {k: v for k, v in data.items() if k != "baseline_profile"}
                elif "analyses" in data:
                    df = pd.DataFrame(data["analyses"])
                    metadata = {k: v for k, v in data.items() if k != "analyses"}
                elif "patients" in data:
                    df = pd.DataFrame(data["patients"])
                    metadata = {k: v for k, v in data.items() if k != "patients"}
                else:
                    scalar_data = {k: v for k, v in data.items() if isinstance(v, (str, int, float, bool, type(None)))}
                    if scalar_data:
                        df = pd.DataFrame([scalar_data])
                    else:
                        df = pd.DataFrame()
                    metadata = {k: v for k, v in data.items() if k not in scalar_data}
                return df, metadata

        if format_type == self.FORMAT_FHIR:
            df = self._parse_fhir(data)
            metadata = {"source": "fhir", "resourceType": data.get("resourceType")}
            return df, metadata

        if format_type == self.FORMAT_RAW_STRING:
            try:
                df = pd.read_csv(StringIO(data))
                return df, metadata
            except:
                raise ValueError("Could not parse raw string as data")

        raise ValueError(f"Cannot convert {format_type} to DataFrame")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map column names to standard names using aliases."""
        if df.empty:
            return df

        column_mapping = {}

        for col in df.columns:
            col_clean = str(col).lower().strip().replace(" ", "_").replace("-", "_")
            matched = False

            for standard_name, aliases in self.field_aliases.items():
                aliases_lower = [a.lower().replace(" ", "_").replace("-", "_") for a in aliases]
                if col_clean in aliases_lower:
                    column_mapping[col] = standard_name
                    matched = True
                    break

            if not matched:
                column_mapping[col] = col_clean

        return df.rename(columns=column_mapping)

    def _transform_for_endpoint(self, df: pd.DataFrame, metadata: Dict, target_endpoint: str) -> Dict[str, Any]:
        """Transform DataFrame to endpoint-specific format."""
        schema = self.endpoint_schemas.get(target_endpoint, {})
        output = {}

        if schema.get("needs_csv", True):
            output["csv"] = df.to_csv(index=False)

        if schema.get("needs_label_column"):
            label = metadata.get("label_column")
            if not label:
                for candidate in ["outcome", "label", "target", "response", "event", "acr20", "responder", "death", "mortality"]:
                    if candidate in df.columns:
                        label = candidate
                        break
            if not label and len(df.columns) > 0:
                label = df.columns[-1]
                self._warnings.append(f"Auto-selected '{label}' as label_column")
            output["label_column"] = label

        if schema.get("needs_treatment_column"):
            treatment = metadata.get("treatment_column")
            if not treatment:
                for candidate in ["treatment", "treatment_arm", "arm", "drug", "intervention", "group", "cohort"]:
                    if candidate in df.columns:
                        treatment = candidate
                        break
            output["treatment_column"] = treatment

        if schema.get("required_fields"):
            for field in schema["required_fields"]:
                if field in metadata:
                    output[field] = metadata[field]
                elif field == "region_column" and "region" in df.columns:
                    output[field] = "region"
                elif field == "time_column" and "time" in df.columns:
                    output[field] = "time"
                elif field == "case_count_column" and "cases" in df.columns:
                    output[field] = "cases"

        for key, value in metadata.items():
            if key not in output and not key.startswith("_"):
                output[key] = value

        return output

    def _handle_custom_format(self, data: Dict, target_endpoint: str, schema: Dict) -> Dict[str, Any]:
        """Handle endpoints with special format requirements."""
        custom_type = schema.get("custom_format")

        if custom_type == "multi_omic":
            return self._transform_multi_omic(data)
        elif custom_type == "baseline_profile":
            return self._transform_baseline_profile(data)
        elif custom_type == "population_risk":
            return self._transform_population_risk(data)
        elif custom_type == "distributions":
            return self._transform_distributions(data)
        elif custom_type == "fluview":
            return self._transform_fluview(data)
        elif custom_type == "medications":
            return self._transform_medications(data)
        elif custom_type == "patient_report":
            return self._transform_patient_report(data)
        elif custom_type == "digital_twin_storage":
            return self._transform_digital_twin_storage(data)
        elif custom_type == "root_cause":
            return self._transform_root_cause(data)
        else:
            return data

    def _transform_multi_omic(self, data: Dict) -> Dict:
        output = {"immune": [], "metabolic": [], "microbiome": []}

        if all(k in data for k in ["immune", "metabolic", "microbiome"]):
            for domain in ["immune", "metabolic", "microbiome"]:
                val = data[domain]
                if isinstance(val, list):
                    output[domain] = [float(v) for v in val if isinstance(v, (int, float))]
                elif isinstance(val, dict):
                    output[domain] = [float(v) for v in val.values() if isinstance(v, (int, float))]
            return output

        for key, value in data.items():
            key_lower = key.lower()
            if isinstance(value, (int, float)):
                if any(x in key_lower for x in ["cd4", "cd8", "nk", "lymph", "immune"]):
                    output["immune"].append(float(value))
                elif any(x in key_lower for x in ["glucose", "hba1c", "insulin", "lipid", "metabol"]):
                    output["metabolic"].append(float(value))
                elif any(x in key_lower for x in ["firmicute", "bacteroid", "microbi", "diversity"]):
                    output["microbiome"].append(float(value))

        for domain in ["immune", "metabolic", "microbiome"]:
            if not output[domain]:
                output[domain] = [0.0]

        return output

    def _transform_baseline_profile(self, data: Dict) -> Dict:
        if "baseline_profile" in data:
            profile = data["baseline_profile"]
        else:
            profile = {}
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    profile[k] = float(v)
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            profile[k2] = float(v2)
        return {"baseline_profile": profile}

    def _transform_population_risk(self, data: Dict) -> Dict:
        if "analyses" in data and "region" in data:
            return data
        analyses = []
        region = data.get("region", "Unknown")
        if "patients" in data:
            analyses = data["patients"]
        elif "data" in data:
            analyses = data["data"]
        elif "csv" in data:
            df = pd.read_csv(StringIO(data["csv"]))
            analyses = df.to_dict(orient="records")
        return {"analyses": analyses, "region": region}

    def _transform_distributions(self, data: Dict) -> Dict:
        if "real_data_distributions" in data and "n_subjects" in data:
            return data
        if "csv" in data:
            df = pd.read_csv(StringIO(data["csv"]))
            distributions = {}
            for col in df.select_dtypes(include=[np.number]).columns:
                distributions[col] = {"mean": float(df[col].mean()), "std": float(df[col].std()), "min": float(df[col].min()), "max": float(df[col].max())}
            return {"real_data_distributions": distributions, "n_subjects": data.get("n_subjects", 10)}
        return data

    def _transform_fluview(self, data: Dict) -> Dict:
        if "fluview_json" in data:
            return data
        return {"fluview_json": data}

    def _transform_medications(self, data: Dict) -> Dict:
        if "medications" in data:
            meds = data["medications"]
            if isinstance(meds, list):
                return {"medications": [str(m) if not isinstance(m, str) else m for m in meds]}
        meds = []
        for key, value in data.items():
            if "med" in key.lower() or "drug" in key.lower():
                if isinstance(value, list):
                    meds.extend(value)
                elif isinstance(value, str):
                    meds.append(value)
        return {"medications": meds if meds else ["unknown"]}

    def _transform_patient_report(self, data: Dict) -> Dict:
        return {"executive_summary": data.get("executive_summary", ""), "clinical_signals": data.get("clinical_signals", []), "recommendations": data.get("recommendations", []), "reading_level": data.get("reading_level", "8th_grade")}

    def _transform_digital_twin_storage(self, data: Dict) -> Dict:
        return {"patient_id": data.get("patient_id", "unknown"), "dataset_id": data.get("dataset_id", f"ds_{datetime.now().strftime('%Y%m%d%H%M%S')}"), "analysis_id": data.get("analysis_id", f"an_{datetime.now().strftime('%Y%m%d%H%M%S')}"), "csv_content": data.get("csv_content", data.get("csv", ""))}

    def _transform_root_cause(self, data: Dict) -> Dict:
        return {"condition": data.get("condition", "unknown")}

    def _parse_fhir(self, data: Dict) -> pd.DataFrame:
        resource_type = data.get("resourceType", "")
        if resource_type == "Bundle":
            entries = data.get("entry", [])
            records = [self._flatten_fhir_resource(entry.get("resource", {})) for entry in entries]
            return pd.DataFrame(records) if records else pd.DataFrame()
        return pd.DataFrame([self._flatten_fhir_resource(data)])

    def _flatten_fhir_resource(self, resource: Dict) -> Dict:
        flat = {"patient_id": resource.get("subject", {}).get("reference", "").replace("Patient/", ""), "resource_type": resource.get("resourceType", "")}
        if resource.get("resourceType") == "Observation":
            code = resource.get("code", {})
            coding = code.get("coding", [{}])[0]
            flat["code"] = coding.get("code", "")
            flat["display"] = coding.get("display", code.get("text", ""))
            value = resource.get("valueQuantity", {})
            flat["value"] = value.get("value", 0)
            flat["unit"] = value.get("unit", "")
            flat["time"] = resource.get("effectiveDateTime", "")
        elif resource.get("resourceType") == "Patient":
            flat["patient_id"] = resource.get("id", "")
            name = resource.get("name", [{}])[0]
            flat["name"] = f"{name.get('given', [''])[0]} {name.get('family', '')}".strip()
            flat["sex"] = 1 if resource.get("gender") == "male" else 0
            flat["birth_date"] = resource.get("birthDate", "")
        return flat


def format_for_endpoint(data: Any, endpoint: str, strict: bool = False) -> Dict:
    """Convenience function to format data for a specific endpoint."""
    formatter = SmartFormatter()
    return formatter.normalize(data, endpoint, strict)
