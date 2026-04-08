"""
Layer 1: Input Normalization
Converts any input format into standardized features.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json
import os


class InputNormalizer:
    """
    Layer 1: Normalize any input into standardized feature vectors.
    """

    def __init__(self, ontology: Dict = None):
        if ontology is None:
            # Load from file
            data_dir = os.path.dirname(os.path.dirname(__file__))
            ontology_path = os.path.join(data_dir, 'data', 'biomarker_ontology.json')
            with open(ontology_path, 'r') as f:
                ontology = json.load(f)
        self.ontology = ontology
        self._build_alias_map()

    def _build_alias_map(self):
        """Build reverse lookup from aliases to canonical names."""
        self.alias_map = {}
        for canonical, info in self.ontology.get('biomarkers', {}).items():
            # Add canonical name
            self.alias_map[canonical.lower()] = canonical
            # Add all aliases
            for alias in info.get('aliases', []):
                self.alias_map[alias.lower()] = canonical

    def normalize(self, raw_patient_data: Dict) -> Dict:
        """
        Takes raw patient data with any column names/units.
        Returns standardized feature dictionary.
        """
        normalized = {
            'patient_id': self._extract_patient_id(raw_patient_data),
            'timestamp': self._extract_timestamp(raw_patient_data),
            'features': {},
            'metadata': {
                'source': 'unknown',
                'completeness': 0.0,
                'normalization_warnings': [],
                'unmapped_fields': []
            }
        }

        # Track unmapped fields
        unmapped = []

        for raw_key, raw_value in raw_patient_data.items():
            # Skip metadata fields
            if raw_key.lower() in ['patient_id', 'timestamp', 'date', 'time', 'visit_id', 'encounter_id']:
                continue

            # Skip None/null values
            if raw_value is None:
                continue

            # Find canonical biomarker name
            canonical = self._resolve_biomarker(raw_key)
            if canonical is None:
                unmapped.append(raw_key)
                continue

            # Convert value to canonical units
            value = self._convert_units(raw_value, raw_key, canonical)
            if value is None:
                continue

            normalized['features'][canonical] = {
                'value': value,
                'unit': self.ontology['biomarkers'][canonical].get('unit_canonical', ''),
                'source': 'measured',
                'raw_key': raw_key,
                'raw_value': raw_value
            }

        # Store unmapped fields in metadata
        normalized['metadata']['unmapped_fields'] = unmapped

        # Calculate completeness (based on typical panel of 20 markers)
        normalized['metadata']['completeness'] = min(len(normalized['features']) / 20, 1.0)

        return normalized

    def _extract_patient_id(self, data: Dict) -> str:
        """Extract patient ID from various possible field names."""
        for key in ['patient_id', 'PatientId', 'PATIENT_ID', 'id', 'ID', 'subject_id', 'mrn']:
            if key in data:
                return str(data[key])
        return 'unknown'

    def _extract_timestamp(self, data: Dict) -> str:
        """Extract timestamp from data."""
        for key in ['timestamp', 'date', 'datetime', 'collection_date', 'collected_at']:
            if key in data:
                val = data[key]
                if isinstance(val, str):
                    return val
                elif hasattr(val, 'isoformat'):
                    return val.isoformat()
        return datetime.utcnow().isoformat()

    def _resolve_biomarker(self, raw_key: str) -> Optional[str]:
        """Map any column name to canonical biomarker name."""
        # Clean the key
        raw_lower = raw_key.lower().strip()
        raw_lower = raw_lower.replace('_value', '').replace('_result', '').replace('_level', '')

        # Direct lookup
        if raw_lower in self.alias_map:
            return self.alias_map[raw_lower]

        # Try partial match
        for alias, canonical in self.alias_map.items():
            if alias in raw_lower or raw_lower in alias:
                return canonical

        return None

    def _convert_units(self, raw_value: Any, raw_key: str, canonical: str) -> Optional[float]:
        """Convert value to canonical units."""
        try:
            # Handle string values
            if isinstance(raw_value, str):
                # Remove common suffixes
                raw_value = raw_value.strip().replace(',', '')
                for suffix in ['mg/dL', 'g/dL', 'U/L', 'mEq/L', '%', 'mmHg', 'bpm']:
                    raw_value = raw_value.replace(suffix, '').strip()

            value = float(raw_value)

            # Get conversion info
            biomarker_info = self.ontology['biomarkers'].get(canonical, {})
            conversions = biomarker_info.get('unit_conversions', {})

            # Check if raw key indicates a unit that needs conversion
            raw_lower = raw_key.lower()
            for unit, factor in conversions.items():
                if unit.lower() in raw_lower:
                    value = value * factor
                    break

            return value

        except (ValueError, TypeError):
            return None

    def get_organ_system(self, canonical: str) -> str:
        """Get the organ system for a biomarker."""
        info = self.ontology['biomarkers'].get(canonical, {})
        return info.get('organ_system', 'unknown')

    def get_pathway(self, canonical: str) -> str:
        """Get the biological pathway for a biomarker."""
        info = self.ontology['biomarkers'].get(canonical, {})
        return info.get('pathway', 'unknown')
