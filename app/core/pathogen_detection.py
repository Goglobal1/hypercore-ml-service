"""
Pathogen Detection Engine for HyperCore

Provides:
1. WHO surveillance data analysis
2. Antimicrobial resistance (AMR) tracking
3. Outbreak detection and alerts
4. Vaccination coverage monitoring
5. Clinical-pathogen correlation analysis

Data sources (configured via environment variables):
- WHO_PATH: WHO surveillance data
- CDC_WONDER_PATH: CDC WONDER data
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import statistics

logger = logging.getLogger(__name__)

# Data paths - use environment variables with fallbacks
_BASE_DIR = Path(__file__).parent.parent
WHO_PATH = Path(os.environ.get('WHO_PATH', _BASE_DIR / 'data' / 'who'))
WHO_INDICATORS_PATH = WHO_PATH / "Indicators"
CDC_WONDER_PATH = Path(os.environ.get('CDC_WONDER_PATH', _BASE_DIR / 'data' / 'cdc_wonder'))

# Known pathogens and their characteristics
PATHOGEN_DATABASE = {
    # Bacteria
    "staphylococcus_aureus": {
        "type": "bacterial",
        "common_name": "Staph aureus",
        "diseases": ["skin infections", "pneumonia", "sepsis", "endocarditis"],
        "amr_concern": True,
        "key_resistances": ["methicillin", "vancomycin"],
        "icd10_codes": ["A49.0", "J15.2", "A41.0"]
    },
    "escherichia_coli": {
        "type": "bacterial",
        "common_name": "E. coli",
        "diseases": ["uti", "sepsis", "gastroenteritis", "meningitis"],
        "amr_concern": True,
        "key_resistances": ["fluoroquinolones", "cephalosporins", "carbapenems"],
        "icd10_codes": ["A04.0", "A41.5", "N39.0"]
    },
    "klebsiella_pneumoniae": {
        "type": "bacterial",
        "common_name": "Klebsiella",
        "diseases": ["pneumonia", "uti", "sepsis"],
        "amr_concern": True,
        "key_resistances": ["carbapenems", "cephalosporins"],
        "icd10_codes": ["J15.0", "A41.8"]
    },
    "pseudomonas_aeruginosa": {
        "type": "bacterial",
        "common_name": "Pseudomonas",
        "diseases": ["pneumonia", "uti", "wound infections"],
        "amr_concern": True,
        "key_resistances": ["carbapenems", "aminoglycosides"],
        "icd10_codes": ["J15.1", "A41.8"]
    },
    "clostridioides_difficile": {
        "type": "bacterial",
        "common_name": "C. diff",
        "diseases": ["colitis", "diarrhea"],
        "amr_concern": True,
        "key_resistances": ["fluoroquinolones"],
        "icd10_codes": ["A04.7"]
    },
    "mycobacterium_tuberculosis": {
        "type": "bacterial",
        "common_name": "TB",
        "diseases": ["tuberculosis"],
        "amr_concern": True,
        "key_resistances": ["rifampicin", "isoniazid"],
        "icd10_codes": ["A15", "A16", "A17", "A18", "A19"]
    },
    # Viruses
    "sars_cov_2": {
        "type": "viral",
        "common_name": "COVID-19",
        "diseases": ["covid-19", "ards", "pneumonia"],
        "amr_concern": False,
        "key_resistances": [],
        "icd10_codes": ["U07.1", "U07.2"]
    },
    "influenza": {
        "type": "viral",
        "common_name": "Flu",
        "diseases": ["influenza", "pneumonia"],
        "amr_concern": False,
        "key_resistances": ["oseltamivir"],
        "icd10_codes": ["J09", "J10", "J11"]
    },
    "hiv": {
        "type": "viral",
        "common_name": "HIV",
        "diseases": ["aids", "opportunistic infections"],
        "amr_concern": True,
        "key_resistances": ["nrti", "nnrti", "pi"],
        "icd10_codes": ["B20", "B21", "B22", "B23", "B24"]
    },
    # Fungi
    "candida_auris": {
        "type": "fungal",
        "common_name": "C. auris",
        "diseases": ["candidemia", "wound infections"],
        "amr_concern": True,
        "key_resistances": ["fluconazole", "amphotericin b", "echinocandins"],
        "icd10_codes": ["B37"]
    },
    "aspergillus": {
        "type": "fungal",
        "common_name": "Aspergillus",
        "diseases": ["aspergillosis", "pneumonia"],
        "amr_concern": True,
        "key_resistances": ["azoles"],
        "icd10_codes": ["B44"]
    },
    # Parasites
    "plasmodium": {
        "type": "parasitic",
        "common_name": "Malaria",
        "diseases": ["malaria"],
        "amr_concern": True,
        "key_resistances": ["chloroquine", "artemisinin"],
        "icd10_codes": ["B50", "B51", "B52", "B53", "B54"]
    }
}

# Disease to pathogen mapping
DISEASE_PATHOGEN_MAP = {
    "sepsis": ["staphylococcus_aureus", "escherichia_coli", "klebsiella_pneumoniae"],
    "pneumonia": ["staphylococcus_aureus", "klebsiella_pneumoniae", "pseudomonas_aeruginosa", "influenza", "sars_cov_2"],
    "uti": ["escherichia_coli", "klebsiella_pneumoniae", "pseudomonas_aeruginosa"],
    "covid": ["sars_cov_2"],
    "tuberculosis": ["mycobacterium_tuberculosis"],
    "malaria": ["plasmodium"],
    "candidiasis": ["candida_auris"],
    "c_diff": ["clostridioides_difficile"],
}


@dataclass
class WHOIndicatorRecord:
    """Single WHO indicator record."""
    indicator_id: str
    indicator_code: str
    indicator_name: str
    country: str
    country_code: str
    year: int
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    category: Optional[str] = None


class WHODataLoader:
    """Loader for WHO surveillance data."""

    def __init__(self, who_path: Path = WHO_INDICATORS_PATH):
        self.who_path = who_path
        self._indicator_cache: Dict[str, List[WHOIndicatorRecord]] = {}
        self._loaded = False

    def list_available_indicators(self) -> List[str]:
        """List available WHO indicator files."""
        if not self.who_path.exists():
            return []

        indicators = []
        for f in self.who_path.glob("*.csv"):
            indicators.append(f.stem)
        return indicators

    def _parse_indicator_file(self, file_path: Path, max_records: int = 5000) -> List[WHOIndicatorRecord]:
        """Parse a WHO indicator CSV file."""
        records = []

        try:
            with open(file_path, 'r', encoding='utf-8-sig', errors='replace') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    if i >= max_records:
                        break

                    # Extract common fields
                    try:
                        value = float(row.get('RATE_PER_100_N') or row.get('RATE_PER_1000_N') or
                                     row.get('RATE_PER_100000_N') or row.get('VALUE') or 0)
                    except (ValueError, TypeError):
                        value = 0.0

                    try:
                        year = int(row.get('DIM_TIME') or row.get('YEAR') or 0)
                    except (ValueError, TypeError):
                        year = 0

                    if year == 0:
                        continue

                    record = WHOIndicatorRecord(
                        indicator_id=row.get('IND_ID', ''),
                        indicator_code=row.get('IND_CODE', ''),
                        indicator_name=row.get('IND_NAME', ''),
                        country=row.get('GEO_NAME_SHORT') or row.get('COUNTRY', ''),
                        country_code=row.get('DIM_GEO_CODE_M49', ''),
                        year=year,
                        value=value,
                        lower_bound=self._safe_float(row.get('RATE_PER_100_NL') or row.get('RATE_PER_1000_NL')),
                        upper_bound=self._safe_float(row.get('RATE_PER_100_NU') or row.get('RATE_PER_1000_NU')),
                        category=row.get('DIM_AMR_GLASS_AWARE') or row.get('CATEGORY')
                    )
                    records.append(record)

        except Exception as e:
            logger.error(f"Error parsing WHO file {file_path}: {e}")

        return records

    def _safe_float(self, value) -> Optional[float]:
        """Safely convert to float."""
        if value is None or value == '':
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def search_indicators(
        self,
        indicator_name: Optional[str] = None,
        country: Optional[str] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search WHO indicators with filters."""
        results = []

        for f in self.who_path.glob("*.csv"):
            records = self._parse_indicator_file(f, max_records=2000)

            for r in records:
                # Apply filters
                if indicator_name and indicator_name.lower() not in r.indicator_name.lower():
                    continue
                if country and country.lower() not in r.country.lower():
                    continue
                if year_from and r.year < year_from:
                    continue
                if year_to and r.year > year_to:
                    continue

                results.append(asdict(r))

                if len(results) >= limit:
                    return results

        return results

    def get_covid_vaccination_data(self, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get COVID vaccination data."""
        results = []

        # Look for COVID vaccination files
        for pattern in ["COV_VAC_*.csv"]:
            for f in self.who_path.glob(pattern):
                try:
                    with open(f, 'r', encoding='utf-8-sig', errors='replace') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if country and country.lower() not in row.get('COUNTRY', '').lower():
                                continue

                            results.append({
                                "country": row.get('COUNTRY', ''),
                                "indicator": row.get('INDICATOR', ''),
                                "value": row.get('VALUE', ''),
                                "category": row.get('CATEGORY', ''),
                                "year": row.get('YEAR', ''),
                                "quarter": row.get('QUARTER', '')
                            })

                            if len(results) >= 500:
                                return results

                except Exception as e:
                    logger.error(f"Error reading COVID file {f}: {e}")

        return results

    def get_amr_data(self, country: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get antimicrobial resistance data."""
        results = []

        # Search for AMR-related indicators
        amr_keywords = ["antibiotic", "antimicrobial", "resistance", "amr", "glass"]

        for f in self.who_path.glob("*.csv"):
            # Quick check if file might contain AMR data
            fname_lower = f.name.lower()

            records = self._parse_indicator_file(f, max_records=1000)

            for r in records:
                if not any(kw in r.indicator_name.lower() for kw in amr_keywords):
                    continue

                if country and country.lower() not in r.country.lower():
                    continue

                results.append(asdict(r))

                if len(results) >= 200:
                    return results

        return results


class CDCWonderLoader:
    """Loader for CDC WONDER mortality data."""

    def __init__(self, cdc_path: Path = CDC_WONDER_PATH):
        self.cdc_path = cdc_path

    def get_mortality_data(self, cause: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get mortality data from CDC WONDER."""
        results = []

        if not self.cdc_path.exists():
            return results

        # CDC WONDER files are typically Excel
        # For now, return metadata about available files
        for f in self.cdc_path.glob("*.xls*"):
            results.append({
                "file": f.name,
                "type": "mortality",
                "format": "excel",
                "size_mb": f.stat().st_size / (1024 * 1024)
            })

        return results


class PathogenDetectionEngine:
    """
    Main pathogen detection and surveillance engine.
    """

    def __init__(self):
        self.who_loader = WHODataLoader()
        self.cdc_loader = CDCWonderLoader()

    def get_pathogen_info(self, pathogen: str) -> Dict[str, Any]:
        """Get information about a pathogen."""
        pathogen_lower = pathogen.lower().replace(" ", "_").replace("-", "_")

        # Direct match
        if pathogen_lower in PATHOGEN_DATABASE:
            info = PATHOGEN_DATABASE[pathogen_lower]
            return {
                "pathogen": pathogen_lower,
                "found": True,
                **info
            }

        # Search by common name
        for key, info in PATHOGEN_DATABASE.items():
            if pathogen_lower in info.get("common_name", "").lower():
                return {
                    "pathogen": key,
                    "found": True,
                    **info
                }

        return {
            "pathogen": pathogen,
            "found": False,
            "note": f"Pathogen '{pathogen}' not in database"
        }

    def get_disease_pathogens(self, disease: str) -> List[Dict[str, Any]]:
        """Get pathogens associated with a disease."""
        disease_lower = disease.lower()

        pathogens = []
        for disease_key, pathogen_list in DISEASE_PATHOGEN_MAP.items():
            if disease_lower in disease_key:
                for p in pathogen_list:
                    if p in PATHOGEN_DATABASE:
                        pathogens.append({
                            "pathogen": p,
                            **PATHOGEN_DATABASE[p]
                        })

        return pathogens

    def detect_outbreaks(
        self,
        regions: Optional[List[str]] = None,
        threshold_multiplier: float = 1.5,
        lookback_years: int = 5
    ) -> Dict[str, Any]:
        """
        Detect potential outbreaks based on surveillance data.

        Uses statistical analysis to identify unusual increases
        in disease indicators.
        """
        alerts = []
        current_year = datetime.now().year

        # Get recent surveillance data
        indicators = self.who_loader.search_indicators(
            year_from=current_year - lookback_years,
            limit=1000
        )

        # Group by indicator and country
        grouped = defaultdict(list)
        for ind in indicators:
            key = (ind.get("indicator_name", ""), ind.get("country", ""))
            grouped[key].append(ind)

        # Analyze each group for anomalies
        for (indicator_name, country), records in grouped.items():
            if regions and not any(r.lower() in country.lower() for r in regions):
                continue

            if len(records) < 3:
                continue

            # Sort by year
            records.sort(key=lambda x: x.get("year", 0))

            # Get historical values and current
            values = [r.get("value", 0) for r in records[:-1] if r.get("value")]
            current = records[-1].get("value", 0) if records else 0

            if not values or current == 0:
                continue

            # Calculate statistics
            try:
                mean_val = statistics.mean(values)
                std_val = statistics.stdev(values) if len(values) > 1 else mean_val * 0.1

                # Check if current value exceeds threshold
                if std_val > 0 and current > mean_val + (threshold_multiplier * std_val):
                    deviation = (current - mean_val) / std_val

                    # Determine alert level
                    if deviation > 3:
                        alert_level = "critical"
                    elif deviation > 2:
                        alert_level = "high"
                    elif deviation > 1.5:
                        alert_level = "moderate"
                    else:
                        alert_level = "low"

                    alerts.append({
                        "indicator": indicator_name,
                        "country": country,
                        "alert_level": alert_level,
                        "current_value": current,
                        "historical_mean": round(mean_val, 2),
                        "deviation_std": round(deviation, 2),
                        "trend": "increasing" if current > values[-1] else "stable",
                        "year": records[-1].get("year"),
                        "confidence": min(0.95, 0.5 + (len(values) * 0.05))
                    })

            except Exception as e:
                continue

        # Sort by alert level
        level_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
        alerts.sort(key=lambda x: level_order.get(x["alert_level"], 4))

        return {
            "alerts": alerts[:20],  # Top 20
            "total_alerts": len(alerts),
            "critical_alerts": sum(1 for a in alerts if a["alert_level"] == "critical"),
            "regions_analyzed": list(set(a["country"] for a in alerts)),
            "analysis_period": f"{current_year - lookback_years}-{current_year}"
        }

    def analyze_amr(
        self,
        pathogen: Optional[str] = None,
        antibiotic: Optional[str] = None,
        country: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze antimicrobial resistance patterns."""
        # Get AMR data from WHO
        amr_data = self.who_loader.get_amr_data(country)

        # Get pathogen-specific AMR info
        high_risk = []
        for p_key, p_info in PATHOGEN_DATABASE.items():
            if pathogen and pathogen.lower() not in p_key:
                continue
            if not p_info.get("amr_concern"):
                continue

            resistances = p_info.get("key_resistances", [])
            if antibiotic and antibiotic.lower() not in [r.lower() for r in resistances]:
                continue

            high_risk.append({
                "pathogen": p_key,
                "common_name": p_info["common_name"],
                "key_resistances": resistances,
                "diseases": p_info["diseases"]
            })

        return {
            "amr_surveillance_data": amr_data[:20],
            "high_risk_pathogens": high_risk,
            "total_amr_records": len(amr_data),
            "recommendations": [
                "Monitor local resistance patterns",
                "Follow antimicrobial stewardship guidelines",
                "Report unusual resistance patterns to public health"
            ]
        }

    def get_vaccination_coverage(self, disease: Optional[str] = None, country: Optional[str] = None) -> Dict[str, Any]:
        """Get vaccination coverage data."""
        covid_data = self.who_loader.get_covid_vaccination_data(country)

        # Filter by disease if specified
        if disease:
            covid_data = [
                d for d in covid_data
                if disease.lower() in d.get("indicator", "").lower()
            ]

        return {
            "vaccination_data": covid_data[:50],
            "total_records": len(covid_data),
            "data_type": "COVID-19 vaccination" if not disease else disease
        }

    def get_clinical_pathogen_correlation(self, pathogen: str) -> Dict[str, Any]:
        """Get correlation between pathogen and clinical outcomes."""
        pathogen_info = self.get_pathogen_info(pathogen)

        if not pathogen_info.get("found"):
            return {
                "pathogen": pathogen,
                "correlations": [],
                "note": "Pathogen not found in database"
            }

        correlations = []
        for disease in pathogen_info.get("diseases", []):
            for icd_code in pathogen_info.get("icd10_codes", []):
                correlations.append({
                    "disease": disease,
                    "icd10_code": icd_code,
                    "pathogen_type": pathogen_info.get("type"),
                    "amr_concern": pathogen_info.get("amr_concern", False)
                })

        return {
            "pathogen": pathogen,
            "common_name": pathogen_info.get("common_name"),
            "pathogen_type": pathogen_info.get("type"),
            "clinical_correlations": correlations,
            "key_resistances": pathogen_info.get("key_resistances", [])
        }


# Singleton instance
_engine = None


def get_engine() -> PathogenDetectionEngine:
    """Get singleton engine instance."""
    global _engine
    if _engine is None:
        _engine = PathogenDetectionEngine()
    return _engine


# API-friendly functions
def get_pathogen_info(pathogen: str) -> Dict[str, Any]:
    """Get pathogen information."""
    return get_engine().get_pathogen_info(pathogen)


def get_disease_pathogens(disease: str) -> List[Dict[str, Any]]:
    """Get pathogens for a disease."""
    return get_engine().get_disease_pathogens(disease)


def detect_outbreaks(
    regions: Optional[List[str]] = None,
    threshold_multiplier: float = 1.5,
    lookback_years: int = 5
) -> Dict[str, Any]:
    """Detect outbreaks."""
    return get_engine().detect_outbreaks(regions, threshold_multiplier, lookback_years)


def analyze_amr(
    pathogen: Optional[str] = None,
    antibiotic: Optional[str] = None,
    country: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze AMR patterns."""
    return get_engine().analyze_amr(pathogen, antibiotic, country)


def get_vaccination_coverage(disease: Optional[str] = None, country: Optional[str] = None) -> Dict[str, Any]:
    """Get vaccination coverage."""
    return get_engine().get_vaccination_coverage(disease, country)


def search_surveillance(
    indicator_name: Optional[str] = None,
    country: Optional[str] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """Search surveillance indicators."""
    engine = get_engine()
    results = engine.who_loader.search_indicators(
        indicator_name, country, year_from, year_to, limit
    )
    return {
        "results": results,
        "count": len(results),
        "query": {
            "indicator_name": indicator_name,
            "country": country,
            "year_from": year_from,
            "year_to": year_to
        }
    }


def get_data_status() -> Dict[str, Any]:
    """Get status of pathogen detection data sources."""
    engine = get_engine()

    who_indicators = engine.who_loader.list_available_indicators()

    return {
        "who_available": WHO_INDICATORS_PATH.exists(),
        "who_indicator_files": len(who_indicators),
        "cdc_wonder_available": CDC_WONDER_PATH.exists(),
        "pathogens_indexed": len(PATHOGEN_DATABASE),
        "disease_mappings": len(DISEASE_PATHOGEN_MAP)
    }
