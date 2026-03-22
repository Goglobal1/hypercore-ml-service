"""
Data Ingestion Module - Bulletproof parsing for HyperCore.

Handles ANY input format and produces structured results.
NEVER crashes. ALWAYS produces useful output.
"""

from .robust_parser import (
    RobustDataParser,
    DataQualityReport,
    ParseResult,
    get_parser,
    parse_any_data,
    extract_lab_data,
    BIOMARKER_MAPPINGS,
    DOMAIN_CRITICAL_BIOMARKERS,
)

__all__ = [
    "RobustDataParser",
    "DataQualityReport",
    "ParseResult",
    "get_parser",
    "parse_any_data",
    "extract_lab_data",
    "BIOMARKER_MAPPINGS",
    "DOMAIN_CRITICAL_BIOMARKERS",
]
