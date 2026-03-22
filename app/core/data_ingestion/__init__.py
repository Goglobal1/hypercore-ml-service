"""
Data Ingestion Module - Bulletproof parsing for HyperCore.

Handles ANY input format and produces structured results.
NEVER crashes. ALWAYS produces useful output.
Includes evidence-based clinical domain classification.
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

from .domain_classifier import (
    ClinicalDomainClassifier,
    DomainClassificationResult,
    DomainScore,
    get_classifier,
    classify_domain,
    infer_risk_domain,
    ALL_DOMAINS,
)

__all__ = [
    # Robust parser
    "RobustDataParser",
    "DataQualityReport",
    "ParseResult",
    "get_parser",
    "parse_any_data",
    "extract_lab_data",
    "BIOMARKER_MAPPINGS",
    "DOMAIN_CRITICAL_BIOMARKERS",
    # Domain classifier
    "ClinicalDomainClassifier",
    "DomainClassificationResult",
    "DomainScore",
    "get_classifier",
    "classify_domain",
    "infer_risk_domain",
    "ALL_DOMAINS",
]
