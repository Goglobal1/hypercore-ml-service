"""
Data Sources for Diagnostic Engine
External medical knowledge bases integration.

- ClinVar: 209K+ genetic disease conditions
- ICD-10: 97K+ diagnosis codes
"""

from .clinvar_loader import ClinVarLoader, get_clinvar_loader
from .icd10_loader import ICD10Loader, get_icd10_loader

__all__ = [
    'ClinVarLoader',
    'get_clinvar_loader',
    'ICD10Loader',
    'get_icd10_loader'
]
