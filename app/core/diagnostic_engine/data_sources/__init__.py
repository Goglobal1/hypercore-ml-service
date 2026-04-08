"""
Data Sources for Diagnostic Engine
External medical knowledge bases integration.
"""

from .clinvar_loader import ClinVarLoader, get_clinvar_loader

__all__ = ['ClinVarLoader', 'get_clinvar_loader']
