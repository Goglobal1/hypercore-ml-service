"""
Evolution Audit Module
======================

FDA-compliant audit trails:
- Immutable logging of all evolution actions
- Cryptographic integrity verification (SHA-256 chain)
- Export for regulatory submission
"""

from .trail import AuditTrail, get_audit_trail, audit_log

__all__ = [
    "AuditTrail",
    "get_audit_trail",
    "audit_log",
]
