"""
Pattern Storage and Retrieval

Stores patterns from all modules, enables efficient querying.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from threading import Lock
import uuid

from .patterns import Pattern, PatternType, PatternSource


class PatternStore:
    """Central storage for all detected patterns. Thread-safe with TTL."""

    DEFAULT_TTL_HOURS = 72

    def __init__(self):
        self._lock = Lock()
        self._patterns: Dict[str, Pattern] = {}
        self._by_patient: Dict[str, Set[str]] = defaultdict(set)
        self._by_type: Dict[PatternType, Set[str]] = defaultdict(set)
        self._by_source: Dict[PatternSource, Set[str]] = defaultdict(set)
        self._by_biomarker: Dict[str, Set[str]] = defaultdict(set)
        self._by_gene: Dict[str, Set[str]] = defaultdict(set)
        self._by_drug: Dict[str, Set[str]] = defaultdict(set)
        self._timestamps: Dict[str, datetime] = {}

    def store(self, pattern: Pattern) -> str:
        """Store a pattern and return its ID."""
        with self._lock:
            if not pattern.id:
                pattern.id = f"{pattern.source.value}_{uuid.uuid4().hex[:12]}"

            self._patterns[pattern.id] = pattern
            self._timestamps[pattern.id] = datetime.now()

            self._by_patient[pattern.patient_id].add(pattern.id)
            self._by_type[pattern.pattern_type].add(pattern.id)
            self._by_source[pattern.source].add(pattern.id)

            for bio in pattern.related_biomarkers:
                self._by_biomarker[bio.lower()].add(pattern.id)
            for gene in pattern.related_genes:
                self._by_gene[gene.upper()].add(pattern.id)
            for drug in pattern.related_drugs:
                self._by_drug[drug.lower()].add(pattern.id)

            return pattern.id

    def get(self, pattern_id: str) -> Optional[Pattern]:
        with self._lock:
            return self._patterns.get(pattern_id)

    def get_by_patient(
        self,
        patient_id: str,
        pattern_types: Optional[List[PatternType]] = None,
        sources: Optional[List[PatternSource]] = None,
        min_confidence: float = 0.0,
        max_age_hours: Optional[float] = None
    ) -> List[Pattern]:
        """Get all patterns for a patient with optional filters."""
        with self._lock:
            pattern_ids = self._by_patient.get(patient_id, set())
            patterns = []
            cutoff = datetime.now() - timedelta(hours=max_age_hours) if max_age_hours else None

            for pid in pattern_ids:
                p = self._patterns.get(pid)
                if not p:
                    continue
                if pattern_types and p.pattern_type not in pattern_types:
                    continue
                if sources and p.source not in sources:
                    continue
                if p.confidence < min_confidence:
                    continue
                if cutoff and self._timestamps.get(pid, datetime.min) < cutoff:
                    continue
                patterns.append(p)

            return sorted(patterns, key=lambda x: x.confidence, reverse=True)

    def get_by_biomarker(self, biomarker: str) -> List[Pattern]:
        with self._lock:
            ids = self._by_biomarker.get(biomarker.lower(), set())
            return [self._patterns[i] for i in ids if i in self._patterns]

    def get_by_gene(self, gene: str) -> List[Pattern]:
        with self._lock:
            ids = self._by_gene.get(gene.upper(), set())
            return [self._patterns[i] for i in ids if i in self._patterns]

    def get_by_drug(self, drug: str) -> List[Pattern]:
        with self._lock:
            ids = self._by_drug.get(drug.lower(), set())
            return [self._patterns[i] for i in ids if i in self._patterns]

    def get_by_type(self, pattern_type: PatternType) -> List[Pattern]:
        with self._lock:
            ids = self._by_type.get(pattern_type, set())
            return [self._patterns[i] for i in ids if i in self._patterns]

    def get_by_source(self, source: PatternSource) -> List[Pattern]:
        with self._lock:
            ids = self._by_source.get(source, set())
            return [self._patterns[i] for i in ids if i in self._patterns]

    def get_recent(self, hours: float = 24, min_severity: float = 0.0) -> List[Pattern]:
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            patterns = []
            for pid, ts in self._timestamps.items():
                if ts >= cutoff:
                    p = self._patterns.get(pid)
                    if p and p.severity >= min_severity:
                        patterns.append(p)
            return sorted(patterns, key=lambda x: x.severity, reverse=True)

    def find_related(self, pattern: Pattern) -> List[Pattern]:
        """Find patterns that might correlate with this one."""
        with self._lock:
            related_ids = set()
            for bio in pattern.related_biomarkers:
                related_ids.update(self._by_biomarker.get(bio.lower(), set()))
            for gene in pattern.related_genes:
                related_ids.update(self._by_gene.get(gene.upper(), set()))
            for drug in pattern.related_drugs:
                related_ids.update(self._by_drug.get(drug.lower(), set()))
            related_ids.discard(pattern.id)

            patterns = [self._patterns[i] for i in related_ids if i in self._patterns]

            def relevance(p):
                score = len(set(p.related_biomarkers) & set(pattern.related_biomarkers))
                score += len(set(p.related_genes) & set(pattern.related_genes)) * 2
                score += len(set(p.related_drugs) & set(pattern.related_drugs)) * 1.5
                return score * p.confidence

            return sorted(patterns, key=relevance, reverse=True)

    def cleanup_expired(self) -> int:
        with self._lock:
            cutoff = datetime.now() - timedelta(hours=self.DEFAULT_TTL_HOURS)
            expired = [pid for pid, ts in self._timestamps.items() if ts < cutoff]

            for pid in expired:
                p = self._patterns.pop(pid, None)
                self._timestamps.pop(pid, None)
                if p:
                    self._by_patient[p.patient_id].discard(pid)
                    self._by_type[p.pattern_type].discard(pid)
                    self._by_source[p.source].discard(pid)
                    for bio in p.related_biomarkers:
                        self._by_biomarker[bio.lower()].discard(pid)
                    for gene in p.related_genes:
                        self._by_gene[gene.upper()].discard(pid)
                    for drug in p.related_drugs:
                        self._by_drug[drug.lower()].discard(pid)

            return len(expired)

    def get_statistics(self) -> Dict:
        with self._lock:
            return {
                "total_patterns": len(self._patterns),
                "unique_patients": len(self._by_patient),
                "by_type": {t.value: len(ids) for t, ids in self._by_type.items() if ids},
                "by_source": {s.value: len(ids) for s, ids in self._by_source.items() if ids},
                "biomarkers_tracked": len(self._by_biomarker),
                "genes_tracked": len(self._by_gene),
                "drugs_tracked": len(self._by_drug)
            }

    def clear(self):
        """Clear all patterns (for testing)."""
        with self._lock:
            self._patterns.clear()
            self._by_patient.clear()
            self._by_type.clear()
            self._by_source.clear()
            self._by_biomarker.clear()
            self._by_gene.clear()
            self._by_drug.clear()
            self._timestamps.clear()
