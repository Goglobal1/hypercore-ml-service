"""
Cross-Domain Correlation Engine

The MAGIC happens here - patterns from different domains are correlated
to produce insights that no single domain could generate alone.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import numpy as np

from .patterns import (
    Pattern, PatternType, PatternSource,
    TrajectoryPattern, GenomicPattern, PharmaPattern,
    PathogenPattern, AlertPattern
)


class CorrelationType(Enum):
    TRAJECTORY_GENOMIC = "trajectory_genomic"
    TRAJECTORY_PHARMA = "trajectory_pharma"
    TRAJECTORY_PATHOGEN = "trajectory_pathogen"
    GENOMIC_PHARMA = "genomic_pharma"
    GENOMIC_PATHOGEN = "genomic_pathogen"
    MULTI_DOMAIN = "multi_domain"
    TEMPORAL_CASCADE = "temporal_cascade"
    POPULATION_CLUSTER = "population_cluster"


@dataclass
class Correlation:
    """A cross-domain correlation between patterns."""
    id: str
    correlation_type: CorrelationType
    patterns_involved: List[str]
    domains_involved: List[str]
    strength: float
    clinical_significance: str
    implication: str
    action_required: str
    urgency: str  # immediate, urgent, routine, informational
    risk_multiplier: float = 1.0
    detection_improvement_days: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.correlation_type.value,
            "patterns": self.patterns_involved,
            "domains": self.domains_involved,
            "strength": self.strength,
            "clinical_significance": self.clinical_significance,
            "implication": self.implication,
            "action": self.action_required,
            "urgency": self.urgency,
            "risk_multiplier": self.risk_multiplier,
            "detection_improvement_days": self.detection_improvement_days,
            "recommendations": self.recommendations
        }


class CrossDomainCorrelator:
    """Engine that finds correlations between patterns from different domains."""

    # Known correlation rules: (domain, gene/drug/condition) -> implications
    RULES = {
        ("sepsis", "CYP2D6"): {
            "impl": "CYP2D6 affects antibiotic metabolism in sepsis",
            "action": "Adjust antibiotic dosing based on CYP2D6 status",
            "mult": 1.3,
            "urg": "urgent"
        },
        ("sepsis", "DPYD"): {
            "impl": "DPYD variant affects sepsis treatment options",
            "action": "Avoid fluoropyrimidines if DPYD deficient",
            "mult": 1.4,
            "urg": "urgent"
        },
        ("sepsis", "IL6"): {
            "impl": "IL6 variants affect inflammatory response in sepsis",
            "action": "Consider IL-6 inhibitor therapy",
            "mult": 1.2,
            "urg": "urgent"
        },
        ("cardiac", "CYP2C19"): {
            "impl": "CYP2C19 affects clopidogrel efficacy",
            "action": "Alternative antiplatelet or dose adjustment needed",
            "mult": 1.5,
            "urg": "immediate"
        },
        ("cardiac", "VKORC1"): {
            "impl": "VKORC1 affects warfarin sensitivity",
            "action": "Pharmacogenomic-guided warfarin dosing required",
            "mult": 1.2,
            "urg": "urgent"
        },
        ("cardiac", "CYP2C9"): {
            "impl": "CYP2C9 affects warfarin metabolism",
            "action": "Adjust warfarin dose based on CYP2C9 status",
            "mult": 1.2,
            "urg": "urgent"
        },
        ("renal", "SLCO1B1"): {
            "impl": "SLCO1B1 affects statin clearance in renal decline",
            "action": "Reduce statin dose, monitor for myopathy",
            "mult": 1.3,
            "urg": "routine"
        },
        ("renal", "ABCB1"): {
            "impl": "ABCB1 affects drug transport in kidney",
            "action": "Adjust doses of ABCB1 substrates",
            "mult": 1.2,
            "urg": "routine"
        },
        ("hepatic", "CYP2D6"): {
            "impl": "CYP2D6 status critical during hepatic dysfunction",
            "action": "Avoid CYP2D6-dependent drugs or reduce dose",
            "mult": 1.4,
            "urg": "urgent"
        },
        ("hepatic", "UGT1A1"): {
            "impl": "UGT1A1 affects bilirubin and drug conjugation",
            "action": "Avoid UGT1A1-dependent drugs",
            "mult": 1.3,
            "urg": "urgent"
        },
        ("oncology", "DPYD"): {
            "impl": "DPYD deficiency causes severe 5-FU toxicity",
            "action": "CONTRAINDICATED: Avoid fluoropyrimidines",
            "mult": 2.0,
            "urg": "immediate"
        },
        ("oncology", "UGT1A1"): {
            "impl": "UGT1A1*28 causes irinotecan toxicity",
            "action": "Reduce irinotecan dose by 30%",
            "mult": 1.5,
            "urg": "immediate"
        },
    }

    def __init__(self):
        self._cache: Dict[str, List[Correlation]] = {}

    def correlate(self, patterns: List[Pattern], patient_id: Optional[str] = None) -> List[Correlation]:
        """Find all correlations among a set of patterns."""
        correlations = []

        # Group by source
        by_source: Dict[PatternSource, List[Pattern]] = {}
        for p in patterns:
            by_source.setdefault(p.source, []).append(p)

        # Trajectory + Genomics
        if PatternSource.TRAJECTORY in by_source and PatternSource.GENOMICS in by_source:
            correlations.extend(self._correlate_trajectory_genomics(
                by_source[PatternSource.TRAJECTORY],
                by_source[PatternSource.GENOMICS]
            ))

        # Trajectory + Pharma
        if PatternSource.TRAJECTORY in by_source and PatternSource.PHARMA in by_source:
            correlations.extend(self._correlate_trajectory_pharma(
                by_source[PatternSource.TRAJECTORY],
                by_source[PatternSource.PHARMA]
            ))

        # Trajectory + Pathogen
        if PatternSource.TRAJECTORY in by_source and PatternSource.PATHOGEN in by_source:
            correlations.extend(self._correlate_trajectory_pathogen(
                by_source[PatternSource.TRAJECTORY],
                by_source[PatternSource.PATHOGEN]
            ))

        # Genomics + Pharma
        if PatternSource.GENOMICS in by_source and PatternSource.PHARMA in by_source:
            correlations.extend(self._correlate_genomics_pharma(
                by_source[PatternSource.GENOMICS],
                by_source[PatternSource.PHARMA]
            ))

        # Temporal cascade detection
        cascade = self._detect_cascade(patterns)
        if cascade:
            correlations.append(cascade)

        # Multi-domain (3+ sources)
        if len(by_source) >= 3:
            multi = self._correlate_multi_domain(patterns, by_source)
            if multi:
                correlations.append(multi)

        if patient_id:
            self._cache[patient_id] = correlations

        return sorted(correlations, key=lambda c: c.strength, reverse=True)

    def _correlate_trajectory_genomics(
        self,
        traj: List[Pattern],
        genomic: List[Pattern]
    ) -> List[Correlation]:
        """Correlate trajectory patterns with genomic findings."""
        corrs = []

        for t in traj:
            domain = t.evidence.get('domain', '') or t.evidence.get('pattern_name', '')
            domain = domain.lower() if domain else ''

            for g in genomic:
                if not isinstance(g, GenomicPattern):
                    continue
                gene = g.gene.upper() if hasattr(g, 'gene') else ''

                # Check against known rules
                for (rule_dom, rule_gene), rule_data in self.RULES.items():
                    if rule_dom in domain and rule_gene in gene:
                        corrs.append(Correlation(
                            id=f"corr_{t.id}_{g.id}",
                            correlation_type=CorrelationType.TRAJECTORY_GENOMIC,
                            patterns_involved=[t.id, g.id],
                            domains_involved=["trajectory", "genomics"],
                            strength=min(t.confidence * g.confidence * 1.2, 1.0),
                            clinical_significance=rule_data["impl"],
                            implication=rule_data["impl"],
                            action_required=rule_data["action"],
                            urgency=rule_data["urg"],
                            risk_multiplier=rule_data["mult"],
                            detection_improvement_days=t.onset_days_ago or 0,
                            recommendations=[
                                rule_data["action"],
                                f"Monitor {gene} implications",
                                "Pharmacogenomic-guided therapy recommended"
                            ]
                        ))

        return corrs

    def _correlate_trajectory_pharma(
        self,
        traj: List[Pattern],
        pharma: List[Pattern]
    ) -> List[Correlation]:
        """Correlate trajectory patterns with drug interactions."""
        corrs = []

        for t in traj:
            domain = (t.evidence.get('domain', '') or '').lower()

            for p in pharma:
                if not isinstance(p, PharmaPattern):
                    continue

                effect = p.effect.lower() if hasattr(p, 'effect') else ''
                interaction_type = p.interaction_type if hasattr(p, 'interaction_type') else ''
                drug_a = p.drug_a if hasattr(p, 'drug_a') else 'unknown'
                management = p.management if hasattr(p, 'management') else ''

                # Renal + nephrotoxic
                if 'renal' in domain and 'nephrotox' in effect:
                    corrs.append(Correlation(
                        id=f"corr_{t.id}_{p.id}_nephro",
                        correlation_type=CorrelationType.TRAJECTORY_PHARMA,
                        patterns_involved=[t.id, p.id],
                        domains_involved=["trajectory", "pharma"],
                        strength=min(t.severity * 1.5, 1.0),
                        clinical_significance="Nephrotoxic drug during renal decline",
                        implication="Drug accelerating kidney injury",
                        action_required="STOP nephrotoxic medication immediately",
                        urgency="immediate",
                        risk_multiplier=2.0,
                        detection_improvement_days=t.onset_days_ago or 0,
                        recommendations=[
                            f"STOP {drug_a}",
                            "Monitor creatinine every 6 hours",
                            "Nephrology consult recommended"
                        ]
                    ))

                # Cardiac + QT prolongation
                if 'cardiac' in domain and 'qt' in effect:
                    corrs.append(Correlation(
                        id=f"corr_{t.id}_{p.id}_qt",
                        correlation_type=CorrelationType.TRAJECTORY_PHARMA,
                        patterns_involved=[t.id, p.id],
                        domains_involved=["trajectory", "pharma"],
                        strength=min(t.severity * 1.3, 1.0),
                        clinical_significance="QT-prolonging drug during cardiac stress",
                        implication="Increased arrhythmia risk",
                        action_required="Discontinue QT-prolonging medications",
                        urgency="immediate",
                        risk_multiplier=1.8,
                        recommendations=[
                            f"Discontinue {drug_a}",
                            "Continuous cardiac monitoring",
                            "Check and correct electrolytes"
                        ]
                    ))

                # Major interaction during critical illness
                if interaction_type == 'major' and t.severity > 0.5:
                    corrs.append(Correlation(
                        id=f"corr_{t.id}_{p.id}_major",
                        correlation_type=CorrelationType.TRAJECTORY_PHARMA,
                        patterns_involved=[t.id, p.id],
                        domains_involved=["trajectory", "pharma"],
                        strength=t.severity * 0.8,
                        clinical_significance=f"Major drug interaction during {domain or 'critical'} deterioration",
                        implication=effect or "Drug interaction detected",
                        action_required=management or "Review medication regimen",
                        urgency="urgent",
                        risk_multiplier=1.4,
                        recommendations=[
                            management or "Review all medications",
                            "Pharmacy consult recommended"
                        ]
                    ))

        return corrs

    def _correlate_trajectory_pathogen(
        self,
        traj: List[Pattern],
        pathogen: List[Pattern]
    ) -> List[Correlation]:
        """Correlate trajectory with pathogen findings."""
        corrs = []

        for t in traj:
            pattern_type = t.pattern_type.value.lower() if t.pattern_type else ''
            if 'sepsis' not in pattern_type and 'infection' not in str(t.evidence):
                continue

            for p in pathogen:
                if not isinstance(p, PathogenPattern):
                    continue

                infection_type = p.infection_type if hasattr(p, 'infection_type') else ''
                pathogen_name = p.pathogen if hasattr(p, 'pathogen') else None
                resistance = p.resistance_genes if hasattr(p, 'resistance_genes') else []

                corr = Correlation(
                    id=f"corr_{t.id}_{p.id}",
                    correlation_type=CorrelationType.TRAJECTORY_PATHOGEN,
                    patterns_involved=[t.id, p.id],
                    domains_involved=["trajectory", "pathogen"],
                    strength=min((t.confidence + p.confidence) / 2 * 1.3, 1.0),
                    clinical_significance=f"Sepsis trajectory with {infection_type or 'suspected'} infection",
                    implication="Confirmed infectious etiology",
                    action_required="Target antimicrobial therapy",
                    urgency="urgent",
                    risk_multiplier=1.2,
                    detection_improvement_days=max(t.onset_days_ago or 0, p.onset_days_ago or 0),
                    recommendations=[
                        f"Target therapy for {pathogen_name}" if pathogen_name else "Obtain cultures urgently",
                        "Ensure source control",
                        "Repeat labs in 6 hours"
                    ]
                )

                if resistance:
                    corr.recommendations.append(f"Resistance detected: {', '.join(resistance)}")
                    corr.risk_multiplier *= 1.3

                corrs.append(corr)

        return corrs

    def _correlate_genomics_pharma(
        self,
        genomic: List[Pattern],
        pharma: List[Pattern]
    ) -> List[Correlation]:
        """Correlate genomic variants with drug metabolism."""
        corrs = []

        for g in genomic:
            if not isinstance(g, GenomicPattern):
                continue

            gene = g.gene.upper() if hasattr(g, 'gene') else ''
            classification = g.classification if hasattr(g, 'classification') else ''
            clinical_sig = g.clinical_significance if hasattr(g, 'clinical_significance') else ''

            for p in pharma:
                if not isinstance(p, PharmaPattern):
                    continue

                affected_gene = p.affected_gene.upper() if hasattr(p, 'affected_gene') and p.affected_gene else ''
                drug_a = p.drug_a if hasattr(p, 'drug_a') else 'unknown'
                management = p.management if hasattr(p, 'management') else ''

                if affected_gene and affected_gene == gene:
                    corrs.append(Correlation(
                        id=f"corr_{g.id}_{p.id}",
                        correlation_type=CorrelationType.GENOMIC_PHARMA,
                        patterns_involved=[g.id, p.id],
                        domains_involved=["genomics", "pharma"],
                        strength=0.9,
                        clinical_significance=f"{gene} variant affects {drug_a} metabolism",
                        implication=f"Patient is {classification} for {gene}",
                        action_required=management or "Adjust dosing per pharmacogenomics",
                        urgency="routine",
                        risk_multiplier=1.2,
                        recommendations=[
                            f"Adjust {drug_a} for {gene} status",
                            f"Patient status: {clinical_sig}",
                            "Consider alternative medications"
                        ]
                    ))

        return corrs

    def _detect_cascade(self, patterns: List[Pattern]) -> Optional[Correlation]:
        """Detect temporal cascade - patterns emerging in sequence across domains."""
        timed = [(p, p.onset_days_ago or 0) for p in patterns if p.onset_days_ago]
        if len(timed) < 3:
            return None

        timed.sort(key=lambda x: x[1], reverse=True)
        sequence = [(p.source.value, days, p.id) for p, days in timed]

        sources = [s[0] for s in sequence]
        # Ideal cascade: genomics -> multiomic -> trajectory -> alert
        ideal_order = ['genomics', 'multiomic', 'trajectory', 'alert_system']

        score = sum(0.25 for i, src in enumerate(sources[:4]) if src in ideal_order and ideal_order.index(src) <= i)

        if score >= 0.5:
            earliest, latest = sequence[0], sequence[-1]
            return Correlation(
                id=f"cascade_{'_'.join(s[2][:8] for s in sequence[:3])}",
                correlation_type=CorrelationType.TEMPORAL_CASCADE,
                patterns_involved=[s[2] for s in sequence],
                domains_involved=list(set(s[0] for s in sequence)),
                strength=score,
                clinical_significance="Temporal cascade detected across biological levels",
                implication=f"Signal propagated {earliest[0]} -> {latest[0]} over {earliest[1] - latest[1]:.1f} days",
                action_required="Early intervention opportunity identified",
                urgency="urgent",
                detection_improvement_days=earliest[1],
                evidence={"sequence": sequence, "duration_days": earliest[1] - latest[1]},
                recommendations=[
                    f"Earliest signal: {earliest[0]} ({earliest[1]:.1f} days ago)",
                    "Multi-level progression confirmed",
                    "Aggressive early intervention recommended"
                ]
            )
        return None

    def _correlate_multi_domain(
        self,
        patterns: List[Pattern],
        by_source: Dict
    ) -> Optional[Correlation]:
        """Generate insight when 3+ domains have patterns."""
        if len(by_source) < 3:
            return None

        severities = [p.severity for p in patterns]
        confidences = [p.confidence for p in patterns]
        avg_sev = float(np.mean(severities))
        avg_conf = float(np.mean(confidences))

        return Correlation(
            id=f"multi_{'_'.join(d.value[:4] for d in list(by_source.keys())[:4])}",
            correlation_type=CorrelationType.MULTI_DOMAIN,
            patterns_involved=[p.id for p in patterns],
            domains_involved=[d.value for d in by_source.keys()],
            strength=min(avg_conf + len(by_source) * 0.1, 1.0),
            clinical_significance=f"Convergent evidence from {len(by_source)} domains",
            implication="Multiple biological systems abnormal simultaneously",
            action_required="Multi-disciplinary review required",
            urgency="urgent" if avg_sev > 0.6 else "routine",
            risk_multiplier=1.0 + len(by_source) * 0.15,
            evidence={
                "domains": [d.value for d in by_source.keys()],
                "pattern_count": len(patterns),
                "avg_severity": avg_sev
            },
            recommendations=[
                f"{len(by_source)} domains involved in patient deterioration",
                "Multi-disciplinary team conference recommended",
                "Develop integrated care plan"
            ]
        )
