"""
Biomarker Agent - Multi-Omic Signal Interpreter

Connects to:
- Multi-Omic Fusion Pipeline
- Genomics Integration Pipeline

Capabilities:
- Interpret biomarker signals across omics layers
- Detect abnormal expression patterns
- Correlate genetic variants with phenotypes
- Generate clinical significance assessments
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from app.agents.base_agent import (
    BaseAgent,
    AgentType,
    AgentFinding,
    AgentRegistry,
)

logger = logging.getLogger(__name__)

# Import pipelines with fallback
try:
    from app.core.multiomic_fusion import (
        get_source_status,
        unified_query,
        fusion_analysis,
        MultiOmicFusionEngine,
    )
    MULTIOMIC_AVAILABLE = True
except ImportError:
    MULTIOMIC_AVAILABLE = False
    logger.warning("Multi-omic fusion pipeline not available")

try:
    from app.core.genomics_integration import (
        get_gene_expression,
        get_gene_variants,
        GenomicsIntegration,
    )
    GENOMICS_AVAILABLE = True
except ImportError:
    GENOMICS_AVAILABLE = False
    logger.warning("Genomics integration pipeline not available")

# Import ChEMBL for drug-target interactions
try:
    from app.core.chembl_integration import (
        get_drug_targets,
        get_drug_mechanisms,
        search_compounds_by_name,
    )
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logger.warning("ChEMBL integration not available")

# Import PharmGKB for pharmacogenomics
try:
    from app.core.pharmgkb_integration import (
        get_drug_gene_interactions,
        get_clinical_guidelines,
        get_variant_annotations,
    )
    PHARMGKB_AVAILABLE = True
except ImportError:
    PHARMGKB_AVAILABLE = False
    logger.warning("PharmGKB integration not available")


# Biomarker reference ranges and clinical significance
BIOMARKER_REFERENCE = {
    "inflammatory": {
        "CRP": {"low": 0, "normal": 3, "elevated": 10, "critical": 100, "unit": "mg/L"},
        "IL6": {"low": 0, "normal": 7, "elevated": 20, "critical": 100, "unit": "pg/mL"},
        "TNF_ALPHA": {"low": 0, "normal": 8.1, "elevated": 20, "critical": 50, "unit": "pg/mL"},
        "PROCALCITONIN": {"low": 0, "normal": 0.5, "elevated": 2, "critical": 10, "unit": "ng/mL"},
    },
    "metabolic": {
        "HBA1C": {"low": 0, "normal": 5.7, "elevated": 6.5, "critical": 10, "unit": "%"},
        "GLUCOSE": {"low": 70, "normal": 100, "elevated": 126, "critical": 400, "unit": "mg/dL"},
        "TRIGLYCERIDES": {"low": 0, "normal": 150, "elevated": 200, "critical": 500, "unit": "mg/dL"},
    },
    "cardiac": {
        "TROPONIN_I": {"low": 0, "normal": 0.04, "elevated": 0.4, "critical": 2, "unit": "ng/mL"},
        "BNP": {"low": 0, "normal": 100, "elevated": 400, "critical": 900, "unit": "pg/mL"},
        "CKMB": {"low": 0, "normal": 5, "elevated": 10, "critical": 25, "unit": "ng/mL"},
    },
    "renal": {
        "CREATININE": {"low": 0.5, "normal": 1.2, "elevated": 2, "critical": 5, "unit": "mg/dL"},
        "BUN": {"low": 7, "normal": 20, "elevated": 40, "critical": 100, "unit": "mg/dL"},
        "EGFR": {"low": 15, "normal": 60, "elevated": 90, "critical": 120, "unit": "mL/min"},
    },
    "hepatic": {
        "ALT": {"low": 0, "normal": 40, "elevated": 80, "critical": 1000, "unit": "U/L"},
        "AST": {"low": 0, "normal": 40, "elevated": 80, "critical": 1000, "unit": "U/L"},
        "BILIRUBIN": {"low": 0, "normal": 1.2, "elevated": 3, "critical": 12, "unit": "mg/dL"},
    },
}

# Gene-biomarker associations
GENE_BIOMARKER_MAP = {
    "APOE": ["cholesterol", "LDL", "HDL", "triglycerides"],
    "CYP2D6": ["drug_metabolism"],
    "BRCA1": ["tumor_markers", "CA125"],
    "BRCA2": ["tumor_markers", "CA125"],
    "TP53": ["tumor_markers", "p53_antibody"],
    "MTHFR": ["homocysteine", "folate", "B12"],
    "HFE": ["ferritin", "iron", "transferrin"],
    "F5": ["D_dimer", "fibrinogen", "PT_INR"],
    "LDLR": ["LDL", "cholesterol"],
}


class BiomarkerAgent(BaseAgent):
    """
    Multi-omic signal interpreter agent.

    Analyzes biomarker patterns across multiple data sources
    and generates clinical significance assessments.
    """

    def __init__(self):
        super().__init__(AgentType.BIOMARKER)
        self._multiomic_engine = None
        self._genomics_engine = None

        # Initialize engines if available
        if MULTIOMIC_AVAILABLE:
            try:
                self._multiomic_engine = MultiOmicFusionEngine()
            except Exception as e:
                logger.error(f"Failed to initialize multi-omic engine: {e}")

        if GENOMICS_AVAILABLE:
            try:
                self._genomics_engine = GenomicsIntegration()
            except Exception as e:
                logger.error(f"Failed to initialize genomics engine: {e}")

    @property
    def name(self) -> str:
        return "Biomarker Signal Interpreter"

    @property
    def description(self) -> str:
        return "Interprets biomarker signals across multi-omic data sources"

    @property
    def capabilities(self) -> List[str]:
        return [
            "biomarker_interpretation",
            "expression_pattern_detection",
            "variant_phenotype_correlation",
            "clinical_significance_assessment",
            "multi_omic_integration",
        ]

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze biomarker data and generate findings.

        Input schema:
        {
            "biomarkers": {"CRP": 15.5, "IL6": 25.0, ...},
            "genes": ["APOE", "BRCA1"],
            "patient_context": {"age": 65, "sex": "M", "conditions": [...]},
            "include_genomics": true,
            "correlation_id": "session_123"
        }
        """
        findings = []
        correlation_id = input_data.get("correlation_id")

        # 1. Analyze biomarker levels
        biomarkers = input_data.get("biomarkers", {})
        if biomarkers:
            biomarker_findings = self._analyze_biomarkers(biomarkers)
            findings.extend(biomarker_findings)

        # 2. Analyze genes if provided
        genes = input_data.get("genes", [])
        if genes and input_data.get("include_genomics", True):
            gene_findings = await self._analyze_genes(genes)
            findings.extend(gene_findings)

        # 3. Analyze medications if provided (auto-query ChEMBL/PharmGKB)
        medications = input_data.get("medications", [])
        patient_context = input_data.get("patient_context", {})
        # Also check for medications in patient context
        if not medications and patient_context:
            medications = patient_context.get("medications", [])
            medications.extend(patient_context.get("current_medications", []))
            medications.extend(patient_context.get("drugs", []))
        if medications:
            medication_findings = await self._analyze_medications(medications, genes)
            findings.extend(medication_findings)

        # 4. Cross-correlate findings
        if len(findings) > 1:
            correlation_findings = self._cross_correlate_findings(findings)
            findings.extend(correlation_findings)

        # 4. Check peer findings from other agents
        peer_findings = self.get_peer_findings(min_confidence=0.6)
        if peer_findings:
            integration_findings = self._integrate_peer_findings(peer_findings, findings)
            findings.extend(integration_findings)

        # 5. Broadcast high-confidence findings
        high_confidence = [f for f in findings if f.confidence >= 0.7]
        if high_confidence:
            self.broadcast_findings(high_confidence, correlation_id)

        # 6. Send alerts for critical findings
        critical = [f for f in findings if "critical" in f.category.lower() or f.confidence > 0.9]
        if critical:
            self.send_alert(
                alert_message=f"Critical biomarker findings detected: {len(critical)} items",
                severity="high",
                related_findings=critical,
                recipient=AgentType.DIAGNOSTIC
            )

        return {
            "agent": self.name,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "findings": [f.to_dict() for f in findings],
            "finding_count": len(findings),
            "high_confidence_count": len(high_confidence),
            "critical_count": len(critical),
            "correlation_id": correlation_id,
            "pipelines_used": {
                "multiomic": MULTIOMIC_AVAILABLE,
                "genomics": GENOMICS_AVAILABLE,
                "chembl": CHEMBL_AVAILABLE,
                "pharmgkb": PHARMGKB_AVAILABLE
            }
        }

    async def _analyze_medications(
        self, medications: List[str], genes: List[str] = None
    ) -> List[AgentFinding]:
        """
        Analyze medications by querying ChEMBL and PharmGKB.

        Args:
            medications: List of medication names
            genes: Optional list of genes for pharmacogenomics correlation

        Returns:
            List of findings from drug databases
        """
        findings = []

        for medication in medications:
            if not medication:
                continue
            medication = medication.strip()

            # Query ChEMBL for drug-target interactions
            if CHEMBL_AVAILABLE:
                try:
                    targets = get_drug_targets(medication)
                    if targets.get("targets"):
                        target_names = [t.get("target_name", "Unknown") for t in targets["targets"][:5]]
                        finding = self.create_finding(
                            category="drug_target_interaction",
                            description=f"{medication} targets: {', '.join(target_names)}",
                            confidence=0.85,
                            evidence=[
                                f"Source: ChEMBL database",
                                f"Total targets: {len(targets['targets'])}",
                                f"Drug: {medication}"
                            ],
                            related_entities={
                                "drug": medication,
                                "targets": target_names,
                                "source": "chembl"
                            }
                        )
                        findings.append(finding)

                    # Get drug mechanisms
                    mechanisms = get_drug_mechanisms(medication)
                    if mechanisms.get("mechanisms"):
                        mech_names = [m.get("mechanism", "Unknown") for m in mechanisms["mechanisms"][:3]]
                        finding = self.create_finding(
                            category="drug_mechanism",
                            description=f"{medication} mechanism: {', '.join(mech_names)}",
                            confidence=0.80,
                            evidence=[
                                f"Source: ChEMBL database",
                                f"Mechanism count: {len(mechanisms['mechanisms'])}"
                            ],
                            related_entities={
                                "drug": medication,
                                "mechanisms": mech_names,
                                "source": "chembl"
                            }
                        )
                        findings.append(finding)
                except Exception as e:
                    logger.debug(f"ChEMBL query failed for {medication}: {e}")

            # Query PharmGKB for pharmacogenomics
            if PHARMGKB_AVAILABLE:
                try:
                    # Get drug-gene interactions
                    interactions = get_drug_gene_interactions(medication)
                    if interactions.get("interactions"):
                        gene_names = [i.get("gene", "Unknown") for i in interactions["interactions"][:5]]
                        finding = self.create_finding(
                            category="pharmacogenomics",
                            description=f"{medication} has interactions with genes: {', '.join(gene_names)}",
                            confidence=0.90,
                            evidence=[
                                f"Source: PharmGKB database",
                                f"Total gene interactions: {len(interactions['interactions'])}",
                                f"Clinical relevance: pharmacogenomic dosing guidance may apply"
                            ],
                            related_entities={
                                "drug": medication,
                                "genes": gene_names,
                                "source": "pharmgkb"
                            }
                        )
                        findings.append(finding)

                        # Check if any patient genes are affected
                        if genes:
                            affected_genes = set(gene_names) & set(g.upper() for g in genes)
                            if affected_genes:
                                finding = self.create_finding(
                                    category="pharmacogenomics_alert",
                                    description=f"Patient genes {affected_genes} may affect {medication} response",
                                    confidence=0.95,
                                    evidence=[
                                        f"Patient has variants in drug-interacting genes",
                                        f"Recommend pharmacogenomic testing",
                                        f"Source: PharmGKB"
                                    ],
                                    related_entities={
                                        "drug": medication,
                                        "affected_genes": list(affected_genes),
                                        "source": "pharmgkb"
                                    }
                                )
                                findings.append(finding)

                    # Get clinical guidelines
                    guidelines = get_clinical_guidelines(medication)
                    if guidelines.get("guidelines"):
                        guideline_summaries = [g.get("summary", "")[:100] for g in guidelines["guidelines"][:2]]
                        finding = self.create_finding(
                            category="clinical_guideline",
                            description=f"Clinical guidelines available for {medication}",
                            confidence=0.85,
                            evidence=guideline_summaries + [f"Source: PharmGKB"],
                            related_entities={
                                "drug": medication,
                                "guideline_count": len(guidelines["guidelines"]),
                                "source": "pharmgkb"
                            }
                        )
                        findings.append(finding)
                except Exception as e:
                    logger.debug(f"PharmGKB query failed for {medication}: {e}")

        return findings

    def _analyze_biomarkers(self, biomarkers: Dict[str, float]) -> List[AgentFinding]:
        """Analyze individual biomarker levels."""
        findings = []

        for marker, value in biomarkers.items():
            marker_upper = marker.upper()

            # Find reference range
            ref = None
            category = None
            for cat, markers in BIOMARKER_REFERENCE.items():
                if marker_upper in markers:
                    ref = markers[marker_upper]
                    category = cat
                    break

            if ref:
                status, confidence = self._evaluate_biomarker(value, ref)

                if status != "normal":
                    finding = self.create_finding(
                        category=f"biomarker_{category}",
                        description=f"{marker_upper} is {status}: {value} {ref['unit']} (normal <{ref['normal']})",
                        confidence=confidence,
                        evidence=[
                            f"Measured value: {value} {ref['unit']}",
                            f"Reference range: <{ref['normal']} {ref['unit']}",
                            f"Critical threshold: {ref['critical']} {ref['unit']}"
                        ],
                        related_entities={
                            "biomarker": marker_upper,
                            "value": value,
                            "unit": ref["unit"],
                            "status": status,
                            "category": category
                        }
                    )
                    findings.append(finding)

        return findings

    def _evaluate_biomarker(self, value: float, ref: Dict) -> tuple:
        """Evaluate biomarker against reference range."""
        if value >= ref["critical"]:
            return "critical", 0.95
        elif value >= ref["elevated"]:
            return "elevated", 0.8
        elif value >= ref["normal"]:
            return "borderline", 0.6
        elif value < ref.get("low", 0):
            return "low", 0.7
        else:
            return "normal", 0.5

    async def _analyze_genes(self, genes: List[str]) -> List[AgentFinding]:
        """Analyze gene expression and variants."""
        findings = []

        for gene in genes:
            # Get gene variants if genomics available
            if GENOMICS_AVAILABLE and self._genomics_engine:
                try:
                    variants = get_gene_variants(gene)
                    if variants.get("variants"):
                        pathogenic = [v for v in variants["variants"]
                                     if "pathogenic" in v.get("clinical_significance", "").lower()]

                        if pathogenic:
                            finding = self.create_finding(
                                category="genetic_variant",
                                description=f"{gene} has {len(pathogenic)} pathogenic variant(s) in ClinVar",
                                confidence=0.85,
                                evidence=[
                                    f"Total variants found: {len(variants['variants'])}",
                                    f"Pathogenic variants: {len(pathogenic)}",
                                    f"Source: ClinVar database"
                                ],
                                related_entities={
                                    "gene": gene,
                                    "pathogenic_count": len(pathogenic),
                                    "variants": pathogenic[:5]  # Top 5
                                }
                            )
                            findings.append(finding)
                except Exception as e:
                    logger.error(f"Error analyzing gene {gene}: {e}")

            # Check gene-biomarker associations
            if gene.upper() in GENE_BIOMARKER_MAP:
                associated_markers = GENE_BIOMARKER_MAP[gene.upper()]
                finding = self.create_finding(
                    category="gene_biomarker_association",
                    description=f"{gene} is associated with biomarkers: {', '.join(associated_markers)}",
                    confidence=0.75,
                    evidence=[
                        f"Gene: {gene}",
                        f"Associated biomarkers: {associated_markers}",
                        "Recommendation: Monitor associated biomarker levels"
                    ],
                    related_entities={
                        "gene": gene,
                        "associated_biomarkers": associated_markers
                    }
                )
                findings.append(finding)

        return findings

    def _cross_correlate_findings(self, findings: List[AgentFinding]) -> List[AgentFinding]:
        """Find correlations between findings."""
        correlation_findings = []

        # Group by category
        by_category = {}
        for f in findings:
            cat = f.category.split("_")[0]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(f)

        # Check for multi-system involvement
        if len(by_category) >= 2:
            categories = list(by_category.keys())
            confidence = min(0.9, 0.5 + len(by_category) * 0.1)

            correlation_findings.append(self.create_finding(
                category="multi_system_pattern",
                description=f"Multi-system involvement detected: {', '.join(categories)}",
                confidence=confidence,
                evidence=[
                    f"Systems involved: {len(categories)}",
                    f"Categories: {categories}",
                    f"Total findings: {len(findings)}"
                ],
                related_entities={
                    "systems": categories,
                    "finding_count": len(findings)
                }
            ))

        return correlation_findings

    def _integrate_peer_findings(
        self,
        peer_findings: List[AgentFinding],
        own_findings: List[AgentFinding]
    ) -> List[AgentFinding]:
        """Integrate findings from other agents."""
        integration_findings = []

        # Look for correlations with diagnostic agent findings
        diagnostic_findings = [f for f in peer_findings
                              if f.agent_type == AgentType.DIAGNOSTIC]

        if diagnostic_findings and own_findings:
            # Check if our biomarker findings support diagnostic hypotheses
            for diag in diagnostic_findings:
                related_evidence = []
                for own in own_findings:
                    if any(entity in str(diag.related_entities)
                          for entity in own.related_entities.keys()):
                        related_evidence.append(own.description)

                if related_evidence:
                    confidence = self.calculate_aggregate_confidence(
                        [diag.confidence] + [0.7] * len(related_evidence)
                    )

                    integration_findings.append(self.create_finding(
                        category="cross_agent_correlation",
                        description=f"Biomarker evidence supports: {diag.description[:100]}",
                        confidence=confidence,
                        evidence=related_evidence[:5],
                        related_entities={
                            "diagnostic_finding": diag.finding_id,
                            "supporting_evidence_count": len(related_evidence)
                        }
                    ))

        return integration_findings

    def get_biomarker_panel(self, category: str) -> Dict[str, Any]:
        """Get reference panel for a biomarker category."""
        if category.lower() in BIOMARKER_REFERENCE:
            return {
                "category": category,
                "markers": BIOMARKER_REFERENCE[category.lower()]
            }
        return {"error": f"Unknown category: {category}"}

    def get_gene_associations(self, gene: str) -> Dict[str, Any]:
        """Get biomarker associations for a gene."""
        gene_upper = gene.upper()
        if gene_upper in GENE_BIOMARKER_MAP:
            return {
                "gene": gene_upper,
                "associated_biomarkers": GENE_BIOMARKER_MAP[gene_upper]
            }
        return {"gene": gene_upper, "associated_biomarkers": []}


# Singleton instance
_biomarker_agent = None


def get_biomarker_agent() -> BiomarkerAgent:
    """Get singleton biomarker agent instance."""
    global _biomarker_agent
    if _biomarker_agent is None:
        _biomarker_agent = BiomarkerAgent()
    return _biomarker_agent
