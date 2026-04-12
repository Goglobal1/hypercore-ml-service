"""
Trial Rescue Agent - Pharma Trial Intelligence

Connects to:
- Drug Response Predictor Pipeline (FAERS, AACT)
- Diagnostic Agent findings

Capabilities:
- Match patients to clinical trials
- Identify rescue therapies for treatment failures
- Assess pharmacogenomic compatibility
- Monitor adverse event profiles
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

# Import drug response predictor pipeline
try:
    from app.core.drug_response_predictor import (
        get_drug_profile,
        predict_drug_response,
        check_drug_interactions,
        get_adverse_events,
        search_trials,
        PHARMACOGENOMIC_MAP,
    )
    PHARMA_AVAILABLE = True
except ImportError:
    PHARMA_AVAILABLE = False
    logger.warning("Drug response predictor pipeline not available")
    PHARMACOGENOMIC_MAP = {}


# Condition to trial category mappings
CONDITION_TRIAL_MAP = {
    "sepsis": ["sepsis", "infection", "critical care", "antibiotics"],
    "acute_coronary_syndrome": ["cardiac", "cardiovascular", "myocardial infarction", "acs"],
    "heart_failure": ["heart failure", "cardiac", "cardiomyopathy", "hfref", "hfpef"],
    "pneumonia": ["pneumonia", "respiratory", "infection", "antibiotics"],
    "diabetes": ["diabetes", "glycemic", "insulin", "metformin"],
    "cancer": ["oncology", "tumor", "chemotherapy", "immunotherapy"],
    "kidney_disease": ["renal", "kidney", "dialysis", "nephrology"],
    "liver_disease": ["hepatic", "liver", "cirrhosis", "hepatology"],
}

# Drug class rescue options
RESCUE_THERAPIES = {
    "antibiotic_failure": {
        "escalation": ["carbapenem", "colistin", "tigecycline"],
        "combination": ["aminoglycoside", "fluoroquinolone"],
        "considerations": ["culture_sensitivity", "renal_adjustment", "pharmacist_consult"],
    },
    "cardiac_failure": {
        "escalation": ["inotropes", "mechanical_support", "transplant_evaluation"],
        "combination": ["diuretic_intensification", "afterload_reduction"],
        "considerations": ["hemodynamic_monitoring", "cardiology_consult"],
    },
    "cancer_progression": {
        "escalation": ["next_line_therapy", "clinical_trial", "immunotherapy"],
        "combination": ["combination_chemotherapy", "targeted_therapy"],
        "considerations": ["molecular_profiling", "tumor_board", "palliative_care"],
    },
    "renal_failure": {
        "escalation": ["dialysis", "continuous_rrt", "transplant_evaluation"],
        "combination": ["volume_optimization", "nephrotoxin_avoidance"],
        "considerations": ["nephrology_consult", "medication_adjustment"],
    },
}


class TrialRescueAgent(BaseAgent):
    """
    Pharma trial intelligence agent.

    Matches patients to clinical trials and identifies
    rescue therapies based on treatment response.

    Evolution Parameters:
    - trial_match_threshold: Min score to recommend trial
    - rescue_urgency_threshold: Urgency level for rescue therapy
    - interaction_severity_threshold: Min severity to flag interactions
    """

    VERSION = "1.1.0"

    def __init__(self):
        super().__init__(AgentType.TRIAL_RESCUE)
        self._trial_cache: Dict[str, Any] = {}

        # Evolution-tunable parameters
        self._trial_match_threshold = 0.6
        self._rescue_urgency_threshold = 0.7
        self._interaction_severity_threshold = 0.5

        # Register for parameter updates
        self.on_parameter_change("trial_match_threshold", self._on_trial_threshold_change)

    def _get_configurable_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Define evolution-tunable parameters."""
        return {
            "trial_match_threshold": {
                "type": "float",
                "min": 0.4,
                "max": 0.9,
                "default": 0.6,
                "description": "Minimum match score to recommend a clinical trial",
            },
            "rescue_urgency_threshold": {
                "type": "float",
                "min": 0.5,
                "max": 0.95,
                "default": 0.7,
                "description": "Urgency threshold for rescue therapy recommendations",
            },
            "interaction_severity_threshold": {
                "type": "float",
                "min": 0.3,
                "max": 0.8,
                "default": 0.5,
                "description": "Minimum severity to flag drug interactions",
            },
        }

    def _on_trial_threshold_change(self, old_value: float, new_value: float) -> None:
        self._trial_match_threshold = new_value
        logger.info(f"TrialRescueAgent: trial_match_threshold updated {old_value} -> {new_value}")

    @property
    def name(self) -> str:
        return "Trial Rescue Intelligence"

    @property
    def description(self) -> str:
        return "Matches patients to trials and identifies rescue therapies"

    @property
    def capabilities(self) -> List[str]:
        return [
            "trial_matching",
            "rescue_therapy_identification",
            "pharmacogenomic_assessment",
            "adverse_event_monitoring",
            "drug_interaction_check",
        ]

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze patient for trial eligibility and rescue options.

        Input schema:
        {
            "diagnosis": "sepsis",
            "current_medications": ["vancomycin", "piperacillin"],
            "failed_therapies": ["ceftriaxone"],
            "patient_profile": {
                "age": 65,
                "genes": {"CYP2D6": "poor_metabolizer"},
                "conditions": ["diabetes", "hypertension"]
            },
            "treatment_response": "inadequate",
            "correlation_id": "session_123"
        }
        """
        correlation_id = input_data.get("correlation_id")
        findings = []

        diagnosis = input_data.get("diagnosis", "")
        current_meds = input_data.get("current_medications", [])
        failed_therapies = input_data.get("failed_therapies", [])
        patient_profile = input_data.get("patient_profile", {})
        treatment_response = input_data.get("treatment_response", "unknown")

        # 1. Check peer findings for diagnostic context
        peer_findings = self.get_peer_findings(
            agent_type=AgentType.DIAGNOSTIC,
            min_confidence=0.5
        )

        # Extract diagnoses from peer findings
        peer_diagnoses = []
        for finding in peer_findings:
            if "diagnosis" in finding.related_entities:
                peer_diagnoses.append(finding.related_entities["diagnosis"])

        # Use peer diagnoses if no direct diagnosis provided
        if not diagnosis and peer_diagnoses:
            diagnosis = peer_diagnoses[0]

        # 2. Search for matching clinical trials
        trial_findings = await self._search_trials(diagnosis, patient_profile)
        findings.extend(trial_findings)

        # 3. Check drug interactions for current medications
        if len(current_meds) >= 2:
            interaction_findings = await self._check_interactions(current_meds)
            findings.extend(interaction_findings)

        # 4. Assess pharmacogenomic factors
        if patient_profile.get("genes"):
            pgx_findings = await self._assess_pharmacogenomics(
                current_meds,
                patient_profile.get("genes", {})
            )
            findings.extend(pgx_findings)

        # 5. Identify rescue therapies if treatment failing
        if treatment_response in ["inadequate", "failure", "progressing"]:
            rescue_findings = self._identify_rescue_options(
                diagnosis,
                failed_therapies,
                current_meds
            )
            findings.extend(rescue_findings)

        # 6. Get adverse event profile for current medications
        ae_findings = await self._get_adverse_events(current_meds)
        findings.extend(ae_findings)

        # 7. Broadcast findings
        if findings:
            self.broadcast_findings(findings, correlation_id)

        # 8. Alert on critical interactions or rescue needs
        critical = [f for f in findings
                   if "major" in f.description.lower() or "critical" in f.category]
        if critical:
            self.send_alert(
                alert_message=f"Critical pharma findings: {len(critical)} items requiring attention",
                severity="high",
                related_findings=critical,
                recipient=AgentType.DIAGNOSTIC
            )

        return {
            "agent": self.name,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "diagnosis_evaluated": diagnosis,
            "current_medications": current_meds,
            "treatment_response": treatment_response,
            "findings": [f.to_dict() for f in findings],
            "trial_matches": len([f for f in findings if "trial" in f.category]),
            "interactions_found": len([f for f in findings if "interaction" in f.category]),
            "rescue_options": len([f for f in findings if "rescue" in f.category]),
            "correlation_id": correlation_id,
            "pipeline_available": PHARMA_AVAILABLE,
        }

    async def _search_trials(
        self,
        diagnosis: str,
        patient_profile: Dict[str, Any]
    ) -> List[AgentFinding]:
        """Search for matching clinical trials."""
        findings = []

        if not PHARMA_AVAILABLE or not diagnosis:
            return findings

        # Get search terms for condition
        search_terms = CONDITION_TRIAL_MAP.get(
            diagnosis.lower().replace(" ", "_"),
            [diagnosis]
        )

        for term in search_terms[:2]:  # Limit searches
            try:
                result = search_trials(condition=term, limit=20)
                trials = result.get("trials", [])

                if trials:
                    # Filter by patient eligibility (basic)
                    age = patient_profile.get("age", 50)
                    eligible_trials = []

                    for trial in trials:
                        # Simple eligibility check
                        status = trial.get("status", "").lower()
                        if "recruiting" in status or "active" in status:
                            eligible_trials.append(trial)

                    if eligible_trials:
                        finding = self.create_finding(
                            category="trial_match",
                            description=f"Found {len(eligible_trials)} recruiting trials for {term}",
                            confidence=0.75,
                            evidence=[
                                f"Search term: {term}",
                                f"Total trials found: {len(trials)}",
                                f"Recruiting trials: {len(eligible_trials)}",
                            ],
                            related_entities={
                                "condition": term,
                                "trial_count": len(eligible_trials),
                                "sample_trials": [
                                    {
                                        "nct_id": t.get("nct_id"),
                                        "title": t.get("title", "")[:100],
                                        "status": t.get("status"),
                                    }
                                    for t in eligible_trials[:5]
                                ]
                            }
                        )
                        findings.append(finding)

            except Exception as e:
                logger.error(f"Error searching trials for {term}: {e}")

        return findings

    async def _check_interactions(self, medications: List[str]) -> List[AgentFinding]:
        """Check for drug-drug interactions."""
        findings = []

        if not PHARMA_AVAILABLE or len(medications) < 2:
            return findings

        try:
            result = check_drug_interactions(medications)
            interactions = result.get("interactions", [])

            for interaction in interactions:
                severity = interaction.get("severity", "unknown")
                confidence = 0.9 if severity == "major" else 0.7 if severity == "moderate" else 0.5

                finding = self.create_finding(
                    category="drug_interaction",
                    description=f"{severity.upper()} interaction: {interaction.get('drug_a')} + {interaction.get('drug_b')}",
                    confidence=confidence,
                    evidence=[
                        f"Drugs: {interaction.get('drug_a')} and {interaction.get('drug_b')}",
                        f"Type: {interaction.get('type', 'unknown')}",
                        f"Effect: {interaction.get('effect', 'unknown')}",
                        f"Management: {interaction.get('management', 'Consult pharmacist')}",
                    ],
                    related_entities={
                        "drug_a": interaction.get("drug_a"),
                        "drug_b": interaction.get("drug_b"),
                        "severity": severity,
                        "management": interaction.get("management"),
                    }
                )
                findings.append(finding)

        except Exception as e:
            logger.error(f"Error checking interactions: {e}")

        return findings

    async def _assess_pharmacogenomics(
        self,
        medications: List[str],
        gene_status: Dict[str, str]
    ) -> List[AgentFinding]:
        """Assess pharmacogenomic implications."""
        findings = []

        if not PHARMA_AVAILABLE:
            return findings

        for gene, status in gene_status.items():
            gene_upper = gene.upper()
            if gene_upper in PHARMACOGENOMIC_MAP:
                affected_drugs = PHARMACOGENOMIC_MAP[gene_upper].get("drugs", [])

                # Check if any current meds are affected
                for med in medications:
                    if med.lower() in [d.lower() for d in affected_drugs]:
                        impact = "high" if "poor" in status.lower() else "moderate"
                        confidence = 0.85 if impact == "high" else 0.7

                        finding = self.create_finding(
                            category="pharmacogenomic_alert",
                            description=f"{gene_upper} {status}: affects {med} metabolism",
                            confidence=confidence,
                            evidence=[
                                f"Gene: {gene_upper}",
                                f"Status: {status}",
                                f"Affected drug: {med}",
                                f"Effect: {PHARMACOGENOMIC_MAP[gene_upper].get('effect', 'altered metabolism')}",
                            ],
                            related_entities={
                                "gene": gene_upper,
                                "status": status,
                                "drug": med,
                                "impact": impact,
                                "recommendation": self._get_pgx_recommendation(gene_upper, status, med)
                            }
                        )
                        findings.append(finding)

        return findings

    def _get_pgx_recommendation(self, gene: str, status: str, drug: str) -> str:
        """Get pharmacogenomic recommendation."""
        if "poor" in status.lower():
            return f"Consider dose reduction or alternative to {drug} due to {gene} poor metabolizer status"
        elif "ultra" in status.lower():
            return f"Consider dose increase for {drug} due to {gene} ultra-rapid metabolizer status"
        else:
            return f"Standard dosing may be appropriate; monitor response"

    def _identify_rescue_options(
        self,
        diagnosis: str,
        failed_therapies: List[str],
        current_meds: List[str]
    ) -> List[AgentFinding]:
        """Identify rescue therapy options."""
        findings = []

        # Map diagnosis to rescue category
        rescue_category = None
        if "sepsis" in diagnosis.lower() or "infection" in diagnosis.lower():
            rescue_category = "antibiotic_failure"
        elif "heart" in diagnosis.lower() or "cardiac" in diagnosis.lower():
            rescue_category = "cardiac_failure"
        elif "cancer" in diagnosis.lower() or "tumor" in diagnosis.lower():
            rescue_category = "cancer_progression"
        elif "kidney" in diagnosis.lower() or "renal" in diagnosis.lower():
            rescue_category = "renal_failure"

        if rescue_category and rescue_category in RESCUE_THERAPIES:
            options = RESCUE_THERAPIES[rescue_category]

            finding = self.create_finding(
                category="rescue_therapy",
                description=f"Rescue options for {rescue_category.replace('_', ' ')}",
                confidence=0.8,
                evidence=[
                    f"Failed therapies: {', '.join(failed_therapies) if failed_therapies else 'None specified'}",
                    f"Current regimen: {', '.join(current_meds) if current_meds else 'None specified'}",
                    f"Category: {rescue_category}",
                ],
                related_entities={
                    "category": rescue_category,
                    "escalation_options": options.get("escalation", []),
                    "combination_options": options.get("combination", []),
                    "considerations": options.get("considerations", []),
                }
            )
            findings.append(finding)

        return findings

    async def _get_adverse_events(self, medications: List[str]) -> List[AgentFinding]:
        """Get adverse event profiles for medications."""
        findings = []

        if not PHARMA_AVAILABLE:
            return findings

        for med in medications[:3]:  # Limit to first 3
            try:
                result = get_adverse_events(med, max_quarters=2)

                if result.get("total_reports", 0) > 100:
                    top_events = result.get("top_adverse_events", [])[:5]

                    finding = self.create_finding(
                        category="adverse_event_profile",
                        description=f"{med}: {result.get('total_reports', 0)} adverse event reports",
                        confidence=0.7,
                        evidence=[
                            f"Drug: {med}",
                            f"Total reports: {result.get('total_reports', 0)}",
                            f"Top events: {', '.join([e.get('reaction', '') for e in top_events])}",
                        ],
                        related_entities={
                            "drug": med,
                            "total_reports": result.get("total_reports", 0),
                            "top_events": top_events,
                        }
                    )
                    findings.append(finding)

            except Exception as e:
                logger.debug(f"Error getting AEs for {med}: {e}")

        return findings

    def get_rescue_options(self, category: str) -> Dict[str, Any]:
        """Get rescue therapy options for a category."""
        if category.lower() in RESCUE_THERAPIES:
            return {
                "category": category,
                "options": RESCUE_THERAPIES[category.lower()]
            }
        return {"error": f"Unknown category: {category}"}

    def list_rescue_categories(self) -> List[str]:
        """List available rescue therapy categories."""
        return list(RESCUE_THERAPIES.keys())


# Singleton instance
_trial_rescue_agent = None


def get_trial_rescue_agent() -> TrialRescueAgent:
    """Get singleton trial rescue agent instance."""
    global _trial_rescue_agent
    if _trial_rescue_agent is None:
        _trial_rescue_agent = TrialRescueAgent()
    return _trial_rescue_agent
