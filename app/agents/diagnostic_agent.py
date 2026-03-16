"""
Diagnostic Agent - Differential Reasoning Engine

Connects to:
- All other agents via message protocol
- Multi-omic and genomics pipelines

Capabilities:
- Generate differential diagnoses
- Rank diagnostic hypotheses by probability
- Integrate evidence from all agents
- Recommend confirmatory tests
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from collections import defaultdict

from app.agents.base_agent import (
    BaseAgent,
    AgentType,
    AgentFinding,
    AgentRegistry,
    ConfidenceLevel,
)

logger = logging.getLogger(__name__)


# Diagnostic knowledge base - symptom/sign to diagnosis mappings
DIAGNOSTIC_PATTERNS = {
    "sepsis": {
        "required": ["fever_or_hypothermia", "tachycardia"],
        "supportive": ["elevated_wbc", "elevated_lactate", "hypotension", "altered_mental_status"],
        "biomarkers": ["CRP", "PROCALCITONIN", "IL6", "LACTATE"],
        "severity_markers": ["lactate", "creatinine", "bilirubin"],
        "icd10": ["A41.9", "R65.20", "R65.21"],
    },
    "acute_coronary_syndrome": {
        "required": ["chest_pain", "ecg_changes"],
        "supportive": ["troponin_elevation", "risk_factors", "radiation_to_arm"],
        "biomarkers": ["TROPONIN_I", "TROPONIN_T", "CKMB", "BNP"],
        "severity_markers": ["troponin_peak", "bnp"],
        "icd10": ["I21.9", "I20.0", "I24.9"],
    },
    "acute_kidney_injury": {
        "required": ["creatinine_elevation"],
        "supportive": ["oliguria", "fluid_overload", "uremia_symptoms"],
        "biomarkers": ["CREATININE", "BUN", "CYSTATIN_C", "NGAL"],
        "severity_markers": ["creatinine", "potassium", "ph"],
        "icd10": ["N17.9", "N17.0", "N17.1"],
    },
    "diabetic_ketoacidosis": {
        "required": ["hyperglycemia", "ketosis", "acidosis"],
        "supportive": ["dehydration", "altered_mental_status", "abdominal_pain"],
        "biomarkers": ["GLUCOSE", "BETA_HYDROXYBUTYRATE", "BICARBONATE", "PH"],
        "severity_markers": ["ph", "bicarbonate", "glucose"],
        "icd10": ["E10.10", "E11.10", "E13.10"],
    },
    "pneumonia": {
        "required": ["cough", "fever", "radiographic_infiltrate"],
        "supportive": ["dyspnea", "sputum_production", "hypoxia", "crackles"],
        "biomarkers": ["CRP", "PROCALCITONIN", "WBC"],
        "severity_markers": ["oxygen_saturation", "respiratory_rate"],
        "icd10": ["J18.9", "J15.9", "J13"],
    },
    "heart_failure": {
        "required": ["dyspnea", "elevated_bnp"],
        "supportive": ["edema", "orthopnea", "cardiomegaly", "elevated_jvp"],
        "biomarkers": ["BNP", "NT_PROBNP", "TROPONIN"],
        "severity_markers": ["bnp", "ejection_fraction"],
        "icd10": ["I50.9", "I50.20", "I50.30"],
    },
    "pulmonary_embolism": {
        "required": ["dyspnea", "risk_factors"],
        "supportive": ["chest_pain", "tachycardia", "hypoxia", "hemoptysis"],
        "biomarkers": ["D_DIMER", "TROPONIN", "BNP"],
        "severity_markers": ["troponin", "bnp", "rv_dysfunction"],
        "icd10": ["I26.99", "I26.92", "I26.09"],
    },
    "hepatic_failure": {
        "required": ["elevated_liver_enzymes", "coagulopathy"],
        "supportive": ["jaundice", "encephalopathy", "ascites"],
        "biomarkers": ["ALT", "AST", "BILIRUBIN", "INR", "ALBUMIN", "AMMONIA"],
        "severity_markers": ["inr", "bilirubin", "encephalopathy_grade"],
        "icd10": ["K72.90", "K72.00", "K72.10"],
    },
}

# Biomarker to clinical finding mappings
BIOMARKER_FINDINGS = {
    "CRP": "elevated_crp",
    "PROCALCITONIN": "elevated_procalcitonin",
    "TROPONIN_I": "troponin_elevation",
    "TROPONIN_T": "troponin_elevation",
    "BNP": "elevated_bnp",
    "CREATININE": "creatinine_elevation",
    "GLUCOSE": "hyperglycemia",
    "LACTATE": "elevated_lactate",
    "D_DIMER": "elevated_d_dimer",
    "ALT": "elevated_liver_enzymes",
    "AST": "elevated_liver_enzymes",
    "WBC": "elevated_wbc",
}


class DiagnosticHypothesis:
    """A diagnostic hypothesis with supporting evidence."""

    def __init__(self, diagnosis: str, pattern: Dict[str, Any]):
        self.diagnosis = diagnosis
        self.pattern = pattern
        self.required_met: List[str] = []
        self.required_missing: List[str] = []
        self.supportive_met: List[str] = []
        self.biomarker_evidence: List[str] = []
        self.agent_evidence: List[AgentFinding] = []
        self.probability: float = 0.0
        self.confidence: float = 0.0

    def calculate_probability(self):
        """Calculate diagnostic probability based on evidence."""
        # Required criteria weight
        required_count = len(self.pattern.get("required", []))
        required_met_count = len(self.required_met)

        if required_count > 0:
            required_score = required_met_count / required_count
        else:
            required_score = 0.5

        # Supportive criteria weight
        supportive_count = len(self.pattern.get("supportive", []))
        supportive_met_count = len(self.supportive_met)

        if supportive_count > 0:
            supportive_score = supportive_met_count / supportive_count
        else:
            supportive_score = 0.5

        # Biomarker evidence weight
        biomarker_count = len(self.pattern.get("biomarkers", []))
        biomarker_met_count = len(self.biomarker_evidence)

        if biomarker_count > 0:
            biomarker_score = biomarker_met_count / biomarker_count
        else:
            biomarker_score = 0.5

        # Agent evidence boost
        agent_boost = min(0.2, len(self.agent_evidence) * 0.05)

        # Weighted calculation
        self.probability = (
            required_score * 0.5 +
            supportive_score * 0.25 +
            biomarker_score * 0.15 +
            agent_boost
        )

        # Confidence based on evidence quality
        evidence_count = (required_met_count + supportive_met_count +
                         biomarker_met_count + len(self.agent_evidence))
        self.confidence = min(0.95, 0.3 + evidence_count * 0.1)

        return self.probability

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "probability": round(self.probability, 3),
            "confidence": round(self.confidence, 3),
            "icd10_codes": self.pattern.get("icd10", []),
            "required_criteria_met": self.required_met,
            "required_criteria_missing": self.required_missing,
            "supportive_criteria_met": self.supportive_met,
            "biomarker_evidence": self.biomarker_evidence,
            "agent_evidence_count": len(self.agent_evidence),
            "recommended_tests": self._get_recommended_tests(),
        }

    def _get_recommended_tests(self) -> List[str]:
        """Get recommended confirmatory tests."""
        tests = []

        # Add missing biomarkers as recommended tests
        all_biomarkers = set(self.pattern.get("biomarkers", []))
        measured_biomarkers = set(self.biomarker_evidence)
        missing = all_biomarkers - measured_biomarkers

        for marker in missing:
            tests.append(f"Measure {marker}")

        # Add severity markers if diagnosis likely
        if self.probability > 0.5:
            for marker in self.pattern.get("severity_markers", []):
                if marker.upper() not in measured_biomarkers:
                    tests.append(f"Assess {marker} for severity staging")

        return tests[:5]  # Top 5 recommendations


class DiagnosticAgent(BaseAgent):
    """
    Differential reasoning engine agent.

    Generates ranked differential diagnoses based on:
    - Clinical findings
    - Biomarker evidence
    - Peer agent findings
    """

    def __init__(self):
        super().__init__(AgentType.DIAGNOSTIC)
        self._hypotheses: List[DiagnosticHypothesis] = []

    @property
    def name(self) -> str:
        return "Differential Reasoning Engine"

    @property
    def description(self) -> str:
        return "Generates and ranks differential diagnoses based on multi-source evidence"

    @property
    def capabilities(self) -> List[str]:
        return [
            "differential_diagnosis",
            "hypothesis_ranking",
            "evidence_integration",
            "test_recommendation",
            "severity_assessment",
        ]

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate differential diagnosis.

        Input schema:
        {
            "clinical_findings": ["fever", "cough", "dyspnea"],
            "biomarkers": {"CRP": 45.0, "PROCALCITONIN": 2.5},
            "vital_signs": {"hr": 110, "temp": 39.2, "rr": 24},
            "history": {"conditions": ["diabetes", "hypertension"]},
            "correlation_id": "session_123"
        }
        """
        correlation_id = input_data.get("correlation_id")
        self._hypotheses = []

        # 1. Extract all available evidence
        clinical_findings = set(input_data.get("clinical_findings", []))
        biomarkers = input_data.get("biomarkers", {})
        vital_signs = input_data.get("vital_signs", {})
        history = input_data.get("history", {})

        # 2. Derive findings from biomarkers
        derived_findings = self._derive_findings_from_biomarkers(biomarkers)
        clinical_findings.update(derived_findings)

        # 3. Derive findings from vital signs
        vital_findings = self._derive_findings_from_vitals(vital_signs)
        clinical_findings.update(vital_findings)

        # 4. Get peer agent findings
        peer_findings = self.get_peer_findings(min_confidence=0.5)
        biomarker_findings = [f for f in peer_findings
                             if f.agent_type == AgentType.BIOMARKER]
        surveillance_findings = [f for f in peer_findings
                                if f.agent_type == AgentType.SURVEILLANCE]

        # 5. Generate hypotheses for each diagnostic pattern
        for diagnosis, pattern in DIAGNOSTIC_PATTERNS.items():
            hypothesis = self._evaluate_diagnosis(
                diagnosis,
                pattern,
                clinical_findings,
                biomarkers,
                biomarker_findings + surveillance_findings
            )
            if hypothesis.probability > 0.1:  # Threshold
                self._hypotheses.append(hypothesis)

        # 6. Rank hypotheses
        self._hypotheses.sort(key=lambda h: h.probability, reverse=True)

        # 7. Create findings for top diagnoses
        findings = []
        for i, hyp in enumerate(self._hypotheses[:5]):
            finding = self.create_finding(
                category="differential_diagnosis",
                description=f"#{i+1} {hyp.diagnosis}: {hyp.probability*100:.1f}% probability",
                confidence=hyp.confidence,
                evidence=hyp.required_met + hyp.supportive_met + hyp.biomarker_evidence,
                related_entities={
                    "diagnosis": hyp.diagnosis,
                    "rank": i + 1,
                    "probability": hyp.probability,
                    "icd10": hyp.pattern.get("icd10", []),
                    "recommended_tests": hyp._get_recommended_tests()
                }
            )
            findings.append(finding)

        # 8. Broadcast findings
        if findings:
            self.broadcast_findings(findings, correlation_id)

        # 9. Alert on high-probability serious diagnoses
        serious = ["sepsis", "acute_coronary_syndrome", "pulmonary_embolism"]
        for hyp in self._hypotheses[:3]:
            if hyp.diagnosis in serious and hyp.probability > 0.6:
                self.send_alert(
                    alert_message=f"High probability {hyp.diagnosis}: {hyp.probability*100:.1f}%",
                    severity="critical",
                    related_findings=[f for f in findings if hyp.diagnosis in f.description]
                )

        # 10. Query trial agent for treatment options
        if self._hypotheses:
            top_diagnosis = self._hypotheses[0].diagnosis
            self.query_agent(
                AgentType.TRIAL_RESCUE,
                f"Find trials for {top_diagnosis}",
                {"diagnosis": top_diagnosis, "icd10": self._hypotheses[0].pattern.get("icd10", [])}
            )

        return {
            "agent": self.name,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "differential_diagnosis": [h.to_dict() for h in self._hypotheses[:10]],
            "top_diagnosis": self._hypotheses[0].to_dict() if self._hypotheses else None,
            "findings": [f.to_dict() for f in findings],
            "clinical_findings_used": list(clinical_findings),
            "biomarkers_evaluated": list(biomarkers.keys()),
            "peer_evidence_integrated": len(peer_findings),
            "correlation_id": correlation_id,
        }

    def _derive_findings_from_biomarkers(self, biomarkers: Dict[str, float]) -> set:
        """Derive clinical findings from biomarker values."""
        findings = set()

        for marker, value in biomarkers.items():
            marker_upper = marker.upper()
            if marker_upper in BIOMARKER_FINDINGS:
                # Simple threshold logic (would be more sophisticated in production)
                if marker_upper == "CRP" and value > 10:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "PROCALCITONIN" and value > 0.5:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper in ["TROPONIN_I", "TROPONIN_T"] and value > 0.04:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "BNP" and value > 100:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "CREATININE" and value > 1.5:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "GLUCOSE" and value > 250:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "LACTATE" and value > 2:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "D_DIMER" and value > 500:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper in ["ALT", "AST"] and value > 80:
                    findings.add(BIOMARKER_FINDINGS[marker_upper])
                elif marker_upper == "WBC" and (value > 12 or value < 4):
                    findings.add(BIOMARKER_FINDINGS[marker_upper])

        return findings

    def _derive_findings_from_vitals(self, vital_signs: Dict[str, float]) -> set:
        """Derive clinical findings from vital signs."""
        findings = set()

        hr = vital_signs.get("hr") or vital_signs.get("heart_rate")
        temp = vital_signs.get("temp") or vital_signs.get("temperature")
        rr = vital_signs.get("rr") or vital_signs.get("respiratory_rate")
        sbp = vital_signs.get("sbp") or vital_signs.get("systolic_bp")
        spo2 = vital_signs.get("spo2") or vital_signs.get("oxygen_saturation")

        if hr and hr > 100:
            findings.add("tachycardia")
        if hr and hr < 60:
            findings.add("bradycardia")
        if temp and temp > 38.3:
            findings.add("fever_or_hypothermia")
        if temp and temp < 36:
            findings.add("fever_or_hypothermia")
        if rr and rr > 20:
            findings.add("tachypnea")
        if sbp and sbp < 90:
            findings.add("hypotension")
        if spo2 and spo2 < 94:
            findings.add("hypoxia")

        return findings

    def _evaluate_diagnosis(
        self,
        diagnosis: str,
        pattern: Dict[str, Any],
        clinical_findings: set,
        biomarkers: Dict[str, float],
        agent_findings: List[AgentFinding]
    ) -> DiagnosticHypothesis:
        """Evaluate a single diagnostic hypothesis."""
        hypothesis = DiagnosticHypothesis(diagnosis, pattern)

        # Check required criteria
        for criterion in pattern.get("required", []):
            if criterion in clinical_findings:
                hypothesis.required_met.append(criterion)
            else:
                hypothesis.required_missing.append(criterion)

        # Check supportive criteria
        for criterion in pattern.get("supportive", []):
            if criterion in clinical_findings:
                hypothesis.supportive_met.append(criterion)

        # Check biomarker evidence
        for marker in pattern.get("biomarkers", []):
            if marker in [b.upper() for b in biomarkers.keys()]:
                hypothesis.biomarker_evidence.append(marker)

        # Integrate agent findings
        for finding in agent_findings:
            # Check if finding relates to this diagnosis
            related = False
            for marker in pattern.get("biomarkers", []):
                if marker.lower() in finding.description.lower():
                    related = True
                    break

            if related:
                hypothesis.agent_evidence.append(finding)

        # Calculate probability
        hypothesis.calculate_probability()

        return hypothesis

    def get_diagnosis_info(self, diagnosis: str) -> Dict[str, Any]:
        """Get information about a specific diagnosis."""
        if diagnosis.lower() in DIAGNOSTIC_PATTERNS:
            pattern = DIAGNOSTIC_PATTERNS[diagnosis.lower()]
            return {
                "diagnosis": diagnosis,
                "required_criteria": pattern.get("required", []),
                "supportive_criteria": pattern.get("supportive", []),
                "key_biomarkers": pattern.get("biomarkers", []),
                "severity_markers": pattern.get("severity_markers", []),
                "icd10_codes": pattern.get("icd10", []),
            }
        return {"error": f"Unknown diagnosis: {diagnosis}"}

    def list_diagnoses(self) -> List[str]:
        """List all available diagnostic patterns."""
        return list(DIAGNOSTIC_PATTERNS.keys())


# Singleton instance
_diagnostic_agent = None


def get_diagnostic_agent() -> DiagnosticAgent:
    """Get singleton diagnostic agent instance."""
    global _diagnostic_agent
    if _diagnostic_agent is None:
        _diagnostic_agent = DiagnosticAgent()
    return _diagnostic_agent
