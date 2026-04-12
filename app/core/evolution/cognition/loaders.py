"""
Cognition Loaders - HyperCore Evolution
========================================

Pre-built knowledge loaders for the cognition store.

Loaders:
- FDA AI/ML guidance
- Clinical trial protocols
- Safety constraints
- Drug interaction rules
- Diagnostic guidelines
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schemas import (
    CognitionItem,
    CognitionItemType,
    DeploymentDomain,
)
from .store import CognitionStore

logger = logging.getLogger(__name__)


class BaseLoader:
    """Base class for cognition loaders."""

    def __init__(self, store: CognitionStore):
        self.store = store

    def load(self) -> int:
        """Load items into the store. Returns count of items added."""
        raise NotImplementedError


class FDAGuidanceLoader(BaseLoader):
    """
    Load FDA AI/ML guidance into cognition store.

    Includes key regulatory principles for AI in healthcare.
    """

    FDA_GUIDANCE = [
        {
            "title": "FDA AI/ML SaMD: Good Machine Learning Practice",
            "content": """
Good Machine Learning Practice (GMLP) principles for AI/ML-based Software as a Medical Device:

1. Multi-Disciplinary Expertise: Leverage diverse expertise throughout the total product life cycle.

2. Good Software Engineering: Implement good software engineering and security practices.

3. Clinical Study Participants Representative: Ensure data collection protocols include populations
   representative of the intended patient population.

4. Training and Test Data Independence: Maintain independence of training and test datasets.

5. Reference Datasets: Use appropriate reference datasets that represent realistic clinical conditions.

6. Model Design Tailored to Data: Design models fit for the available data and intended context.

7. Focus on Human Factors: Consider human-AI team performance including user interface design.

8. Testing in Real World Conditions: Test deployed models with clinically relevant conditions.

9. Provide Clear User Information: Provide users with clear, essential information including
   intended use, known limitations, and performance data.

10. Monitor Deployed Models: Monitor deployed models and manage retraining risks.
""",
            "source": "FDA - Good Machine Learning Practice for Medical Device Development",
            "category": "regulatory",
        },
        {
            "title": "FDA AI/ML: Predetermined Change Control Plan",
            "content": """
Predetermined Change Control Plan (PCCP) for AI/ML-based medical devices:

A PCCP describes the types of anticipated modifications (ML-DSP) that may be made to the device
without requiring separate marketing submission:

1. SaMD Pre-Specifications (SPS): Description of what aspects of the device the manufacturer
   intends to change through the ML update.

2. Algorithm Change Protocol (ACP): Methodology used to implement those changes in a controlled
   manner that manages risks to patients.

Key Elements:
- Description of intended modifications
- Method for developing, validating, and implementing modifications
- Assessment of modification impact on safety and effectiveness
- Update procedures for labeling

Performance Monitoring:
- Continuous performance assessment against pre-specified criteria
- Real-world performance data collection
- Bias monitoring across subpopulations
- Drift detection for input data and model performance
""",
            "source": "FDA - Marketing Submission Recommendations for PCCP",
            "category": "regulatory",
        },
        {
            "title": "FDA: Clinical Decision Support Software Guidance",
            "content": """
FDA guidance on Clinical Decision Support (CDS) software:

Not a Medical Device (criteria must ALL be met):
1. Not intended to acquire, process, or analyze medical data
2. Intended for displaying, storing, transmitting, or converting data formats
3. Not intended for active patient monitoring
4. Not intended for analyzing medical device data

Device Software Functions:
- Software intended to generate alerts/alarms for urgent situations requiring immediate action
- Software providing diagnostic or treatment recommendations where clinical reasoning is not transparent
- Software intended as a sole basis for clinical decisions without independent practitioner review

Risk-Based Approach:
- Consider significance of healthcare situation or condition
- Consider state of healthcare science
- Consider role of software in clinical workflow
- Consider user's ability to review basis for recommendations
""",
            "source": "FDA - Clinical Decision Support Software Guidance",
            "category": "regulatory",
        },
        {
            "title": "FDA 21 CFR Part 11: Electronic Records Compliance",
            "content": """
21 CFR Part 11 requirements for electronic records and signatures:

Validation Requirements:
- Systems must be validated to ensure accuracy, reliability, consistent intended performance
- Ability to discern invalid or altered records

Audit Trail Requirements:
- Computer-generated, time-stamped audit trails
- Record date/time of operator entries and actions
- Record changes do not obscure previously recorded information
- Audit trail documentation retained at least as long as subject records

Access Controls:
- Limit system access to authorized individuals
- Use of authority checks to ensure appropriate access levels
- Use of device checks to determine validity of source

Electronic Signatures:
- Signatures must be unique to one individual
- Identity verification before establishing signatures
- Signatures must include printed name, date/time, and meaning
""",
            "source": "FDA 21 CFR Part 11 - Electronic Records; Electronic Signatures",
            "category": "regulatory",
        },
        {
            "title": "FDA: Risk Management for AI/ML Medical Devices",
            "content": """
Risk management considerations for AI/ML-based medical devices:

Hazard Identification:
- Model performance degradation over time
- Distribution shift in input data
- Bias in training data affecting subpopulations
- Adversarial inputs or edge cases
- Integration failures with clinical workflow

Risk Analysis:
- Severity of potential harm from incorrect output
- Probability of hazardous situation occurring
- Probability of hazard leading to harm
- Detectability of failure modes

Risk Controls:
- Input data validation and preprocessing
- Output confidence thresholds
- Human-in-the-loop verification requirements
- Fallback mechanisms for model failures
- Continuous monitoring and alerting

Residual Risk Evaluation:
- Overall residual risk acceptability
- Risk-benefit analysis
- Post-market surveillance requirements
""",
            "source": "FDA - AI/ML Risk Management Framework",
            "category": "safety",
        },
    ]

    def load(self) -> int:
        """Load FDA guidance into cognition store."""
        added = 0

        for guidance in self.FDA_GUIDANCE:
            item = CognitionItem(
                title=guidance["title"],
                content=guidance["content"].strip(),
                source=guidance["source"],
                item_type=CognitionItemType.REGULATORY,
                category=guidance.get("category", "regulatory"),
                domain=DeploymentDomain.CLINICAL,
            )
            self.store.add(item, generate_embedding=True)
            added += 1

        logger.info(f"Loaded {added} FDA guidance items")
        return added


class ClinicalProtocolLoader(BaseLoader):
    """
    Load clinical protocol templates into cognition store.
    """

    PROTOCOLS = [
        {
            "title": "Diagnostic AI Validation Protocol",
            "content": """
Standard protocol for validating diagnostic AI systems:

1. Dataset Requirements:
   - Minimum N=1000 cases for primary endpoint
   - Stratified sampling across relevant subgroups
   - Independent test set not used in development
   - Gold standard labels from qualified reviewers

2. Performance Metrics:
   - Sensitivity (minimum threshold based on clinical context)
   - Specificity (minimum threshold based on clinical context)
   - Positive/Negative Predictive Values
   - Area Under ROC Curve (AUC)
   - Calibration metrics (Brier score, calibration curves)

3. Subgroup Analysis:
   - Age groups (pediatric, adult, geriatric)
   - Sex/Gender
   - Race/Ethnicity
   - Disease severity levels
   - Comorbidity profiles

4. Failure Mode Analysis:
   - False positive characterization
   - False negative characterization
   - Edge case identification
   - Confidence calibration assessment

5. Clinical Integration:
   - Workflow integration testing
   - User interface usability testing
   - Time-to-result impact assessment
   - Downstream decision impact analysis
""",
            "category": "methodology",
        },
        {
            "title": "Model Performance Monitoring Protocol",
            "content": """
Continuous monitoring protocol for deployed AI models:

1. Performance Metrics Tracking:
   - Daily/weekly performance metric calculation
   - Statistical process control charts
   - Alert thresholds for significant deviation
   - Trend analysis for gradual degradation

2. Data Distribution Monitoring:
   - Input feature distribution tracking
   - Covariate shift detection
   - Prior probability shift detection
   - Missing data pattern monitoring

3. Prediction Distribution Monitoring:
   - Output distribution tracking
   - Confidence score distribution
   - Prediction class balance
   - Comparison to historical baselines

4. Ground Truth Collection:
   - Systematic outcome data collection
   - Sampling strategy for verification
   - Time-to-outcome tracking
   - Feedback loop implementation

5. Bias Monitoring:
   - Performance metrics by demographic groups
   - Disparity detection algorithms
   - Regular equity audits
   - Mitigation action triggers

6. Response Procedures:
   - Alert escalation pathways
   - Model pause/rollback criteria
   - Retraining triggers
   - Stakeholder notification requirements
""",
            "category": "methodology",
        },
        {
            "title": "Clinical Endpoint Selection Guidelines",
            "content": """
Guidelines for selecting clinical endpoints in AI validation:

Primary Endpoints:
- Must be clinically meaningful
- Should be measurable and reproducible
- Gold standard should be clearly defined
- Time horizon must be specified

Surrogate Endpoints:
- Validated relationship to clinical outcome
- Biological plausibility
- Evidence from prior studies
- Regulatory acceptance considerations

Composite Endpoints:
- Components should be similar in clinical importance
- Direction of effect should be consistent
- Components should be measured reliably
- Clinical interpretability required

Time-to-Event Endpoints:
- Clear event definition
- Censoring rules specified
- Competing risks addressed
- Follow-up duration justified

Patient-Reported Outcomes:
- Validated instruments
- Appropriate for target population
- Missing data handling specified
- Clinically meaningful differences defined
""",
            "category": "methodology",
        },
    ]

    def load(self) -> int:
        """Load clinical protocols into cognition store."""
        added = 0

        for protocol in self.PROTOCOLS:
            item = CognitionItem(
                title=protocol["title"],
                content=protocol["content"].strip(),
                source="HyperCore Clinical Protocol Library",
                item_type=CognitionItemType.CLINICAL,
                category=protocol.get("category", "clinical"),
                domain=DeploymentDomain.CLINICAL,
            )
            self.store.add(item, generate_embedding=True)
            added += 1

        logger.info(f"Loaded {added} clinical protocol items")
        return added


class SafetyConstraintLoader(BaseLoader):
    """
    Load safety constraints and rules into cognition store.
    """

    CONSTRAINTS = [
        {
            "title": "AI Output Confidence Requirements",
            "content": """
Safety requirements for AI prediction confidence:

High-Stakes Decisions (Tier 1):
- Minimum confidence threshold: 95%
- Below threshold: MUST escalate to human reviewer
- Uncertainty quantification required
- Cannot be sole basis for decision

Standard Clinical Decisions (Tier 2):
- Minimum confidence threshold: 85%
- Below threshold: Flag for clinician attention
- Provide differential considerations
- Document alternative possibilities

Screening/Triage (Tier 3):
- Minimum confidence threshold: 75%
- Below threshold: Include in review queue
- Prioritization may be adjusted
- Follow-up protocols apply

Never Automated:
- Life-threatening immediate decisions
- Irreversible treatment decisions
- Emergency situations without override capability
- Decisions requiring patient consent
""",
            "category": "safety",
        },
        {
            "title": "PHI Protection Requirements",
            "content": """
Protected Health Information (PHI) handling requirements:

HIPAA Minimum Necessary Standard:
- Access only PHI necessary for specific purpose
- Implement role-based access controls
- Log all PHI access events
- Regular access pattern audits

De-identification Requirements:
- Remove 18 HIPAA identifiers
- Expert determination or Safe Harbor method
- Re-identification risk assessment
- De-identification documentation

Data Transmission:
- Encryption in transit (TLS 1.2+)
- Encryption at rest (AES-256)
- Secure key management
- Audit trail for data movement

AI-Specific Considerations:
- Training data de-identification verification
- Model memorization testing
- Inference output PHI checking
- Embedding privacy protection
""",
            "category": "safety",
        },
        {
            "title": "Model Failure Handling Protocol",
            "content": """
Protocol for handling AI model failures:

Failure Detection:
- Runtime error monitoring
- Output validity checking
- Latency threshold monitoring
- Confidence anomaly detection

Immediate Response:
- Log failure with full context
- Fall back to safe default behavior
- Alert appropriate personnel
- Do not expose internal errors to users

Graceful Degradation:
- Disable affected functionality
- Enable alternative pathways
- Maintain audit trail
- Preserve patient safety

Recovery Procedures:
- Root cause analysis required
- Fix verification before re-enabling
- Staged rollout after recovery
- Post-incident review within 24 hours

Communication:
- Notify affected clinicians
- Document in system status
- Patient notification if appropriate
- Regulatory reporting if required
""",
            "category": "safety",
        },
        {
            "title": "Human Override Requirements",
            "content": """
Requirements for human override capability:

Override Availability:
- All AI recommendations must be overridable
- Override mechanism clearly visible
- No friction added to override action
- Override effective immediately

Override Documentation:
- Automatic logging of override events
- Reason for override (structured + free text)
- Alternative decision captured
- Outcome tracking for overridden decisions

Override Analysis:
- Regular review of override patterns
- Identify systematic issues
- Feed back to model improvement
- Clinician feedback collection

Override Restrictions:
- Only qualified personnel can override
- Role-appropriate override levels
- Audit trail for override authority
- Regular competency verification

Never Block Override:
- System must never prevent clinician override
- Technical failures must not lock decisions
- Override pathway independent of AI system
- Emergency override always available
""",
            "category": "safety",
        },
    ]

    def load(self) -> int:
        """Load safety constraints into cognition store."""
        added = 0

        for constraint in self.CONSTRAINTS:
            item = CognitionItem(
                title=constraint["title"],
                content=constraint["content"].strip(),
                source="HyperCore Safety Framework",
                item_type=CognitionItemType.SAFETY,
                category=constraint.get("category", "safety"),
                domain=DeploymentDomain.CLINICAL,
            )
            self.store.add(item, generate_embedding=True)
            added += 1

        logger.info(f"Loaded {added} safety constraint items")
        return added


class EvolutionLessonLoader(BaseLoader):
    """
    Load seed lessons for evolution system.
    """

    SEED_LESSONS = [
        {
            "title": "Confidence Calibration Improves Trust",
            "content": """
Lesson: Well-calibrated confidence scores significantly improve clinician trust and adoption.

Evidence:
- Models with calibrated probabilities show higher clinician acceptance
- Overconfident predictions lead to alert fatigue
- Underconfident predictions lead to unnecessary escalations
- Calibration should be monitored continuously in production

Recommendation:
- Use temperature scaling or Platt scaling for calibration
- Monitor Expected Calibration Error (ECE) in production
- Stratify calibration by subpopulations
- Recalibrate when performance drift detected
""",
            "confidence": 0.85,
        },
        {
            "title": "Subgroup Performance Varies Significantly",
            "content": """
Lesson: Model performance often varies substantially across patient subgroups.

Evidence:
- Training data imbalance leads to disparate performance
- Rare conditions may have significantly worse accuracy
- Age extremes (very young, very old) often underperform
- Comorbidity patterns affect prediction reliability

Recommendation:
- Always conduct stratified performance analysis
- Set minimum subgroup sample sizes for validation
- Implement subgroup-specific confidence adjustments
- Monitor subgroup performance in production
""",
            "confidence": 0.9,
        },
        {
            "title": "Integration Workflow Critical for Adoption",
            "content": """
Lesson: Clinical workflow integration is as important as model accuracy for success.

Evidence:
- Poorly integrated tools are ignored by clinicians
- Interruption-based alerts have low compliance
- Information overload reduces effectiveness
- Timing of recommendations affects action rates

Recommendation:
- Design with clinical workflow in mind from start
- Minimize clicks and cognitive load
- Present at decision points, not randomly
- Provide actionable recommendations, not just predictions
""",
            "confidence": 0.88,
        },
    ]

    def load(self) -> int:
        """Load seed lessons into cognition store."""
        added = 0

        for lesson in self.SEED_LESSONS:
            item = CognitionItem(
                title=lesson["title"],
                content=lesson["content"].strip(),
                source="HyperCore Evolution Seed Lessons",
                item_type=CognitionItemType.LESSON,
                category="lesson",
                domain=DeploymentDomain.RESEARCH,
                metadata={"confidence": lesson.get("confidence", 0.7)},
            )
            self.store.add(item, generate_embedding=True)
            added += 1

        logger.info(f"Loaded {added} seed lesson items")
        return added


class JSONFileLoader(BaseLoader):
    """
    Load cognition items from a JSON file.

    Expected format:
    [
        {
            "title": "...",
            "content": "...",
            "source": "...",
            "item_type": "regulatory|clinical|safety|lesson|methodology|evidence|prior",
            "category": "...",
            "domain": "pharma|clinical|research|administrative"
        },
        ...
    ]
    """

    def __init__(self, store: CognitionStore, file_path: Path):
        super().__init__(store)
        self.file_path = Path(file_path)

    def load(self) -> int:
        """Load items from JSON file."""
        if not self.file_path.exists():
            logger.warning(f"JSON file not found: {self.file_path}")
            return 0

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            return 0

        if not isinstance(data, list):
            data = [data]

        added = 0
        for item_data in data:
            try:
                item = CognitionItem(
                    title=item_data.get("title", ""),
                    content=item_data.get("content", ""),
                    source=item_data.get("source", str(self.file_path)),
                    item_type=CognitionItemType(item_data.get("item_type", "prior")),
                    category=item_data.get("category", ""),
                    domain=DeploymentDomain(item_data.get("domain", "research")),
                    metadata=item_data.get("metadata", {}),
                )
                self.store.add(item, generate_embedding=True)
                added += 1
            except Exception as e:
                logger.error(f"Failed to add item: {e}")

        logger.info(f"Loaded {added} items from {self.file_path}")
        return added


def load_all_default_knowledge(store: CognitionStore) -> Dict[str, int]:
    """
    Load all default knowledge into the cognition store.

    Returns:
        Dictionary with counts of items loaded by category
    """
    results = {}

    loaders = [
        ("fda_guidance", FDAGuidanceLoader(store)),
        ("clinical_protocols", ClinicalProtocolLoader(store)),
        ("safety_constraints", SafetyConstraintLoader(store)),
        ("evolution_lessons", EvolutionLessonLoader(store)),
    ]

    for name, loader in loaders:
        try:
            count = loader.load()
            results[name] = count
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            results[name] = 0

    total = sum(results.values())
    logger.info(f"Loaded {total} total cognition items")

    return results
