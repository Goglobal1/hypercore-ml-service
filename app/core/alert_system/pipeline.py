"""
Alert System Pipeline - Unified Integration Layer
Connects all components: scoring, CSE, agents, TTH, routing, persistence.
"""

from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
import logging
import asyncio

from .models import (
    AlertEvent,
    EvaluationResult,
    ClinicalState,
    AlertType,
    AlertSeverity,
    EventType,
    ConfidenceLevel,
)
from .engine import ClinicalStateEngine, get_engine
from .storage import StorageBackend, get_storage
from .routing import AlertRouter, EscalationManager, get_router, get_escalation_manager
from .config import DOMAIN_CONFIGS, get_domain_config
from .risk_calculator import calculate_risk_score as calc_risk, quick_risk_score

# Specialized module imports for integrated pipeline
try:
    from ..genomics_integration import analyze_genomics, get_gene_variants
    GENOMICS_AVAILABLE = True
except ImportError:
    GENOMICS_AVAILABLE = False

try:
    from ..drug_response_predictor import check_drug_interactions, get_drug_profile
    PHARMA_AVAILABLE = True
except ImportError:
    PHARMA_AVAILABLE = False

try:
    from ..pathogen_detection import detect_outbreaks, get_pathogen_info
    PATHOGEN_AVAILABLE = True
except ImportError:
    PATHOGEN_AVAILABLE = False

try:
    from ..multiomic_fusion import fusion_analysis, unified_query
    MULTIOMIC_AVAILABLE = True
except ImportError:
    MULTIOMIC_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE STEP DEFINITIONS
# =============================================================================

@dataclass
class PipelineStepResult:
    """Result of a single pipeline step."""
    step_name: str
    success: bool
    duration_ms: float
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_name": self.step_name,
            "success": self.success,
            "duration_ms": round(self.duration_ms, 2),
            "data": self.data,
            "error": self.error,
        }


@dataclass
class PipelineResult:
    """Complete result of pipeline execution."""
    patient_id: str
    risk_domain: str
    timestamp: datetime
    success: bool
    steps: List[PipelineStepResult]
    evaluation_result: Optional[EvaluationResult]
    routing_result: Optional[Dict[str, Any]]
    total_duration_ms: float
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "risk_domain": self.risk_domain,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "steps": [s.to_dict() for s in self.steps],
            "evaluation_result": self.evaluation_result.to_dict() if self.evaluation_result else None,
            "routing_result": self.routing_result,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "error": self.error,
        }


# =============================================================================
# AGENT INTEGRATION
# =============================================================================

class AgentIntegration:
    """
    Integrates with the 4 clinical agents:
    - BiomarkerAgent: Analyzes lab values and trends
    - DiagnosticAgent: Clinical reasoning and differential diagnosis
    - TrialRescueAgent: Clinical trial eligibility
    - SurveillanceAgent: Population-level outbreak detection
    """

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._agent_callbacks: Dict[str, Callable] = {}

    def register_agent(self, agent_name: str, agent_instance: Any) -> None:
        """Register an agent instance."""
        self._agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")

    def register_callback(self, agent_name: str, callback: Callable) -> None:
        """Register a callback for agent invocation."""
        self._agent_callbacks[agent_name] = callback

    async def invoke_biomarker_agent(
        self,
        patient_id: str,
        risk_domain: str,
        lab_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke BiomarkerAgent for lab analysis."""
        start = datetime.now(timezone.utc)

        try:
            if "biomarker" in self._agent_callbacks:
                result = await self._agent_callbacks["biomarker"](
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    lab_data=lab_data,
                )
            elif "biomarker" in self._agents:
                agent = self._agents["biomarker"]
                if hasattr(agent, "analyze_async"):
                    result = await agent.analyze_async(patient_id, lab_data)
                elif hasattr(agent, "analyze"):
                    result = agent.analyze(patient_id, lab_data)
                else:
                    result = {"status": "no_method"}
            else:
                # Default analysis when agent not available
                result = self._default_biomarker_analysis(lab_data, risk_domain)

            return {
                "agent": "biomarker",
                "success": True,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "findings": result,
            }
        except Exception as e:
            logger.error(f"BiomarkerAgent failed: {e}")
            return {
                "agent": "biomarker",
                "success": False,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "error": str(e),
            }

    async def invoke_diagnostic_agent(
        self,
        patient_id: str,
        risk_domain: str,
        clinical_data: Dict[str, Any],
        biomarker_findings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Invoke DiagnosticAgent for clinical reasoning."""
        start = datetime.now(timezone.utc)

        try:
            if "diagnostic" in self._agent_callbacks:
                result = await self._agent_callbacks["diagnostic"](
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    clinical_data=clinical_data,
                    biomarker_findings=biomarker_findings,
                )
            elif "diagnostic" in self._agents:
                agent = self._agents["diagnostic"]
                if hasattr(agent, "diagnose_async"):
                    result = await agent.diagnose_async(patient_id, clinical_data)
                elif hasattr(agent, "diagnose"):
                    result = agent.diagnose(patient_id, clinical_data)
                else:
                    result = {"status": "no_method"}
            else:
                result = self._default_diagnostic_analysis(clinical_data, biomarker_findings)

            return {
                "agent": "diagnostic",
                "success": True,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "findings": result,
            }
        except Exception as e:
            logger.error(f"DiagnosticAgent failed: {e}")
            return {
                "agent": "diagnostic",
                "success": False,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "error": str(e),
            }

    async def invoke_trial_rescue_agent(
        self,
        patient_id: str,
        risk_domain: str,
        evaluation_result: EvaluationResult,
    ) -> Dict[str, Any]:
        """Invoke TrialRescueAgent to check trial eligibility."""
        start = datetime.now(timezone.utc)

        try:
            if "trial_rescue" in self._agent_callbacks:
                result = await self._agent_callbacks["trial_rescue"](
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    evaluation_result=evaluation_result,
                )
            elif "trial_rescue" in self._agents:
                agent = self._agents["trial_rescue"]
                if hasattr(agent, "check_eligibility_async"):
                    result = await agent.check_eligibility_async(patient_id)
                elif hasattr(agent, "check_eligibility"):
                    result = agent.check_eligibility(patient_id)
                else:
                    result = {"status": "no_method"}
            else:
                result = {"eligible_trials": [], "status": "agent_not_available"}

            return {
                "agent": "trial_rescue",
                "success": True,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "findings": result,
            }
        except Exception as e:
            logger.error(f"TrialRescueAgent failed: {e}")
            return {
                "agent": "trial_rescue",
                "success": False,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "error": str(e),
            }

    async def invoke_surveillance_agent(
        self,
        patient_id: str,
        risk_domain: str,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invoke SurveillanceAgent for outbreak detection."""
        start = datetime.now(timezone.utc)

        try:
            if "surveillance" in self._agent_callbacks:
                result = await self._agent_callbacks["surveillance"](
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    location=location,
                )
            elif "surveillance" in self._agents:
                agent = self._agents["surveillance"]
                if hasattr(agent, "check_outbreak_async"):
                    result = await agent.check_outbreak_async(location)
                elif hasattr(agent, "check_outbreak"):
                    result = agent.check_outbreak(location)
                else:
                    result = {"status": "no_method"}
            else:
                result = {"outbreak_detected": False, "status": "agent_not_available"}

            return {
                "agent": "surveillance",
                "success": True,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "findings": result,
            }
        except Exception as e:
            logger.error(f"SurveillanceAgent failed: {e}")
            return {
                "agent": "surveillance",
                "success": False,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "error": str(e),
            }

    def _default_biomarker_analysis(
        self,
        lab_data: Dict[str, Any],
        risk_domain: str,
    ) -> Dict[str, Any]:
        """Default biomarker analysis when agent not available."""
        from .risk_calculator import calculate_risk_score, normalize_biomarker_name, get_domain_thresholds

        # Use the risk calculator for detailed analysis
        result = calculate_risk_score(
            risk_domain=risk_domain,
            lab_data=lab_data,
        )

        return {
            "abnormal_biomarkers": result.get("warning_biomarkers", []),
            "critical_biomarkers": result.get("critical_biomarkers", []),
            "total_checked": result.get("matched_biomarkers", len(lab_data)),
            "source": "default_analysis",
        }

    def _default_diagnostic_analysis(
        self,
        clinical_data: Dict[str, Any],
        biomarker_findings: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Default diagnostic analysis when agent not available."""
        critical_count = len(biomarker_findings.get("findings", {}).get("critical_biomarkers", []))
        abnormal_count = len(biomarker_findings.get("findings", {}).get("abnormal_biomarkers", []))

        if critical_count > 0:
            assessment = "Critical abnormalities detected requiring immediate attention"
            confidence = 0.85
        elif abnormal_count > 2:
            assessment = "Multiple abnormalities detected, close monitoring recommended"
            confidence = 0.70
        elif abnormal_count > 0:
            assessment = "Minor abnormalities detected, routine follow-up"
            confidence = 0.60
        else:
            assessment = "No significant abnormalities detected"
            confidence = 0.90

        return {
            "assessment": assessment,
            "confidence": confidence,
            "differentials": [],
            "source": "default_analysis",
        }

    async def run_all_agents(
        self,
        patient_id: str,
        risk_domain: str,
        lab_data: Dict[str, Any],
        clinical_data: Dict[str, Any],
        evaluation_result: EvaluationResult,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run all 4 agents with cross-validation."""
        start = datetime.now(timezone.utc)

        # Run biomarker agent first
        biomarker_result = await self.invoke_biomarker_agent(
            patient_id, risk_domain, lab_data
        )

        # Run diagnostic and surveillance in parallel
        diagnostic_task = self.invoke_diagnostic_agent(
            patient_id, risk_domain, clinical_data, biomarker_result
        )
        surveillance_task = self.invoke_surveillance_agent(
            patient_id, risk_domain, location
        )

        diagnostic_result, surveillance_result = await asyncio.gather(
            diagnostic_task, surveillance_task
        )

        # Run trial rescue (depends on evaluation)
        trial_result = await self.invoke_trial_rescue_agent(
            patient_id, risk_domain, evaluation_result
        )

        # Cross-validate findings
        cross_validation = self._cross_validate_agents(
            biomarker_result,
            diagnostic_result,
            surveillance_result,
            trial_result,
        )

        return {
            "biomarker": biomarker_result,
            "diagnostic": diagnostic_result,
            "surveillance": surveillance_result,
            "trial_rescue": trial_result,
            "cross_validation": cross_validation,
            "total_duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
        }

    def _cross_validate_agents(
        self,
        biomarker: Dict[str, Any],
        diagnostic: Dict[str, Any],
        surveillance: Dict[str, Any],
        trial: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Cross-validate findings across agents."""
        agreements = 0
        conflicts = []
        total_confidence = 0.0
        confidence_count = 0

        # Check if biomarker and diagnostic agree on severity
        biomarker_findings = biomarker.get("findings", {})
        diagnostic_findings = diagnostic.get("findings", {})

        biomarker_critical = len(biomarker_findings.get("critical_biomarkers", []))
        diagnostic_confidence = diagnostic_findings.get("confidence", 0.5)

        if biomarker_critical > 0 and diagnostic_confidence > 0.7:
            agreements += 1
        elif biomarker_critical == 0 and diagnostic_confidence < 0.5:
            agreements += 1
        elif biomarker_critical > 0 and diagnostic_confidence < 0.3:
            conflicts.append("Biomarker shows critical values but diagnostic confidence low")

        # Track confidence scores
        if "confidence" in diagnostic_findings:
            total_confidence += diagnostic_findings["confidence"]
            confidence_count += 1

        # Surveillance agreement
        surveillance_findings = surveillance.get("findings", {})
        if surveillance_findings.get("outbreak_detected"):
            if biomarker_critical > 0:
                agreements += 1
            else:
                conflicts.append("Outbreak detected but no critical biomarkers")

        return {
            "agreements": agreements,
            "conflicts": conflicts,
            "consensus_score": agreements / max(agreements + len(conflicts), 1),
            "average_confidence": total_confidence / max(confidence_count, 1),
        }


# =============================================================================
# TIME-TO-HARM INTEGRATION
# =============================================================================

class TTHIntegration:
    """Integrates with Time-to-Harm prediction system."""

    def __init__(self):
        self._tth_callback: Optional[Callable] = None

    def register_callback(self, callback: Callable) -> None:
        """Register TTH prediction callback."""
        self._tth_callback = callback

    async def predict_tth(
        self,
        patient_id: str,
        risk_domain: str,
        risk_score: float,
        velocity: float,
        clinical_state: ClinicalState,
    ) -> Dict[str, Any]:
        """Predict time-to-harm."""
        start = datetime.now(timezone.utc)

        try:
            if self._tth_callback:
                result = await self._tth_callback(
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    risk_score=risk_score,
                    velocity=velocity,
                    clinical_state=clinical_state,
                )
            else:
                # Default TTH estimation based on state and velocity
                result = self._default_tth_prediction(
                    risk_score, velocity, clinical_state
                )

            return {
                "success": True,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "prediction": result,
            }
        except Exception as e:
            logger.error(f"TTH prediction failed: {e}")
            return {
                "success": False,
                "duration_ms": (datetime.now(timezone.utc) - start).total_seconds() * 1000,
                "error": str(e),
            }

    def _default_tth_prediction(
        self,
        risk_score: float,
        velocity: float,
        clinical_state: ClinicalState,
    ) -> Dict[str, Any]:
        """Default TTH prediction when callback not available."""
        # Base hours by state
        base_hours = {
            ClinicalState.S0_STABLE: 72.0,
            ClinicalState.S1_WATCH: 24.0,
            ClinicalState.S2_ESCALATING: 8.0,
            ClinicalState.S3_CRITICAL: 2.0,
        }

        hours = base_hours.get(clinical_state, 24.0)

        # Adjust by velocity
        if velocity > 0.15:
            hours *= 0.5  # Rapid deterioration
        elif velocity > 0.05:
            hours *= 0.75

        # Adjust by risk score
        hours *= (1.5 - risk_score)

        # Determine intervention window
        if hours <= 2:
            window = "IMMEDIATE"
        elif hours <= 6:
            window = "URGENT"
        elif hours <= 24:
            window = "SOON"
        else:
            window = "ROUTINE"

        # Confidence based on available data
        confidence = 0.65 if velocity > 0 else 0.50

        return {
            "hours": round(max(hours, 0.5), 1),
            "intervention_window": window,
            "confidence": confidence,
            "source": "default_model",
        }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class AlertPipeline:
    """
    Main alert pipeline implementing the 11-step flow:
    1. Patient data arrives (labs, vitals, etc.)
    2. Risk score calculated
    3. CSE evaluates state transition
    4. If state change -> trigger all 4 agents
    5. Agents cross-validate
    6. Time-to-harm prediction runs
    7. Confidence scored across sources
    8. Alert decision made (fire/suppress)
    9. If fire -> route to appropriate clinician
    10. Audit log updated
    11. Dashboard notified (WebSocket/SSE)
    """

    def __init__(
        self,
        engine: Optional[ClinicalStateEngine] = None,
        storage: Optional[StorageBackend] = None,
        router: Optional[AlertRouter] = None,
        escalation_manager: Optional[EscalationManager] = None,
    ):
        self.engine = engine or get_engine()
        self.storage = storage or get_storage()
        self.router = router or get_router()
        self.escalation_manager = escalation_manager or get_escalation_manager()

        self.agents = AgentIntegration()
        self.tth = TTHIntegration()

        # Callbacks for real-time notifications
        self._dashboard_callbacks: List[Callable] = []
        self._risk_score_callback: Optional[Callable] = None

    def register_dashboard_callback(self, callback: Callable) -> None:
        """Register callback for dashboard notifications (step 11)."""
        self._dashboard_callbacks.append(callback)

    def register_risk_score_callback(self, callback: Callable) -> None:
        """Register callback for risk score calculation (step 2)."""
        self._risk_score_callback = callback

    async def process_patient_data(
        self,
        patient_id: str,
        risk_domain: str,
        lab_data: Optional[Dict[str, Any]] = None,
        vitals_data: Optional[Dict[str, Any]] = None,
        clinical_data: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
        location: Optional[str] = None,
        encounter_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Process patient data through the complete pipeline.

        This is the main entry point for the unified /patient/intake endpoint.
        """
        pipeline_start = datetime.now(timezone.utc)
        steps: List[PipelineStepResult] = []
        evaluation_result: Optional[EvaluationResult] = None
        routing_result: Optional[Dict[str, Any]] = None

        try:
            # Step 1: Data arrives (implicit - we have it)
            steps.append(PipelineStepResult(
                step_name="data_received",
                success=True,
                duration_ms=0,
                data={
                    "has_labs": lab_data is not None,
                    "has_vitals": vitals_data is not None,
                    "has_clinical": clinical_data is not None,
                    "has_risk_score": risk_score is not None,
                }
            ))

            # Step 2: Calculate risk score (if not provided)
            step_start = datetime.now(timezone.utc)
            if risk_score is None:
                risk_score = await self._calculate_risk_score(
                    patient_id, risk_domain, lab_data, vitals_data
                )
            steps.append(PipelineStepResult(
                step_name="risk_score_calculated",
                success=True,
                duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                data={"risk_score": risk_score}
            ))

            # Step 3: CSE evaluates state transition
            step_start = datetime.now(timezone.utc)
            evaluation_result = self.engine.evaluate(
                patient_id=patient_id,
                timestamp=datetime.now(timezone.utc),
                risk_domain=risk_domain,
                current_scores={"primary": risk_score},
                contributing_biomarkers=list((lab_data or {}).keys()),
                current_tth_hours=None,  # Will be filled in step 6
            )
            steps.append(PipelineStepResult(
                step_name="cse_evaluation",
                success=True,
                duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                data={
                    "state_previous": evaluation_result.state_previous.value if evaluation_result.state_previous else None,
                    "state_current": evaluation_result.state_now.value,
                    "state_transition": evaluation_result.state_transition,
                    "alert_fired": evaluation_result.alert_fired,
                }
            ))

            # Step 4 & 5: If state change, trigger agents and cross-validate
            agent_results = None
            if evaluation_result.state_transition or evaluation_result.state_now.severity_level >= 2:
                step_start = datetime.now(timezone.utc)
                agent_results = await self.agents.run_all_agents(
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    lab_data=lab_data or {},
                    clinical_data=clinical_data or {},
                    evaluation_result=evaluation_result,
                    location=location,
                )
                steps.append(PipelineStepResult(
                    step_name="agent_analysis",
                    success=True,
                    duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                    data={
                        "agents_invoked": ["biomarker", "diagnostic", "surveillance", "trial_rescue"],
                        "cross_validation": agent_results.get("cross_validation", {}),
                    }
                ))
                evaluation_result.agent_findings = agent_results
            else:
                steps.append(PipelineStepResult(
                    step_name="agent_analysis",
                    success=True,
                    duration_ms=0,
                    data={"skipped": True, "reason": "no_state_transition_or_low_severity"}
                ))

            # Step 5.5: Run specialized modules (genomics, pharma, pathogen, multiomic)
            step_start = datetime.now(timezone.utc)
            specialized_results = await self._run_specialized_modules(
                patient_id=patient_id,
                risk_domain=risk_domain,
                lab_data=lab_data or {},
                clinical_data=clinical_data or {},
                contributing_biomarkers=evaluation_result.contributing_biomarkers or [],
            )
            modules_invoked = specialized_results.get("modules_invoked", [])
            if modules_invoked:
                steps.append(PipelineStepResult(
                    step_name="specialized_modules",
                    success=True,
                    duration_ms=specialized_results.get("duration_ms", 0),
                    data={
                        "modules_invoked": modules_invoked,
                        "genomics": specialized_results.get("genomics", {}).get("invoked", False),
                        "pharma": specialized_results.get("pharma", {}).get("invoked", False),
                        "pathogen": specialized_results.get("pathogen", {}).get("invoked", False),
                        "multiomic": specialized_results.get("multiomic", {}).get("invoked", False),
                    }
                ))
                # Add specialized results to agent_findings
                if evaluation_result.agent_findings is None:
                    evaluation_result.agent_findings = {}
                evaluation_result.agent_findings["specialized_modules"] = specialized_results
            else:
                steps.append(PipelineStepResult(
                    step_name="specialized_modules",
                    success=True,
                    duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                    data={"skipped": True, "reason": "no_modules_triggered"}
                ))

            # Step 6: Time-to-harm prediction
            step_start = datetime.now(timezone.utc)
            tth_result = await self.tth.predict_tth(
                patient_id=patient_id,
                risk_domain=risk_domain,
                risk_score=risk_score,
                velocity=evaluation_result.alert_event.velocity if evaluation_result.alert_event else 0,
                clinical_state=evaluation_result.state_now,
            )
            tth_prediction = tth_result.get("prediction", {})
            steps.append(PipelineStepResult(
                step_name="tth_prediction",
                success=tth_result.get("success", False),
                duration_ms=tth_result.get("duration_ms", 0),
                data=tth_prediction
            ))

            # Update evaluation with TTH
            if evaluation_result.alert_event:
                evaluation_result.alert_event.time_to_harm_hours = tth_prediction.get("hours")
                evaluation_result.alert_event.intervention_window = tth_prediction.get("intervention_window")
            evaluation_result.time_to_harm = tth_prediction

            # Step 6.5: Cascade detection - extend detection window with multi-omic signals
            if specialized_results.get("modules_invoked"):
                cascade_detection = self._calculate_cascade_detection(
                    specialized_results=specialized_results,
                    traditional_tth_hours=tth_prediction.get("hours", 24.0),
                    lab_data=lab_data or {},
                )

                # Add cascade detection step
                steps.append(PipelineStepResult(
                    step_name="cascade_detection",
                    success=True,
                    duration_ms=0.01,
                    data={
                        "earliest_level": cascade_detection.get("earliest_signal_level"),
                        "improvement_days": cascade_detection.get("detection_improvement_days"),
                        "levels_detected": cascade_detection.get("levels_detected"),
                    }
                ))

                # Store cascade detection in evaluation result
                evaluation_result.cascade_detection = cascade_detection

                # Update time_to_harm with integrated values
                if cascade_detection.get("integrated_detection_hours"):
                    evaluation_result.time_to_harm["integrated_hours"] = cascade_detection["integrated_detection_hours"]
                    evaluation_result.time_to_harm["traditional_hours"] = cascade_detection["traditional_detection_hours"]
                    evaluation_result.time_to_harm["improvement_hours"] = cascade_detection["detection_improvement_hours"]
                    evaluation_result.time_to_harm["improvement_days"] = cascade_detection["detection_improvement_days"]

                # Add cascade recommendations to alert
                cascade_recs = self._generate_cascade_recommendations(cascade_detection)
                if evaluation_result.alert_event and cascade_recs:
                    existing_recs = evaluation_result.alert_event.recommendations or []
                    evaluation_result.alert_event.recommendations = cascade_recs + existing_recs

            # Step 7: Confidence scoring across sources
            step_start = datetime.now(timezone.utc)
            final_confidence = self._calculate_combined_confidence(
                evaluation_result,
                agent_results,
                tth_prediction,
            )
            steps.append(PipelineStepResult(
                step_name="confidence_scoring",
                success=True,
                duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                data={"final_confidence": final_confidence}
            ))
            evaluation_result.confidence = final_confidence

            # Step 8: Alert decision (already made in CSE, but can be adjusted)
            steps.append(PipelineStepResult(
                step_name="alert_decision",
                success=True,
                duration_ms=0,
                data={
                    "alert_fired": evaluation_result.alert_fired,
                    "alert_type": evaluation_result.alert_type.value,
                    "severity": evaluation_result.severity.value,
                }
            ))

            # Step 9: Route alert if fired
            step_start = datetime.now(timezone.utc)
            if evaluation_result.alert_fired and evaluation_result.alert_event:
                routing_result = self.router.route_alert(evaluation_result.alert_event)

                # Schedule escalation if needed
                if routing_result.get("escalation_minutes"):
                    self.escalation_manager.schedule_escalation(
                        evaluation_result.alert_event,
                        routing_result["escalation_minutes"],
                    )

                steps.append(PipelineStepResult(
                    step_name="alert_routing",
                    success=True,
                    duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                    data={
                        "routed_to_roles": routing_result.get("routed_to_roles", []),
                        "notification_channels": routing_result.get("notification_channels", []),
                        "escalation_minutes": routing_result.get("escalation_minutes"),
                    }
                ))
            else:
                steps.append(PipelineStepResult(
                    step_name="alert_routing",
                    success=True,
                    duration_ms=0,
                    data={"skipped": True, "reason": "alert_not_fired"}
                ))

            # Step 10: Audit log (already done in CSE, but log pipeline completion)
            step_start = datetime.now(timezone.utc)
            if evaluation_result.alert_event:
                self.storage.log_event(evaluation_result.alert_event)
            steps.append(PipelineStepResult(
                step_name="audit_log",
                success=True,
                duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                data={"event_id": evaluation_result.alert_event.event_id if evaluation_result.alert_event else None}
            ))

            # Step 11: Dashboard notification
            step_start = datetime.now(timezone.utc)
            notification_count = await self._notify_dashboards(evaluation_result, routing_result)
            steps.append(PipelineStepResult(
                step_name="dashboard_notification",
                success=True,
                duration_ms=(datetime.now(timezone.utc) - step_start).total_seconds() * 1000,
                data={"callbacks_notified": notification_count}
            ))

            total_duration = (datetime.now(timezone.utc) - pipeline_start).total_seconds() * 1000

            return PipelineResult(
                patient_id=patient_id,
                risk_domain=risk_domain,
                timestamp=pipeline_start,
                success=True,
                steps=steps,
                evaluation_result=evaluation_result,
                routing_result=routing_result,
                total_duration_ms=total_duration,
            )

        except Exception as e:
            logger.exception(f"Pipeline failed for patient {patient_id}: {e}")
            total_duration = (datetime.now(timezone.utc) - pipeline_start).total_seconds() * 1000

            return PipelineResult(
                patient_id=patient_id,
                risk_domain=risk_domain,
                timestamp=pipeline_start,
                success=False,
                steps=steps,
                evaluation_result=evaluation_result,
                routing_result=routing_result,
                total_duration_ms=total_duration,
                error=str(e),
            )

    async def _calculate_risk_score(
        self,
        patient_id: str,
        risk_domain: str,
        lab_data: Optional[Dict[str, Any]],
        vitals_data: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate risk score from lab and vitals data."""
        if self._risk_score_callback:
            try:
                return await self._risk_score_callback(
                    patient_id=patient_id,
                    risk_domain=risk_domain,
                    lab_data=lab_data,
                    vitals_data=vitals_data,
                )
            except Exception as e:
                logger.error(f"Risk score callback failed: {e}")

        # Default score calculation based on biomarker thresholds
        return self._default_risk_score(risk_domain, lab_data, vitals_data)

    def _default_risk_score(
        self,
        risk_domain: str,
        lab_data: Optional[Dict[str, Any]],
        vitals_data: Optional[Dict[str, Any]],
    ) -> float:
        """
        Default risk score calculation using the risk calculator.

        Uses weighted threshold analysis with:
        - Direction-aware scoring (rising vs falling)
        - Case-insensitive biomarker matching
        - Weight-based composite scoring
        """
        # Use the risk calculator module
        return quick_risk_score(
            risk_domain=risk_domain,
            lab_data=lab_data,
            vital_signs=vitals_data,
        )

    def _calculate_combined_confidence(
        self,
        evaluation: EvaluationResult,
        agent_results: Optional[Dict[str, Any]],
        tth_prediction: Dict[str, Any],
    ) -> float:
        """Calculate combined confidence from all sources."""
        confidences = []
        weights = []

        # CSE confidence
        if evaluation.confidence > 0:
            confidences.append(evaluation.confidence)
            weights.append(1.0)

        # Agent cross-validation confidence
        if agent_results:
            cross_val = agent_results.get("cross_validation", {})
            if "consensus_score" in cross_val:
                confidences.append(cross_val["consensus_score"])
                weights.append(0.8)
            if "average_confidence" in cross_val:
                confidences.append(cross_val["average_confidence"])
                weights.append(0.6)

        # TTH confidence
        if "confidence" in tth_prediction:
            confidences.append(tth_prediction["confidence"])
            weights.append(0.5)

        if not confidences:
            return 0.5

        # Weighted average
        total = sum(c * w for c, w in zip(confidences, weights))
        return total / sum(weights)

    async def _notify_dashboards(
        self,
        evaluation: EvaluationResult,
        routing: Optional[Dict[str, Any]],
    ) -> int:
        """Notify all registered dashboard callbacks."""
        notification_data = {
            "type": "alert_update",
            "patient_id": evaluation.patient_id,
            "risk_domain": evaluation.risk_domain,
            "timestamp": evaluation.timestamp.isoformat(),
            "state": evaluation.state_now.value,
            "alert_fired": evaluation.alert_fired,
            "severity": evaluation.severity.value,
            "routing": routing,
        }

        notified = 0
        for callback in self._dashboard_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification_data)
                else:
                    callback(notification_data)
                notified += 1
            except Exception as e:
                logger.error(f"Dashboard callback failed: {e}")

        return notified

    # =========================================================================
    # SPECIALIZED MODULE INTEGRATION
    # =========================================================================

    def _should_invoke_genomics(
        self,
        clinical_data: Dict[str, Any],
        biomarkers: List[str],
    ) -> bool:
        """Check if genomics module should be invoked."""
        if not GENOMICS_AVAILABLE:
            return False
        genetic_markers = {"CYP2D6", "CYP2C19", "BRCA1", "BRCA2", "HLA-B", "TPMT", "DPYD", "VKORC1"}
        return bool(
            clinical_data.get("genetic_data") or
            clinical_data.get("genes") or
            any(bm.upper() in genetic_markers for bm in biomarkers)
        )

    def _should_invoke_pharma(self, clinical_data: Dict[str, Any]) -> bool:
        """Check if pharma module should be invoked."""
        if not PHARMA_AVAILABLE:
            return False
        meds = clinical_data.get("medications", [])
        return len(meds) >= 2  # Need at least 2 drugs for interaction check

    def _should_invoke_pathogen(self, lab_data: Dict[str, Any]) -> bool:
        """Check if pathogen module should be invoked based on infection markers."""
        if not PATHOGEN_AVAILABLE:
            return False
        infection_thresholds = {
            "procalcitonin": 0.5,
            "pct": 0.5,
            "wbc": 12.0,
            "crp": 10.0,
            "lactate": 2.0,
        }
        for marker, threshold in infection_thresholds.items():
            value = lab_data.get(marker, 0)
            if isinstance(value, (int, float)) and value > threshold:
                return True
        return False

    def _should_invoke_multiomic(
        self,
        risk_domain: str,
        clinical_data: Dict[str, Any],
    ) -> bool:
        """Check if multiomic module should be invoked."""
        if not MULTIOMIC_AVAILABLE:
            return False
        genomic_domains = {"oncology", "hematologic", "metabolic", "oncology_inception"}
        return bool(
            risk_domain in genomic_domains or
            clinical_data.get("genetic_data") or
            clinical_data.get("genes") or
            len(clinical_data) > 2
        )

    async def _run_specialized_modules(
        self,
        patient_id: str,
        risk_domain: str,
        lab_data: Dict[str, Any],
        clinical_data: Dict[str, Any],
        contributing_biomarkers: List[str],
    ) -> Dict[str, Any]:
        """
        Run specialized modules (genomics, pharma, pathogen, multiomic) based on data.

        Returns results from all invoked modules.
        """
        start_time = datetime.now(timezone.utc)
        results = {}
        modules_invoked = []

        # Genomics analysis
        if self._should_invoke_genomics(clinical_data, contributing_biomarkers):
            modules_invoked.append("genomics")
            try:
                genes = clinical_data.get("genes", [])
                if not genes:
                    # Extract genes from genetic_data if present
                    genetic_data = clinical_data.get("genetic_data", {})
                    genes = list(genetic_data.keys()) if isinstance(genetic_data, dict) else []
                if genes:
                    genomics_result = analyze_genomics(
                        genes=genes[:10],  # Limit to 10 genes
                        include_variants=True,
                        include_expression=True,
                        max_expression_files=5,
                    )
                    results["genomics"] = {
                        "invoked": True,
                        "genes_analyzed": genes[:10],
                        "variant_impacts": genomics_result.get("variant_impacts", [])[:5],
                        "expression_patterns": genomics_result.get("expression_patterns", [])[:3],
                        "recommendations": self._extract_genomics_recommendations(genomics_result),
                    }
                else:
                    results["genomics"] = {"invoked": False, "reason": "no_genes_specified"}
            except Exception as e:
                logger.warning(f"Genomics analysis failed: {e}")
                results["genomics"] = {"invoked": False, "error": str(e)}

        # Pharma interactions check
        if self._should_invoke_pharma(clinical_data):
            modules_invoked.append("pharma")
            try:
                medications = clinical_data.get("medications", [])
                pharma_result = check_drug_interactions(medications)
                results["pharma"] = {
                    "invoked": True,
                    "drugs_checked": medications,
                    "interactions_found": pharma_result.get("interactions", [])[:5],
                    "interaction_count": pharma_result.get("total_interactions", 0),
                    "max_severity": pharma_result.get("max_severity", "none"),
                    "recommendations": pharma_result.get("recommendations", []),
                }
            except Exception as e:
                logger.warning(f"Pharma interaction check failed: {e}")
                results["pharma"] = {"invoked": False, "error": str(e)}

        # Pathogen detection
        if self._should_invoke_pathogen(lab_data):
            modules_invoked.append("pathogen")
            try:
                pathogen_result = detect_outbreaks(
                    threshold_multiplier=1.5,
                    lookback_years=3,
                )
                alerts = pathogen_result.get("alerts", [])
                results["pathogen"] = {
                    "invoked": True,
                    "alerts": alerts[:3],
                    "alert_count": len(alerts),
                    "outbreak_risk": "high" if len(alerts) > 2 else "moderate" if alerts else "low",
                    "infection_markers_elevated": True,
                    "recommendations": self._extract_pathogen_recommendations(pathogen_result, lab_data),
                }
            except Exception as e:
                logger.warning(f"Pathogen detection failed: {e}")
                results["pathogen"] = {"invoked": False, "error": str(e)}

        # Multi-omic fusion
        if self._should_invoke_multiomic(risk_domain, clinical_data):
            modules_invoked.append("multiomic")
            try:
                # Build query based on available data
                genes = clinical_data.get("genes", [])
                target_gene = genes[0] if genes else None

                multiomic_result = fusion_analysis(
                    target_gene=target_gene,
                    target_disease=risk_domain,
                    include_genomic=True,
                    include_clinical=True,
                    include_pharmacological=True,
                )
                results["multiomic"] = {
                    "invoked": True,
                    "target": target_gene or risk_domain,
                    "layers_analyzed": multiomic_result.get("layers_analyzed", []),
                    "integrated_score": multiomic_result.get("integrated_score", 0),
                    "biomarker_candidates": multiomic_result.get("biomarker_candidates", [])[:5],
                    "drug_candidates": multiomic_result.get("drug_candidates", [])[:5],
                    "clinical_implications": multiomic_result.get("clinical_implications", [])[:3],
                }
            except Exception as e:
                logger.warning(f"Multi-omic fusion failed: {e}")
                results["multiomic"] = {"invoked": False, "error": str(e)}

        # Calculate duration
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        results["modules_invoked"] = modules_invoked
        results["duration_ms"] = round(duration_ms, 2)

        return results

    def _extract_genomics_recommendations(self, genomics_result: Dict[str, Any]) -> List[str]:
        """Extract clinical recommendations from genomics analysis."""
        recommendations = []

        variants = genomics_result.get("variant_impacts", [])
        for variant in variants[:3]:
            if variant.get("clinical_significance") in ["Pathogenic", "Likely_pathogenic"]:
                gene = variant.get("gene", "Unknown")
                recommendations.append(f"Pathogenic variant in {gene} - consider genetic counseling")

        if not recommendations:
            recommendations.append("No actionable genomic variants detected")

        return recommendations

    def _extract_pathogen_recommendations(
        self,
        pathogen_result: Dict[str, Any],
        lab_data: Dict[str, Any],
    ) -> List[str]:
        """Extract clinical recommendations from pathogen analysis."""
        recommendations = []

        # Check infection markers
        if lab_data.get("procalcitonin", 0) > 2.0 or lab_data.get("pct", 0) > 2.0:
            recommendations.append("High procalcitonin suggests bacterial infection - consider broad-spectrum antibiotics")
        elif lab_data.get("procalcitonin", 0) > 0.5 or lab_data.get("pct", 0) > 0.5:
            recommendations.append("Elevated procalcitonin - monitor for bacterial infection")

        if lab_data.get("lactate", 0) > 2.0:
            recommendations.append("Elevated lactate - assess tissue perfusion and consider sepsis protocol")

        alerts = pathogen_result.get("alerts", [])
        for alert in alerts[:2]:
            pathogen_name = alert.get("pathogen", "Unknown pathogen")
            recommendations.append(f"Regional alert for {pathogen_name} - consider in differential")

        if not recommendations:
            recommendations.append("No specific pathogen alerts - continue standard infection workup")

        return recommendations

    # =========================================================================
    # CASCADE DETECTION - MULTI-OMIC EARLY WARNING
    # =========================================================================

    def _calculate_cascade_detection(
        self,
        specialized_results: Dict[str, Any],
        traditional_tth_hours: float,
        lab_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate extended detection window based on cascade signals.

        CASCADE LEVELS (disease progression order):
        - Level 1 (Genomic): Genetic predisposition - always present
        - Level 2 (Metabolic): Lactate, amino acid shifts - Day 1-2
        - Level 3 (Proteomic): Cytokine precursors - Day 2-3
        - Level 4 (Inflammatory): CRP, WBC, PCT elevated - Day 3-4 (current)
        - Level 5 (Clinical): Vitals deteriorating - Day 5+ (too late)

        Returns extended detection window when multi-omic signals are found.
        """
        levels_detected = []
        contributing_factors = []
        detection_extension_hours = 0.0
        earliest_level = "inflammatory"  # Default - where we typically detect
        confounders = []

        # Check genomics (Level 1 - earliest possible detection)
        genomics = specialized_results.get("genomics", {})
        if genomics.get("invoked"):
            variants = genomics.get("variant_impacts", [])
            genes = genomics.get("genes_analyzed", [])

            # Pathogenic variants = +72 hours (3 days earlier detection)
            pathogenic = [
                v for v in variants
                if "pathogenic" in str(v.get("significance", "")).lower()
            ]
            if pathogenic:
                levels_detected.append("genomic")
                detection_extension_hours += 72.0
                for v in pathogenic[:3]:
                    contributing_factors.append(
                        f"pathogenic_variant_{v.get('gene', 'unknown')}"
                    )
                earliest_level = "genomic"

            # Drug metabolism variants = +48 hours (2 days earlier)
            metabolism_genes = {"CYP2D6", "CYP2C19", "CYP2C9", "DPYD", "TPMT", "UGT1A1"}
            metabolism_found = [g for g in genes if g.upper() in metabolism_genes]
            if metabolism_found and not pathogenic:
                levels_detected.append("genomic")
                detection_extension_hours += 48.0
                for g in metabolism_found[:2]:
                    contributing_factors.append(f"metabolism_variant_{g}")
                earliest_level = "genomic"

        # Check multiomic (Level 2-3 - metabolic/proteomic patterns)
        multiomic = specialized_results.get("multiomic", {})
        if multiomic.get("invoked"):
            integrated_score = multiomic.get("integrated_score", 0)
            biomarker_candidates = multiomic.get("biomarker_candidates", [])

            # High integrated score = +48 hours (cross-layer correlation)
            if integrated_score > 0.5:
                if "metabolic" not in levels_detected:
                    levels_detected.append("metabolic")
                detection_extension_hours += 48.0
                contributing_factors.append(f"multiomic_score_{integrated_score:.2f}")
                if earliest_level == "inflammatory":
                    earliest_level = "metabolic"

            # Biomarker candidates identified = +24 hours
            if biomarker_candidates:
                detection_extension_hours += 24.0
                for bc in biomarker_candidates[:2]:
                    contributing_factors.append(f"biomarker_candidate_{bc}")

        # Check pathogen (Level 3-4 - regional surveillance adds lead time)
        pathogen = specialized_results.get("pathogen", {})
        if pathogen.get("invoked"):
            alert_count = pathogen.get("alert_count", 0)
            outbreak_risk = pathogen.get("outbreak_risk", "low")

            # Regional outbreak alerts = +36 hours (1.5 days)
            if alert_count > 0 or outbreak_risk in ["high", "moderate"]:
                if "proteomic" not in levels_detected:
                    levels_detected.append("proteomic")
                detection_extension_hours += 36.0
                contributing_factors.append(f"regional_pathogen_alert_{outbreak_risk}")
                if earliest_level == "inflammatory":
                    earliest_level = "proteomic"

            # Elevated infection markers (inflammatory level - standard detection)
            if pathogen.get("infection_markers_elevated"):
                if "inflammatory" not in levels_detected:
                    levels_detected.append("inflammatory")
                for marker in ["procalcitonin", "pct", "wbc", "crp"]:
                    val = lab_data.get(marker, 0)
                    if isinstance(val, (int, float)) and val > 0:
                        contributing_factors.append(f"elevated_{marker}")

        # Check pharma for confounders (drug interactions may mask symptoms)
        pharma = specialized_results.get("pharma", {})
        if pharma.get("invoked"):
            interactions = pharma.get("interactions_found", [])
            max_severity = pharma.get("max_severity", "none")

            if max_severity in ["major", "contraindicated"]:
                confounders.append("major_drug_interaction")
                contributing_factors.append(f"drug_interaction_{max_severity}")

            if interactions:
                confounders.append("potential_symptom_masking")

        # Ensure inflammatory is in levels if we have lab data with values
        if lab_data and "inflammatory" not in levels_detected:
            for marker in ["procalcitonin", "pct", "wbc", "crp", "lactate"]:
                val = lab_data.get(marker, 0)
                if isinstance(val, (int, float)) and val > 0:
                    levels_detected.append("inflammatory")
                    break

        # Calculate final integrated detection hours
        integrated_hours = traditional_tth_hours + detection_extension_hours
        improvement_hours = detection_extension_hours

        return {
            "primary_detection_level": "inflammatory",
            "earliest_signal_level": earliest_level,
            "levels_detected": levels_detected,
            "traditional_detection_hours": round(traditional_tth_hours, 1),
            "integrated_detection_hours": round(integrated_hours, 1),
            "detection_improvement_hours": round(improvement_hours, 1),
            "detection_improvement_days": round(improvement_hours / 24.0, 1),
            "contributing_factors": contributing_factors[:10],
            "confounders": confounders,
        }

    def _generate_cascade_recommendations(
        self,
        cascade_detection: Dict[str, Any],
    ) -> List[str]:
        """Generate clinical recommendations based on cascade detection results."""
        recommendations = []

        improvement_days = cascade_detection.get("detection_improvement_days", 0)
        improvement_hours = cascade_detection.get("detection_improvement_hours", 0)
        earliest = cascade_detection.get("earliest_signal_level", "inflammatory")

        if improvement_hours > 0:
            if earliest == "genomic":
                recommendations.append(
                    f"Early genomic signal detected - {improvement_days} days earlier "
                    f"detection than inflammatory markers alone"
                )
            elif earliest in ["metabolic", "proteomic"]:
                recommendations.append(
                    f"Multi-omic fusion identified risk pattern {improvement_hours:.0f} hours "
                    f"before clinical deterioration"
                )

        # Add confounder warnings
        confounders = cascade_detection.get("confounders", [])
        if "major_drug_interaction" in confounders:
            recommendations.append(
                "CAUTION: Major drug interaction may mask or alter symptom presentation"
            )

        return recommendations


# =============================================================================
# GLOBAL PIPELINE INSTANCE
# =============================================================================

_pipeline: Optional[AlertPipeline] = None


def get_pipeline() -> AlertPipeline:
    """Get the global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AlertPipeline()
    return _pipeline


def set_pipeline(pipeline: AlertPipeline) -> None:
    """Set a custom pipeline instance."""
    global _pipeline
    _pipeline = pipeline


async def process_patient_intake(
    patient_id: str,
    risk_domain: str,
    lab_data: Optional[Dict[str, Any]] = None,
    vitals_data: Optional[Dict[str, Any]] = None,
    clinical_data: Optional[Dict[str, Any]] = None,
    risk_score: Optional[float] = None,
    location: Optional[str] = None,
    encounter_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PipelineResult:
    """
    Unified patient intake processing.

    This is the main API for the /patient/intake endpoint.
    """
    pipeline = get_pipeline()
    return await pipeline.process_patient_data(
        patient_id=patient_id,
        risk_domain=risk_domain,
        lab_data=lab_data,
        vitals_data=vitals_data,
        clinical_data=clinical_data,
        risk_score=risk_score,
        location=location,
        encounter_id=encounter_id,
        metadata=metadata,
    )
