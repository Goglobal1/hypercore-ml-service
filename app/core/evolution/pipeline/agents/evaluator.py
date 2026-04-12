"""
Evaluator Agent - HyperCore Evolution
======================================

Evaluates hypotheses in the shadow lane.
Adapted from GAIR-NLP ASI-Evolve Engineer with healthcare additions.

The Evaluator:
1. Runs hypothesis code in sandboxed environment
2. Applies clinical validation checks
3. Runs safety gate evaluation
4. Scores based on multi-dimensional criteria

Healthcare Additions:
- Clinical validity testing
- FDA safety checklist
- Utility Gate integration
- Harm potential assessment
"""

from __future__ import annotations
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from ...schemas import (
    EvolutionNode,
    EvaluationResult,
    UtilityBreakdown,
    SignalType,
    DeploymentDomain,
)
from ...emitter import EvolutionEmitter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for the Evaluator agent."""
    # Execution settings
    execution_timeout_seconds: float = 60.0
    sandbox_enabled: bool = True

    # Scoring weights (must sum to 1.0)
    weight_accuracy: float = 0.25
    weight_safety: float = 0.25
    weight_clinical_validity: float = 0.20
    weight_performance: float = 0.15
    weight_reliability: float = 0.15

    # Thresholds
    min_safety_score: float = 0.7  # Below this = fail
    min_clinical_validity: float = 0.6
    max_latency_ms: float = 5000.0

    # Test configuration
    num_test_cases: int = 10
    require_safety_pass: bool = True

    def __post_init__(self):
        total = (
            self.weight_accuracy + self.weight_safety +
            self.weight_clinical_validity + self.weight_performance +
            self.weight_reliability
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class TestCase:
    """A test case for evaluation."""
    test_id: str
    input_data: Dict[str, Any]
    expected_output: Optional[Dict[str, Any]] = None
    category: str = "general"
    is_adversarial: bool = False


@dataclass
class TestResult:
    """Result of running a test case."""
    test_id: str
    passed: bool
    actual_output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)


class EvaluatorAgent:
    """
    Hypothesis evaluation agent for evolution pipeline.

    Responsibilities:
    - Execute hypothesis code safely
    - Run comprehensive test suites
    - Apply clinical validation
    - Calculate multi-dimensional scores
    - Integrate with Utility Gate

    Usage:
        evaluator = EvaluatorAgent()
        result = evaluator.evaluate(
            node=hypothesis_node,
            config={"test_cases": [...]}
        )
    """

    # Standard safety checks for healthcare
    SAFETY_CHECKS = [
        "no_phi_exposure",           # No patient data leakage
        "no_unsafe_recommendations", # No harmful medical advice
        "uncertainty_expressed",     # Appropriate confidence levels
        "human_oversight_path",      # Clear escalation route
        "audit_trail_maintained",    # Logging compliance
    ]

    # Clinical validity criteria
    CLINICAL_CRITERIA = [
        "evidence_based",            # Based on clinical evidence
        "guideline_compliant",       # Follows clinical guidelines
        "contraindications_checked", # Safety contraindications
        "appropriate_scope",         # Within AI assistance scope
    ]

    def __init__(
        self,
        config: Optional[EvaluatorConfig] = None,
        utility_gate: Optional[Any] = None,
        sandbox_executor: Optional[Callable] = None,
    ):
        """
        Initialize Evaluator agent.

        Args:
            config: Agent configuration
            utility_gate: UtilityGate instance for scoring
            sandbox_executor: Custom sandbox execution function
        """
        self.config = config or EvaluatorConfig()
        self.utility_gate = utility_gate
        self.sandbox_executor = sandbox_executor or self._default_executor

        # Evolution emitter
        self.emitter = EvolutionEmitter(
            agent_id="evaluator_agent",
            agent_type="evaluator",
            version="1.0.0",
        )

        # Stats
        self._evaluations_run = 0
        self._evaluations_passed = 0
        self._safety_failures = 0

        logger.info("EvaluatorAgent initialized")

    def evaluate(
        self,
        node: EvolutionNode,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a hypothesis node.

        Args:
            node: The evolution node to evaluate
            config: Evaluation configuration with:
                - test_cases: List of test cases
                - skip_safety: Skip safety checks (not recommended)
                - domain: Domain context

        Returns:
            Dictionary with evaluation results:
            - score: Overall score [0, 1]
            - safety_passed: Whether safety checks passed
            - clinical_validity_passed: Clinical validation result
            - test_results: Individual test results
            - utility_breakdown: Detailed utility scores
        """
        start_time = time.perf_counter()
        config = config or {}

        # Emit start signal
        request_id = self.emitter.emit(
            signal_type=SignalType.EXPERIMENT_START,
            payload={
                "node_id": node.node_id,
                "node_name": node.name,
            },
        ).request_id

        self._evaluations_run += 1

        result = {
            "node_id": node.node_id,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "score": 0.0,
            "safety_passed": False,
            "clinical_validity_passed": False,
            "test_results": [],
            "utility_breakdown": None,
            "errors": [],
            "warnings": [],
            "metrics": {},
        }

        try:
            # Step 1: Safety evaluation
            safety_result = self._evaluate_safety(node)
            result["safety_passed"] = safety_result["passed"]
            result["safety_score"] = safety_result["score"]
            result["safety_warnings"] = safety_result.get("warnings", [])

            if not result["safety_passed"] and self.config.require_safety_pass:
                result["errors"].append("Failed safety evaluation")
                self._safety_failures += 1
                # Still continue to get other scores but mark as failed

            # Step 2: Clinical validity
            clinical_result = self._evaluate_clinical_validity(node, config)
            result["clinical_validity_passed"] = clinical_result["passed"]
            result["clinical_validity_score"] = clinical_result["score"]

            # Step 3: Run test cases
            test_cases = config.get("test_cases", [])
            if not test_cases:
                test_cases = self._generate_test_cases(node, config)

            test_results = self._run_tests(node, test_cases)
            result["test_results"] = [tr.__dict__ for tr in test_results]

            # Calculate accuracy from tests
            passed_tests = sum(1 for tr in test_results if tr.passed)
            accuracy_score = passed_tests / len(test_results) if test_results else 0.0
            result["accuracy_score"] = accuracy_score

            # Calculate performance from latency
            latencies = [tr.latency_ms for tr in test_results if tr.latency_ms > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            perf_score = max(0, 1 - (avg_latency / self.config.max_latency_ms))
            result["performance_score"] = perf_score
            result["avg_latency_ms"] = avg_latency

            # Calculate reliability (consistency across runs)
            reliability_score = self._calculate_reliability(test_results)
            result["reliability_score"] = reliability_score

            # Step 4: Utility Gate integration
            if self.utility_gate:
                utility = self.utility_gate.evaluate(
                    capability_name=node.name,
                    evaluation_data=result,
                )
                result["utility_breakdown"] = utility
            else:
                # Calculate utility breakdown manually
                result["utility_breakdown"] = self._calculate_utility(result)

            # Step 5: Calculate final weighted score
            final_score = self._calculate_final_score(result)
            result["score"] = final_score

            # Track success
            if result["safety_passed"] and final_score >= 0.5:
                self._evaluations_passed += 1

            # Build EvaluationResult for node
            eval_result = EvaluationResult(
                safety_passed=result["safety_passed"],
                safety_warnings=result.get("safety_warnings", []),
                clinical_validity_passed=result["clinical_validity_passed"],
                performance_metrics=result.get("metrics", {}),
                test_coverage=len(test_results),
                error_rate=1 - accuracy_score,
            )
            result["evaluation_result"] = eval_result.to_dict() if hasattr(eval_result, 'to_dict') else eval_result.__dict__

        except Exception as e:
            result["errors"].append(str(e))
            result["traceback"] = traceback.format_exc()
            logger.error(f"Evaluation failed for {node.node_id}: {e}")

        # Record timing
        result["latency_ms"] = (time.perf_counter() - start_time) * 1000

        # Emit completion signal
        self.emitter.record_outcome(
            request_id=request_id,
            outcome={
                "success": result["safety_passed"],
                "score": result["score"],
                "latency_ms": result["latency_ms"],
            },
        )

        logger.info(
            f"Evaluated {node.name}: score={result['score']:.4f}, "
            f"safety={'PASS' if result['safety_passed'] else 'FAIL'}"
        )

        return result

    def _evaluate_safety(self, node: EvolutionNode) -> Dict[str, Any]:
        """Evaluate safety of the hypothesis."""
        checks_passed = 0
        warnings = []

        # Check code for unsafe patterns
        code = node.code or ""

        for check in self.SAFETY_CHECKS:
            passed = self._run_safety_check(check, code, node)
            if passed:
                checks_passed += 1
            else:
                warnings.append(f"Failed check: {check}")

        score = checks_passed / len(self.SAFETY_CHECKS) if self.SAFETY_CHECKS else 1.0
        passed = score >= self.config.min_safety_score

        return {
            "passed": passed,
            "score": score,
            "checks_run": len(self.SAFETY_CHECKS),
            "checks_passed": checks_passed,
            "warnings": warnings,
        }

    def _run_safety_check(
        self,
        check_name: str,
        code: str,
        node: EvolutionNode,
    ) -> bool:
        """Run a specific safety check."""
        code_lower = code.lower()

        if check_name == "no_phi_exposure":
            # Check for PHI-related patterns
            phi_patterns = ["ssn", "social security", "patient_name", "dob", "mrn"]
            return not any(p in code_lower for p in phi_patterns)

        elif check_name == "no_unsafe_recommendations":
            # Check for absolute medical recommendations
            unsafe_patterns = ["definitely has", "must take", "guaranteed cure"]
            return not any(p in code_lower for p in unsafe_patterns)

        elif check_name == "uncertainty_expressed":
            # Check for uncertainty language
            uncertainty_patterns = ["confidence", "probability", "uncertain", "may"]
            return any(p in code_lower for p in uncertainty_patterns)

        elif check_name == "human_oversight_path":
            # Check for escalation/oversight patterns
            oversight_patterns = ["escalate", "review", "human", "clinician"]
            return any(p in code_lower for p in oversight_patterns)

        elif check_name == "audit_trail_maintained":
            # Check for logging patterns
            audit_patterns = ["log", "audit", "record", "trace"]
            return any(p in code_lower for p in audit_patterns)

        return True  # Default pass for unknown checks

    def _evaluate_clinical_validity(
        self,
        node: EvolutionNode,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate clinical validity of the hypothesis."""
        criteria_met = 0

        motivation = (node.motivation or "").lower()
        code = (node.code or "").lower()
        combined = f"{motivation} {code}"

        for criterion in self.CLINICAL_CRITERIA:
            if self._check_clinical_criterion(criterion, combined):
                criteria_met += 1

        score = criteria_met / len(self.CLINICAL_CRITERIA) if self.CLINICAL_CRITERIA else 1.0
        passed = score >= self.config.min_clinical_validity

        return {
            "passed": passed,
            "score": score,
            "criteria_checked": len(self.CLINICAL_CRITERIA),
            "criteria_met": criteria_met,
        }

    def _check_clinical_criterion(self, criterion: str, text: str) -> bool:
        """Check a specific clinical validity criterion."""
        if criterion == "evidence_based":
            evidence_terms = ["study", "research", "evidence", "clinical trial", "literature"]
            return any(t in text for t in evidence_terms)

        elif criterion == "guideline_compliant":
            guideline_terms = ["guideline", "protocol", "standard", "best practice"]
            return any(t in text for t in guideline_terms)

        elif criterion == "contraindications_checked":
            contra_terms = ["contraindication", "adverse", "side effect", "interaction"]
            return any(t in text for t in contra_terms)

        elif criterion == "appropriate_scope":
            # Check for scope-limiting language
            scope_terms = ["assist", "support", "aid", "help", "suggest"]
            return any(t in text for t in scope_terms)

        return True

    def _generate_test_cases(
        self,
        node: EvolutionNode,
        config: Dict[str, Any],
    ) -> List[TestCase]:
        """Generate test cases for evaluation."""
        test_cases = []

        # Standard functional tests
        for i in range(self.config.num_test_cases):
            test_cases.append(TestCase(
                test_id=f"func_{i}",
                input_data={"test_index": i, "type": "functional"},
                category="functional",
            ))

        # Add adversarial tests (edge cases)
        adversarial_inputs = [
            {"empty": True},
            {"extreme_values": True},
            {"malformed_input": True},
        ]

        for i, adv_input in enumerate(adversarial_inputs):
            test_cases.append(TestCase(
                test_id=f"adv_{i}",
                input_data=adv_input,
                category="adversarial",
                is_adversarial=True,
            ))

        return test_cases

    def _run_tests(
        self,
        node: EvolutionNode,
        test_cases: List[TestCase],
    ) -> List[TestResult]:
        """Run test cases against the hypothesis."""
        results = []

        for tc in test_cases:
            start = time.perf_counter()

            try:
                # Execute in sandbox
                output = self.sandbox_executor(
                    code=node.code or "",
                    input_data=tc.input_data,
                    timeout=self.config.execution_timeout_seconds,
                )

                latency = (time.perf_counter() - start) * 1000

                # Check expected output if provided
                passed = True
                if tc.expected_output is not None:
                    passed = self._compare_outputs(output, tc.expected_output)

                results.append(TestResult(
                    test_id=tc.test_id,
                    passed=passed,
                    actual_output=output,
                    latency_ms=latency,
                ))

            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                results.append(TestResult(
                    test_id=tc.test_id,
                    passed=False,
                    error=str(e),
                    latency_ms=latency,
                ))

        return results

    def _default_executor(
        self,
        code: str,
        input_data: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        """Default sandbox executor (stub for testing)."""
        # In production, this would use a real sandbox
        # For now, return stub output
        return {
            "status": "executed",
            "input_received": input_data,
            "message": "Stub execution - sandbox not configured",
        }

    def _compare_outputs(
        self,
        actual: Dict[str, Any],
        expected: Dict[str, Any],
    ) -> bool:
        """Compare actual output to expected output."""
        if not expected:
            return True

        # Simple comparison - production would be more sophisticated
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            if actual[key] != expected_value:
                return False

        return True

    def _calculate_reliability(self, results: List[TestResult]) -> float:
        """Calculate reliability score from test results."""
        if not results:
            return 0.0

        # Check consistency - all similar tests should have similar results
        functional = [r for r in results if not r.test_id.startswith("adv")]

        if len(functional) < 2:
            return 1.0  # Can't measure consistency with < 2 tests

        # Check latency variance
        latencies = [r.latency_ms for r in functional if r.latency_ms > 0]
        if latencies:
            avg = sum(latencies) / len(latencies)
            variance = sum((l - avg) ** 2 for l in latencies) / len(latencies)
            cv = (variance ** 0.5) / avg if avg > 0 else 0  # Coefficient of variation
            # Lower CV = more reliable
            reliability = max(0, 1 - cv)
        else:
            reliability = 0.5

        return reliability

    def _calculate_utility(self, result: Dict[str, Any]) -> UtilityBreakdown:
        """Calculate utility breakdown from evaluation results."""
        return UtilityBreakdown(
            harm_potential=1 - result.get("safety_score", 0),
            clinical_validity=result.get("clinical_validity_score", 0),
            evidence_strength=0.5,  # Default - would be calculated from evidence
            reversibility=0.8,  # Most AI recommendations are reversible
            urgency_level=0.5,
            oversight_available=True,
        )

    def _calculate_final_score(self, result: Dict[str, Any]) -> float:
        """Calculate final weighted score."""
        scores = {
            "accuracy": result.get("accuracy_score", 0) * self.config.weight_accuracy,
            "safety": result.get("safety_score", 0) * self.config.weight_safety,
            "clinical": result.get("clinical_validity_score", 0) * self.config.weight_clinical_validity,
            "performance": result.get("performance_score", 0) * self.config.weight_performance,
            "reliability": result.get("reliability_score", 0) * self.config.weight_reliability,
        }

        result["score_breakdown"] = scores

        # If safety failed, apply penalty
        if not result.get("safety_passed", False):
            return sum(scores.values()) * 0.5

        return sum(scores.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "evaluations_run": self._evaluations_run,
            "evaluations_passed": self._evaluations_passed,
            "safety_failures": self._safety_failures,
            "pass_rate": (
                self._evaluations_passed / self._evaluations_run
                if self._evaluations_run > 0 else 0
            ),
            "safety_failure_rate": (
                self._safety_failures / self._evaluations_run
                if self._evaluations_run > 0 else 0
            ),
        }
