"""
Analyzer Agent - HyperCore Evolution
=====================================

Extracts lessons from evaluation results.
Adapted from GAIR-NLP ASI-Evolve Analyzer with healthcare additions.

The Analyzer:
1. Compares hypothesis results to parents
2. Identifies success factors and failure modes
3. Generates actionable lessons for cognition store
4. Updates node analysis with insights

Healthcare Additions:
- Clinical outcome analysis
- Safety pattern detection
- Evidence synthesis
- Regulatory compliance insights
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...schemas import (
    EvolutionNode,
    CognitionItem,
    CognitionItemType,
    SignalType,
)
from ...emitter import EvolutionEmitter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for the Analyzer agent."""
    # Analysis settings
    min_improvement_threshold: float = 0.05  # 5% improvement to note
    significant_improvement_threshold: float = 0.15  # 15% = significant

    # Lesson generation
    max_lessons_per_analysis: int = 5
    min_confidence_for_lesson: float = 0.6

    # LLM settings
    use_llm_synthesis: bool = False
    llm_model: str = "gpt-4"

    # Output settings
    include_recommendations: bool = True
    include_safety_analysis: bool = True


@dataclass
class Lesson:
    """A lesson extracted from evolution analysis."""
    lesson_id: str
    lesson_type: str  # "success", "failure", "safety", "optimization"
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    applicable_domains: List[str] = field(default_factory=list)

    def to_cognition_item(self) -> CognitionItem:
        """Convert to cognition item for storage."""
        return CognitionItem(
            content=self.content,
            item_type=CognitionItemType.LESSON,
            metadata={
                "lesson_id": self.lesson_id,
                "lesson_type": self.lesson_type,
                "confidence": self.confidence,
                "evidence": self.evidence,
                "domains": self.applicable_domains,
            },
        )


@dataclass
class AnalysisResult:
    """Result of analyzing an evolution step."""
    node_id: str
    analyzed_at: str

    # Comparison
    score_delta: float = 0.0  # vs best parent
    improved: bool = False
    significant_improvement: bool = False

    # Insights
    success_factors: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    safety_insights: List[str] = field(default_factory=list)

    # Lessons
    lessons: List[Lesson] = field(default_factory=list)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Summary
    analysis_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "analyzed_at": self.analyzed_at,
            "score_delta": self.score_delta,
            "improved": self.improved,
            "significant_improvement": self.significant_improvement,
            "success_factors": self.success_factors,
            "failure_modes": self.failure_modes,
            "safety_insights": self.safety_insights,
            "lessons": [
                {
                    "lesson_id": l.lesson_id,
                    "type": l.lesson_type,
                    "content": l.content,
                    "confidence": l.confidence,
                }
                for l in self.lessons
            ],
            "recommendations": self.recommendations,
            "analysis": self.analysis_text,
        }


class AnalyzerAgent:
    """
    Lesson extraction agent for evolution pipeline.

    Responsibilities:
    - Compare hypothesis results to baselines
    - Identify patterns in successes and failures
    - Generate actionable lessons
    - Synthesize clinical insights
    - Update cognition store

    Usage:
        analyzer = AnalyzerAgent()
        result = analyzer.analyze(
            node=evaluated_node,
            eval_result=evaluation_result,
            best_parent=parent_node,
            task_description="..."
        )
    """

    def __init__(
        self,
        config: Optional[AnalyzerConfig] = None,
        cognition_store: Optional[Any] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize Analyzer agent.

        Args:
            config: Agent configuration
            cognition_store: Store for persisting lessons
            llm_client: LLM for synthesis (optional)
        """
        self.config = config or AnalyzerConfig()
        self.cognition_store = cognition_store
        self.llm_client = llm_client

        # Evolution emitter
        self.emitter = EvolutionEmitter(
            agent_id="analyzer_agent",
            agent_type="analyzer",
            version="1.0.0",
        )

        # Stats
        self._analyses_run = 0
        self._lessons_generated = 0
        self._improvements_found = 0

        logger.info("AnalyzerAgent initialized")

    def analyze(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
        best_parent: Optional[EvolutionNode] = None,
        task_description: str = "",
    ) -> Dict[str, Any]:
        """
        Analyze evaluation results and extract lessons.

        Args:
            node: The evaluated node
            eval_result: Evaluation results
            best_parent: Best performing parent for comparison
            task_description: Context for the analysis

        Returns:
            Dictionary with analysis and lessons
        """
        start_time = time.perf_counter()

        # Emit start signal
        request_id = self.emitter.emit(
            signal_type=SignalType.LESSON_EXTRACTED,
            payload={
                "node_id": node.node_id,
                "has_parent": best_parent is not None,
            },
        ).request_id

        self._analyses_run += 1

        result = AnalysisResult(
            node_id=node.node_id,
            analyzed_at=datetime.now(timezone.utc).isoformat(),
        )

        try:
            # Step 1: Compare to parent
            if best_parent:
                comparison = self._compare_to_parent(node, eval_result, best_parent)
                result.score_delta = comparison["delta"]
                result.improved = comparison["improved"]
                result.significant_improvement = comparison["significant"]

                if result.improved:
                    self._improvements_found += 1

            # Step 2: Identify success factors
            if eval_result.get("score", 0) >= 0.7:
                result.success_factors = self._identify_success_factors(
                    node, eval_result
                )

            # Step 3: Identify failure modes
            if eval_result.get("score", 0) < 0.5 or not eval_result.get("safety_passed"):
                result.failure_modes = self._identify_failure_modes(
                    node, eval_result
                )

            # Step 4: Safety analysis
            if self.config.include_safety_analysis:
                result.safety_insights = self._analyze_safety(node, eval_result)

            # Step 5: Generate lessons
            lessons = self._generate_lessons(
                node=node,
                eval_result=eval_result,
                comparison_result=result,
                task_description=task_description,
            )
            result.lessons = lessons
            self._lessons_generated += len(lessons)

            # Store lessons in cognition store
            if self.cognition_store and lessons:
                for lesson in lessons:
                    self.cognition_store.add(lesson.to_cognition_item())

            # Step 6: Generate recommendations
            if self.config.include_recommendations:
                result.recommendations = self._generate_recommendations(
                    node, eval_result, result
                )

            # Step 7: Generate summary
            result.analysis_text = self._generate_summary(node, eval_result, result)

        except Exception as e:
            logger.error(f"Analysis failed for {node.node_id}: {e}")
            result.analysis_text = f"Analysis failed: {str(e)}"

        # Record timing
        latency = (time.perf_counter() - start_time) * 1000

        # Emit completion
        self.emitter.record_outcome(
            request_id=request_id,
            outcome={
                "success": True,
                "lessons_generated": len(result.lessons),
                "latency_ms": latency,
            },
        )

        logger.info(
            f"Analyzed {node.name}: delta={result.score_delta:+.4f}, "
            f"lessons={len(result.lessons)}"
        )

        return {
            "analysis": result.analysis_text,
            "lessons": [l.content for l in result.lessons],
            "full_result": result.to_dict(),
        }

    def _compare_to_parent(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
        parent: EvolutionNode,
    ) -> Dict[str, Any]:
        """Compare node performance to parent."""
        node_score = eval_result.get("score", node.score)
        parent_score = parent.score

        delta = node_score - parent_score
        improved = delta > self.config.min_improvement_threshold
        significant = delta > self.config.significant_improvement_threshold

        return {
            "node_score": node_score,
            "parent_score": parent_score,
            "delta": delta,
            "improved": improved,
            "significant": significant,
            "comparison_details": {
                "safety_improvement": (
                    eval_result.get("safety_passed", False) and
                    not (parent.evaluation_result and parent.evaluation_result.safety_passed)
                ),
                "accuracy_delta": (
                    eval_result.get("accuracy_score", 0) -
                    (parent.evaluation_result.error_rate if parent.evaluation_result else 0.5)
                ),
            }
        }

    def _identify_success_factors(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
    ) -> List[str]:
        """Identify what made this hypothesis successful."""
        factors = []

        # Check safety success
        if eval_result.get("safety_passed"):
            factors.append("Passed all safety checks")

        # Check clinical validity
        if eval_result.get("clinical_validity_passed"):
            factors.append("Achieved clinical validity threshold")

        # Check score breakdown
        breakdown = eval_result.get("score_breakdown", {})
        for component, score in breakdown.items():
            if score > 0.8 * (eval_result.get("score", 0) / len(breakdown)):
                factors.append(f"Strong {component} performance ({score:.2f})")

        # Check test results
        test_results = eval_result.get("test_results", [])
        passed = sum(1 for t in test_results if t.get("passed"))
        if passed == len(test_results) and test_results:
            factors.append(f"Passed all {len(test_results)} test cases")

        # Analyze motivation patterns
        if node.motivation:
            if "evidence" in node.motivation.lower():
                factors.append("Evidence-based approach in motivation")
            if "safety" in node.motivation.lower():
                factors.append("Safety-conscious design")

        return factors[:self.config.max_lessons_per_analysis]

    def _identify_failure_modes(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
    ) -> List[str]:
        """Identify what caused failure or poor performance."""
        modes = []

        # Safety failures
        if not eval_result.get("safety_passed"):
            warnings = eval_result.get("safety_warnings", [])
            for warning in warnings[:3]:
                modes.append(f"Safety issue: {warning}")

        # Clinical validity failures
        if not eval_result.get("clinical_validity_passed"):
            modes.append(
                f"Clinical validity below threshold "
                f"({eval_result.get('clinical_validity_score', 0):.2f})"
            )

        # Test failures
        test_results = eval_result.get("test_results", [])
        failed = [t for t in test_results if not t.get("passed")]
        if failed:
            adversarial_fails = [t for t in failed if "adv" in t.get("test_id", "")]
            if adversarial_fails:
                modes.append(f"Failed {len(adversarial_fails)} adversarial test(s)")
            functional_fails = len(failed) - len(adversarial_fails)
            if functional_fails:
                modes.append(f"Failed {functional_fails} functional test(s)")

        # Performance issues
        if eval_result.get("avg_latency_ms", 0) > 1000:
            modes.append(
                f"High latency: {eval_result.get('avg_latency_ms', 0):.0f}ms average"
            )

        # Errors
        errors = eval_result.get("errors", [])
        for error in errors[:2]:
            modes.append(f"Error: {error[:100]}")

        return modes[:self.config.max_lessons_per_analysis]

    def _analyze_safety(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
    ) -> List[str]:
        """Extract safety-specific insights."""
        insights = []

        safety_score = eval_result.get("safety_score", 0)

        if safety_score >= 0.9:
            insights.append("Excellent safety profile - suitable for clinical review")
        elif safety_score >= 0.7:
            insights.append("Acceptable safety with minor concerns noted")
        elif safety_score >= 0.5:
            insights.append("Moderate safety issues require attention before promotion")
        else:
            insights.append("Critical safety concerns - not suitable for promotion")

        # Check specific safety aspects
        warnings = eval_result.get("safety_warnings", [])
        if "no_phi_exposure" in str(warnings):
            insights.append("PHI exposure risk detected - requires data handling review")
        if "uncertainty_expressed" in str(warnings):
            insights.append("Uncertainty expression missing - add confidence indicators")

        return insights

    def _generate_lessons(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
        comparison_result: AnalysisResult,
        task_description: str,
    ) -> List[Lesson]:
        """Generate actionable lessons from the analysis."""
        lessons = []
        lesson_num = 0

        # Lesson from improvement
        if comparison_result.significant_improvement:
            lesson_num += 1
            lesson = Lesson(
                lesson_id=f"{node.node_id}_lesson_{lesson_num}",
                lesson_type="success",
                content=(
                    f"Significant improvement ({comparison_result.score_delta:+.2%}) achieved by: "
                    f"{'; '.join(comparison_result.success_factors[:2])}"
                ),
                confidence=min(0.9, 0.5 + comparison_result.score_delta),
                evidence=comparison_result.success_factors,
            )
            lessons.append(lesson)

        # Lessons from failure modes
        for mode in comparison_result.failure_modes[:2]:
            lesson_num += 1
            lesson = Lesson(
                lesson_id=f"{node.node_id}_lesson_{lesson_num}",
                lesson_type="failure",
                content=f"Avoid pattern: {mode}",
                confidence=0.7,
                evidence=[mode],
            )
            lessons.append(lesson)

        # Safety lessons
        if not eval_result.get("safety_passed"):
            lesson_num += 1
            lesson = Lesson(
                lesson_id=f"{node.node_id}_lesson_{lesson_num}",
                lesson_type="safety",
                content=(
                    "Safety gate failure indicates need for: "
                    f"{'; '.join(eval_result.get('safety_warnings', ['review'])[:2])}"
                ),
                confidence=0.85,
                evidence=eval_result.get("safety_warnings", []),
            )
            lessons.append(lesson)

        # Performance optimization lessons
        if eval_result.get("avg_latency_ms", 0) > 500:
            lesson_num += 1
            lesson = Lesson(
                lesson_id=f"{node.node_id}_lesson_{lesson_num}",
                lesson_type="optimization",
                content=(
                    f"Performance bottleneck: {eval_result.get('avg_latency_ms', 0):.0f}ms "
                    "average - consider optimization"
                ),
                confidence=0.6,
                evidence=[f"avg_latency: {eval_result.get('avg_latency_ms', 0)}ms"],
            )
            lessons.append(lesson)

        # Filter by confidence
        lessons = [
            l for l in lessons
            if l.confidence >= self.config.min_confidence_for_lesson
        ]

        return lessons[:self.config.max_lessons_per_analysis]

    def _generate_recommendations(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
        analysis: AnalysisResult,
    ) -> List[str]:
        """Generate recommendations for next steps."""
        recs = []

        score = eval_result.get("score", 0)

        # Based on score
        if score >= 0.85:
            recs.append("Consider queuing for promotion review")
        elif score >= 0.7:
            recs.append("Good candidate for further iteration")
        elif score >= 0.5:
            recs.append("Moderate potential - address failure modes before iteration")
        else:
            recs.append("Consider different approach - current direction shows limited promise")

        # Based on safety
        if not eval_result.get("safety_passed"):
            recs.append("REQUIRED: Address safety issues before any promotion consideration")

        # Based on improvement
        if analysis.significant_improvement:
            recs.append("Build on successful patterns in next iteration")

        # Based on failure modes
        if analysis.failure_modes:
            recs.append(f"Prioritize fixing: {analysis.failure_modes[0]}")

        return recs

    def _generate_summary(
        self,
        node: EvolutionNode,
        eval_result: Dict[str, Any],
        analysis: AnalysisResult,
    ) -> str:
        """Generate human-readable analysis summary."""
        lines = [
            f"## Analysis: {node.name}",
            "",
            f"**Overall Score:** {eval_result.get('score', 0):.4f}",
            f"**Safety Status:** {'PASS' if eval_result.get('safety_passed') else 'FAIL'}",
        ]

        if analysis.score_delta != 0:
            direction = "improved" if analysis.improved else "declined"
            lines.append(
                f"**vs Parent:** {direction} by {abs(analysis.score_delta):.2%}"
            )

        if analysis.success_factors:
            lines.extend([
                "",
                "### Success Factors",
                *[f"- {f}" for f in analysis.success_factors],
            ])

        if analysis.failure_modes:
            lines.extend([
                "",
                "### Failure Modes",
                *[f"- {m}" for m in analysis.failure_modes],
            ])

        if analysis.lessons:
            lines.extend([
                "",
                "### Key Lessons",
                *[f"- {l.content}" for l in analysis.lessons],
            ])

        if analysis.recommendations:
            lines.extend([
                "",
                "### Recommendations",
                *[f"- {r}" for r in analysis.recommendations],
            ])

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "analyses_run": self._analyses_run,
            "lessons_generated": self._lessons_generated,
            "improvements_found": self._improvements_found,
            "avg_lessons_per_analysis": (
                self._lessons_generated / self._analyses_run
                if self._analyses_run > 0 else 0
            ),
            "improvement_rate": (
                self._improvements_found / self._analyses_run
                if self._analyses_run > 0 else 0
            ),
        }
