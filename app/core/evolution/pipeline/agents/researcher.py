"""
Researcher Agent - HyperCore Evolution
=======================================

Generates hypotheses for capability improvements.
Adapted from GAIR-NLP ASI-Evolve with healthcare additions.

The Researcher:
1. Analyzes prior knowledge (cognition store)
2. Samples parent nodes for inspiration
3. Generates novel hypotheses combining insights
4. Ensures clinical safety in hypothesis formulation

Healthcare Additions:
- Clinical context awareness
- Evidence-based reasoning
- FDA guideline compliance
- Risk stratification in hypotheses
"""

from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone

from ...schemas import (
    EvolutionNode,
    CognitionItem,
    DeploymentDomain,
    SignalType,
)
from ...emitter import EvolutionEmitter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTemplate:
    """Template for hypothesis generation."""
    name: str
    template: str
    required_fields: List[str] = field(default_factory=list)
    clinical_context: str = ""


@dataclass
class ResearcherConfig:
    """Configuration for the Researcher agent."""
    # LLM settings
    model_name: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2048

    # Research settings
    max_parents_to_consider: int = 5
    max_cognition_items: int = 10
    require_clinical_reasoning: bool = True

    # Output validation
    min_motivation_length: int = 50
    max_code_length: int = 10000

    # Domain constraints
    allowed_domains: List[DeploymentDomain] = field(
        default_factory=lambda: list(DeploymentDomain)
    )


class ResearcherAgent:
    """
    Hypothesis generation agent for evolution pipeline.

    Responsibilities:
    - Synthesize insights from parent nodes
    - Integrate prior knowledge from cognition store
    - Generate novel hypotheses with clinical awareness
    - Ensure regulatory compliance in formulations

    Usage:
        researcher = ResearcherAgent()
        hypothesis = researcher.generate_hypothesis(
            task_description="Improve early cancer detection",
            parent_nodes=[...],
            cognition_items=[...],
        )
    """

    # Standard templates for healthcare hypotheses
    TEMPLATES = {
        "diagnostic_improvement": HypothesisTemplate(
            name="diagnostic_improvement",
            template="""
Based on analysis of parent capabilities and prior research:

## Hypothesis: {name}

### Clinical Motivation
{motivation}

### Proposed Improvement
{description}

### Expected Impact
- Sensitivity improvement: {sensitivity_delta}
- Specificity impact: {specificity_delta}
- Clinical utility: {utility_rationale}

### Implementation
```python
{code}
```

### Safety Considerations
{safety_notes}

### Evidence Basis
{evidence}
""",
            required_fields=["name", "motivation", "description", "code"],
            clinical_context="diagnostic",
        ),

        "therapeutic_recommendation": HypothesisTemplate(
            name="therapeutic_recommendation",
            template="""
Based on clinical evidence synthesis:

## Hypothesis: {name}

### Clinical Problem
{problem_statement}

### Proposed Solution
{description}

### Mechanism
{mechanism}

### Implementation
```python
{code}
```

### Risk Stratification
{risk_notes}
""",
            required_fields=["name", "problem_statement", "description", "code"],
            clinical_context="therapeutic",
        ),

        "workflow_optimization": HypothesisTemplate(
            name="workflow_optimization",
            template="""
## Hypothesis: {name}

### Current Limitation
{limitation}

### Proposed Optimization
{description}

### Implementation
```python
{code}
```

### Performance Impact
{performance_notes}
""",
            required_fields=["name", "limitation", "description", "code"],
            clinical_context="operational",
        ),
    }

    def __init__(
        self,
        config: Optional[ResearcherConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize Researcher agent.

        Args:
            config: Agent configuration
            llm_client: LLM client for generation (optional, uses mock if None)
        """
        self.config = config or ResearcherConfig()
        self.llm_client = llm_client

        # Evolution emitter
        self.emitter = EvolutionEmitter(
            agent_id="researcher_agent",
            agent_type="researcher",
            version="1.0.0",
        )

        # Stats
        self._hypotheses_generated = 0
        self._failed_generations = 0

        logger.info("ResearcherAgent initialized")

    def generate_hypothesis(
        self,
        task_description: str,
        parent_nodes: Optional[List[EvolutionNode]] = None,
        cognition_items: Optional[List[CognitionItem]] = None,
        template_name: Optional[str] = None,
        domain: Optional[DeploymentDomain] = None,
    ) -> Dict[str, Any]:
        """
        Generate a hypothesis for capability improvement.

        Args:
            task_description: What we're trying to improve
            parent_nodes: Parent nodes to draw inspiration from
            cognition_items: Prior knowledge items
            template_name: Specific template to use (auto-selects if None)
            domain: Domain context

        Returns:
            Dictionary with hypothesis details:
            - name: Hypothesis name
            - motivation: Why this hypothesis
            - code: Implementation code
            - evidence: Supporting evidence
            - safety_notes: Safety considerations
        """
        start_time = time.perf_counter()

        parent_nodes = parent_nodes or []
        cognition_items = cognition_items or []

        # Emit start signal
        request_id = self.emitter.emit(
            signal_type=SignalType.HYPOTHESIS_GENERATED,
            payload={
                "task": task_description[:100],
                "num_parents": len(parent_nodes),
                "num_cognition": len(cognition_items),
            },
        ).request_id

        try:
            # Step 1: Synthesize insights from parents
            parent_insights = self._extract_parent_insights(parent_nodes)

            # Step 2: Retrieve relevant prior knowledge
            relevant_knowledge = self._select_relevant_knowledge(
                task_description, cognition_items
            )

            # Step 3: Select template
            template = self._select_template(
                task_description, template_name
            )

            # Step 4: Generate hypothesis via LLM or fallback
            if self.llm_client:
                hypothesis = self._generate_with_llm(
                    task_description=task_description,
                    parent_insights=parent_insights,
                    relevant_knowledge=relevant_knowledge,
                    template=template,
                )
            else:
                hypothesis = self._generate_fallback(
                    task_description=task_description,
                    parent_insights=parent_insights,
                    relevant_knowledge=relevant_knowledge,
                    template=template,
                )

            # Step 5: Validate hypothesis
            hypothesis = self._validate_hypothesis(hypothesis)

            # Step 6: Add clinical reasoning if required
            if self.config.require_clinical_reasoning:
                hypothesis = self._add_clinical_reasoning(hypothesis, domain)

            self._hypotheses_generated += 1

            # Record outcome
            latency = (time.perf_counter() - start_time) * 1000
            self.emitter.record_outcome(
                request_id=request_id,
                outcome={
                    "success": True,
                    "hypothesis_name": hypothesis.get("name", ""),
                    "latency_ms": latency,
                },
            )

            logger.info(f"Generated hypothesis: {hypothesis.get('name', 'unnamed')}")

            return hypothesis

        except Exception as e:
            self._failed_generations += 1

            latency = (time.perf_counter() - start_time) * 1000
            self.emitter.record_outcome(
                request_id=request_id,
                outcome={"success": False, "error": str(e), "latency_ms": latency},
            )

            logger.error(f"Hypothesis generation failed: {e}")
            raise

    def _extract_parent_insights(
        self,
        parents: List[EvolutionNode],
    ) -> List[Dict[str, Any]]:
        """Extract key insights from parent nodes."""
        insights = []

        for parent in parents[:self.config.max_parents_to_consider]:
            insight = {
                "node_id": parent.node_id,
                "name": parent.name,
                "score": parent.score,
                "motivation": parent.motivation,
                "analysis": parent.analysis or "",
            }

            # Extract what worked
            if parent.score > 0.7:
                insight["success_factors"] = self._extract_success_factors(parent)

            # Extract what didn't work
            if parent.analysis and "failed" in parent.analysis.lower():
                insight["failure_modes"] = self._extract_failure_modes(parent)

            insights.append(insight)

        return insights

    def _extract_success_factors(self, node: EvolutionNode) -> List[str]:
        """Extract success factors from a high-scoring node."""
        factors = []

        if node.evaluation_result:
            if node.evaluation_result.safety_passed:
                factors.append("Passed safety validation")
            if hasattr(node.evaluation_result, 'clinical_validity_passed'):
                if node.evaluation_result.clinical_validity_passed:
                    factors.append("Clinical validity confirmed")

        # Parse analysis for positive indicators
        if node.analysis:
            for indicator in ["improved", "successful", "effective", "accurate"]:
                if indicator in node.analysis.lower():
                    factors.append(f"Analysis indicates: {indicator}")
                    break

        return factors

    def _extract_failure_modes(self, node: EvolutionNode) -> List[str]:
        """Extract failure modes from node analysis."""
        modes = []

        if node.analysis:
            # Look for failure patterns
            failure_patterns = [
                r"failed (to|because|due)",
                r"error in",
                r"incorrect",
                r"low (accuracy|sensitivity|specificity)",
            ]
            for pattern in failure_patterns:
                if re.search(pattern, node.analysis.lower()):
                    modes.append(f"Pattern: {pattern}")

        return modes

    def _select_relevant_knowledge(
        self,
        task: str,
        items: List[CognitionItem],
    ) -> List[CognitionItem]:
        """Select most relevant cognition items for the task."""
        if not items:
            return []

        # Simple relevance scoring (production would use embeddings)
        task_words = set(task.lower().split())

        scored = []
        for item in items:
            content_words = set(item.content.lower().split())
            overlap = len(task_words & content_words)
            scored.append((item, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [item for item, _ in scored[:self.config.max_cognition_items]]

    def _select_template(
        self,
        task: str,
        template_name: Optional[str],
    ) -> HypothesisTemplate:
        """Select appropriate template for the task."""
        if template_name and template_name in self.TEMPLATES:
            return self.TEMPLATES[template_name]

        # Auto-select based on task keywords
        task_lower = task.lower()

        if any(kw in task_lower for kw in ["diagnos", "detect", "screen"]):
            return self.TEMPLATES["diagnostic_improvement"]

        if any(kw in task_lower for kw in ["treat", "therap", "recommend"]):
            return self.TEMPLATES["therapeutic_recommendation"]

        return self.TEMPLATES["workflow_optimization"]

    def _generate_with_llm(
        self,
        task_description: str,
        parent_insights: List[Dict[str, Any]],
        relevant_knowledge: List[CognitionItem],
        template: HypothesisTemplate,
    ) -> Dict[str, Any]:
        """Generate hypothesis using LLM."""
        # Build prompt
        prompt = self._build_prompt(
            task_description, parent_insights, relevant_knowledge, template
        )

        # Call LLM
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Parse response
        return self._parse_llm_response(response, template)

    def _generate_fallback(
        self,
        task_description: str,
        parent_insights: List[Dict[str, Any]],
        relevant_knowledge: List[CognitionItem],
        template: HypothesisTemplate,
    ) -> Dict[str, Any]:
        """Generate hypothesis without LLM (for testing/fallback)."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # Synthesize from parents
        parent_summary = ""
        if parent_insights:
            best_parent = max(parent_insights, key=lambda x: x["score"])
            parent_summary = f"Building on {best_parent['name']} (score: {best_parent['score']:.3f})"

        # Synthesize from knowledge
        knowledge_summary = ""
        if relevant_knowledge:
            knowledge_summary = f"Incorporating {len(relevant_knowledge)} prior findings"

        hypothesis = {
            "name": f"hypothesis_{timestamp}",
            "motivation": (
                f"Task: {task_description}\n\n"
                f"{parent_summary}\n"
                f"{knowledge_summary}\n\n"
                f"Generated via fallback mechanism - LLM integration pending"
            ),
            "description": f"Hypothesis for: {task_description[:100]}",
            "code": self._generate_stub_code(task_description),
            "safety_notes": "Requires clinical validation before deployment",
            "evidence": [
                f"Based on {len(parent_insights)} parent nodes",
                f"Informed by {len(relevant_knowledge)} prior knowledge items",
            ],
            "template_used": template.name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        return hypothesis

    def _generate_stub_code(self, task: str) -> str:
        """Generate stub implementation code."""
        return f'''"""
Auto-generated hypothesis stub for: {task[:50]}
TODO: Implement actual capability
"""

class HypothesisCapability:
    """Capability generated by evolution pipeline."""

    def __init__(self):
        self.name = "hypothesis_capability"
        self.version = "0.1.0"

    def execute(self, input_data: dict) -> dict:
        """
        Execute the capability.

        Args:
            input_data: Input data for processing

        Returns:
            Processing result
        """
        # TODO: Implement based on hypothesis
        return {{
            "status": "stub",
            "message": "Implementation pending",
        }}

    def validate(self) -> bool:
        """Validate capability is safe to run."""
        return True
'''

    def _build_prompt(
        self,
        task: str,
        parent_insights: List[Dict[str, Any]],
        knowledge: List[CognitionItem],
        template: HypothesisTemplate,
    ) -> str:
        """Build LLM prompt for hypothesis generation."""
        prompt_parts = [
            f"# Task\n{task}\n",
            "\n# Parent Node Insights\n",
        ]

        for i, insight in enumerate(parent_insights):
            prompt_parts.append(
                f"{i+1}. {insight['name']} (score: {insight['score']:.3f})\n"
                f"   Motivation: {insight['motivation'][:200]}\n"
            )

        prompt_parts.append("\n# Prior Knowledge\n")
        for i, item in enumerate(knowledge):
            prompt_parts.append(f"{i+1}. {item.content[:200]}\n")

        prompt_parts.append(f"\n# Template: {template.name}\n")
        prompt_parts.append(template.template)

        prompt_parts.append(
            "\n\nGenerate a novel hypothesis that improves upon the parents. "
            "Return a JSON object with: name, motivation, description, code, "
            "safety_notes, and evidence (list)."
        )

        return "".join(prompt_parts)

    def _parse_llm_response(
        self,
        response: str,
        template: HypothesisTemplate,
    ) -> Dict[str, Any]:
        """Parse LLM response into hypothesis dict."""
        # Try to extract JSON
        try:
            # Find JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try direct parse
            return json.loads(response)

        except json.JSONDecodeError:
            # Fall back to regex extraction
            hypothesis = {}

            # Extract name
            name_match = re.search(r'name["\s:]+([^\n"]+)', response, re.I)
            if name_match:
                hypothesis["name"] = name_match.group(1).strip()

            # Extract code block
            code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
            if code_match:
                hypothesis["code"] = code_match.group(1)

            hypothesis["motivation"] = response[:500]  # Use truncated response

            return hypothesis

    def _validate_hypothesis(self, hypothesis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean hypothesis."""
        # Ensure required fields
        if "name" not in hypothesis or not hypothesis["name"]:
            hypothesis["name"] = f"hypothesis_{int(time.time())}"

        if "motivation" not in hypothesis:
            hypothesis["motivation"] = ""

        # Validate lengths
        if len(hypothesis.get("motivation", "")) < self.config.min_motivation_length:
            hypothesis["motivation"] += (
                "\n[Note: Motivation auto-extended - original was too brief]"
            )

        if len(hypothesis.get("code", "")) > self.config.max_code_length:
            hypothesis["code"] = hypothesis["code"][:self.config.max_code_length]
            hypothesis["code"] += "\n# [Truncated - exceeded max length]"

        # Add metadata
        hypothesis["validated_at"] = datetime.now(timezone.utc).isoformat()

        return hypothesis

    def _add_clinical_reasoning(
        self,
        hypothesis: Dict[str, Any],
        domain: Optional[DeploymentDomain],
    ) -> Dict[str, Any]:
        """Add clinical reasoning context to hypothesis."""
        if "clinical_reasoning" not in hypothesis:
            hypothesis["clinical_reasoning"] = {
                "domain": domain.value if domain else "general",
                "evidence_level": "hypothesis",
                "requires_validation": True,
                "safety_classification": "unvalidated",
            }

        # Add standard safety disclaimer
        if "safety_notes" not in hypothesis or not hypothesis["safety_notes"]:
            hypothesis["safety_notes"] = (
                "This is an AI-generated hypothesis requiring human review "
                "and clinical validation before any deployment consideration."
            )

        return hypothesis

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "hypotheses_generated": self._hypotheses_generated,
            "failed_generations": self._failed_generations,
            "success_rate": (
                self._hypotheses_generated /
                (self._hypotheses_generated + self._failed_generations)
                if (self._hypotheses_generated + self._failed_generations) > 0
                else 0
            ),
        }
