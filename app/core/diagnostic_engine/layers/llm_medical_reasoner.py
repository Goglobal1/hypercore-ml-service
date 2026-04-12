"""
Layer 4e: LLM Medical Reasoning
Uses Claude API to reason about unexplained abnormalities.

The Key Insight: Claude already knows medical patterns. Instead of coding
97,000 rules, USE CLAUDE AS THE REASONING ENGINE.

How It Works:
1. Existing Layer 4a-4d finds abnormal values
2. If unexplained abnormalities remain -> Send to Claude API
3. Claude reasons: "These markers suggest X, Y, Z diseases"
4. Results merged back into diagnostic output as layer_4e_llm_diagnoses
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Check for Anthropic API availability
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("[LLMReasoner] anthropic package not installed")


@dataclass
class LLMDiagnosis:
    """Structured diagnosis from LLM reasoning."""
    disease_name: str
    icd10_code: Optional[str]
    confidence: str  # high, medium, low
    confidence_score: float
    supporting_evidence: List[str]
    reasoning: str
    recommended_tests: List[str]
    source: str = "llm_reasoning"


class LLMMedicalReasoner:
    """
    Layer 4e: Uses Claude API to reason about unexplained abnormalities.

    This layer activates when existing rule-based and ML layers find
    abnormal values that don't match known disease patterns.
    """

    def __init__(self, model: str = None, api_key: str = None):
        """
        Initialize the LLM Medical Reasoner.

        Args:
            model: Model to use (default: claude-sonnet-4-20250514)
            api_key: Anthropic API key (default: from ANTHROPIC_API_KEY env var)
        """
        self.model = model or "claude-sonnet-4-20250514"
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self.available = False

        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.available = True
                logger.info(f"[LLMReasoner] Initialized with model {self.model}")
            except Exception as e:
                logger.warning(f"[LLMReasoner] Failed to initialize: {e}")
        else:
            if not ANTHROPIC_AVAILABLE:
                logger.warning("[LLMReasoner] anthropic package not available")
            if not self.api_key:
                logger.warning("[LLMReasoner] ANTHROPIC_API_KEY not set")

    def reason_about_abnormalities(
        self,
        abnormal_labs: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]] = None,
        axis_scores: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Send unexplained abnormalities to Claude for medical reasoning.

        Args:
            abnormal_labs: Dict of lab values flagged as abnormal
                Format: {lab_name: {value: X, unit: Y, reference_range: Z, status: 'high'/'low'}}
            patient_context: Optional context (age, sex, history, medications)
            axis_scores: Optional axis scores from Layer 3 for additional context

        Returns:
            List of potential diagnoses with reasoning
        """
        if not self.available:
            logger.warning("[LLMReasoner] Not available, returning empty results")
            return []

        if not abnormal_labs:
            return []

        patient_context = patient_context or {}

        try:
            prompt = self._build_medical_reasoning_prompt(
                abnormal_labs,
                patient_context,
                axis_scores
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self._get_system_prompt(),
                messages=[{"role": "user", "content": prompt}]
            )

            diagnoses = self._parse_diagnoses(response.content[0].text)
            logger.info(f"[LLMReasoner] Generated {len(diagnoses)} diagnoses")
            return diagnoses

        except Exception as e:
            logger.error(f"[LLMReasoner] Error during reasoning: {e}")
            return [{
                "source": "llm_reasoning",
                "error": str(e),
                "confidence": "none"
            }]

    def _get_system_prompt(self) -> str:
        """Get the system prompt for medical reasoning."""
        return """You are a clinical reasoning engine integrated into a diagnostic support system.
Given abnormal lab values and patient context, identify potential diagnoses.

CRITICAL RULES:
1. You are a DECISION SUPPORT tool, not a replacement for clinical judgment
2. All outputs will be reviewed by clinicians
3. Prioritize patient safety - flag urgent/critical findings prominently
4. Be thorough but prioritize the most likely diagnoses

For each diagnosis, provide:
1. disease_name: Standard medical term
2. icd10_code: ICD-10 code if known (or null)
3. confidence: "high", "medium", or "low"
4. confidence_score: 0.0 to 1.0
5. supporting_evidence: List of labs that support this diagnosis
6. reasoning: Brief clinical reasoning
7. recommended_tests: Tests to confirm/rule out

Format response as a JSON array of diagnosis objects.
Example:
[
  {
    "disease_name": "Diabetic Ketoacidosis",
    "icd10_code": "E10.10",
    "confidence": "high",
    "confidence_score": 0.85,
    "supporting_evidence": ["glucose: 450 mg/dL", "pH: 7.25", "ketones: positive"],
    "reasoning": "Severe hyperglycemia with metabolic acidosis and ketosis indicates DKA",
    "recommended_tests": ["arterial blood gas", "serum ketones", "HbA1c"]
  }
]"""

    def _build_medical_reasoning_prompt(
        self,
        abnormal_labs: Dict[str, Any],
        patient_context: Dict[str, Any],
        axis_scores: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build the prompt for Claude to reason about abnormalities."""
        prompt = "## Patient Context\n"
        prompt += f"- Age: {patient_context.get('age', 'Unknown')}\n"
        prompt += f"- Sex: {patient_context.get('sex', 'Unknown')}\n"
        prompt += f"- Medical History: {patient_context.get('history', 'None provided')}\n"
        prompt += f"- Current Medications: {patient_context.get('medications', 'None')}\n"

        if axis_scores:
            prompt += "\n## System Status (from diagnostic engine)\n"
            for axis_name, axis_data in axis_scores.items():
                if isinstance(axis_data, dict):
                    status = axis_data.get('status', 'unknown')
                    score = axis_data.get('score', 0)
                    if status != 'normal' or score > 0.3:
                        prompt += f"- {axis_name}: {status} (score: {score:.2f})\n"

        prompt += "\n## Abnormal Lab Values\n"
        prompt += "These values are abnormal and not fully explained by existing diagnostic rules:\n\n"

        for lab, data in abnormal_labs.items():
            if isinstance(data, dict):
                value = data.get('value', 'N/A')
                unit = data.get('unit', '')
                ref_range = data.get('reference_range', data.get('reference', 'N/A'))
                status = data.get('status', 'abnormal')
                prompt += f"- **{lab}**: {value} {unit} (Reference: {ref_range}) [{status}]\n"
            else:
                prompt += f"- **{lab}**: {data}\n"

        prompt += """
## Task
Analyze these abnormalities and identify potential diagnoses that could explain this pattern.
Consider both common and rare conditions. Return as a JSON array of diagnosis objects."""

        return prompt

    def _parse_diagnoses(self, response_text: str) -> List[Dict]:
        """Parse Claude's response into structured diagnoses."""
        try:
            # Handle potential markdown code blocks
            text = response_text.strip()
            if "```" in text:
                # Extract content between code blocks
                parts = text.split("```")
                for part in parts:
                    if part.strip().startswith("json"):
                        text = part.strip()[4:].strip()
                        break
                    elif part.strip().startswith("["):
                        text = part.strip()
                        break

            diagnoses = json.loads(text)

            # Normalize and validate each diagnosis
            normalized = []
            for diag in diagnoses:
                if isinstance(diag, dict):
                    normalized.append({
                        "disease_name": diag.get("disease_name", "Unknown"),
                        "icd10_code": diag.get("icd10_code"),
                        "confidence": diag.get("confidence", "medium"),
                        "confidence_score": float(diag.get("confidence_score", 0.5)),
                        "supporting_evidence": diag.get("supporting_evidence", []),
                        "reasoning": diag.get("reasoning", ""),
                        "recommended_tests": diag.get("recommended_tests", []),
                        "source": "llm_reasoning"
                    })

            return normalized

        except json.JSONDecodeError as e:
            logger.warning(f"[LLMReasoner] Failed to parse JSON: {e}")
            # Fallback: return as single insight
            return [{
                "source": "llm_reasoning",
                "reasoning": response_text,
                "confidence": "medium",
                "confidence_score": 0.5,
                "disease_name": "LLM Analysis",
                "supporting_evidence": [],
                "recommended_tests": []
            }]

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the LLM reasoner."""
        return {
            "available": self.available,
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "anthropic_installed": ANTHROPIC_AVAILABLE
        }


# Singleton instance
_llm_reasoner_instance = None

def get_llm_reasoner() -> LLMMedicalReasoner:
    """Get or create the singleton LLM reasoner instance."""
    global _llm_reasoner_instance
    if _llm_reasoner_instance is None:
        _llm_reasoner_instance = LLMMedicalReasoner()
    return _llm_reasoner_instance
