"""
Cross-System Convergence Detection
The core innovation of HyperCore
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class ConvergenceType(Enum):
    NONE = "none"
    EARLY = "early"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class ConvergenceResult:
    convergence_type: ConvergenceType
    convergence_score: float
    systems_involved: List[str]
    system_count: int
    estimated_time_to_harm: Optional[Dict]
    velocity: str
    description: str
    details: Dict

    def to_dict(self) -> Dict:
        return {
            'convergence_type': self.convergence_type.value,
            'convergence_score': self.convergence_score,
            'systems_involved': self.systems_involved,
            'system_count': self.system_count,
            'estimated_time_to_harm': self.estimated_time_to_harm,
            'velocity': self.velocity,
            'description': self.description,
            'details': self.details
        }


CONVERGENCE_PATTERNS = {
    'sepsis': {
        'required': ['sepsis', 'inflammatory'],
        'associated': ['perfusion', 'respiratory', 'renal', 'coagulation'],
        'min_systems': 2
    },
    'cardiogenic_shock': {
        'required': ['cardiac', 'perfusion'],
        'associated': ['respiratory', 'renal', 'acid_base'],
        'min_systems': 2
    },
    'acute_kidney_injury': {
        'required': ['renal'],
        'associated': ['fluid_balance', 'acid_base', 'metabolic'],
        'min_systems': 1
    },
    'diabetic_ketoacidosis': {
        'required': ['metabolic', 'acid_base'],
        'associated': ['fluid_balance', 'renal'],
        'min_systems': 2
    },
    'respiratory_failure': {
        'required': ['respiratory'],
        'associated': ['acid_base', 'perfusion', 'cardiac'],
        'min_systems': 1
    },
    'liver_failure': {
        'required': ['hepatic'],
        'associated': ['coagulation', 'metabolic', 'neurologic'],
        'min_systems': 1
    },
    'multi_organ_failure': {
        'required': [],
        'associated': ['cardiac', 'renal', 'hepatic', 'respiratory', 'coagulation'],
        'min_systems': 3
    },
    'thyroid_storm': {
        'required': ['endocrine'],
        'associated': ['cardiac', 'vitals', 'neurologic'],
        'min_systems': 2
    },
    'adrenal_crisis': {
        'required': ['endocrine'],
        'associated': ['perfusion', 'fluid_balance', 'metabolic'],
        'min_systems': 2
    }
}


class ConvergenceDetector:
    """
    Detects cross-system convergence patterns.
    """

    def __init__(self):
        self.patterns = CONVERGENCE_PATTERNS

    def detect(self, endpoint_results: Dict[str, Any]) -> ConvergenceResult:
        """
        Detect convergence across analyzed endpoints.
        """
        elevated_endpoints = []
        risk_scores = {}

        for endpoint, result in endpoint_results.items():
            if hasattr(result, 'risk_score'):
                score = result.risk_score
            else:
                score = result.get('risk_score', 0)

            risk_scores[endpoint] = score
            if score >= 40:
                elevated_endpoints.append(endpoint)

        if not elevated_endpoints:
            return ConvergenceResult(
                convergence_type=ConvergenceType.NONE,
                convergence_score=0,
                systems_involved=[],
                system_count=0,
                estimated_time_to_harm=None,
                velocity="stable",
                description="No significant convergence detected",
                details={'endpoint_risks': risk_scores}
            )

        matched_patterns = []
        for pattern_name, pattern_config in self.patterns.items():
            if self._matches_pattern(elevated_endpoints, pattern_config):
                matched_patterns.append({
                    'name': pattern_name,
                    'confidence': self._calculate_pattern_confidence(
                        elevated_endpoints, pattern_config, risk_scores
                    )
                })

        convergence_score = self._calculate_convergence_score(
            elevated_endpoints, risk_scores
        )

        if len(elevated_endpoints) >= 4 or convergence_score >= 80:
            conv_type = ConvergenceType.CRITICAL
        elif len(elevated_endpoints) >= 3 or convergence_score >= 60:
            conv_type = ConvergenceType.SEVERE
        elif len(elevated_endpoints) >= 2 or convergence_score >= 40:
            conv_type = ConvergenceType.MODERATE
        elif elevated_endpoints:
            conv_type = ConvergenceType.EARLY
        else:
            conv_type = ConvergenceType.NONE

        time_to_harm = self._estimate_time_to_harm(convergence_score, elevated_endpoints)
        velocity = self._determine_velocity(convergence_score)
        description = self._build_description(conv_type, elevated_endpoints, matched_patterns)

        return ConvergenceResult(
            convergence_type=conv_type,
            convergence_score=convergence_score,
            systems_involved=elevated_endpoints,
            system_count=len(elevated_endpoints),
            estimated_time_to_harm=time_to_harm,
            velocity=velocity,
            description=description,
            details={
                'endpoint_risks': risk_scores,
                'matched_patterns': matched_patterns,
                'elevated_endpoints': elevated_endpoints
            }
        )

    def _matches_pattern(self, endpoints: List[str], pattern: Dict) -> bool:
        """Check if endpoints match a known pattern."""
        required = pattern.get('required', [])
        associated = pattern.get('associated', [])
        min_systems = pattern.get('min_systems', 1)

        for req in required:
            if req not in endpoints:
                return False

        associated_count = sum(1 for a in associated if a in endpoints)
        total = len(required) + associated_count
        return total >= min_systems

    def _calculate_pattern_confidence(
        self,
        endpoints: List[str],
        pattern: Dict,
        risk_scores: Dict
    ) -> float:
        """Calculate confidence that a pattern matches."""
        required = pattern.get('required', [])
        associated = pattern.get('associated', [])

        confidence = 0
        for req in required:
            if req in endpoints:
                confidence += risk_scores.get(req, 50) / 100 * 0.4

        for assoc in associated:
            if assoc in endpoints:
                confidence += risk_scores.get(assoc, 50) / 100 * 0.1

        return min(1.0, confidence)

    def _calculate_convergence_score(
        self,
        endpoints: List[str],
        risk_scores: Dict
    ) -> float:
        """Calculate overall convergence score."""
        if not endpoints:
            return 0

        scores = [risk_scores.get(ep, 0) for ep in endpoints]
        max_score = max(scores)
        mean_score = np.mean(scores)
        count_bonus = min(100, len(endpoints) * 20)

        convergence = max_score * 0.5 + mean_score * 0.3 + count_bonus * 0.2
        return min(100, convergence)

    def _estimate_time_to_harm(
        self,
        convergence_score: float,
        endpoints: List[str]
    ) -> Optional[Dict]:
        """Estimate time to clinical harm."""
        if convergence_score < 40:
            return {'min': 48, 'max': 72, 'unit': 'hours'}
        elif convergence_score < 60:
            return {'min': 24, 'max': 48, 'unit': 'hours'}
        elif convergence_score < 80:
            return {'min': 12, 'max': 24, 'unit': 'hours'}
        else:
            return {'min': 4, 'max': 12, 'unit': 'hours'}

    def _determine_velocity(self, convergence_score: float) -> str:
        """Determine rate of deterioration."""
        if convergence_score >= 80:
            return "rapid"
        elif convergence_score >= 60:
            return "moderate"
        elif convergence_score >= 40:
            return "slow"
        else:
            return "stable"

    def _build_description(
        self,
        conv_type: ConvergenceType,
        endpoints: List[str],
        patterns: List[Dict]
    ) -> str:
        """Build human-readable description."""
        if conv_type == ConvergenceType.NONE:
            return "No significant cross-system convergence detected."

        pattern_names = [p['name'].replace('_', ' ').title() for p in patterns]

        if patterns:
            pattern_str = ", ".join(pattern_names)
            return f"{conv_type.value.title()} convergence detected across {len(endpoints)} systems. " \
                   f"Pattern suggests: {pattern_str}."
        else:
            return f"{conv_type.value.title()} convergence detected across {len(endpoints)} systems: " \
                   f"{', '.join(endpoints)}. No specific disease pattern matched - investigate further."
