"""
Utility Engine - Core calculation logic
The 25th layer that decides: fire / suppress / delay / downgrade
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta


class EmissionDecision(Enum):
    FIRE = "fire"
    SUPPRESS = "suppress"
    DELAY = "delay"
    DOWNGRADE = "downgrade"
    CONDITIONAL = "conditional"


@dataclass
class UtilityComponents:
    information_gain: float      # 0-1
    urgency_factor: float        # 0-1
    actionability: float         # 0-1
    redundancy_penalty: float    # 0-1
    interruption_cost: float     # 0-1


@dataclass
class UtilityResult:
    utility_score: float
    decision: EmissionDecision
    components: UtilityComponents
    event_id: Optional[str]
    is_new_event: bool
    alert_sequence: int
    explanation: Dict[str, str]
    recommended_action: Optional[Dict]
    delay_minutes: Optional[int]
    downgrade_to_tier: Optional[str]


class UtilityEngine:
    """
    The Utility Engine evaluates whether a proposed alert is worth
    the cognitive interruption to a clinician.
    """

    # Default weights - can be overridden by UtilityConfig
    DEFAULT_WEIGHTS = {
        'information_gain': 0.30,
        'urgency_factor': 0.25,
        'actionability': 0.25,
        'redundancy_penalty': 0.15,
        'interruption_cost': 0.05
    }

    # Decision thresholds
    THRESHOLDS = {
        'fire': 0.70,
        'conditional': 0.50,
        'delay': 0.30,
        'downgrade': 0.10
    }

    def __init__(self, config: Optional[Dict] = None):
        self.weights = config.get('weights', self.DEFAULT_WEIGHTS) if config else self.DEFAULT_WEIGHTS
        self.thresholds = config.get('thresholds', self.THRESHOLDS) if config else self.THRESHOLDS

    def calculate_utility(
        self,
        patient_id: str,
        proposed_alert: Dict,
        analysis: Dict,
        user_id: str,
        user_alert_load: int,
        alert_history: List[Dict],
        event_history: List[Dict]
    ) -> UtilityResult:
        """
        Main entry point - calculate utility score and make emission decision.

        Args:
            patient_id: Patient identifier
            proposed_alert: The alert we're considering emitting
                - tier: 'critical', 'warning', 'advisory'
                - type: 'renal_deterioration', 'respiratory_decline', etc.
                - endpoint: Primary endpoint triggering alert
                - score: Current HyperCore score
                - message: Alert message
            analysis: HyperCore analysis result
                - hypercore_score: 0-100
                - clinical_state: 'stable', 'watch', 'warning', 'critical'
                - time_to_harm_min: hours
                - time_to_harm_max: hours
                - top_endpoints: list
                - velocity: 'rapid_worsening', 'worsening', 'stable', etc.
            user_id: User who would receive alert
            user_alert_load: Current number of active alerts for user
            alert_history: Recent alerts for this patient
            event_history: Active clinical events for this patient

        Returns:
            UtilityResult with decision and explanation
        """

        # Step 1: Detect or link to clinical event
        event, is_new_event = self._detect_or_link_event(
            patient_id, proposed_alert, analysis, event_history
        )

        # Step 2: Calculate each component
        information_gain = self._calculate_information_gain(
            proposed_alert, analysis, event, is_new_event, alert_history
        )

        urgency_factor = self._calculate_urgency_factor(analysis)

        actionability = self._calculate_actionability(
            proposed_alert, analysis, user_id
        )

        redundancy_penalty = self._calculate_redundancy_penalty(
            proposed_alert, event, alert_history
        )

        interruption_cost = self._calculate_interruption_cost(
            user_alert_load, user_id
        )

        # Step 3: Calculate composite utility score
        components = UtilityComponents(
            information_gain=information_gain,
            urgency_factor=urgency_factor,
            actionability=actionability,
            redundancy_penalty=redundancy_penalty,
            interruption_cost=interruption_cost
        )

        utility_score = self._calculate_composite_score(components)

        # Step 4: Make emission decision
        decision, delay_minutes, downgrade_tier = self._make_decision(
            utility_score, user_alert_load, proposed_alert
        )

        # Step 5: Generate explanation
        explanation = self._generate_explanation(
            components, decision, analysis, is_new_event
        )

        # Step 6: Determine recommended action
        recommended_action = self._get_recommended_action(
            proposed_alert, analysis
        )

        return UtilityResult(
            utility_score=utility_score,
            decision=decision,
            components=components,
            event_id=event.get('id') if event else None,
            is_new_event=is_new_event,
            alert_sequence=event.get('alert_count', 1) if event else 1,
            explanation=explanation,
            recommended_action=recommended_action,
            delay_minutes=delay_minutes,
            downgrade_to_tier=downgrade_tier
        )

    def _detect_or_link_event(
        self,
        patient_id: str,
        proposed_alert: Dict,
        analysis: Dict,
        event_history: List[Dict]
    ) -> Tuple[Optional[Dict], bool]:
        """
        Check if this alert belongs to an existing clinical event
        or if it represents a new event.
        """
        alert_type = proposed_alert.get('type', '')
        endpoint = proposed_alert.get('endpoint', '')

        # Look for active event of same type
        for event in event_history:
            if event.get('status') != 'active':
                continue

            # Match by type or primary endpoint
            if (event.get('event_type') == alert_type or
                event.get('primary_endpoint') == endpoint):
                # This is an update to existing event
                return event, False

        # No matching event - this is new
        new_event = {
            'id': f"EVT-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'patient_id': patient_id,
            'event_type': alert_type,
            'primary_endpoint': endpoint,
            'first_detected_at': datetime.now().isoformat(),
            'alert_count': 1,
            'current_severity': analysis.get('clinical_state', 'warning'),
            'status': 'active'
        }

        return new_event, True

    def _calculate_information_gain(
        self,
        proposed_alert: Dict,
        analysis: Dict,
        event: Optional[Dict],
        is_new_event: bool,
        alert_history: List[Dict]
    ) -> float:
        """
        How much NEW information does this alert provide?
        1.0 = First alert for this event
        0.0 = Completely redundant
        """

        # New event = maximum information gain
        if is_new_event:
            return 1.0

        if not event:
            return 0.8

        # Get last alert for this event
        event_alerts = [a for a in alert_history if a.get('event_id') == event.get('id')]
        if not event_alerts:
            return 0.9

        last_alert = max(event_alerts, key=lambda a: a.get('created_at', ''))

        # Compare severity/state
        last_state = last_alert.get('clinical_state', 'stable')
        current_state = analysis.get('clinical_state', 'stable')

        state_order = {'stable': 0, 'watch': 1, 'warning': 2, 'critical': 3}
        last_level = state_order.get(last_state, 0)
        current_level = state_order.get(current_state, 0)

        # Escalation = high information gain
        if current_level > last_level:
            return 0.85

        # Same level but score changed significantly
        last_score = last_alert.get('hypercore_score', 50)
        current_score = analysis.get('hypercore_score', 50)
        score_delta = abs(current_score - last_score)

        if score_delta > 15:
            return 0.6
        elif score_delta > 10:
            return 0.4
        elif score_delta > 5:
            return 0.2
        else:
            return 0.05  # Essentially redundant

    def _calculate_urgency_factor(self, analysis: Dict) -> float:
        """
        How time-sensitive is this alert?
        Based on time-to-harm prediction.
        """
        time_to_harm_min = analysis.get('time_to_harm_min', 48)

        if time_to_harm_min < 4:
            return 1.0
        elif time_to_harm_min < 8:
            return 0.85
        elif time_to_harm_min < 12:
            return 0.7
        elif time_to_harm_min < 24:
            return 0.5
        elif time_to_harm_min < 48:
            return 0.3
        else:
            return 0.1

    def _calculate_actionability(
        self,
        proposed_alert: Dict,
        analysis: Dict,
        user_id: str
    ) -> float:
        """
        Can the clinician actually DO something about this?
        """
        score = 0.5  # Start neutral

        endpoint = proposed_alert.get('endpoint', '')
        clinical_state = analysis.get('clinical_state', 'stable')

        # High-actionability endpoints
        high_action_endpoints = [
            'renal', 'respiratory', 'cardiac', 'sepsis',
            'fluid_balance', 'electrolyte'
        ]

        # Medium-actionability endpoints
        medium_action_endpoints = [
            'metabolic', 'acid_base', 'oxygenation', 'perfusion'
        ]

        if endpoint in high_action_endpoints:
            score += 0.25
        elif endpoint in medium_action_endpoints:
            score += 0.15

        # Earlier states are more actionable
        if clinical_state in ['watch', 'warning']:
            score += 0.15  # Can still intervene
        elif clinical_state == 'critical':
            score += 0.05  # May be late

        # Time to act
        time_to_harm_min = analysis.get('time_to_harm_min', 48)
        if time_to_harm_min > 12:
            score += 0.1  # Enough time
        elif time_to_harm_min < 4:
            score -= 0.1  # May be too late

        return max(0, min(1, score))

    def _calculate_redundancy_penalty(
        self,
        proposed_alert: Dict,
        event: Optional[Dict],
        alert_history: List[Dict]
    ) -> float:
        """
        Have we already communicated this information?
        """
        if not event:
            return 0.0

        penalty = 0.0

        # Get hours since last alert for this event
        event_id = event.get('id')
        event_alerts = [a for a in alert_history if a.get('event_id') == event_id]

        if not event_alerts:
            return 0.0

        last_alert = max(event_alerts, key=lambda a: a.get('created_at', ''))
        last_alert_time_str = last_alert.get('created_at', datetime.now().isoformat())

        try:
            last_alert_time = datetime.fromisoformat(last_alert_time_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            last_alert_time = datetime.now()

        hours_since = (datetime.now() - last_alert_time.replace(tzinfo=None)).total_seconds() / 3600

        # Time-based penalty
        if hours_since < 2:
            penalty += 0.4
        elif hours_since < 4:
            penalty += 0.2
        elif hours_since < 8:
            penalty += 0.1

        # Check if already acknowledged
        if last_alert.get('acknowledged_at'):
            ack_type = last_alert.get('acknowledgment_type', '')
            if ack_type == 'monitoring':
                penalty += 0.2  # They said they're watching
            elif ack_type == 'intervened':
                penalty += 0.3  # They already acted

        return min(1.0, penalty)

    def _calculate_interruption_cost(
        self,
        user_alert_load: int,
        user_id: str
    ) -> float:
        """
        What's the cognitive cost of this interruption?
        Based on user's current alert load.
        """
        if user_alert_load <= 2:
            return 0.0  # Low load, good time
        elif user_alert_load <= 4:
            return 0.2  # Moderate load
        elif user_alert_load <= 6:
            return 0.4  # High load
        elif user_alert_load <= 8:
            return 0.6  # Very high load
        else:
            return 0.8  # Overwhelmed

    def _calculate_composite_score(self, components: UtilityComponents) -> float:
        """
        Combine components into final utility score.
        """
        score = (
            components.information_gain * self.weights['information_gain']
            + components.urgency_factor * self.weights['urgency_factor']
            + components.actionability * self.weights['actionability']
            - components.redundancy_penalty * self.weights['redundancy_penalty']
            - components.interruption_cost * self.weights['interruption_cost']
        )

        return max(0, min(1, score))

    def _make_decision(
        self,
        utility_score: float,
        user_alert_load: int,
        proposed_alert: Dict
    ) -> Tuple[EmissionDecision, Optional[int], Optional[str]]:
        """
        Determine emission decision based on utility score.
        Returns: (decision, delay_minutes, downgrade_tier)
        """
        delay_minutes = None
        downgrade_tier = None

        if utility_score >= self.thresholds['fire']:
            decision = EmissionDecision.FIRE

        elif utility_score >= self.thresholds['conditional']:
            # Fire if load is low, otherwise delay
            if user_alert_load <= 3:
                decision = EmissionDecision.FIRE
            else:
                decision = EmissionDecision.DELAY
                delay_minutes = 30

        elif utility_score >= self.thresholds['delay']:
            decision = EmissionDecision.DELAY
            delay_minutes = 60

        elif utility_score >= self.thresholds['downgrade']:
            decision = EmissionDecision.DOWNGRADE
            current_tier = proposed_alert.get('tier', 'warning')
            tier_order = ['advisory', 'warning', 'critical']
            current_idx = tier_order.index(current_tier) if current_tier in tier_order else 1
            downgrade_tier = tier_order[max(0, current_idx - 1)]
            delay_minutes = 45

        else:
            decision = EmissionDecision.SUPPRESS

        return decision, delay_minutes, downgrade_tier

    def _generate_explanation(
        self,
        components: UtilityComponents,
        decision: EmissionDecision,
        analysis: Dict,
        is_new_event: bool
    ) -> Dict[str, str]:
        """
        Generate human-readable explanation for the decision.
        """
        explanations = {}

        # Why firing (or not)
        if decision == EmissionDecision.FIRE:
            reasons = []
            if is_new_event:
                reasons.append("First alert for this clinical issue")
            if components.information_gain > 0.7:
                reasons.append("Significant new information")
            if components.urgency_factor > 0.7:
                reasons.append(f"Time-sensitive - action needed within {analysis.get('time_to_harm_min', '?')} hours")
            if components.actionability > 0.6:
                reasons.append("Clear intervention available")
            explanations['why_firing'] = ". ".join(reasons) if reasons else "Utility score exceeds threshold"

        elif decision == EmissionDecision.SUPPRESS:
            reasons = []
            if components.redundancy_penalty > 0.5:
                reasons.append("Redundant - same information already communicated")
            if components.information_gain < 0.2:
                reasons.append("No significant new information")
            if components.actionability < 0.3:
                reasons.append("Low actionability - no clear intervention")
            explanations['why_suppressed'] = ". ".join(reasons) if reasons else "Utility score below threshold"

        elif decision == EmissionDecision.DELAY:
            explanations['why_delayed'] = "Moderate utility - will reassess shortly"

        elif decision == EmissionDecision.DOWNGRADE:
            explanations['why_downgraded'] = "Lower priority - reduced tier and delayed"

        # Why now
        if components.urgency_factor > 0.6:
            explanations['why_now'] = f"Time to harm: {analysis.get('time_to_harm_min', '?')}-{analysis.get('time_to_harm_max', '?')} hours"

        return explanations

    def _get_recommended_action(
        self,
        proposed_alert: Dict,
        analysis: Dict
    ) -> Optional[Dict]:
        """
        Determine recommended clinical action based on alert type.
        """
        endpoint = proposed_alert.get('endpoint', '')

        # Action recommendations by endpoint
        actions = {
            'renal': {
                'primary': 'Order repeat BMP',
                'secondary': ['Review nephrotoxic medications', 'Check fluid status', 'Consider nephrology consult'],
                'one_click_available': True
            },
            'respiratory': {
                'primary': 'Assess respiratory status',
                'secondary': ['Check SpO2 on room air', 'Order ABG if indicated', 'Consider chest X-ray'],
                'one_click_available': False
            },
            'cardiac': {
                'primary': 'Order ECG',
                'secondary': ['Review telemetry', 'Check troponin trend', 'Assess fluid status'],
                'one_click_available': True
            },
            'sepsis': {
                'primary': 'Initiate sepsis workup',
                'secondary': ['Blood cultures x2', 'Lactate level', 'Consider broad-spectrum antibiotics'],
                'one_click_available': True
            },
            'fluid_balance': {
                'primary': 'Review I/O balance',
                'secondary': ['Daily weight', 'Assess for edema', 'Consider diuretic adjustment'],
                'one_click_available': False
            }
        }

        return actions.get(endpoint, {
            'primary': 'Assess patient',
            'secondary': ['Review recent vitals', 'Consider additional workup'],
            'one_click_available': False
        })


# Singleton instance
_engine = None


def get_utility_engine(config: Optional[Dict] = None) -> UtilityEngine:
    global _engine
    if _engine is None:
        _engine = UtilityEngine(config)
    return _engine
