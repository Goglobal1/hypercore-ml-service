"""
Alert Routing System - Routes alerts to appropriate clinicians.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field, asdict
import logging

from .models import (
    AlertEvent,
    AlertType,
    AlertSeverity,
    ClinicalState,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ROUTING RULES
# =============================================================================

@dataclass
class RoutingRule:
    """A single routing rule."""
    rule_id: str
    name: str
    # Matching conditions
    risk_domains: List[str] = field(default_factory=list)  # Empty = all
    min_severity: Optional[AlertSeverity] = None
    alert_types: List[AlertType] = field(default_factory=list)  # Empty = all
    # Routing targets
    clinician_roles: List[str] = field(default_factory=list)
    clinician_ids: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)  # pager, sms, email, dashboard
    # Timing
    escalation_minutes: Optional[int] = None  # Auto-escalate if no ack

    def matches(self, event: AlertEvent) -> bool:
        """Check if this rule matches the event."""
        # Check domain
        if self.risk_domains and event.risk_domain not in self.risk_domains:
            return False

        # Check severity
        if self.min_severity:
            severity_order = ["INFO", "WARNING", "URGENT", "CRITICAL"]
            event_idx = severity_order.index(event.severity.value)
            min_idx = severity_order.index(self.min_severity.value)
            if event_idx < min_idx:
                return False

        # Check alert type
        if self.alert_types and event.alert_type not in self.alert_types:
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "risk_domains": self.risk_domains,
            "min_severity": self.min_severity.value if self.min_severity else None,
            "alert_types": [at.value for at in self.alert_types],
            "clinician_roles": self.clinician_roles,
            "clinician_ids": self.clinician_ids,
            "notification_channels": self.notification_channels,
            "escalation_minutes": self.escalation_minutes,
        }


# =============================================================================
# DEFAULT ROUTING RULES
# =============================================================================

DEFAULT_ROUTING_RULES: List[RoutingRule] = [
    # Critical alerts - pager + dashboard
    RoutingRule(
        rule_id="critical_all",
        name="Critical Alerts - All Domains",
        min_severity=AlertSeverity.CRITICAL,
        alert_types=[AlertType.INTERRUPTIVE],
        clinician_roles=["attending", "resident", "charge_nurse"],
        notification_channels=["pager", "dashboard", "sms"],
        escalation_minutes=5,
    ),

    # Sepsis alerts - infectious disease team
    RoutingRule(
        rule_id="sepsis_urgent",
        name="Sepsis Urgent Alerts",
        risk_domains=["sepsis", "infection"],
        min_severity=AlertSeverity.URGENT,
        clinician_roles=["id_attending", "resident"],
        notification_channels=["pager", "dashboard"],
        escalation_minutes=15,
    ),

    # Cardiac alerts - cardiology team
    RoutingRule(
        rule_id="cardiac_urgent",
        name="Cardiac Urgent Alerts",
        risk_domains=["cardiac", "deterioration_cardiac"],
        min_severity=AlertSeverity.URGENT,
        clinician_roles=["cardiology", "resident"],
        notification_channels=["pager", "dashboard"],
        escalation_minutes=10,
    ),

    # Neurological alerts - neuro team
    RoutingRule(
        rule_id="neuro_urgent",
        name="Neurological Urgent Alerts",
        risk_domains=["neurological"],
        min_severity=AlertSeverity.URGENT,
        clinician_roles=["neurology", "neurosurgery", "resident"],
        notification_channels=["pager", "dashboard"],
        escalation_minutes=5,  # Very time-sensitive
    ),

    # Respiratory alerts - RT + pulmonology
    RoutingRule(
        rule_id="respiratory_urgent",
        name="Respiratory Urgent Alerts",
        risk_domains=["respiratory", "respiratory_failure"],
        min_severity=AlertSeverity.URGENT,
        clinician_roles=["pulmonology", "respiratory_therapy", "resident"],
        notification_channels=["pager", "dashboard"],
        escalation_minutes=10,
    ),

    # Kidney alerts - nephrology
    RoutingRule(
        rule_id="kidney_urgent",
        name="Kidney Urgent Alerts",
        risk_domains=["kidney", "kidney_injury"],
        min_severity=AlertSeverity.URGENT,
        clinician_roles=["nephrology", "resident"],
        notification_channels=["dashboard", "email"],
        escalation_minutes=30,
    ),

    # Watch alerts - bedside nurse
    RoutingRule(
        rule_id="watch_all",
        name="Watch Alerts - All Domains",
        min_severity=AlertSeverity.WARNING,
        alert_types=[AlertType.NON_INTERRUPTIVE],
        clinician_roles=["bedside_nurse", "charge_nurse"],
        notification_channels=["dashboard"],
        escalation_minutes=60,
    ),

    # Trial rescue alerts - research team
    RoutingRule(
        rule_id="trial_rescue",
        name="Trial Rescue Opportunities",
        risk_domains=["trial_confounder"],
        clinician_roles=["research_coordinator", "principal_investigator"],
        notification_channels=["email", "dashboard"],
        escalation_minutes=None,  # No escalation for informational
    ),

    # Outbreak alerts - infection control
    RoutingRule(
        rule_id="outbreak",
        name="Outbreak Alerts",
        risk_domains=["outbreak"],
        min_severity=AlertSeverity.WARNING,
        clinician_roles=["infection_control", "epidemiology"],
        notification_channels=["email", "dashboard", "sms"],
        escalation_minutes=30,
    ),
]


# =============================================================================
# ALERT ROUTER
# =============================================================================

class AlertRouter:
    """Routes alerts to appropriate clinicians based on rules."""

    def __init__(self, rules: Optional[List[RoutingRule]] = None):
        self.rules = rules or DEFAULT_ROUTING_RULES
        # Callbacks for actual notification delivery
        self._notification_callbacks: Dict[str, callable] = {}

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a routing rule by ID."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                return True
        return False

    def get_rules(self) -> List[RoutingRule]:
        """Get all routing rules."""
        return self.rules

    def register_notification_callback(self, channel: str, callback: callable) -> None:
        """
        Register a callback for a notification channel.

        The callback should accept: (alert: AlertEvent, targets: List[str]) -> bool
        """
        self._notification_callbacks[channel] = callback

    def route_alert(self, event: AlertEvent) -> Dict[str, Any]:
        """
        Route an alert to appropriate clinicians.

        Returns routing result with matched rules and targets.
        """
        matched_rules = []
        all_roles = set()
        all_ids = set()
        all_channels = set()
        min_escalation = None

        for rule in self.rules:
            if rule.matches(event):
                matched_rules.append(rule)
                all_roles.update(rule.clinician_roles)
                all_ids.update(rule.clinician_ids)
                all_channels.update(rule.notification_channels)

                if rule.escalation_minutes:
                    if min_escalation is None or rule.escalation_minutes < min_escalation:
                        min_escalation = rule.escalation_minutes

        # Update event with routing info
        event.routed_to = list(all_roles) + list(all_ids)

        # Trigger notification callbacks
        notifications_sent = {}
        for channel in all_channels:
            if channel in self._notification_callbacks:
                try:
                    callback = self._notification_callbacks[channel]
                    success = callback(event, list(all_roles) + list(all_ids))
                    notifications_sent[channel] = success
                except Exception as e:
                    logger.error(f"Notification callback failed for {channel}: {e}")
                    notifications_sent[channel] = False
            else:
                notifications_sent[channel] = "no_callback_registered"

        return {
            "alert_id": event.event_id,
            "patient_id": event.patient_id,
            "risk_domain": event.risk_domain,
            "severity": event.severity.value,
            "alert_type": event.alert_type.value,
            "matched_rules": [r.to_dict() for r in matched_rules],
            "routed_to_roles": list(all_roles),
            "routed_to_ids": list(all_ids),
            "notification_channels": list(all_channels),
            "notifications_sent": notifications_sent,
            "escalation_minutes": min_escalation,
            "escalation_deadline": (
                (event.timestamp + timedelta(minutes=min_escalation)).isoformat()
                if min_escalation else None
            ),
        }


# =============================================================================
# ESCALATION MANAGER
# =============================================================================

@dataclass
class PendingEscalation:
    """Tracks a pending escalation."""
    alert_id: str
    patient_id: str
    episode_id: str
    original_severity: AlertSeverity
    deadline: datetime
    escalation_level: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "patient_id": self.patient_id,
            "episode_id": self.episode_id,
            "original_severity": self.original_severity.value,
            "deadline": self.deadline.isoformat(),
            "escalation_level": self.escalation_level,
        }


class EscalationManager:
    """
    Manages escalation timers for unacknowledged alerts.

    If alert is not acknowledged within escalation window:
    - Level 1: Notify additional clinicians
    - Level 2: Page attending physician
    - Level 3: Activate rapid response team (for critical)
    """

    def __init__(self, router: AlertRouter):
        self.router = router
        self.pending: Dict[str, PendingEscalation] = {}  # alert_id -> PendingEscalation

    def schedule_escalation(
        self,
        event: AlertEvent,
        escalation_minutes: int,
    ) -> None:
        """Schedule an escalation for an alert."""
        deadline = event.timestamp + timedelta(minutes=escalation_minutes)

        self.pending[event.event_id] = PendingEscalation(
            alert_id=event.event_id,
            patient_id=event.patient_id,
            episode_id=event.episode_id or "",
            original_severity=event.severity,
            deadline=deadline,
        )

        logger.info(f"Scheduled escalation for alert {event.event_id} at {deadline}")

    def cancel_escalation(self, alert_id: str) -> bool:
        """Cancel a pending escalation (e.g., on acknowledgment)."""
        if alert_id in self.pending:
            del self.pending[alert_id]
            logger.info(f"Cancelled escalation for alert {alert_id}")
            return True
        return False

    def check_escalations(self) -> List[PendingEscalation]:
        """
        Check for alerts that need escalation.

        Should be called periodically (e.g., every minute).
        """
        now = datetime.now(timezone.utc)
        to_escalate = []

        for alert_id, pending in list(self.pending.items()):
            if now >= pending.deadline:
                to_escalate.append(pending)
                pending.escalation_level += 1

                # Calculate new deadline based on level
                if pending.escalation_level == 1:
                    new_deadline = now + timedelta(minutes=5)
                elif pending.escalation_level == 2:
                    new_deadline = now + timedelta(minutes=3)
                else:
                    # Max escalation reached - remove from pending
                    del self.pending[alert_id]
                    continue

                pending.deadline = new_deadline

        return to_escalate

    def get_pending_count(self) -> int:
        """Get count of pending escalations."""
        return len(self.pending)

    def get_pending_for_patient(self, patient_id: str) -> List[PendingEscalation]:
        """Get all pending escalations for a patient."""
        return [p for p in self.pending.values() if p.patient_id == patient_id]


# =============================================================================
# GLOBAL INSTANCES
# =============================================================================

_router: Optional[AlertRouter] = None
_escalation_manager: Optional[EscalationManager] = None


def get_router() -> AlertRouter:
    """Get the global alert router."""
    global _router
    if _router is None:
        _router = AlertRouter()
    return _router


def get_escalation_manager() -> EscalationManager:
    """Get the global escalation manager."""
    global _escalation_manager
    if _escalation_manager is None:
        _escalation_manager = EscalationManager(get_router())
    return _escalation_manager
