"""
Alert System Configuration - Unified Implementation
All biomarker thresholds, domain configurations, and site overrides.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from .models import RiskDomain


# =============================================================================
# BIOMARKER THRESHOLDS
# =============================================================================

@dataclass
class BiomarkerThreshold:
    """Threshold configuration for a single biomarker."""
    warning: float
    critical: float
    unit: str
    direction: str  # "rising" or "falling"
    weight: float = 1.0  # Importance weight for scoring

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Complete biomarker thresholds from hypercore (ALL preserved)
BIOMARKER_THRESHOLDS: Dict[str, Dict[str, BiomarkerThreshold]] = {
    "sepsis": {
        "lactate": BiomarkerThreshold(warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=1.0),
        "map": BiomarkerThreshold(warning=70.0, critical=65.0, unit="mmHg", direction="falling", weight=0.9),
        "crp": BiomarkerThreshold(warning=50.0, critical=100.0, unit="mg/L", direction="rising", weight=0.7),
        "wbc": BiomarkerThreshold(warning=10.0, critical=12.0, unit="K/uL", direction="rising", weight=0.6),
        "temperature": BiomarkerThreshold(warning=37.8, critical=38.3, unit="C", direction="rising", weight=0.5),
        "procalcitonin": BiomarkerThreshold(warning=0.5, critical=2.0, unit="ng/mL", direction="rising", weight=0.8),
        "heart_rate": BiomarkerThreshold(warning=100.0, critical=120.0, unit="bpm", direction="rising", weight=0.5),
        "respiratory_rate": BiomarkerThreshold(warning=20.0, critical=24.0, unit="/min", direction="rising", weight=0.5),
    },
    "cardiac": {
        "troponin": BiomarkerThreshold(warning=0.01, critical=0.04, unit="ng/mL", direction="rising", weight=1.0),
        "troponin_i": BiomarkerThreshold(warning=0.01, critical=0.04, unit="ng/mL", direction="rising", weight=1.0),
        "troponin_t": BiomarkerThreshold(warning=0.03, critical=0.1, unit="ng/mL", direction="rising", weight=1.0),
        "bnp": BiomarkerThreshold(warning=100.0, critical=400.0, unit="pg/mL", direction="rising", weight=0.9),
        "nt_probnp": BiomarkerThreshold(warning=300.0, critical=900.0, unit="pg/mL", direction="rising", weight=0.9),
        "heart_rate": BiomarkerThreshold(warning=110.0, critical=130.0, unit="bpm", direction="rising", weight=0.7),
        "systolic_bp": BiomarkerThreshold(warning=100.0, critical=90.0, unit="mmHg", direction="falling", weight=0.8),
        "diastolic_bp": BiomarkerThreshold(warning=65.0, critical=60.0, unit="mmHg", direction="falling", weight=0.6),
        "ck_mb": BiomarkerThreshold(warning=5.0, critical=25.0, unit="ng/mL", direction="rising", weight=0.7),
    },
    "kidney": {
        "creatinine": BiomarkerThreshold(warning=1.3, critical=2.0, unit="mg/dL", direction="rising", weight=1.0),
        "bun": BiomarkerThreshold(warning=25.0, critical=40.0, unit="mg/dL", direction="rising", weight=0.8),
        "potassium": BiomarkerThreshold(warning=5.0, critical=5.5, unit="mEq/L", direction="rising", weight=0.9),
        "gfr": BiomarkerThreshold(warning=60.0, critical=30.0, unit="mL/min", direction="falling", weight=0.9),
        "urine_output": BiomarkerThreshold(warning=1.0, critical=0.5, unit="mL/kg/h", direction="falling", weight=0.8),
        "sodium": BiomarkerThreshold(warning=130.0, critical=125.0, unit="mEq/L", direction="falling", weight=0.5),
    },
    "respiratory": {
        "spo2": BiomarkerThreshold(warning=94.0, critical=90.0, unit="%", direction="falling", weight=1.0),
        "pao2": BiomarkerThreshold(warning=80.0, critical=60.0, unit="mmHg", direction="falling", weight=0.9),
        "respiratory_rate": BiomarkerThreshold(warning=24.0, critical=30.0, unit="/min", direction="rising", weight=0.8),
        "fio2": BiomarkerThreshold(warning=0.4, critical=0.6, unit="fraction", direction="rising", weight=0.7),
        "pao2_fio2": BiomarkerThreshold(warning=300.0, critical=200.0, unit="ratio", direction="falling", weight=0.9),
        "paco2": BiomarkerThreshold(warning=45.0, critical=50.0, unit="mmHg", direction="rising", weight=0.6),
    },
    "hepatic": {
        "alt": BiomarkerThreshold(warning=200.0, critical=1000.0, unit="U/L", direction="rising", weight=0.8),
        "ast": BiomarkerThreshold(warning=200.0, critical=1000.0, unit="U/L", direction="rising", weight=0.8),
        "bilirubin": BiomarkerThreshold(warning=2.0, critical=4.0, unit="mg/dL", direction="rising", weight=0.9),
        "inr": BiomarkerThreshold(warning=1.5, critical=2.0, unit="ratio", direction="rising", weight=0.9),
        "albumin": BiomarkerThreshold(warning=3.0, critical=2.5, unit="g/dL", direction="falling", weight=0.7),
        "ammonia": BiomarkerThreshold(warning=50.0, critical=100.0, unit="umol/L", direction="rising", weight=0.8),
    },
    "neurological": {
        "gcs": BiomarkerThreshold(warning=12.0, critical=8.0, unit="score", direction="falling", weight=1.0),
        "icp": BiomarkerThreshold(warning=15.0, critical=22.0, unit="mmHg", direction="rising", weight=0.9),
        "cpp": BiomarkerThreshold(warning=70.0, critical=60.0, unit="mmHg", direction="falling", weight=0.9),
    },
    "hematologic": {
        "hemoglobin": BiomarkerThreshold(warning=9.0, critical=7.0, unit="g/dL", direction="falling", weight=0.9),
        "platelets": BiomarkerThreshold(warning=100.0, critical=50.0, unit="K/uL", direction="falling", weight=0.8),
        "inr": BiomarkerThreshold(warning=1.5, critical=2.5, unit="ratio", direction="rising", weight=0.8),
        "fibrinogen": BiomarkerThreshold(warning=200.0, critical=100.0, unit="mg/dL", direction="falling", weight=0.7),
    },
    "metabolic": {
        "glucose": BiomarkerThreshold(warning=180.0, critical=400.0, unit="mg/dL", direction="rising", weight=0.8),
        "glucose_low": BiomarkerThreshold(warning=70.0, critical=50.0, unit="mg/dL", direction="falling", weight=0.9),
        "sodium": BiomarkerThreshold(warning=130.0, critical=125.0, unit="mEq/L", direction="falling", weight=0.7),
        "sodium_high": BiomarkerThreshold(warning=148.0, critical=155.0, unit="mEq/L", direction="rising", weight=0.7),
        "potassium": BiomarkerThreshold(warning=5.0, critical=6.0, unit="mEq/L", direction="rising", weight=0.9),
        "potassium_low": BiomarkerThreshold(warning=3.5, critical=3.0, unit="mEq/L", direction="falling", weight=0.8),
        "ph": BiomarkerThreshold(warning=7.32, critical=7.25, unit="pH", direction="falling", weight=0.9),
        "ph_high": BiomarkerThreshold(warning=7.48, critical=7.55, unit="pH", direction="rising", weight=0.8),
        "lactate": BiomarkerThreshold(warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=0.9),
        "anion_gap": BiomarkerThreshold(warning=14.0, critical=20.0, unit="mEq/L", direction="rising", weight=0.7),
    },
    "oncology": {
        "cea": BiomarkerThreshold(warning=5.0, critical=20.0, unit="ng/mL", direction="rising", weight=0.7),
        "ca125": BiomarkerThreshold(warning=35.0, critical=100.0, unit="U/mL", direction="rising", weight=0.7),
        "ca199": BiomarkerThreshold(warning=37.0, critical=100.0, unit="U/mL", direction="rising", weight=0.7),
        "psa": BiomarkerThreshold(warning=4.0, critical=10.0, unit="ng/mL", direction="rising", weight=0.7),
        "afp": BiomarkerThreshold(warning=10.0, critical=400.0, unit="ng/mL", direction="rising", weight=0.8),
        "ldh": BiomarkerThreshold(warning=250.0, critical=500.0, unit="U/L", direction="rising", weight=0.6),
    },
}


# =============================================================================
# DOMAIN CONFIGURATIONS
# =============================================================================

@dataclass
class DomainConfig:
    """Configuration for a clinical risk domain."""
    # State thresholds (upper bounds)
    s0_upper: float = 0.30
    s1_upper: float = 0.55
    s2_upper: float = 0.80
    s3_upper: float = 1.00

    # Cooldown settings (minutes)
    default_cooldown_minutes: int = 60
    escalation_cooldown_minutes: int = 15
    critical_cooldown_minutes: int = 5

    # Velocity thresholds
    velocity_threshold: float = 0.15
    velocity_window_hours: float = 1.0

    # Episode settings
    episode_break_hours: float = 4.0
    max_episode_duration_hours: float = 24.0

    # Break rule settings
    novelty_detection_enabled: bool = True
    velocity_override_enabled: bool = True
    tth_shortening_enabled: bool = True
    tth_shortening_threshold: float = 0.25  # 25% decrease triggers re-alert
    dwell_escalation_enabled: bool = True
    dwell_escalation_hours: float = 4.0  # S2+ for >4 hours without ack

    # Escalation timer
    escalation_timer_enabled: bool = True
    escalation_timer_s2_minutes: int = 60  # Auto-escalate S2 if no ack
    escalation_timer_s3_minutes: int = 15  # Auto-escalate S3 if no ack

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def get_cooldown_for_state(self, state_severity: int) -> int:
        """Get appropriate cooldown minutes for state severity (0-3)."""
        if state_severity >= 3:
            return self.critical_cooldown_minutes
        elif state_severity >= 2:
            return self.escalation_cooldown_minutes
        return self.default_cooldown_minutes


# Domain-specific configurations (ALL from hypercore + enhancements)
DOMAIN_CONFIGS: Dict[str, DomainConfig] = {
    # Sepsis - aggressive alerting due to rapid deterioration risk
    "sepsis": DomainConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=30,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=5,
        velocity_threshold=0.12,
        dwell_escalation_hours=2.0,  # More aggressive for sepsis
    ),

    # Cardiac - balanced thresholds
    "deterioration_cardiac": DomainConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=45,
        escalation_cooldown_minutes=15,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15,
    ),
    "cardiac": DomainConfig(  # Alias
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=45,
        escalation_cooldown_minutes=15,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15,
    ),

    # Kidney injury - slightly higher thresholds (slower progression typical)
    "kidney_injury": DomainConfig(
        s0_upper=0.35,
        s1_upper=0.60,
        s2_upper=0.82,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.10,
    ),
    "kidney": DomainConfig(  # Alias
        s0_upper=0.35,
        s1_upper=0.60,
        s2_upper=0.82,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.10,
    ),

    # Respiratory failure - aggressive like sepsis
    "respiratory_failure": DomainConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=30,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15,
    ),
    "respiratory": DomainConfig(  # Alias
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=30,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15,
    ),

    # Hepatic dysfunction - moderate alerting
    "hepatic_dysfunction": DomainConfig(
        s0_upper=0.32,
        s1_upper=0.58,
        s2_upper=0.82,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.12,
    ),
    "hepatic": DomainConfig(  # Alias
        s0_upper=0.32,
        s1_upper=0.58,
        s2_upper=0.82,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.12,
    ),

    # Neurological - aggressive due to time-sensitive nature
    "neurological": DomainConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=20,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=3,
        velocity_threshold=0.10,
        dwell_escalation_hours=1.0,  # Very aggressive for neuro
    ),

    # Metabolic - moderate alerting
    "metabolic": DomainConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=45,
        escalation_cooldown_minutes=15,
        critical_cooldown_minutes=10,
        velocity_threshold=0.15,
    ),

    # Hematologic - moderate alerting
    "hematologic": DomainConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=20,
        critical_cooldown_minutes=10,
        velocity_threshold=0.12,
    ),

    # Oncology inception - higher thresholds (requires pattern confirmation)
    "oncology_inception": DomainConfig(
        s0_upper=0.40,
        s1_upper=0.65,
        s2_upper=0.85,
        default_cooldown_minutes=120,
        escalation_cooldown_minutes=60,
        critical_cooldown_minutes=30,
        velocity_threshold=0.08,
        dwell_escalation_hours=24.0,  # Longer for oncology
    ),
    "oncology": DomainConfig(  # Alias
        s0_upper=0.40,
        s1_upper=0.65,
        s2_upper=0.85,
        default_cooldown_minutes=120,
        escalation_cooldown_minutes=60,
        critical_cooldown_minutes=30,
        velocity_threshold=0.08,
    ),

    # Multi-system - most aggressive (multiple organ involvement)
    "multi_system": DomainConfig(
        s0_upper=0.20,
        s1_upper=0.45,
        s2_upper=0.70,
        default_cooldown_minutes=20,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=3,
        velocity_threshold=0.10,
        dwell_escalation_hours=1.0,
    ),

    # New domains from cse.py
    "deterioration": DomainConfig(
        s0_upper=0.30,
        s1_upper=0.55,
        s2_upper=0.80,
        default_cooldown_minutes=45,
        escalation_cooldown_minutes=15,
        critical_cooldown_minutes=5,
        velocity_threshold=0.15,
    ),

    "infection": DomainConfig(
        s0_upper=0.25,
        s1_upper=0.50,
        s2_upper=0.75,
        default_cooldown_minutes=30,
        escalation_cooldown_minutes=10,
        critical_cooldown_minutes=5,
        velocity_threshold=0.12,
    ),

    "outbreak": DomainConfig(
        s0_upper=0.35,
        s1_upper=0.60,
        s2_upper=0.80,
        default_cooldown_minutes=60,
        escalation_cooldown_minutes=30,
        critical_cooldown_minutes=15,
        velocity_threshold=0.10,
    ),

    "trial_confounder": DomainConfig(
        s0_upper=0.40,
        s1_upper=0.65,
        s2_upper=0.85,
        default_cooldown_minutes=120,
        escalation_cooldown_minutes=60,
        critical_cooldown_minutes=30,
        velocity_threshold=0.08,
    ),

    "custom": DomainConfig(),  # Default values

    "unknown": DomainConfig(),  # Default values
}

# Default configuration
DEFAULT_CONFIG = DomainConfig()


def get_domain_config(domain: str) -> DomainConfig:
    """Get configuration for a specific clinical domain."""
    domain_lower = domain.lower().replace("-", "_").replace(" ", "_")
    return DOMAIN_CONFIGS.get(domain_lower, DEFAULT_CONFIG)


def get_biomarker_thresholds(domain: str) -> Dict[str, BiomarkerThreshold]:
    """Get biomarker thresholds for a domain."""
    domain_lower = domain.lower().replace("-", "_").replace(" ", "_")

    # Direct match
    if domain_lower in BIOMARKER_THRESHOLDS:
        return BIOMARKER_THRESHOLDS[domain_lower]

    # Try to extract base domain from compound names
    for key in BIOMARKER_THRESHOLDS:
        if key in domain_lower or domain_lower in key:
            return BIOMARKER_THRESHOLDS[key]

    # Default to sepsis (most comprehensive)
    return BIOMARKER_THRESHOLDS.get("sepsis", {})


# =============================================================================
# SITE CONFIGURATION (Per-Site Overrides)
# =============================================================================

@dataclass
class SiteConfig:
    """Per-site configuration overrides."""
    site_id: str
    site_name: str

    # Override domain configs
    domain_overrides: Dict[str, DomainConfig] = field(default_factory=dict)

    # Override biomarker thresholds
    biomarker_overrides: Dict[str, Dict[str, BiomarkerThreshold]] = field(default_factory=dict)

    # Routing configuration
    routing_rules: Dict[str, List[str]] = field(default_factory=dict)  # domain -> [clinician_ids]
    escalation_contacts: Dict[str, List[str]] = field(default_factory=dict)  # severity -> [contact_ids]

    # Feature flags
    realtime_push_enabled: bool = True
    websocket_enabled: bool = True
    sse_enabled: bool = True

    def get_domain_config(self, domain: str) -> DomainConfig:
        """Get domain config with site overrides."""
        base_config = get_domain_config(domain)
        override = self.domain_overrides.get(domain.lower())
        if override:
            return override
        return base_config

    def get_biomarker_thresholds(self, domain: str) -> Dict[str, BiomarkerThreshold]:
        """Get biomarker thresholds with site overrides."""
        base_thresholds = get_biomarker_thresholds(domain)
        overrides = self.biomarker_overrides.get(domain.lower(), {})

        # Merge overrides
        merged = dict(base_thresholds)
        merged.update(overrides)
        return merged


# Global site config (can be loaded from database/config file)
_current_site_config: Optional[SiteConfig] = None


def set_site_config(config: SiteConfig):
    """Set the current site configuration."""
    global _current_site_config
    _current_site_config = config


def get_site_config() -> Optional[SiteConfig]:
    """Get the current site configuration."""
    return _current_site_config


# =============================================================================
# CLINICAL RATIONALE TEMPLATES
# =============================================================================

RATIONALE_TEMPLATES: Dict[str, Dict[str, str]] = {
    "sepsis": {
        "S1": "Early infection markers detected - recommend enhanced monitoring",
        "S2": "Sepsis indicators escalating - consider cultures and source control",
        "S3": "Severe sepsis pattern - immediate sepsis bundle initiation recommended",
        "escalation": "Sepsis markers worsening: {drivers} trending up ({velocity}/hr)",
        "velocity": "Rapid inflammatory marker rise detected ({velocity}/hr)",
        "novelty": "New sepsis marker active: {new_markers} now contributing",
        "tth_shortening": "Time-to-harm decreased by {pct}% - accelerated deterioration",
        "dwell": "Sustained elevated state for {hours}h without acknowledgment",
    },
    "cardiac": {
        "S1": "Cardiac stress indicators elevated - monitor for ACS/HF",
        "S2": "Significant myocardial injury pattern - cardiology consultation advised",
        "S3": "Critical cardiac event likely - immediate evaluation required",
        "escalation": "Cardiac markers rising: {drivers} ({velocity}/hr)",
        "velocity": "Rapid cardiac marker elevation ({velocity}/hr)",
        "novelty": "New cardiac marker active: {new_markers}",
        "tth_shortening": "Cardiac trajectory accelerating - {pct}% faster deterioration",
        "dwell": "Persistent cardiac stress for {hours}h - requires assessment",
    },
    "kidney": {
        "S1": "Early renal function decline - optimize volume/nephrotoxin exposure",
        "S2": "Acute kidney injury developing - consider nephrology input",
        "S3": "Severe AKI - urgent renal support evaluation needed",
        "escalation": "Renal function declining: {drivers} ({velocity}/hr)",
        "velocity": "Rapid creatinine/urea rise ({velocity}/hr)",
        "novelty": "New renal marker involved: {new_markers}",
        "tth_shortening": "AKI progression accelerating - {pct}% faster",
        "dwell": "Sustained renal decline for {hours}h - nephrology consult",
    },
    "respiratory": {
        "S1": "Oxygenation impairment noted - assess respiratory status",
        "S2": "Respiratory failure developing - prepare for escalation",
        "S3": "Critical hypoxemia - immediate ventilatory support needed",
        "escalation": "Oxygenation worsening: {drivers} ({velocity}/hr)",
        "velocity": "Rapid respiratory decline ({velocity}/hr)",
        "novelty": "New respiratory marker: {new_markers}",
        "tth_shortening": "Respiratory decline accelerating - {pct}% faster",
        "dwell": "Persistent hypoxemia for {hours}h - RT evaluation",
    },
    "neurological": {
        "S1": "Neurological changes detected - frequent neuro checks",
        "S2": "Significant neurological decline - urgent evaluation needed",
        "S3": "Critical neurological event - immediate neurology/neurosurgery",
        "escalation": "Neuro status declining: {drivers} ({velocity}/hr)",
        "velocity": "Rapid neurological change ({velocity}/hr)",
        "novelty": "New neurological finding: {new_markers}",
        "tth_shortening": "Neuro deterioration accelerating - {pct}% faster",
        "dwell": "Sustained neuro decline for {hours}h - stat evaluation",
    },
    "multi_system": {
        "S1": "Multi-organ stress detected - comprehensive assessment needed",
        "S2": "Multi-organ dysfunction developing - ICU evaluation",
        "S3": "Multi-organ failure - immediate critical care intervention",
        "escalation": "Multiple systems declining: {drivers}",
        "velocity": "Rapid multi-system deterioration ({velocity}/hr)",
        "novelty": "Additional organ system involved: {new_markers}",
        "tth_shortening": "Multi-organ cascade accelerating - {pct}% faster",
        "dwell": "Sustained multi-system dysfunction for {hours}h - critical care",
    },
    "default": {
        "S1": "Risk indicators elevated - enhanced monitoring recommended",
        "S2": "Risk score escalating - clinical evaluation advised",
        "S3": "Critical risk level - immediate evaluation required",
        "escalation": "Risk markers worsening: {drivers} ({velocity}/hr)",
        "velocity": "Rapid risk increase ({velocity}/hr)",
        "novelty": "New risk factor identified: {new_markers}",
        "tth_shortening": "Deterioration accelerating - {pct}% faster progression",
        "dwell": "Elevated risk state sustained for {hours}h - requires review",
    },
}

SUGGESTED_ACTIONS: Dict[str, Dict[str, str]] = {
    "S0": "Continue routine monitoring.",
    "S1": "Increase monitoring frequency. Review recent labs and vitals.",
    "S2": "Evaluate patient bedside. Consider specialist consultation.",
    "S3": "Immediate bedside evaluation. Activate rapid response if appropriate.",
}

RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    "sepsis": {
        "immediate": [
            "Initiate Sepsis Bundle within 1 hour",
            "Obtain blood cultures before antibiotics",
            "Administer broad-spectrum antibiotics",
            "Begin fluid resuscitation (30 mL/kg crystalloid)",
            "Consider vasopressors if MAP < 65 after fluids",
        ],
        "urgent": [
            "Repeat lactate measurement in 2-4 hours",
            "Monitor urine output hourly",
            "Reassess fluid responsiveness",
            "Consider infectious disease consult",
        ],
        "monitor": [
            "Continue trending inflammatory markers",
            "Reassess in 4-6 hours",
            "Monitor for clinical deterioration",
        ],
    },
    "cardiac": {
        "immediate": [
            "Obtain 12-lead ECG stat",
            "Activate cardiac catheterization lab if STEMI",
            "Administer aspirin and anticoagulation",
            "Cardiology consult stat",
        ],
        "urgent": [
            "Serial troponins every 3-6 hours",
            "Continuous cardiac monitoring",
            "Echocardiogram within 24 hours",
            "Risk stratify with HEART or TIMI score",
        ],
        "monitor": [
            "Continue cardiac monitoring",
            "Trend cardiac biomarkers",
            "Optimize rate and rhythm control",
        ],
    },
    "respiratory": {
        "immediate": [
            "Prepare for intubation",
            "Apply high-flow oxygen or NIV",
            "Obtain ABG stat",
            "Chest X-ray and consider CT if PE suspected",
        ],
        "urgent": [
            "Increase oxygen supplementation",
            "Respiratory therapy evaluation",
            "Consider bronchodilators if wheezing",
            "Monitor closely for fatigue",
        ],
        "monitor": [
            "Continue SpO2 monitoring",
            "Incentive spirometry",
            "Mobilize if appropriate",
        ],
    },
    "kidney": {
        "immediate": [
            "Evaluate for obstruction (bladder scan, renal ultrasound)",
            "Hold nephrotoxic medications",
            "Nephrology consult for dialysis evaluation",
            "Assess volume status and optimize perfusion",
        ],
        "urgent": [
            "Strict I/O monitoring",
            "Avoid contrast if possible",
            "Renally dose all medications",
            "Trend creatinine every 6-12 hours",
        ],
        "monitor": [
            "Daily creatinine and electrolytes",
            "Maintain euvolemia",
            "Review medication list for nephrotoxins",
        ],
    },
}


def get_recommendations(domain: str, intervention_window: str) -> List[str]:
    """Get recommendations for domain and intervention window."""
    domain_lower = domain.lower()
    domain_recs = RECOMMENDATIONS.get(domain_lower, {})
    return domain_recs.get(intervention_window, [
        "Evaluate patient status",
        "Review recent data trends",
        "Consider specialist consultation if needed",
    ])
