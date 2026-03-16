"""
Surveillance Agent - Population Anomaly Detection

Connects to:
- Pathogen Detection Pipeline (WHO, CDC)
- Multi-omic Fusion Pipeline

Capabilities:
- Detect population-level anomalies
- Track outbreak patterns
- Monitor antimicrobial resistance trends
- Correlate environmental and clinical data
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from collections import defaultdict

from app.agents.base_agent import (
    BaseAgent,
    AgentType,
    AgentFinding,
    AgentRegistry,
)

logger = logging.getLogger(__name__)

# Import pathogen detection pipeline
try:
    from app.core.pathogen_detection import (
        get_pathogen_info,
        get_disease_pathogens,
        detect_outbreaks,
        analyze_amr,
        get_vaccination_coverage,
        search_surveillance,
        PATHOGEN_DATABASE,
        DISEASE_PATHOGEN_MAP,
    )
    PATHOGEN_AVAILABLE = True
except ImportError:
    PATHOGEN_AVAILABLE = False
    logger.warning("Pathogen detection pipeline not available")
    PATHOGEN_DATABASE = {}
    DISEASE_PATHOGEN_MAP = {}

# Import multi-omic for population data
try:
    from app.core.multiomic_fusion import get_source_status
    MULTIOMIC_AVAILABLE = True
except ImportError:
    MULTIOMIC_AVAILABLE = False


# Regional risk profiles
REGIONAL_PROFILES = {
    "north_america": {
        "endemic_diseases": ["influenza", "lyme_disease", "west_nile"],
        "amr_concerns": ["mrsa", "vre", "cre"],
        "vaccination_priorities": ["influenza", "pneumococcal", "shingles"],
    },
    "europe": {
        "endemic_diseases": ["influenza", "tick_borne_encephalitis"],
        "amr_concerns": ["mrsa", "esbl", "cre"],
        "vaccination_priorities": ["influenza", "pneumococcal", "measles"],
    },
    "asia": {
        "endemic_diseases": ["dengue", "malaria", "tuberculosis", "japanese_encephalitis"],
        "amr_concerns": ["mdr_tb", "esbl", "carbapenem_resistance"],
        "vaccination_priorities": ["hepatitis_b", "japanese_encephalitis", "typhoid"],
    },
    "africa": {
        "endemic_diseases": ["malaria", "tuberculosis", "hiv", "cholera", "ebola"],
        "amr_concerns": ["mdr_tb", "chloroquine_resistance"],
        "vaccination_priorities": ["measles", "polio", "yellow_fever"],
    },
    "south_america": {
        "endemic_diseases": ["dengue", "zika", "chikungunya", "yellow_fever"],
        "amr_concerns": ["esbl", "carbapenem_resistance"],
        "vaccination_priorities": ["yellow_fever", "hepatitis_a", "typhoid"],
    },
}

# Outbreak severity thresholds
OUTBREAK_THRESHOLDS = {
    "critical": {"min_deviation": 3.0, "min_cases": 100, "trend": "increasing"},
    "high": {"min_deviation": 2.0, "min_cases": 50, "trend": "increasing"},
    "moderate": {"min_deviation": 1.5, "min_cases": 20, "trend": "any"},
    "low": {"min_deviation": 1.0, "min_cases": 10, "trend": "any"},
}


class SurveillanceAgent(BaseAgent):
    """
    Population anomaly detection agent.

    Monitors epidemiological data for outbreak patterns,
    AMR trends, and population health anomalies.
    """

    def __init__(self):
        super().__init__(AgentType.SURVEILLANCE)
        self._outbreak_cache: Dict[str, Any] = {}
        self._amr_cache: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "Population Surveillance Intelligence"

    @property
    def description(self) -> str:
        return "Detects population-level anomalies and outbreak patterns"

    @property
    def capabilities(self) -> List[str]:
        return [
            "outbreak_detection",
            "amr_trend_analysis",
            "vaccination_coverage_monitoring",
            "regional_risk_assessment",
            "epidemic_forecasting",
        ]

    async def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze population surveillance data.

        Input schema:
        {
            "region": "north_america",
            "pathogens": ["influenza", "sars_cov_2"],
            "time_period": {"years": 5},
            "include_amr": true,
            "include_vaccination": true,
            "patient_location": "United States",
            "correlation_id": "session_123"
        }
        """
        correlation_id = input_data.get("correlation_id")
        findings = []

        region = input_data.get("region", "").lower().replace(" ", "_")
        pathogens = input_data.get("pathogens", [])
        time_period = input_data.get("time_period", {"years": 5})
        include_amr = input_data.get("include_amr", True)
        include_vaccination = input_data.get("include_vaccination", True)
        patient_location = input_data.get("patient_location")

        # 1. Detect active outbreaks
        outbreak_findings = await self._detect_outbreaks(
            region,
            time_period.get("years", 5)
        )
        findings.extend(outbreak_findings)

        # 2. Analyze AMR patterns if requested
        if include_amr:
            amr_findings = await self._analyze_amr(pathogens, patient_location)
            findings.extend(amr_findings)

        # 3. Check vaccination coverage
        if include_vaccination:
            vax_findings = await self._check_vaccination(region, patient_location)
            findings.extend(vax_findings)

        # 4. Get regional risk profile
        if region:
            risk_findings = self._assess_regional_risk(region)
            findings.extend(risk_findings)

        # 5. Analyze specific pathogens
        if pathogens:
            pathogen_findings = await self._analyze_pathogens(pathogens)
            findings.extend(pathogen_findings)

        # 6. Integrate with peer agent findings
        peer_findings = self.get_peer_findings(min_confidence=0.6)
        diagnostic_findings = [f for f in peer_findings
                              if f.agent_type == AgentType.DIAGNOSTIC]

        if diagnostic_findings:
            correlation_findings = self._correlate_with_diagnostics(
                diagnostic_findings,
                findings
            )
            findings.extend(correlation_findings)

        # 7. Broadcast findings
        if findings:
            self.broadcast_findings(findings, correlation_id)

        # 8. Alert on critical outbreaks
        critical = [f for f in findings if "critical" in f.category.lower()]
        if critical:
            self.send_alert(
                alert_message=f"Critical surveillance alert: {len(critical)} items",
                severity="critical",
                related_findings=critical
            )

        return {
            "agent": self.name,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "region": region,
            "pathogens_analyzed": pathogens,
            "findings": [f.to_dict() for f in findings],
            "outbreak_alerts": len([f for f in findings if "outbreak" in f.category]),
            "amr_alerts": len([f for f in findings if "amr" in f.category]),
            "correlation_id": correlation_id,
            "pipeline_available": PATHOGEN_AVAILABLE,
        }

    async def _detect_outbreaks(
        self,
        region: Optional[str],
        lookback_years: int
    ) -> List[AgentFinding]:
        """Detect active outbreaks from surveillance data."""
        findings = []

        if not PATHOGEN_AVAILABLE:
            return findings

        try:
            regions = [region] if region else None
            result = detect_outbreaks(
                regions=regions,
                threshold_multiplier=1.5,
                lookback_years=lookback_years
            )

            alerts = result.get("alerts", [])

            for alert in alerts[:10]:  # Top 10 alerts
                level = alert.get("alert_level", "low")
                confidence = {
                    "critical": 0.95,
                    "high": 0.85,
                    "moderate": 0.7,
                    "low": 0.5
                }.get(level, 0.5)

                category = f"outbreak_{level}"

                finding = self.create_finding(
                    category=category,
                    description=f"{level.upper()} outbreak alert: {alert.get('indicator', 'Unknown')} in {alert.get('country', 'Unknown')}",
                    confidence=confidence,
                    evidence=[
                        f"Indicator: {alert.get('indicator', 'Unknown')}",
                        f"Region: {alert.get('country', 'Unknown')}",
                        f"Current value: {alert.get('current_value', 0):.2f}",
                        f"Historical mean: {alert.get('historical_mean', 0):.2f}",
                        f"Deviation: {alert.get('deviation_std', 0):.2f} std",
                        f"Trend: {alert.get('trend', 'unknown')}",
                    ],
                    related_entities={
                        "indicator": alert.get("indicator"),
                        "country": alert.get("country"),
                        "alert_level": level,
                        "deviation": alert.get("deviation_std"),
                        "year": alert.get("year"),
                    }
                )
                findings.append(finding)

                # Cache for later reference
                self._outbreak_cache[f"{alert.get('indicator')}_{alert.get('country')}"] = alert

        except Exception as e:
            logger.error(f"Error detecting outbreaks: {e}")

        return findings

    async def _analyze_amr(
        self,
        pathogens: List[str],
        country: Optional[str]
    ) -> List[AgentFinding]:
        """Analyze antimicrobial resistance patterns."""
        findings = []

        if not PATHOGEN_AVAILABLE:
            return findings

        try:
            for pathogen in pathogens[:3]:  # Limit searches
                result = analyze_amr(pathogen=pathogen, country=country)

                high_risk = result.get("high_risk_pathogens", [])

                for risk in high_risk:
                    finding = self.create_finding(
                        category="amr_concern",
                        description=f"AMR concern: {risk.get('pathogen', pathogen)} - {', '.join(risk.get('key_resistances', [])[:3])}",
                        confidence=0.8,
                        evidence=[
                            f"Pathogen: {risk.get('pathogen', pathogen)}",
                            f"Common name: {risk.get('common_name', 'Unknown')}",
                            f"Key resistances: {', '.join(risk.get('key_resistances', []))}",
                            f"Associated diseases: {', '.join(risk.get('diseases', [])[:3])}",
                        ],
                        related_entities={
                            "pathogen": risk.get("pathogen"),
                            "resistances": risk.get("key_resistances", []),
                            "diseases": risk.get("diseases", []),
                        }
                    )
                    findings.append(finding)

                    # Cache for later
                    self._amr_cache[risk.get("pathogen", pathogen)] = risk

        except Exception as e:
            logger.error(f"Error analyzing AMR: {e}")

        return findings

    async def _check_vaccination(
        self,
        region: Optional[str],
        country: Optional[str]
    ) -> List[AgentFinding]:
        """Check vaccination coverage data."""
        findings = []

        if not PATHOGEN_AVAILABLE:
            return findings

        try:
            result = get_vaccination_coverage(country=country)
            vax_data = result.get("vaccination_data", [])

            if vax_data:
                # Group by country/category
                by_country = defaultdict(list)
                for record in vax_data[:50]:
                    by_country[record.get("country", "unknown")].append(record)

                # Summarize coverage
                for country_code, records in list(by_country.items())[:5]:
                    yes_count = sum(1 for r in records if r.get("value") == "YES")
                    total = len(records)

                    if total > 0:
                        coverage_pct = (yes_count / total) * 100

                        category = "vaccination_coverage"
                        if coverage_pct < 50:
                            category = "vaccination_gap"

                        finding = self.create_finding(
                            category=category,
                            description=f"Vaccination coverage in {country_code}: {coverage_pct:.1f}% of tracked programs active",
                            confidence=0.75,
                            evidence=[
                                f"Country: {country_code}",
                                f"Programs tracked: {total}",
                                f"Active programs: {yes_count}",
                                f"Coverage rate: {coverage_pct:.1f}%",
                            ],
                            related_entities={
                                "country": country_code,
                                "active_programs": yes_count,
                                "total_programs": total,
                                "coverage_percentage": coverage_pct,
                            }
                        )
                        findings.append(finding)

        except Exception as e:
            logger.error(f"Error checking vaccination: {e}")

        return findings

    def _assess_regional_risk(self, region: str) -> List[AgentFinding]:
        """Assess regional disease risk profile."""
        findings = []

        if region in REGIONAL_PROFILES:
            profile = REGIONAL_PROFILES[region]

            finding = self.create_finding(
                category="regional_risk_profile",
                description=f"Regional profile for {region.replace('_', ' ').title()}",
                confidence=0.85,
                evidence=[
                    f"Endemic diseases: {', '.join(profile.get('endemic_diseases', []))}",
                    f"AMR concerns: {', '.join(profile.get('amr_concerns', []))}",
                    f"Vaccination priorities: {', '.join(profile.get('vaccination_priorities', []))}",
                ],
                related_entities={
                    "region": region,
                    "endemic_diseases": profile.get("endemic_diseases", []),
                    "amr_concerns": profile.get("amr_concerns", []),
                    "vaccination_priorities": profile.get("vaccination_priorities", []),
                }
            )
            findings.append(finding)

        return findings

    async def _analyze_pathogens(self, pathogens: List[str]) -> List[AgentFinding]:
        """Analyze specific pathogens."""
        findings = []

        if not PATHOGEN_AVAILABLE:
            return findings

        for pathogen in pathogens:
            try:
                info = get_pathogen_info(pathogen)

                if info.get("found"):
                    amr_status = "HIGH RISK" if info.get("amr_concern") else "Standard"

                    finding = self.create_finding(
                        category="pathogen_profile",
                        description=f"{info.get('common_name', pathogen)}: {amr_status} - {', '.join(info.get('diseases', [])[:3])}",
                        confidence=0.9,
                        evidence=[
                            f"Pathogen: {pathogen}",
                            f"Type: {info.get('type', 'unknown')}",
                            f"Diseases: {', '.join(info.get('diseases', []))}",
                            f"AMR concern: {info.get('amr_concern', False)}",
                            f"Key resistances: {', '.join(info.get('key_resistances', []))}",
                        ],
                        related_entities={
                            "pathogen": pathogen,
                            "type": info.get("type"),
                            "diseases": info.get("diseases", []),
                            "amr_concern": info.get("amr_concern"),
                            "icd10_codes": info.get("icd10_codes", []),
                        }
                    )
                    findings.append(finding)

            except Exception as e:
                logger.error(f"Error analyzing pathogen {pathogen}: {e}")

        return findings

    def _correlate_with_diagnostics(
        self,
        diagnostic_findings: List[AgentFinding],
        surveillance_findings: List[AgentFinding]
    ) -> List[AgentFinding]:
        """Correlate surveillance data with diagnostic findings."""
        correlation_findings = []

        for diag in diagnostic_findings:
            diagnosis = diag.related_entities.get("diagnosis", "")

            # Check if any pathogens are relevant
            relevant_pathogens = []
            for key, pathogens in DISEASE_PATHOGEN_MAP.items():
                if key in diagnosis.lower():
                    relevant_pathogens.extend(pathogens)

            if relevant_pathogens:
                # Check if we have surveillance data for these pathogens
                matching_surveillance = []
                for surv in surveillance_findings:
                    pathogen = surv.related_entities.get("pathogen", "")
                    if pathogen in relevant_pathogens:
                        matching_surveillance.append(surv)

                if matching_surveillance:
                    confidence = self.calculate_aggregate_confidence(
                        [diag.confidence] + [s.confidence for s in matching_surveillance]
                    )

                    finding = self.create_finding(
                        category="diagnostic_surveillance_correlation",
                        description=f"Surveillance data supports diagnosis: {diagnosis}",
                        confidence=confidence,
                        evidence=[
                            f"Diagnosis: {diagnosis}",
                            f"Related pathogens: {', '.join(relevant_pathogens[:3])}",
                            f"Matching surveillance alerts: {len(matching_surveillance)}",
                        ],
                        related_entities={
                            "diagnosis": diagnosis,
                            "pathogens": relevant_pathogens,
                            "surveillance_matches": len(matching_surveillance),
                        }
                    )
                    correlation_findings.append(finding)

        return correlation_findings

    def get_regional_profile(self, region: str) -> Dict[str, Any]:
        """Get risk profile for a region."""
        region_key = region.lower().replace(" ", "_")
        if region_key in REGIONAL_PROFILES:
            return {
                "region": region,
                "profile": REGIONAL_PROFILES[region_key]
            }
        return {"error": f"Unknown region: {region}"}

    def list_regions(self) -> List[str]:
        """List available regional profiles."""
        return list(REGIONAL_PROFILES.keys())


# Singleton instance
_surveillance_agent = None


def get_surveillance_agent() -> SurveillanceAgent:
    """Get singleton surveillance agent instance."""
    global _surveillance_agent
    if _surveillance_agent is None:
        _surveillance_agent = SurveillanceAgent()
    return _surveillance_agent
