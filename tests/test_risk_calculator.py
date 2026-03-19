"""
Tests for the Risk Calculator module.

Tests automatic risk score calculation from raw biomarker values.
"""

import pytest
from app.core.alert_system.risk_calculator import (
    calculate_risk_score,
    quick_risk_score,
    calculate_biomarker_score,
    normalize_biomarker_name,
    get_domain_thresholds,
    BIOMARKER_ALIASES,
)
from app.core.alert_system.config import BiomarkerThreshold


# =============================================================================
# BIOMARKER NAME NORMALIZATION TESTS
# =============================================================================

class TestNormalizeBiomarkerName:
    """Tests for normalize_biomarker_name function."""

    def test_lowercase_conversion(self):
        """Test that names are lowercased."""
        assert normalize_biomarker_name("LACTATE") == "lactate"
        assert normalize_biomarker_name("WBC") == "wbc"
        assert normalize_biomarker_name("CRP") == "crp"

    def test_alias_mapping(self):
        """Test common biomarker aliases."""
        assert normalize_biomarker_name("white_blood_cell") == "wbc"
        assert normalize_biomarker_name("c_reactive_protein") == "crp"
        assert normalize_biomarker_name("lactic_acid") == "lactate"
        assert normalize_biomarker_name("heart_rate") == "heart_rate"
        assert normalize_biomarker_name("HR") == "heart_rate"
        assert normalize_biomarker_name("pulse") == "heart_rate"

    def test_hyphen_underscore_normalization(self):
        """Test that hyphens are converted to underscores."""
        assert normalize_biomarker_name("c-reactive-protein") == "crp"
        assert normalize_biomarker_name("nt-probnp") == "nt_probnp"

    def test_space_normalization(self):
        """Test that spaces are converted to underscores."""
        assert normalize_biomarker_name("heart rate") == "heart_rate"
        assert normalize_biomarker_name("white blood cell") == "wbc"

    def test_unknown_biomarker_passthrough(self):
        """Test that unknown biomarkers pass through normalized."""
        assert normalize_biomarker_name("unknown_marker") == "unknown_marker"
        assert normalize_biomarker_name("CUSTOM_VALUE") == "custom_value"

    def test_cardiac_aliases(self):
        """Test cardiac biomarker aliases."""
        assert normalize_biomarker_name("troponin_i") == "troponin_i"
        assert normalize_biomarker_name("tropi") == "troponin_i"
        assert normalize_biomarker_name("bnp") == "bnp"
        assert normalize_biomarker_name("nt_probnp") == "nt_probnp"
        assert normalize_biomarker_name("ntprobnp") == "nt_probnp"

    def test_kidney_aliases(self):
        """Test kidney biomarker aliases."""
        assert normalize_biomarker_name("creatinine") == "creatinine"
        assert normalize_biomarker_name("creat") == "creatinine"
        assert normalize_biomarker_name("bun") == "bun"
        assert normalize_biomarker_name("blood_urea_nitrogen") == "bun"
        assert normalize_biomarker_name("gfr") == "gfr"
        assert normalize_biomarker_name("egfr") == "gfr"

    def test_respiratory_aliases(self):
        """Test respiratory biomarker aliases."""
        assert normalize_biomarker_name("spo2") == "spo2"
        assert normalize_biomarker_name("oxygen_saturation") == "spo2"
        assert normalize_biomarker_name("o2_sat") == "spo2"
        assert normalize_biomarker_name("respiratory_rate") == "respiratory_rate"
        assert normalize_biomarker_name("rr") == "respiratory_rate"


# =============================================================================
# BIOMARKER SCORE CALCULATION TESTS
# =============================================================================

class TestCalculateBiomarkerScore:
    """Tests for calculate_biomarker_score function."""

    def test_rising_normal_value(self):
        """Test normal value for rising biomarker."""
        threshold = BiomarkerThreshold(
            warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=1.0
        )
        score, level = calculate_biomarker_score(1.0, threshold)
        assert level == "normal"
        assert score < 0.3

    def test_rising_warning_value(self):
        """Test warning value for rising biomarker."""
        threshold = BiomarkerThreshold(
            warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=1.0
        )
        score, level = calculate_biomarker_score(3.0, threshold)
        assert level == "warning"
        assert 0.5 <= score < 1.0

    def test_rising_critical_value(self):
        """Test critical value for rising biomarker."""
        threshold = BiomarkerThreshold(
            warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=1.0
        )
        score, level = calculate_biomarker_score(5.0, threshold)
        assert level == "critical"
        assert score == 1.0

    def test_falling_normal_value(self):
        """Test normal value for falling biomarker (like SpO2)."""
        threshold = BiomarkerThreshold(
            warning=94.0, critical=90.0, unit="%", direction="falling", weight=1.0
        )
        score, level = calculate_biomarker_score(98.0, threshold)
        assert level == "normal"
        assert score < 0.3

    def test_falling_warning_value(self):
        """Test warning value for falling biomarker."""
        threshold = BiomarkerThreshold(
            warning=94.0, critical=90.0, unit="%", direction="falling", weight=1.0
        )
        score, level = calculate_biomarker_score(92.0, threshold)
        assert level == "warning"
        assert 0.5 <= score < 1.0

    def test_falling_critical_value(self):
        """Test critical value for falling biomarker."""
        threshold = BiomarkerThreshold(
            warning=94.0, critical=90.0, unit="%", direction="falling", weight=1.0
        )
        score, level = calculate_biomarker_score(88.0, threshold)
        assert level == "critical"
        assert score == 1.0

    def test_exactly_at_warning_threshold(self):
        """Test value exactly at warning threshold."""
        threshold = BiomarkerThreshold(
            warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=1.0
        )
        score, level = calculate_biomarker_score(2.0, threshold)
        assert level == "warning"
        assert score == 0.5

    def test_exactly_at_critical_threshold(self):
        """Test value exactly at critical threshold."""
        threshold = BiomarkerThreshold(
            warning=2.0, critical=4.0, unit="mmol/L", direction="rising", weight=1.0
        )
        score, level = calculate_biomarker_score(4.0, threshold)
        assert level == "critical"
        assert score == 1.0


# =============================================================================
# SEPSIS DOMAIN TESTS
# =============================================================================

class TestSepsisRiskCalculation:
    """Tests for sepsis domain risk calculation."""

    def test_normal_sepsis_values(self):
        """Test normal sepsis biomarkers return low score."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 1.0,
                "WBC": 7.0,
                "CRP": 5.0,
            }
        )
        assert result["risk_score"] < 0.25
        assert len(result["critical_biomarkers"]) == 0
        assert len(result["warning_biomarkers"]) == 0

    def test_warning_sepsis_values(self):
        """Test warning sepsis biomarkers return moderate score."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 2.5,  # Warning: >2.0
                "WBC": 11.0,     # Warning: >10.0
                "CRP": 75.0,     # Warning: >50.0
            }
        )
        assert 0.25 <= result["risk_score"] < 0.75
        assert len(result["warning_biomarkers"]) >= 2

    def test_critical_sepsis_values(self):
        """Test critical sepsis biomarkers return high score."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 5.0,       # Critical: >4.0
                "WBC": 16.0,          # Critical: >12.0
                "CRP": 150.0,         # Critical: >100.0
                "procalcitonin": 5.0, # Critical: >2.0
            }
        )
        assert result["risk_score"] >= 0.75
        assert len(result["critical_biomarkers"]) >= 3

    def test_sepsis_with_vitals(self):
        """Test sepsis with vital signs included."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 4.5,
                "WBC": 15.0,
            },
            vital_signs={
                "heart_rate": 125,      # Critical: >120
                "temperature": 39.0,    # Critical: >38.3
                "respiratory_rate": 28, # Critical: >24
            }
        )
        assert result["risk_score"] >= 0.7
        assert "lactate" in result["critical_biomarkers"] or "WBC" in result["critical_biomarkers"]


# =============================================================================
# CARDIAC DOMAIN TESTS
# =============================================================================

class TestCardiacRiskCalculation:
    """Tests for cardiac domain risk calculation."""

    def test_normal_cardiac_values(self):
        """Test normal cardiac biomarkers return low score."""
        result = calculate_risk_score(
            risk_domain="cardiac",
            lab_data={
                "troponin": 0.005,
                "bnp": 50.0,
            },
            vital_signs={
                "heart_rate": 75,
            }
        )
        assert result["risk_score"] < 0.25

    def test_critical_cardiac_values(self):
        """Test critical cardiac biomarkers return high score."""
        result = calculate_risk_score(
            risk_domain="cardiac",
            lab_data={
                "troponin": 0.1,   # Critical: >0.04
                "bnp": 500.0,      # Critical: >400
            },
            vital_signs={
                "heart_rate": 140, # Critical: >130
            }
        )
        assert result["risk_score"] >= 0.7
        assert len(result["critical_biomarkers"]) >= 2

    def test_cardiac_with_low_bp(self):
        """Test cardiac with falling blood pressure."""
        result = calculate_risk_score(
            risk_domain="cardiac",
            lab_data={
                "troponin": 0.05,
            },
            vital_signs={
                "systolic_bp": 85,  # Critical: <90
            }
        )
        assert result["risk_score"] >= 0.5
        assert "systolic_bp" in result["critical_biomarkers"]


# =============================================================================
# KIDNEY DOMAIN TESTS
# =============================================================================

class TestKidneyRiskCalculation:
    """Tests for kidney domain risk calculation."""

    def test_normal_kidney_values(self):
        """Test normal kidney biomarkers return low score."""
        result = calculate_risk_score(
            risk_domain="kidney",
            lab_data={
                "creatinine": 0.9,
                "bun": 15.0,
                "potassium": 4.0,
            }
        )
        assert result["risk_score"] < 0.25

    def test_critical_kidney_values(self):
        """Test critical kidney biomarkers return high score."""
        result = calculate_risk_score(
            risk_domain="kidney",
            lab_data={
                "creatinine": 3.0,  # Critical: >2.0
                "bun": 50.0,        # Critical: >40
                "potassium": 6.0,   # Critical: >5.5
            }
        )
        assert result["risk_score"] >= 0.75
        assert len(result["critical_biomarkers"]) >= 2

    def test_kidney_with_low_gfr(self):
        """Test kidney with falling GFR."""
        result = calculate_risk_score(
            risk_domain="kidney",
            lab_data={
                "creatinine": 2.5,
                "gfr": 25.0,  # Critical: <30 (falling direction)
            }
        )
        assert result["risk_score"] >= 0.6


# =============================================================================
# RESPIRATORY DOMAIN TESTS
# =============================================================================

class TestRespiratoryRiskCalculation:
    """Tests for respiratory domain risk calculation."""

    def test_normal_respiratory_values(self):
        """Test normal respiratory biomarkers return low score."""
        result = calculate_risk_score(
            risk_domain="respiratory",
            vital_signs={
                "spo2": 98,
                "respiratory_rate": 14,
            }
        )
        assert result["risk_score"] < 0.25

    def test_critical_respiratory_values(self):
        """Test critical respiratory biomarkers return high score."""
        result = calculate_risk_score(
            risk_domain="respiratory",
            lab_data={
                "pao2": 55.0,  # Critical: <60
            },
            vital_signs={
                "spo2": 88,             # Critical: <90
                "respiratory_rate": 32, # Critical: >30
            }
        )
        assert result["risk_score"] >= 0.7
        assert "spo2" in result["critical_biomarkers"] or "SpO2" in [b.lower() for b in result["critical_biomarkers"]]


# =============================================================================
# HEPATIC DOMAIN TESTS
# =============================================================================

class TestHepaticRiskCalculation:
    """Tests for hepatic domain risk calculation."""

    def test_normal_hepatic_values(self):
        """Test normal hepatic biomarkers return low score."""
        result = calculate_risk_score(
            risk_domain="hepatic",
            lab_data={
                "alt": 30.0,
                "ast": 25.0,
                "bilirubin": 0.8,
            }
        )
        assert result["risk_score"] < 0.25

    def test_critical_hepatic_values(self):
        """Test critical hepatic biomarkers return high score."""
        result = calculate_risk_score(
            risk_domain="hepatic",
            lab_data={
                "alt": 1500.0,     # Critical: >1000
                "ast": 1200.0,     # Critical: >1000
                "bilirubin": 6.0,  # Critical: >4.0
                "inr": 2.5,        # Critical: >2.0
            }
        )
        assert result["risk_score"] >= 0.75
        assert len(result["critical_biomarkers"]) >= 3


# =============================================================================
# METABOLIC DOMAIN TESTS
# =============================================================================

class TestMetabolicRiskCalculation:
    """Tests for metabolic domain risk calculation."""

    def test_normal_metabolic_values(self):
        """Test normal metabolic biomarkers return low score."""
        result = calculate_risk_score(
            risk_domain="metabolic",
            lab_data={
                "glucose": 100.0,
                "ph": 7.40,
                "lactate": 1.0,
            }
        )
        assert result["risk_score"] < 0.25

    def test_critical_high_glucose(self):
        """Test critical high glucose."""
        result = calculate_risk_score(
            risk_domain="metabolic",
            lab_data={
                "glucose": 500.0,  # Critical: >400
            }
        )
        assert result["risk_score"] >= 0.5
        assert "glucose" in result["critical_biomarkers"]

    def test_critical_acidosis(self):
        """Test critical low pH (acidosis)."""
        result = calculate_risk_score(
            risk_domain="metabolic",
            lab_data={
                "ph": 7.15,      # Critical: <7.25
                "lactate": 5.0,  # Critical: >4.0
            }
        )
        assert result["risk_score"] >= 0.6


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self):
        """Test with no data provided."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data=None,
            vital_signs=None,
        )
        assert result["risk_score"] == 0.0
        assert result["calculation_method"] == "no_data"

    def test_empty_dict(self):
        """Test with empty dictionaries."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={},
            vital_signs={},
        )
        assert result["risk_score"] == 0.0

    def test_unknown_biomarkers_only(self):
        """Test with only unknown biomarkers."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "unknown_marker_1": 100.0,
                "custom_value_xyz": 50.0,
            }
        )
        assert result["risk_score"] == 0.0
        assert result["matched_biomarkers"] == 0

    def test_mixed_known_unknown_biomarkers(self):
        """Test with mix of known and unknown biomarkers."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 5.0,          # Known, critical
                "unknown_marker": 100.0,  # Unknown
            }
        )
        assert result["risk_score"] > 0
        assert result["matched_biomarkers"] == 1

    def test_non_numeric_values_ignored(self):
        """Test that non-numeric values are ignored."""
        result = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 5.0,
                "notes": "Patient is febrile",
                "status": None,
            }
        )
        assert result["matched_biomarkers"] == 1

    def test_unknown_domain_uses_fallback(self):
        """Test unknown domain uses sepsis as fallback."""
        result = calculate_risk_score(
            risk_domain="unknown_domain_xyz",
            lab_data={
                "lactate": 5.0,
            }
        )
        # Should still work because sepsis thresholds are used as fallback
        assert result["risk_score"] > 0

    def test_case_insensitive_domain(self):
        """Test domain name is case insensitive."""
        result1 = calculate_risk_score(
            risk_domain="SEPSIS",
            lab_data={"lactate": 5.0}
        )
        result2 = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={"lactate": 5.0}
        )
        assert result1["risk_score"] == result2["risk_score"]


# =============================================================================
# QUICK RISK SCORE HELPER
# =============================================================================

class TestQuickRiskScore:
    """Tests for quick_risk_score helper function."""

    def test_returns_float(self):
        """Test quick_risk_score returns a float."""
        score = quick_risk_score(
            risk_domain="sepsis",
            lab_data={"lactate": 3.0}
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_matches_full_calculation(self):
        """Test quick_risk_score matches calculate_risk_score."""
        lab_data = {"lactate": 4.5, "WBC": 15.0}

        quick = quick_risk_score("sepsis", lab_data)
        full = calculate_risk_score("sepsis", lab_data)

        assert quick == full["risk_score"]


# =============================================================================
# DOMAIN THRESHOLDS
# =============================================================================

class TestGetDomainThresholds:
    """Tests for get_domain_thresholds function."""

    def test_sepsis_thresholds_exist(self):
        """Test sepsis domain has thresholds."""
        thresholds = get_domain_thresholds("sepsis")
        assert "lactate" in thresholds
        assert "wbc" in thresholds
        assert "crp" in thresholds

    def test_cardiac_thresholds_exist(self):
        """Test cardiac domain has thresholds."""
        thresholds = get_domain_thresholds("cardiac")
        assert "troponin" in thresholds
        assert "bnp" in thresholds

    def test_kidney_thresholds_exist(self):
        """Test kidney domain has thresholds."""
        thresholds = get_domain_thresholds("kidney")
        assert "creatinine" in thresholds
        assert "bun" in thresholds

    def test_unknown_domain_returns_fallback(self):
        """Test unknown domain returns sepsis as fallback."""
        thresholds = get_domain_thresholds("unknown_xyz")
        assert "lactate" in thresholds  # Sepsis marker in fallback


# =============================================================================
# WEIGHT FACTORS
# =============================================================================

class TestWeightFactors:
    """Tests for biomarker weight factors."""

    def test_higher_weight_increases_contribution(self):
        """Test that higher weight biomarkers contribute more."""
        # Lactate has weight 1.0, temperature has weight 0.5
        result_lactate = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={"lactate": 5.0}  # Critical, weight 1.0
        )
        result_temp = calculate_risk_score(
            risk_domain="sepsis",
            vital_signs={"temperature": 39.5}  # Critical, weight 0.5
        )

        # Both critical, but lactate should have higher score due to weight
        assert result_lactate["risk_score"] >= result_temp["risk_score"]

    def test_multiple_critical_boosts_score(self):
        """Test that multiple critical values boost the score."""
        result_single = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={"lactate": 5.0}
        )
        result_multiple = calculate_risk_score(
            risk_domain="sepsis",
            lab_data={
                "lactate": 5.0,
                "WBC": 16.0,
                "CRP": 150.0,
            }
        )

        # Multiple criticals should generally result in higher score
        assert result_multiple["risk_score"] >= result_single["risk_score"]
