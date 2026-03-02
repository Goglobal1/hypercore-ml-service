# test_domain_classifier.py
"""
Tests for Domain Discovery Engine

Run with: pytest tests/test_domain_classifier.py -v
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.domain_classifier import (
    classify_domains,
    get_primary_domain,
    get_domain_from_endpoint,
    ClinicalDomain,
    DOMAIN_SIGNATURES
)


class TestSepsisClassification:
    """Test sepsis domain classification."""

    def test_lactate_crp_wbc_classified_as_sepsis(self):
        domains = classify_domains(["lactate", "crp", "wbc"])

        assert len(domains) >= 1
        sepsis_domains = [d for d in domains if d["domain"] == "sepsis"]
        assert len(sepsis_domains) == 1
        assert sepsis_domains[0]["confidence"] >= 0.6

    def test_procalcitonin_triggers_sepsis(self):
        domains = classify_domains(["procalcitonin", "temperature", "heart_rate"])

        sepsis_domains = [d for d in domains if d["domain"] == "sepsis"]
        assert len(sepsis_domains) == 1
        assert "procalcitonin" in sepsis_domains[0]["primary_drivers"]

    def test_il6_sepsis_marker(self):
        domains = classify_domains(["il6", "sofa_score"])

        sepsis_domains = [d for d in domains if d["domain"] == "sepsis"]
        assert len(sepsis_domains) == 1


class TestCardiacClassification:
    """Test cardiac domain classification."""

    def test_troponin_bnp_cardiac(self):
        domains = classify_domains(["troponin", "bnp", "chest_pain"])

        cardiac_domains = [d for d in domains if d["domain"] == "deterioration_cardiac"]
        assert len(cardiac_domains) == 1
        assert cardiac_domains[0]["confidence"] >= 0.65

    def test_nt_probnp_triggers_cardiac(self):
        domains = classify_domains(["nt_probnp", "dyspnea"])

        cardiac_domains = [d for d in domains if d["domain"] == "deterioration_cardiac"]
        assert len(cardiac_domains) == 1

    def test_ck_mb_cardiac_marker(self):
        domains = classify_domains(["ck_mb", "myoglobin"])

        cardiac_domains = [d for d in domains if d["domain"] == "deterioration_cardiac"]
        assert len(cardiac_domains) == 1


class TestKidneyClassification:
    """Test kidney injury domain classification."""

    def test_creatinine_bun_kidney(self):
        domains = classify_domains(["creatinine", "bun", "potassium"])

        kidney_domains = [d for d in domains if d["domain"] == "kidney_injury"]
        assert len(kidney_domains) == 1
        assert kidney_domains[0]["confidence"] >= 0.65

    def test_egfr_kidney_marker(self):
        domains = classify_domains(["egfr", "cystatin_c"])

        kidney_domains = [d for d in domains if d["domain"] == "kidney_injury"]
        assert len(kidney_domains) == 1

    def test_oliguria_kidney(self):
        domains = classify_domains(["oliguria", "creatinine"])

        kidney_domains = [d for d in domains if d["domain"] == "kidney_injury"]
        assert len(kidney_domains) == 1


class TestRespiratoryClassification:
    """Test respiratory failure domain classification."""

    def test_pao2_fio2_respiratory(self):
        domains = classify_domains(["pao2", "fio2", "spo2"])

        resp_domains = [d for d in domains if d["domain"] == "respiratory_failure"]
        assert len(resp_domains) == 1
        assert resp_domains[0]["confidence"] >= 0.6

    def test_pf_ratio_respiratory(self):
        domains = classify_domains(["pf_ratio", "peep"])

        resp_domains = [d for d in domains if d["domain"] == "respiratory_failure"]
        assert len(resp_domains) == 1


class TestHepaticClassification:
    """Test hepatic dysfunction domain classification."""

    def test_alt_ast_bilirubin_hepatic(self):
        domains = classify_domains(["alt", "ast", "bilirubin"])

        hepatic_domains = [d for d in domains if d["domain"] == "hepatic_dysfunction"]
        assert len(hepatic_domains) == 1
        assert hepatic_domains[0]["confidence"] >= 0.6

    def test_inr_albumin_hepatic(self):
        domains = classify_domains(["inr", "albumin", "ammonia"])

        hepatic_domains = [d for d in domains if d["domain"] == "hepatic_dysfunction"]
        assert len(hepatic_domains) == 1


class TestNeurologicalClassification:
    """Test neurological domain classification."""

    def test_gcs_neurological(self):
        domains = classify_domains(["gcs", "pupils"])

        neuro_domains = [d for d in domains if d["domain"] == "neurological"]
        assert len(neuro_domains) == 1

    def test_nihss_neurological(self):
        domains = classify_domains(["nihss", "icp"])

        neuro_domains = [d for d in domains if d["domain"] == "neurological"]
        assert len(neuro_domains) == 1


class TestMultiSystemDetection:
    """Test multi-system domain detection."""

    def test_sepsis_and_kidney_triggers_multi_system(self):
        # Markers from both sepsis and kidney
        domains = classify_domains(["lactate", "crp", "creatinine", "bun"])

        multi_domains = [d for d in domains if d["domain"] == "multi_system"]
        assert len(multi_domains) == 1

        # Component domains should be listed
        assert "component_domains" in multi_domains[0]
        component_domains = multi_domains[0]["component_domains"]
        assert "sepsis" in component_domains or "kidney_injury" in component_domains

    def test_multi_system_highest_confidence(self):
        # When multiple systems involved, multi_system should rank high
        domains = classify_domains(["troponin", "lactate", "creatinine", "alt"])

        # Multi-system should be first or second
        domain_names = [d["domain"] for d in domains[:3]]
        assert "multi_system" in domain_names


class TestUnknownDomain:
    """Test unknown domain handling."""

    def test_unrecognized_markers_return_unknown(self):
        domains = classify_domains(["xyz_marker", "unknown_lab", "random_test"])

        assert len(domains) == 1
        assert domains[0]["domain"] == "unknown"
        assert domains[0]["confidence"] == 0.0

    def test_empty_features_return_unknown(self):
        domains = classify_domains([])

        assert len(domains) == 1
        assert domains[0]["domain"] == "unknown"


class TestGetPrimaryDomain:
    """Test get_primary_domain convenience function."""

    def test_primary_domain_returns_highest_confidence(self):
        domain, confidence, drivers = get_primary_domain(["lactate", "crp", "wbc"])

        assert domain == "sepsis"
        assert confidence >= 0.6
        assert len(drivers) > 0

    def test_primary_domain_with_unknown(self):
        domain, confidence, drivers = get_primary_domain(["unknown_marker"])

        assert domain == "unknown"
        assert confidence == 0.0


class TestGetDomainFromEndpoint:
    """Test endpoint-to-domain mapping."""

    def test_analyze_endpoint_maps_to_cohort_analysis(self):
        domain = get_domain_from_endpoint("analyze")
        assert domain == "cohort_analysis"

    def test_outbreak_detection_maps_to_outbreak(self):
        domain = get_domain_from_endpoint("outbreak_detection")
        assert domain == "outbreak"

    def test_unknown_endpoint_with_biomarkers(self):
        domain = get_domain_from_endpoint("unknown_endpoint", ["lactate", "crp"])
        assert domain == "sepsis"

    def test_unknown_endpoint_without_biomarkers(self):
        domain = get_domain_from_endpoint("unknown_endpoint")
        assert domain == "unknown"


class TestFeatureNormalization:
    """Test feature name normalization."""

    def test_case_insensitive(self):
        domains1 = classify_domains(["LACTATE", "CRP"])
        domains2 = classify_domains(["lactate", "crp"])

        assert domains1[0]["domain"] == domains2[0]["domain"]

    def test_hyphen_underscore_equivalent(self):
        domains1 = classify_domains(["nt-probnp"])
        domains2 = classify_domains(["nt_probnp"])

        # Both should detect cardiac
        cardiac1 = [d for d in domains1 if d["domain"] == "deterioration_cardiac"]
        cardiac2 = [d for d in domains2 if d["domain"] == "deterioration_cardiac"]

        assert len(cardiac1) == len(cardiac2)

    def test_spaces_normalized(self):
        domains1 = classify_domains(["c reactive protein"])
        domains2 = classify_domains(["c_reactive_protein"])

        sepsis1 = [d for d in domains1 if d["domain"] == "sepsis"]
        sepsis2 = [d for d in domains2 if d["domain"] == "sepsis"]

        assert len(sepsis1) == len(sepsis2)


class TestConfidenceBoosts:
    """Test confidence score calculations."""

    def test_more_markers_increases_confidence(self):
        domains_single = classify_domains(["lactate"])
        domains_multiple = classify_domains(["lactate", "crp", "wbc", "procalcitonin"])

        single_conf = [d["confidence"] for d in domains_single if d["domain"] == "sepsis"][0]
        multi_conf = [d["confidence"] for d in domains_multiple if d["domain"] == "sepsis"][0]

        assert multi_conf > single_conf

    def test_secondary_markers_add_small_boost(self):
        domains_primary = classify_domains(["lactate", "crp"])
        domains_with_secondary = classify_domains(["lactate", "crp", "temperature", "heart_rate"])

        primary_conf = [d["confidence"] for d in domains_primary if d["domain"] == "sepsis"][0]
        secondary_conf = [d["confidence"] for d in domains_with_secondary if d["domain"] == "sepsis"][0]

        assert secondary_conf >= primary_conf


class TestDomainSignatures:
    """Test domain signature definitions."""

    def test_all_domains_have_signatures(self):
        expected_domains = [
            ClinicalDomain.SEPSIS,
            ClinicalDomain.CARDIAC,
            ClinicalDomain.KIDNEY,
            ClinicalDomain.RESPIRATORY,
            ClinicalDomain.HEPATIC,
            ClinicalDomain.NEUROLOGICAL,
            ClinicalDomain.METABOLIC,
            ClinicalDomain.HEMATOLOGIC,
            ClinicalDomain.ONCOLOGY
        ]

        for domain in expected_domains:
            assert domain in DOMAIN_SIGNATURES
            sig = DOMAIN_SIGNATURES[domain]
            assert len(sig.primary_markers) > 0
            assert sig.min_primary_match >= 1
            assert 0 < sig.base_confidence < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
