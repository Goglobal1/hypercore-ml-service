"""
HyperCore Edge Case Testing
============================
Tests 6 specific edge case scenarios to verify robustness.
"""

import requests
import json
import time

API_BASE = "https://hypercore-ml-service-production.up.railway.app"

# ============================================================================
# SCENARIO 1: Rapidly Improving Patient
# Patient starts in crisis but responding to treatment
# Expected: Risk score should DECREASE over time
# ============================================================================
SCENARIO_1_IMPROVING = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
IMPROVING_RAPID,2025-01-01T00:00:00Z,145,38,70,39.8,82,4.5,8.0,28.0,45
IMPROVING_RAPID,2025-01-01T02:00:00Z,130,32,80,39.2,85,3.8,6.0,24.0,60
IMPROVING_RAPID,2025-01-01T04:00:00Z,115,26,90,38.5,89,3.0,4.0,18.0,90
IMPROVING_RAPID,2025-01-01T06:00:00Z,100,22,100,38.0,93,2.2,2.5,14.0,130
IMPROVING_RAPID,2025-01-01T08:00:00Z,88,18,110,37.5,96,1.5,1.5,10.0,180
IMPROVING_RAPID,2025-01-01T10:00:00Z,78,15,118,37.0,98,1.1,1.0,8.0,220"""

# ============================================================================
# SCENARIO 2: Oscillating Values (Noise)
# Patient has biomarkers that go up and down randomly
# Expected: Should NOT trigger high alert unless sustained trend
# ============================================================================
SCENARIO_2_OSCILLATING = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
OSCILLATING,2025-01-01T00:00:00Z,85,16,120,37.0,97,1.0,1.2,8.0,240
OSCILLATING,2025-01-01T02:00:00Z,105,22,105,37.8,93,1.4,1.8,12.0,200
OSCILLATING,2025-01-01T04:00:00Z,78,14,125,36.9,98,0.9,1.0,7.5,250
OSCILLATING,2025-01-01T06:00:00Z,110,24,100,38.0,92,1.5,2.0,13.0,190
OSCILLATING,2025-01-01T08:00:00Z,82,15,122,37.1,97,1.0,1.1,8.2,245
OSCILLATING,2025-01-01T10:00:00Z,95,19,115,37.5,95,1.2,1.4,10.0,220"""

# ============================================================================
# SCENARIO 3: Single Extreme Value
# One biomarker extremely high, all others completely normal
# Expected: Flag but don't over-score (could be lab error)
# ============================================================================
SCENARIO_3_SINGLE_EXTREME = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
SINGLE_EXTREME,2025-01-01T00:00:00Z,72,14,120,36.8,98,0.9,1.0,7.5,250
SINGLE_EXTREME,2025-01-01T02:00:00Z,74,14,118,36.9,98,0.9,1.0,7.8,248
SINGLE_EXTREME,2025-01-01T04:00:00Z,73,15,119,36.8,98,0.9,1.0,45.0,245
SINGLE_EXTREME,2025-01-01T06:00:00Z,75,14,120,36.9,97,1.0,1.0,7.5,250"""

# ============================================================================
# SCENARIO 4: Slow Gradual Decline
# Very slow deterioration over 24+ hours
# Expected: Eventually flag even if each individual change is small
# ============================================================================
SCENARIO_4_SLOW_DECLINE = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
SLOW_DECLINE,2025-01-01T00:00:00Z,72,14,125,36.8,98,0.9,1.0,7.0,260
SLOW_DECLINE,2025-01-01T04:00:00Z,76,15,122,36.9,97,1.0,1.1,7.5,255
SLOW_DECLINE,2025-01-01T08:00:00Z,80,16,118,37.1,96,1.1,1.2,8.0,248
SLOW_DECLINE,2025-01-01T12:00:00Z,84,17,114,37.3,95,1.2,1.4,8.8,240
SLOW_DECLINE,2025-01-01T16:00:00Z,88,18,110,37.5,94,1.3,1.6,9.5,230
SLOW_DECLINE,2025-01-01T20:00:00Z,92,19,106,37.7,93,1.5,1.8,10.5,218
SLOW_DECLINE,2025-01-02T00:00:00Z,96,20,102,37.9,92,1.7,2.0,11.5,205"""

# ============================================================================
# SCENARIO 5: Missing Data
# Patient has some biomarkers but not others
# Expected: Score based on available data, don't crash
# ============================================================================
SCENARIO_5A_MISSING_RENAL = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,wbc,platelets
MISSING_RENAL,2025-01-01T00:00:00Z,75,14,120,36.8,98,7.5,250
MISSING_RENAL,2025-01-01T04:00:00Z,95,20,100,38.0,92,14.0,180
MISSING_RENAL,2025-01-01T08:00:00Z,115,28,85,38.8,87,20.0,120"""

SCENARIO_5B_MINIMAL_DATA = """patient_id,timestamp,heart_rate,respiratory_rate,sbp
MINIMAL_DATA,2025-01-01T00:00:00Z,75,14,120
MINIMAL_DATA,2025-01-01T04:00:00Z,110,26,90
MINIMAL_DATA,2025-01-01T08:00:00Z,135,34,70"""

SCENARIO_5C_ONLY_VITALS = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2
ONLY_VITALS,2025-01-01T00:00:00Z,75,14,120,36.8,98
ONLY_VITALS,2025-01-01T04:00:00Z,100,22,100,38.2,91
ONLY_VITALS,2025-01-01T08:00:00Z,125,32,80,39.0,85"""

# ============================================================================
# SCENARIO 6: All Values at Threshold
# Every biomarker right at warning threshold, none above
# Expected: Moderate alert (multiple systems at edge)
# ============================================================================
SCENARIO_6_AT_THRESHOLD = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
AT_THRESHOLD,2025-01-01T00:00:00Z,75,14,125,36.8,98,0.9,1.0,7.5,250
AT_THRESHOLD,2025-01-01T04:00:00Z,85,17,115,37.3,95,1.1,1.3,9.0,220
AT_THRESHOLD,2025-01-01T08:00:00Z,90,20,105,37.8,92,1.3,1.8,11.0,180
AT_THRESHOLD,2025-01-01T12:00:00Z,100,22,95,38.0,90,1.5,2.0,12.5,150"""


def test_scenario(name: str, description: str, csv_data: str, expected: str, mode: str = "balanced"):
    """Test a single scenario against the API."""
    url = f"{API_BASE}/early_risk_discovery"

    try:
        response = requests.post(
            url,
            json={"csv": csv_data, "scoring_mode": mode},
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code != 200:
            return {
                "name": name,
                "status": "ERROR",
                "http_code": response.status_code,
                "message": response.text[:300]
            }

        result = response.json()
        hybrid = result.get("comparator_performance", {}).get("hybrid_multisignal", {})

        return {
            "name": name,
            "description": description,
            "expected": expected,
            "status": "OK",
            "risk_score": hybrid.get("risk_score", 0),
            "risk_level": hybrid.get("risk_level", "unknown"),
            "domains_alerting": hybrid.get("domains_alerting", 0),
            "domain_counts": hybrid.get("domain_alert_counts", {}),
            "patients_alerting": hybrid.get("patients_alerting", 0),
            "patients_analyzed": hybrid.get("patients_analyzed", 0)
        }

    except Exception as e:
        return {
            "name": name,
            "status": "EXCEPTION",
            "error": str(e)
        }


def run_edge_case_tests():
    """Run all edge case scenarios."""
    print("=" * 80)
    print("HYPERCORE EDGE CASE TESTING")
    print("=" * 80)
    print()

    scenarios = [
        ("SCENARIO 1: Rapidly Improving Patient",
         "Patient starts in crisis, responds to treatment",
         SCENARIO_1_IMPROVING,
         "LOW risk - improving trajectory should override initial severity"),

        ("SCENARIO 2: Oscillating Values (Noise)",
         "Biomarkers go up and down randomly",
         SCENARIO_2_OSCILLATING,
         "LOW/WATCH - no sustained trend, should not trigger high alert"),

        ("SCENARIO 3: Single Extreme Value",
         "One biomarker extremely high (WBC=45), others normal",
         SCENARIO_3_SINGLE_EXTREME,
         "WATCH/MODERATE - flag single domain, don't over-score"),

        ("SCENARIO 4: Slow Gradual Decline (24h)",
         "Very slow deterioration over 24 hours",
         SCENARIO_4_SLOW_DECLINE,
         "MODERATE/HIGH - should detect cumulative decline"),

        ("SCENARIO 5A: Missing Renal Data",
         "No creatinine/lactate, has other biomarkers",
         SCENARIO_5A_MISSING_RENAL,
         "Should score based on available data, no crash"),

        ("SCENARIO 5B: Minimal Data (HR, RR, SBP only)",
         "Only basic vitals available",
         SCENARIO_5B_MINIMAL_DATA,
         "Should score hemodynamic domain only, no crash"),

        ("SCENARIO 5C: Only Vitals (no labs)",
         "Vitals only, no laboratory values",
         SCENARIO_5C_ONLY_VITALS,
         "Should score vitals domains, no crash"),

        ("SCENARIO 6: All Values at Threshold",
         "Every biomarker at warning edge, none above",
         SCENARIO_6_AT_THRESHOLD,
         "MODERATE - multiple systems at edge should flag"),
    ]

    results = []

    for name, description, csv_data, expected in scenarios:
        print(f"Testing: {name}")
        print(f"  {description}")

        result = test_scenario(name, description, csv_data, expected)
        results.append(result)

        if result["status"] == "OK":
            score = result["risk_score"]
            level = result["risk_level"]
            domains = result["domains_alerting"]
            domain_counts = result["domain_counts"]

            print(f"  Result: Score={score:.3f} Level={level.upper()} Domains={domains}")
            print(f"  Domain breakdown: {domain_counts}")
            print(f"  Expected: {expected}")

            # Evaluate if result matches expectation
            if "LOW" in expected.upper() and level.lower() in ["low", "watch"]:
                print(f"  [PASS] Result aligns with expectation")
            elif "MODERATE" in expected.upper() and level.lower() in ["moderate", "watch", "high"]:
                print(f"  [PASS] Result aligns with expectation")
            elif "HIGH" in expected.upper() and level.lower() in ["high", "critical", "moderate"]:
                print(f"  [PASS] Result aligns with expectation")
            elif "no crash" in expected.lower() and result["status"] == "OK":
                print(f"  [PASS] No crash, handled gracefully")
            else:
                print(f"  [CHECK] Review if result matches expectation")
        else:
            print(f"  [FAIL] {result.get('status')}: {result.get('message', result.get('error', 'Unknown'))}")

        print()
        time.sleep(0.5)

    # Summary table
    print("=" * 80)
    print("EDGE CASE TESTING SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Scenario':<45} {'Score':<8} {'Level':<12} {'Domains':<8} {'Status'}")
    print("-" * 80)

    for r in results:
        if r["status"] == "OK":
            print(f"{r['name'][:44]:<45} {r['risk_score']:<8.3f} {r['risk_level'].upper():<12} {r['domains_alerting']:<8} {r['status']}")
        else:
            print(f"{r['name'][:44]:<45} {'N/A':<8} {'N/A':<12} {'N/A':<8} {r['status']}")

    print("-" * 80)

    # Key findings
    print()
    print("=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check Scenario 1 - Improving should be low
    s1 = next((r for r in results if "SCENARIO 1" in r["name"]), None)
    if s1 and s1["status"] == "OK":
        if s1["risk_level"].lower() in ["low", "watch"]:
            print("[PASS] Scenario 1: Improving patient correctly scored LOW/WATCH")
        else:
            print(f"[WARN] Scenario 1: Improving patient scored {s1['risk_level'].upper()} - expected LOW")

    # Check Scenario 2 - Oscillating should be low/watch
    s2 = next((r for r in results if "SCENARIO 2" in r["name"]), None)
    if s2 and s2["status"] == "OK":
        if s2["risk_level"].lower() in ["low", "watch", "moderate"]:
            print(f"[PASS] Scenario 2: Oscillating values scored {s2['risk_level'].upper()} - handled noise appropriately")
        else:
            print(f"[WARN] Scenario 2: Oscillating scored {s2['risk_level'].upper()} - may be over-triggering")

    # Check Scenario 3 - Single extreme should not over-score
    s3 = next((r for r in results if "SCENARIO 3" in r["name"]), None)
    if s3 and s3["status"] == "OK":
        if s3["risk_score"] < 0.5 and s3["domains_alerting"] <= 2:
            print(f"[PASS] Scenario 3: Single extreme value correctly limited (score={s3['risk_score']:.3f}, domains={s3['domains_alerting']})")
        else:
            print(f"[WARN] Scenario 3: Single extreme may be over-scored (score={s3['risk_score']:.3f})")

    # Check Scenario 4 - Slow decline should eventually flag
    s4 = next((r for r in results if "SCENARIO 4" in r["name"]), None)
    if s4 and s4["status"] == "OK":
        if s4["risk_level"].lower() in ["moderate", "high", "watch"]:
            print(f"[PASS] Scenario 4: Slow decline detected ({s4['risk_level'].upper()}) over 24h")
        else:
            print(f"[WARN] Scenario 4: Slow decline may be under-detected ({s4['risk_level'].upper()})")

    # Check Scenario 5 - Missing data should not crash
    s5_count = sum(1 for r in results if "SCENARIO 5" in r["name"] and r["status"] == "OK")
    if s5_count == 3:
        print(f"[PASS] Scenario 5: All missing data variants handled without crash (3/3)")
    else:
        print(f"[WARN] Scenario 5: Some missing data scenarios failed ({s5_count}/3)")

    # Check Scenario 6 - Threshold values should flag moderate
    s6 = next((r for r in results if "SCENARIO 6" in r["name"]), None)
    if s6 and s6["status"] == "OK":
        if s6["risk_level"].lower() in ["moderate", "high", "watch"]:
            print(f"[PASS] Scenario 6: Threshold values flagged appropriately ({s6['risk_level'].upper()})")
        else:
            print(f"[WARN] Scenario 6: Threshold values may be under-flagged ({s6['risk_level'].upper()})")

    print()

    return results


if __name__ == "__main__":
    results = run_edge_case_tests()

    # Save results
    with open("edge_case_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Results saved to edge_case_results.json")
