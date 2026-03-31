"""
HyperCore Patient-Level Validation
===================================
Tests individual patients to verify risk scores correlate with clinical severity.
"""

import requests
import json
import time
from typing import Dict, List

API_BASE = "https://hypercore-ml-service-production.up.railway.app"

# Test patients with known severity levels
TEST_PATIENTS = {
    # CRITICAL - Multi-organ failure (4 domains)
    "CRITICAL_001": {
        "csv": """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
CRITICAL_001,2025-01-01T00:00:00Z,75,14,125,36.8,98,0.9,1.0,7.5,250
CRITICAL_001,2025-01-01T04:00:00Z,95,20,105,37.8,93,1.5,2.0,12.0,180
CRITICAL_001,2025-01-01T08:00:00Z,118,28,85,38.8,88,2.5,4.0,20.0,100
CRITICAL_001,2025-01-01T12:00:00Z,140,38,65,39.5,82,4.0,7.0,28.0,50""",
        "expected_level": "critical",
        "expected_domains": 4,
        "description": "Severe multi-organ failure trajectory"
    },

    # HIGH - 3 domains alerting
    "HIGH_001": {
        "csv": """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
HIGH_001,2025-01-01T00:00:00Z,78,15,120,37.0,97,1.0,1.1,8.0,230
HIGH_001,2025-01-01T04:00:00Z,92,20,108,37.8,94,1.4,1.8,11.0,190
HIGH_001,2025-01-01T08:00:00Z,108,26,95,38.4,90,2.0,2.8,15.0,140
HIGH_001,2025-01-01T12:00:00Z,120,32,85,39.0,86,2.8,4.0,20.0,95""",
        "expected_level": "high",
        "expected_domains": 3,
        "description": "Significant deterioration, 3+ domains"
    },

    # MODERATE - 2 domains alerting
    "MODERATE_001": {
        "csv": """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
MODERATE_001,2025-01-01T00:00:00Z,80,16,118,37.2,96,1.0,1.2,8.5,220
MODERATE_001,2025-01-01T04:00:00Z,88,19,110,37.6,94,1.2,1.5,10.0,200
MODERATE_001,2025-01-01T08:00:00Z,95,22,105,38.0,92,1.4,1.9,12.0,180
MODERATE_001,2025-01-01T12:00:00Z,102,25,100,38.3,90,1.6,2.3,14.0,160""",
        "expected_level": "moderate",
        "expected_domains": 2,
        "description": "Gradual deterioration, 2 domains"
    },

    # WATCH - 1 domain alerting
    "WATCH_001": {
        "csv": """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
WATCH_001,2025-01-01T00:00:00Z,75,14,122,36.9,97,0.9,1.0,7.8,240
WATCH_001,2025-01-01T04:00:00Z,80,16,118,37.2,96,1.0,1.1,8.2,235
WATCH_001,2025-01-01T08:00:00Z,88,18,112,37.5,95,1.1,1.3,9.0,225
WATCH_001,2025-01-01T12:00:00Z,95,20,108,37.8,94,1.2,1.5,10.0,210""",
        "expected_level": "watch",
        "expected_domains": 1,
        "description": "Mild deterioration, 1 domain"
    },

    # LOW - Stable patient
    "STABLE_001": {
        "csv": """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
STABLE_001,2025-01-01T00:00:00Z,72,14,125,36.8,98,0.9,0.9,7.0,260
STABLE_001,2025-01-01T04:00:00Z,74,14,123,36.8,98,0.9,0.9,7.2,258
STABLE_001,2025-01-01T08:00:00Z,73,14,124,36.9,98,0.9,1.0,7.3,255
STABLE_001,2025-01-01T12:00:00Z,75,15,122,36.9,97,1.0,1.0,7.5,252""",
        "expected_level": "low",
        "expected_domains": 0,
        "description": "Stable patient, no deterioration"
    },

    # IMPROVING - Getting better
    "IMPROVING_001": {
        "csv": """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
IMPROVING_001,2025-01-01T00:00:00Z,110,26,92,38.5,89,2.2,3.0,16.0,120
IMPROVING_001,2025-01-01T04:00:00Z,95,22,105,38.0,92,1.8,2.2,13.0,150
IMPROVING_001,2025-01-01T08:00:00Z,82,18,115,37.4,95,1.4,1.5,10.0,185
IMPROVING_001,2025-01-01T12:00:00Z,75,15,122,36.9,97,1.0,1.0,8.0,220""",
        "expected_level": "low",  # Should be low since improving
        "expected_domains": 0,
        "description": "Improving patient - started bad, getting better"
    },
}


def test_patient(patient_id: str, patient_data: Dict, mode: str = "balanced") -> Dict:
    """Test a single patient and return results."""
    url = f"{API_BASE}/early_risk_discovery"

    try:
        response = requests.post(
            url,
            json={"csv": patient_data["csv"], "scoring_mode": mode},
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code != 200:
            return {"error": response.status_code, "message": response.text[:200]}

        result = response.json()
        hybrid = result.get("comparator_performance", {}).get("hybrid_multisignal", {})

        return {
            "patient_id": patient_id,
            "expected_level": patient_data["expected_level"],
            "expected_domains": patient_data["expected_domains"],
            "description": patient_data["description"],
            "actual_risk_score": hybrid.get("risk_score", 0),
            "actual_risk_level": hybrid.get("risk_level", "unknown"),
            "actual_domains": hybrid.get("domains_alerting", 0),
            "domain_counts": hybrid.get("domain_alert_counts", {}),
            "patients_alerting": hybrid.get("patients_alerting", 0),
            "patients_analyzed": hybrid.get("patients_analyzed", 0),
            "mode": mode
        }

    except Exception as e:
        return {"error": str(e)}


def run_patient_validation():
    """Run validation on individual patients."""
    print("="*70)
    print("HYPERCORE PATIENT-LEVEL VALIDATION")
    print("="*70)
    print(f"Testing {len(TEST_PATIENTS)} patients to verify risk scoring")
    print()

    results = []

    for patient_id, patient_data in TEST_PATIENTS.items():
        print(f"Testing: {patient_id} - {patient_data['description']}")
        result = test_patient(patient_id, patient_data)
        results.append(result)

        if "error" not in result:
            match = "[OK]" if result["actual_risk_level"] == result["expected_level"] else "[~]" if abs(["low", "watch", "moderate", "high", "critical"].index(result["actual_risk_level"]) - ["low", "watch", "moderate", "high", "critical"].index(result["expected_level"])) <= 1 else "[X]"
            print(f"  Score: {result['actual_risk_score']:.3f} | Level: {result['actual_risk_level']} (expected: {result['expected_level']}) {match}")
            print(f"  Domains: {result['actual_domains']} alerting | Breakdown: {result['domain_counts']}")
        else:
            print(f"  ERROR: {result.get('error', 'Unknown')}")
        print()

        time.sleep(0.5)

    # Summary table
    print("="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print()
    print(f"{'Patient':<15} {'Expected':<12} {'Actual':<12} {'Score':<8} {'Domains':<8} {'Match'}")
    print("-"*70)

    correct = 0
    close = 0
    for r in results:
        if "error" in r:
            continue

        expected = r["expected_level"]
        actual = r["actual_risk_level"]

        levels = ["low", "watch", "moderate", "high", "critical"]
        exp_idx = levels.index(expected) if expected in levels else -1
        act_idx = levels.index(actual) if actual in levels else -1

        if expected == actual:
            match = "[OK] EXACT"
            correct += 1
            close += 1
        elif abs(exp_idx - act_idx) <= 1:
            match = "[~] CLOSE"
            close += 1
        else:
            match = "[X] MISS"

        print(f"{r['patient_id']:<15} {expected:<12} {actual:<12} {r['actual_risk_score']:<8.3f} {r['actual_domains']:<8.1f} {match}")

    print("-"*70)
    print(f"Exact matches: {correct}/{len(results)} ({correct/len(results)*100:.0f}%)")
    print(f"Close matches (±1 level): {close}/{len(results)} ({close/len(results)*100:.0f}%)")

    # Verify ordering
    print()
    print("="*70)
    print("ORDERING VERIFICATION")
    print("="*70)
    print()
    print("Expected: CRITICAL > HIGH > MODERATE > WATCH > LOW")
    print()

    # Sort by actual score
    scored = [(r["patient_id"], r["actual_risk_score"], r["expected_level"]) for r in results if "error" not in r]
    scored.sort(key=lambda x: x[1], reverse=True)

    print("Actual ranking by score:")
    for i, (pid, score, expected) in enumerate(scored, 1):
        print(f"  {i}. {pid}: {score:.3f} (expected: {expected})")

    # Verify critical > low
    critical_scores = [r["actual_risk_score"] for r in results if r.get("expected_level") == "critical"]
    low_scores = [r["actual_risk_score"] for r in results if r.get("expected_level") == "low"]

    if critical_scores and low_scores:
        min_critical = min(critical_scores)
        max_low = max(low_scores)

        print()
        if min_critical > max_low:
            print(f"[OK] CRITICAL patients ({min_critical:.3f}) score higher than LOW patients ({max_low:.3f})")
        else:
            print(f"[X] Overlap detected: Critical={min_critical:.3f}, Low={max_low:.3f}")

    return results


if __name__ == "__main__":
    results = run_patient_validation()

    # Save results
    with open("patient_validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("Results saved to patient_validation_results.json")
