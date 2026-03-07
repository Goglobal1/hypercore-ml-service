"""Test Time-to-Harm with real hematological MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "hematological_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_anemia_cases():
    """Load severe anemia cases."""
    path = os.path.join(OUTPUT_PATH, "hematological_cohort", "trajectories", "severe_anemia_cases.json")
    with open(path) as f:
        return json.load(f)


def load_thrombocytopenia_cases():
    """Load thrombocytopenia cases."""
    path = os.path.join(OUTPUT_PATH, "hematological_cohort", "trajectories", "thrombocytopenia_cases.json")
    with open(path) as f:
        return json.load(f)


def load_coagulopathy_cases():
    """Load coagulopathy cases."""
    path = os.path.join(OUTPUT_PATH, "hematological_cohort", "trajectories", "coagulopathy_cases.json")
    with open(path) as f:
        return json.load(f)


def load_neutropenia_cases():
    """Load neutropenia cases."""
    path = os.path.join(OUTPUT_PATH, "hematological_cohort", "trajectories", "neutropenia_cases.json")
    with open(path) as f:
        return json.load(f)


def format_for_tth(trajectory, max_measurements=20):
    """Format a trajectory for TTH API (limited to avoid command line length issues)."""
    biomarker_trajectories = {}
    for biomarker, measurements in trajectory["biomarkers"].items():
        # Take last N measurements to keep payload size manageable
        recent = measurements[-max_measurements:] if len(measurements) > max_measurements else measurements
        biomarker_trajectories[biomarker] = [
            {"timestamp": m["timestamp"], "value": m["value"]}
            for m in recent
        ]
    return biomarker_trajectories


def call_api(url, payload):
    """Call API using curl."""
    payload_json = json.dumps(payload)
    result = subprocess.run(
        ["curl", "-s", "-X", "POST", url, "-H", "Content-Type: application/json", "-d", payload_json],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        try:
            return json.loads(result.stdout)
        except:
            print(f"Failed to parse: {result.stdout[:200]}")
            return None
    return None


def run_single_patient(hadm_id, trajectories):
    """Run TTH prediction for a single patient."""
    traj = trajectories.get(str(hadm_id))
    if not traj:
        print(f"No trajectory found for hadm_id {hadm_id}")
        return None

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "domain": "hematological",
        "biomarker_trajectories": format_for_tth(traj)
    }

    print(f"\nTesting patient MIMIC-{traj['subject_id']} (hadm_id: {hadm_id})")
    print(f"Biomarkers available: {list(traj['biomarkers'].keys())}")

    return call_api(TTH_URL, payload)


def run_with_alerts(hadm_id, trajectories):
    """Run /alerts/evaluate endpoint (CSE + TTH)."""
    traj = trajectories.get(str(hadm_id))
    if not traj:
        return None

    # Get latest values for current_scores
    current_scores = {}
    for biomarker, measurements in traj["biomarkers"].items():
        if measurements:
            current_scores[biomarker] = measurements[-1]["value"]

    # Calculate risk based on hemoglobin (lower is worse)
    hgb = current_scores.get("hemoglobin", 12.0)
    # Normalize: Hgb < 7 = highest risk
    risk_score = max(0, (12 - hgb) / 5.0)
    risk_score = min(risk_score, 1.0)

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "hematological",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Hematological Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    anemia_cases = load_anemia_cases()
    thrombocytopenia_cases = load_thrombocytopenia_cases()
    coagulopathy_cases = load_coagulopathy_cases()
    neutropenia_cases = load_neutropenia_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(anemia_cases):,} severe anemia cases")
    print(f"Loaded {len(thrombocytopenia_cases):,} thrombocytopenia cases")
    print(f"Loaded {len(coagulopathy_cases):,} coagulopathy cases")
    print(f"Loaded {len(neutropenia_cases):,} neutropenia cases")

    # Test 3 severe anemia cases
    print("\n" + "=" * 70)
    print("TESTING SEVERE ANEMIA CASES (CRITICAL HEMOGLOBIN)")
    print("=" * 70)

    for i, case in enumerate(anemia_cases[:3]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial hemoglobin: {case['initial_hemoglobin']} g/dL")
        print(f"  Min hemoglobin: {case['min_hemoglobin']} g/dL")
        print(f"  Hours to critical: {case['hours_to_critical']:.1f}")
        print(f"{'-' * 70}")

        result = run_single_patient(hadm_id, trajectories)
        if result:
            print(f"\n  TTH PREDICTION:")
            print(f"    Hours to Harm: {result.get('hours_to_harm', 'N/A')}")
            print(f"    Confidence: {result.get('confidence', 'N/A')}")
            print(f"    Intervention Window: {result.get('intervention_window', 'N/A')}")
            print(f"    Key Drivers: {result.get('key_drivers', [])[:3]}")
            rationale = result.get('rationale', 'N/A')
            print(f"    Rationale: {rationale[:100]}..." if len(rationale) > 100 else f"    Rationale: {rationale}")
            if result.get('recommendations'):
                print(f"    Recommendations:")
                for rec in result.get('recommendations', [])[:3]:
                    print(f"      - {rec}")

    # Test thrombocytopenia cases
    print("\n" + "=" * 70)
    print("TESTING THROMBOCYTOPENIA CASES (LOW PLATELETS)")
    print("=" * 70)

    for i, case in enumerate(thrombocytopenia_cases[:2]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial platelets: {case['initial_platelets']:.0f} K/uL")
        print(f"  Min platelets: {case['min_platelets']:.0f} K/uL")
        print(f"  Hours to critical: {case['hours_to_critical']:.1f}")
        print(f"{'-' * 70}")

        result = run_single_patient(hadm_id, trajectories)
        if result:
            print(f"\n  TTH PREDICTION:")
            print(f"    Hours to Harm: {result.get('hours_to_harm', 'N/A')}")
            print(f"    Confidence: {result.get('confidence', 'N/A')}")
            print(f"    Intervention Window: {result.get('intervention_window', 'N/A')}")
            if result.get('recommendations'):
                print(f"    Recommendations:")
                for rec in result.get('recommendations', [])[:3]:
                    print(f"      - {rec}")

    # Test coagulopathy cases
    print("\n" + "=" * 70)
    print("TESTING COAGULOPATHY CASES (ELEVATED INR)")
    print("=" * 70)

    for i, case in enumerate(coagulopathy_cases[:2]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial INR: {case['initial_inr']}")
        print(f"  Max INR: {case['max_inr']}")
        print(f"  Hours to coagulopathy: {case['hours_to_coagulopathy']:.1f}")
        print(f"{'-' * 70}")

        result = run_single_patient(hadm_id, trajectories)
        if result:
            print(f"\n  TTH PREDICTION:")
            print(f"    Hours to Harm: {result.get('hours_to_harm', 'N/A')}")
            print(f"    Confidence: {result.get('confidence', 'N/A')}")
            print(f"    Intervention Window: {result.get('intervention_window', 'N/A')}")
            if result.get('recommendations'):
                print(f"    Recommendations:")
                for rec in result.get('recommendations', [])[:3]:
                    print(f"      - {rec}")

    # Test neutropenia cases if available
    if neutropenia_cases:
        print("\n" + "=" * 70)
        print("TESTING NEUTROPENIA CASES (INFECTION RISK)")
        print("=" * 70)

        case = neutropenia_cases[0]
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE 1: hadm_id={hadm_id}")
        print(f"  Marker type: {case['marker_type']}")
        print(f"  Initial value: {case['initial_value']}")
        print(f"  Min value: {case['min_value']}")
        print(f"  Hours to critical: {case['hours_to_critical']:.1f}")
        print(f"{'-' * 70}")

        result = run_single_patient(hadm_id, trajectories)
        if result:
            print(f"\n  TTH PREDICTION:")
            print(f"    Hours to Harm: {result.get('hours_to_harm', 'N/A')}")
            print(f"    Confidence: {result.get('confidence', 'N/A')}")
            print(f"    Intervention Window: {result.get('intervention_window', 'N/A')}")
            if result.get('recommendations'):
                print(f"    Recommendations:")
                for rec in result.get('recommendations', [])[:3]:
                    print(f"      - {rec}")

    # Test integrated CSE + TTH
    print("\n" + "=" * 70)
    print("TESTING INTEGRATED CSE + TTH (/alerts/evaluate)")
    print("=" * 70)

    # Find a severe anemia case
    severe_case = next((c for c in anemia_cases if c['min_hemoglobin'] <= 6), anemia_cases[0])
    hadm_id = severe_case["hadm_id"]
    print(f"\nPatient hadm_id={hadm_id} (hemoglobin: {severe_case['initial_hemoglobin']} -> {severe_case['min_hemoglobin']} g/dL)")

    result = run_with_alerts(hadm_id, trajectories)
    if result:
        print(f"\n  CLINICAL STATE ENGINE:")
        print(f"    State: {result.get('state_now')} ({result.get('state_name')})")
        print(f"    Severity: {result.get('severity')}")
        print(f"    Clinical Headline: {result.get('clinical_headline')}")
        print(f"    Suggested Action: {result.get('suggested_action')}")

        if result.get('time_to_harm'):
            tth = result['time_to_harm']
            print(f"\n  TIME-TO-HARM:")
            print(f"    Hours to Harm: {tth.get('hours_to_harm')}")
            print(f"    Confidence: {tth.get('confidence')}")
            print(f"    Intervention Window: {tth.get('intervention_window')}")
            if tth.get('recommendations'):
                print(f"    Recommendations:")
                for rec in tth.get('recommendations', [])[:3]:
                    print(f"      - {rec}")

    print("\n" + "=" * 70)
    print("TESTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
