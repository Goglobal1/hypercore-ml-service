"""Test Time-to-Harm with real endocrine MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_hyperglycemia_cases():
    """Load severe hyperglycemia cases."""
    path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "trajectories", "hyperglycemia_cases.json")
    with open(path) as f:
        return json.load(f)


def load_hypoglycemia_cases():
    """Load severe hypoglycemia cases."""
    path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "trajectories", "hypoglycemia_cases.json")
    with open(path) as f:
        return json.load(f)


def load_dka_cases():
    """Load DKA cases."""
    path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "trajectories", "dka_cases.json")
    with open(path) as f:
        return json.load(f)


def load_electrolyte_cases():
    """Load electrolyte crisis cases."""
    path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "trajectories", "electrolyte_crisis_cases.json")
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
        "domain": "endocrine",
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

    # Calculate risk based on glucose (higher or lower is worse)
    glucose = current_scores.get("glucose", 100)
    # Normalize: glucose < 50 or > 400 = highest risk
    if glucose < 50:
        risk_score = (50 - glucose) / 50.0
    elif glucose > 200:
        risk_score = min((glucose - 200) / 200.0, 1.0)
    else:
        risk_score = 0.1

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "endocrine",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Endocrine Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    hyperglycemia_cases = load_hyperglycemia_cases()
    hypoglycemia_cases = load_hypoglycemia_cases()
    dka_cases = load_dka_cases()
    electrolyte_cases = load_electrolyte_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(hyperglycemia_cases):,} severe hyperglycemia cases")
    print(f"Loaded {len(hypoglycemia_cases):,} severe hypoglycemia cases")
    print(f"Loaded {len(dka_cases):,} DKA cases")
    print(f"Loaded {len(electrolyte_cases):,} electrolyte crisis cases")

    # Test 3 severe hyperglycemia cases
    print("\n" + "=" * 70)
    print("TESTING SEVERE HYPERGLYCEMIA CASES (DKA/HHS RISK)")
    print("=" * 70)

    for i, case in enumerate(hyperglycemia_cases[:3]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial glucose: {case['initial_glucose']} mg/dL")
        print(f"  Max glucose: {case['max_glucose']} mg/dL")
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

    # Test hypoglycemia cases
    print("\n" + "=" * 70)
    print("TESTING SEVERE HYPOGLYCEMIA CASES")
    print("=" * 70)

    for i, case in enumerate(hypoglycemia_cases[:2]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial glucose: {case['initial_glucose']} mg/dL")
        print(f"  Min glucose: {case['min_glucose']} mg/dL")
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

    # Test DKA cases
    if dka_cases:
        print("\n" + "=" * 70)
        print("TESTING DKA CASES (HIGH ANION GAP + LOW BICARBONATE)")
        print("=" * 70)

        for i, case in enumerate(dka_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Initial anion gap: {case['initial_anion_gap']} mEq/L")
            print(f"  Max anion gap: {case['max_anion_gap']} mEq/L")
            print(f"  Initial bicarbonate: {case['initial_bicarbonate']} mEq/L")
            print(f"  Min bicarbonate: {case['min_bicarbonate']} mEq/L")
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

    # Test electrolyte crisis cases
    if electrolyte_cases:
        print("\n" + "=" * 70)
        print("TESTING ELECTROLYTE CRISIS CASES")
        print("=" * 70)

        for i, case in enumerate(electrolyte_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Crisis type: {case['crisis_type']}")
            for key, value in case.items():
                if key not in ['hadm_id', 'subject_id', 'crisis_type']:
                    print(f"  {key}: {value}")
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

    # Find a severe hyperglycemia case
    severe_case = next((c for c in hyperglycemia_cases if c['max_glucose'] >= 500), hyperglycemia_cases[0]) if hyperglycemia_cases else None
    if severe_case:
        hadm_id = severe_case["hadm_id"]
        print(f"\nPatient hadm_id={hadm_id} (glucose: {severe_case['initial_glucose']} -> {severe_case['max_glucose']} mg/dL)")

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
