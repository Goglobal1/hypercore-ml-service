"""Test Time-to-Harm with real neurological MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "neurological_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_gcs_cases():
    """Load GCS deterioration cases."""
    path = os.path.join(OUTPUT_PATH, "neurological_cohort", "trajectories", "gcs_deterioration_cases.json")
    with open(path) as f:
        return json.load(f)


def load_sodium_cases():
    """Load sodium abnormality cases."""
    path = os.path.join(OUTPUT_PATH, "neurological_cohort", "trajectories", "sodium_abnormality_cases.json")
    with open(path) as f:
        return json.load(f)


def load_glucose_cases():
    """Load glucose abnormality cases."""
    path = os.path.join(OUTPUT_PATH, "neurological_cohort", "trajectories", "glucose_abnormality_cases.json")
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
        "domain": "neurological",
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

    # Calculate risk based on GCS motor score (lower is worse)
    gcs_motor = current_scores.get("gcs_motor", 6)
    # Normalize: GCS motor 1 = highest risk, 6 = lowest risk
    risk_score = max(0, (6 - gcs_motor) / 5.0)

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "neurological",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Neurological Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    gcs_cases = load_gcs_cases()
    sodium_cases = load_sodium_cases()
    glucose_cases = load_glucose_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(gcs_cases):,} GCS deterioration cases")
    print(f"Loaded {len(sodium_cases):,} sodium abnormality cases")
    print(f"Loaded {len(glucose_cases):,} glucose abnormality cases")

    # Test 3 GCS deterioration cases
    print("\n" + "=" * 70)
    print("TESTING GCS DETERIORATION CASES (SEVERE BRAIN INJURY/COMA)")
    print("=" * 70)

    for i, case in enumerate(gcs_cases[:3]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial GCS Motor: {case['initial_gcs_motor']}")
        print(f"  Min GCS Motor: {case['min_gcs_motor']}")
        print(f"  Hours to deterioration: {case['hours_to_deterioration']:.1f}")
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

    # Test sodium abnormality cases
    print("\n" + "=" * 70)
    print("TESTING SODIUM ABNORMALITY CASES (HYPO/HYPERNATREMIA)")
    print("=" * 70)

    for i, case in enumerate(sodium_cases[:2]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Abnormality type: {case['abnormality_type']}")
        print(f"  Initial sodium: {case['initial_sodium']} mEq/L")
        print(f"  Min sodium: {case['min_sodium']} mEq/L")
        print(f"  Max sodium: {case['max_sodium']} mEq/L")
        print(f"  Hours to abnormality: {case['hours_to_abnormality']:.1f}")
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

    # Test glucose abnormality cases
    print("\n" + "=" * 70)
    print("TESTING GLUCOSE ABNORMALITY CASES (HYPO/HYPERGLYCEMIA)")
    print("=" * 70)

    for i, case in enumerate(glucose_cases[:2]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Abnormality type: {case['abnormality_type']}")
        print(f"  Initial glucose: {case['initial_glucose']} mg/dL")
        print(f"  Min glucose: {case['min_glucose']} mg/dL")
        print(f"  Max glucose: {case['max_glucose']} mg/dL")
        print(f"  Hours to abnormality: {case['hours_to_abnormality']:.1f}")
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

    # Find a severe GCS case
    severe_case = next((c for c in gcs_cases if c['min_gcs_motor'] <= 2), gcs_cases[0])
    hadm_id = severe_case["hadm_id"]
    print(f"\nPatient hadm_id={hadm_id} (GCS Motor: {severe_case['initial_gcs_motor']} -> {severe_case['min_gcs_motor']})")

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
