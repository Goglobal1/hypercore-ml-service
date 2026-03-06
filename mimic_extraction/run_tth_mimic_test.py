"""Test Time-to-Harm with real MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "sepsis_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_deterioration_cases():
    """Load deterioration cases."""
    path = os.path.join(OUTPUT_PATH, "sepsis_cohort", "trajectories", "deterioration_cases.json")
    with open(path) as f:
        return json.load(f)


def format_for_tth(trajectory):
    """Format a trajectory for TTH API."""
    biomarker_trajectories = {}
    for biomarker, measurements in trajectory["biomarkers"].items():
        biomarker_trajectories[biomarker] = [
            {"timestamp": m["timestamp"], "value": m["value"]}
            for m in measurements
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
        "domain": "sepsis",
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

    # Use lactate as risk_score if available, else use 0.5
    risk_score = current_scores.get("lactate", 0.5)
    if risk_score > 10:
        risk_score = risk_score / 20  # Normalize high lactate

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "sepsis",
        "current_scores": {"risk_score": min(risk_score, 1.0), **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Sepsis Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    deterioration = load_deterioration_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(deterioration):,} deterioration cases")

    # Test 3 deterioration cases
    print("\n" + "=" * 70)
    print("TESTING DETERIORATION CASES")
    print("=" * 70)

    for i, case in enumerate(deterioration[:3]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Initial lactate: {case['initial_lactate']}")
        print(f"  Max lactate: {case['max_lactate']}")
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

    # Test integrated CSE + TTH
    print("\n" + "=" * 70)
    print("TESTING INTEGRATED CSE + TTH (/alerts/evaluate)")
    print("=" * 70)

    case = deterioration[0]
    hadm_id = case["hadm_id"]
    print(f"\nPatient hadm_id={hadm_id}")

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
