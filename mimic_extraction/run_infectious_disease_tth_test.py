"""Test Time-to-Harm with real infectious disease MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_sepsis_cases():
    """Load sepsis cases."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "trajectories", "sepsis_cases.json")
    with open(path) as f:
        return json.load(f)


def load_severe_sepsis_cases():
    """Load severe sepsis cases."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "trajectories", "severe_sepsis_cases.json")
    with open(path) as f:
        return json.load(f)


def load_leukocytosis_cases():
    """Load leukocytosis cases."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "trajectories", "leukocytosis_cases.json")
    with open(path) as f:
        return json.load(f)


def load_leukopenia_cases():
    """Load leukopenia cases."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "trajectories", "leukopenia_cases.json")
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
        "domain": "sepsis",  # Use sepsis domain for infectious disease
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

    # Calculate risk based on lactate and WBC
    lactate = current_scores.get("lactate", 1.0)
    wbc = current_scores.get("wbc", 8.0)

    # Higher lactate = higher risk, abnormal WBC = higher risk
    lactate_risk = min(lactate / 4.0, 1.0) if lactate > 2 else 0.1
    wbc_risk = 0.5 if wbc > 12 or wbc < 4 else 0.1
    risk_score = max(lactate_risk, wbc_risk)

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "sepsis",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Infectious Disease Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    sepsis_cases = load_sepsis_cases()
    severe_sepsis_cases = load_severe_sepsis_cases()
    leukocytosis_cases = load_leukocytosis_cases()
    leukopenia_cases = load_leukopenia_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(sepsis_cases):,} sepsis cases")
    print(f"Loaded {len(severe_sepsis_cases):,} severe sepsis cases")
    print(f"Loaded {len(leukocytosis_cases):,} leukocytosis cases")
    print(f"Loaded {len(leukopenia_cases):,} leukopenia cases")

    # Test sepsis cases
    print("\n" + "=" * 70)
    print("TESTING SEPSIS CASES (ELEVATED LACTATE + ABNORMAL WBC)")
    print("=" * 70)

    for i, case in enumerate(sepsis_cases[:3]):
        hadm_id = case["hadm_id"]
        print(f"\n{'-' * 70}")
        print(f"CASE {i+1}: hadm_id={hadm_id}")
        print(f"  Max lactate: {case['max_lactate']} mmol/L")
        print(f"  Max WBC: {case['max_wbc']} K/uL")
        print(f"  Min WBC: {case['min_wbc']} K/uL")
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

    # Test severe sepsis cases
    if severe_sepsis_cases:
        print("\n" + "=" * 70)
        print("TESTING SEVERE SEPSIS CASES (HIGH LACTATE + ORGAN DYSFUNCTION)")
        print("=" * 70)

        for i, case in enumerate(severe_sepsis_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Max lactate: {case['max_lactate']} mmol/L")
            print(f"  Organ dysfunction: {case.get('organ_dysfunction', [])}")
            if 'max_creatinine' in case:
                print(f"  Max creatinine: {case['max_creatinine']} mg/dL")
            if 'min_platelets' in case:
                print(f"  Min platelets: {case['min_platelets']} K/uL")
            if 'max_bilirubin' in case:
                print(f"  Max bilirubin: {case['max_bilirubin']} mg/dL")
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

    # Test leukocytosis cases
    if leukocytosis_cases:
        print("\n" + "=" * 70)
        print("TESTING SEVERE LEUKOCYTOSIS CASES")
        print("=" * 70)

        for i, case in enumerate(leukocytosis_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Initial WBC: {case['initial_wbc']} K/uL")
            print(f"  Max WBC: {case['max_wbc']} K/uL")
            print(f"  Hours to peak: {case['hours_to_peak']}")
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

    # Test leukopenia cases
    if leukopenia_cases:
        print("\n" + "=" * 70)
        print("TESTING SEVERE LEUKOPENIA CASES (IMMUNOCOMPROMISED)")
        print("=" * 70)

        for i, case in enumerate(leukopenia_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Initial WBC: {case['initial_wbc']} K/uL")
            print(f"  Min WBC: {case['min_wbc']} K/uL")
            print(f"  Hours to nadir: {case['hours_to_nadir']}")
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

    # Find a severe case
    severe_case = next((c for c in severe_sepsis_cases if c['max_lactate'] >= 6.0), severe_sepsis_cases[0]) if severe_sepsis_cases else sepsis_cases[0] if sepsis_cases else None
    if severe_case:
        hadm_id = severe_case["hadm_id"]
        print(f"\nPatient hadm_id={hadm_id} (lactate: {severe_case['max_lactate']} mmol/L)")

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
