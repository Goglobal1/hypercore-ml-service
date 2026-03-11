"""Test Time-to-Harm with real psychiatric MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_lithium_toxicity_cases():
    """Load lithium toxicity cases."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "trajectories", "lithium_toxicity_cases.json")
    with open(path) as f:
        return json.load(f)


def load_nms_cases():
    """Load NMS cases."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "trajectories", "nms_cases.json")
    with open(path) as f:
        return json.load(f)


def load_metabolic_syndrome_cases():
    """Load metabolic syndrome cases."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "trajectories", "metabolic_syndrome_cases.json")
    with open(path) as f:
        return json.load(f)


def load_neutropenia_cases():
    """Load neutropenia cases."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "trajectories", "neutropenia_cases.json")
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
        "domain": "psychiatric",
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

    # Calculate risk based on lithium level and renal function
    lithium = current_scores.get("lithium", 0.8)
    creatinine = current_scores.get("creatinine", 1.0)

    # Higher lithium = higher risk, elevated creatinine = higher risk
    lithium_risk = min(lithium / 1.5, 1.0) if lithium > 1.0 else 0.1
    renal_risk = min(creatinine / 2.0, 1.0) if creatinine > 1.5 else 0.1
    risk_score = max(lithium_risk, renal_risk)

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "psychiatric",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Psychiatric Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    lithium_cases = load_lithium_toxicity_cases()
    nms_cases = load_nms_cases()
    metabolic_cases = load_metabolic_syndrome_cases()
    neutropenia_cases = load_neutropenia_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(lithium_cases):,} lithium toxicity cases")
    print(f"Loaded {len(nms_cases):,} NMS cases")
    print(f"Loaded {len(metabolic_cases):,} metabolic syndrome cases")
    print(f"Loaded {len(neutropenia_cases):,} neutropenia cases")

    # Test lithium toxicity cases
    if lithium_cases:
        print("\n" + "=" * 70)
        print("TESTING LITHIUM TOXICITY CASES")
        print("=" * 70)

        for i, case in enumerate(lithium_cases[:3]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Max lithium: {case['max_lithium']} mEq/L")
            if case.get("renal_impairment"):
                print(f"  Max creatinine: {case.get('max_creatinine')} mg/dL (renal impairment)")
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

    # Test NMS cases
    if nms_cases:
        print("\n" + "=" * 70)
        print("TESTING NMS CASES (ELEVATED CK + LEUKOCYTOSIS)")
        print("=" * 70)

        for i, case in enumerate(nms_cases[:3]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Max CK: {case['max_ck']} U/L")
            if case.get("leukocytosis"):
                print(f"  Max WBC: {case.get('max_wbc')} K/uL (leukocytosis)")
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

    # Test metabolic syndrome cases
    if metabolic_cases:
        print("\n" + "=" * 70)
        print("TESTING METABOLIC SYNDROME CASES (ANTIPSYCHOTIC SIDE EFFECTS)")
        print("=" * 70)

        for i, case in enumerate(metabolic_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Abnormal markers: {case.get('abnormal_markers', [])}")
            if 'max_glucose' in case:
                print(f"  Max glucose: {case['max_glucose']} mg/dL")
            if 'max_triglycerides' in case:
                print(f"  Max triglycerides: {case['max_triglycerides']} mg/dL")
            if 'max_cholesterol' in case:
                print(f"  Max cholesterol: {case['max_cholesterol']} mg/dL")
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

    # Test neutropenia cases
    if neutropenia_cases:
        print("\n" + "=" * 70)
        print("TESTING NEUTROPENIA CASES (CLOZAPINE MONITORING)")
        print("=" * 70)

        for i, case in enumerate(neutropenia_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            if 'min_neutrophils' in case:
                print(f"  Min neutrophils: {case['min_neutrophils']} K/uL")
            if 'min_wbc' in case:
                print(f"  Min WBC: {case['min_wbc']} K/uL")
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

    # Find a severe lithium toxicity case
    severe_case = next((c for c in lithium_cases if c['max_lithium'] >= 2.0 and c.get('renal_impairment')),
                       lithium_cases[0]) if lithium_cases else None
    if severe_case:
        hadm_id = severe_case["hadm_id"]
        print(f"\nPatient hadm_id={hadm_id} (lithium: {severe_case['max_lithium']} mEq/L)")

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
