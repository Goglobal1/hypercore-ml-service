"""Test Time-to-Harm with real pediatric MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_jaundice_cases():
    """Load neonatal jaundice cases."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories", "neonatal_jaundice_cases.json")
    with open(path) as f:
        return json.load(f)


def load_sepsis_cases():
    """Load pediatric sepsis cases."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories", "pediatric_sepsis_cases.json")
    with open(path) as f:
        return json.load(f)


def load_respiratory_cases():
    """Load respiratory distress cases."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories", "respiratory_distress_cases.json")
    with open(path) as f:
        return json.load(f)


def load_anemia_cases():
    """Load anemia cases."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories", "anemia_cases.json")
    with open(path) as f:
        return json.load(f)


def load_electrolyte_cases():
    """Load electrolyte imbalance cases."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories", "electrolyte_imbalance_cases.json")
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
        "domain": "pediatric",
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

    # Calculate risk based on key pediatric markers
    bilirubin = current_scores.get("bilirubin_total", 5)
    lactate = current_scores.get("lactate", 1)
    ph = current_scores.get("ph", 7.4)

    # High bilirubin, elevated lactate, or low pH = higher risk
    bili_risk = min(bilirubin / 20, 1.0) if bilirubin > 15 else 0.1
    lactate_risk = min(lactate / 4, 1.0) if lactate > 2 else 0.1
    ph_risk = 0.7 if ph < 7.25 else 0.1
    risk_score = max(bili_risk, lactate_risk, ph_risk)

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "pediatric",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Pediatric Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    jaundice_cases = load_jaundice_cases()
    sepsis_cases = load_sepsis_cases()
    respiratory_cases = load_respiratory_cases()
    anemia_cases = load_anemia_cases()
    electrolyte_cases = load_electrolyte_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(jaundice_cases):,} neonatal jaundice cases")
    print(f"Loaded {len(sepsis_cases):,} pediatric sepsis cases")
    print(f"Loaded {len(respiratory_cases):,} respiratory distress cases")
    print(f"Loaded {len(anemia_cases):,} anemia cases")
    print(f"Loaded {len(electrolyte_cases):,} electrolyte imbalance cases")

    # Test neonatal jaundice cases
    if jaundice_cases:
        print("\n" + "=" * 70)
        print("TESTING NEONATAL JAUNDICE CASES")
        print("=" * 70)

        for i, case in enumerate(jaundice_cases[:3]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Max bilirubin: {case['max_bilirubin_total']} mg/dL")
            if case.get("conjugated"):
                print(f"  Conjugated hyperbilirubinemia detected")
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

    # Test pediatric sepsis cases
    if sepsis_cases:
        print("\n" + "=" * 70)
        print("TESTING PEDIATRIC SEPSIS CASES")
        print("=" * 70)

        for i, case in enumerate(sepsis_cases[:3]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Sepsis markers: {case.get('sepsis_markers', [])}")
            if 'max_lactate' in case:
                print(f"  Max lactate: {case['max_lactate']} mmol/L")
            if 'max_wbc' in case:
                print(f"  Max WBC: {case['max_wbc']} K/uL")
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

    # Test respiratory distress cases
    if respiratory_cases:
        print("\n" + "=" * 70)
        print("TESTING RESPIRATORY DISTRESS CASES")
        print("=" * 70)

        for i, case in enumerate(respiratory_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Respiratory markers: {case.get('respiratory_markers', [])}")
            if 'min_ph' in case:
                print(f"  Min pH: {case['min_ph']}")
            if 'min_pao2' in case:
                print(f"  Min pO2: {case['min_pao2']} mmHg")
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

    # Test anemia cases
    if anemia_cases:
        print("\n" + "=" * 70)
        print("TESTING ANEMIA CASES")
        print("=" * 70)

        for i, case in enumerate(anemia_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Min hemoglobin: {case['min_hemoglobin']} g/dL")
            print(f"  Severity: {case.get('severity', 'unknown')}")
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
    severe_case = next((c for c in jaundice_cases if c['max_bilirubin_total'] >= 20),
                       jaundice_cases[0]) if jaundice_cases else None
    if severe_case:
        hadm_id = severe_case["hadm_id"]
        print(f"\nPatient hadm_id={hadm_id} (bilirubin: {severe_case['max_bilirubin_total']} mg/dL)")

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
