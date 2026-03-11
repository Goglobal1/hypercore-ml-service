"""Test Time-to-Harm with real obstetric MIMIC data."""
import json
import subprocess
import os
from config import OUTPUT_PATH

TTH_URL = "http://localhost:8000/predict/time-to-harm"
ALERTS_URL = "http://localhost:8000/alerts/evaluate"


def load_trajectories():
    """Load full trajectories."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories", "all_trajectories.json")
    print(f"Loading trajectories from {path}...")
    with open(path) as f:
        return json.load(f)


def load_hellp_cases():
    """Load HELLP syndrome cases."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories", "hellp_cases.json")
    with open(path) as f:
        return json.load(f)


def load_preeclampsia_cases():
    """Load severe preeclampsia cases."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories", "severe_preeclampsia_cases.json")
    with open(path) as f:
        return json.load(f)


def load_hemorrhage_cases():
    """Load obstetric hemorrhage cases."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories", "obstetric_hemorrhage_cases.json")
    with open(path) as f:
        return json.load(f)


def load_dic_cases():
    """Load DIC cases."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories", "dic_cases.json")
    with open(path) as f:
        return json.load(f)


def load_magnesium_cases():
    """Load magnesium monitoring cases."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories", "magnesium_monitoring_cases.json")
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
        "domain": "obstetric",
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

    # Calculate risk based on key obstetric markers
    platelets = current_scores.get("platelets", 200)
    ast = current_scores.get("ast", 30)
    hemoglobin = current_scores.get("hemoglobin", 12)

    # Low platelets, high AST, or low hemoglobin = higher risk
    plt_risk = 0.8 if platelets < 100 else (0.4 if platelets < 150 else 0.1)
    ast_risk = 0.6 if ast > 70 else (0.3 if ast > 40 else 0.1)
    hgb_risk = 0.7 if hemoglobin < 8 else (0.3 if hemoglobin < 10 else 0.1)
    risk_score = max(plt_risk, ast_risk, hgb_risk)

    # Get timestamp from available data
    first_biomarker = list(traj["biomarkers"].keys())[0]
    timestamp = traj["biomarkers"][first_biomarker][-1]["timestamp"]

    payload = {
        "patient_id": f"MIMIC-{traj['subject_id']}",
        "timestamp": timestamp,
        "risk_domain": "obstetric",
        "current_scores": {"risk_score": risk_score, **current_scores},
        "biomarker_trajectories": format_for_tth(traj)
    }

    return call_api(ALERTS_URL, payload)


def main():
    print("=" * 70)
    print("Testing Time-to-Harm with Real MIMIC-IV Obstetric Data")
    print("=" * 70)

    # Load data
    trajectories = load_trajectories()
    hellp_cases = load_hellp_cases()
    preeclampsia_cases = load_preeclampsia_cases()
    hemorrhage_cases = load_hemorrhage_cases()
    dic_cases = load_dic_cases()
    mg_cases = load_magnesium_cases()

    print(f"Loaded {len(trajectories):,} trajectories")
    print(f"Loaded {len(hellp_cases):,} HELLP syndrome cases")
    print(f"Loaded {len(preeclampsia_cases):,} severe preeclampsia cases")
    print(f"Loaded {len(hemorrhage_cases):,} obstetric hemorrhage cases")
    print(f"Loaded {len(dic_cases):,} DIC cases")
    print(f"Loaded {len(mg_cases):,} magnesium monitoring cases")

    # Test HELLP syndrome cases
    if hellp_cases:
        print("\n" + "=" * 70)
        print("TESTING HELLP SYNDROME CASES")
        print("=" * 70)

        for i, case in enumerate(hellp_cases[:3]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  HELLP markers: {case.get('hellp_markers', [])}")
            if 'min_platelets' in case:
                print(f"  Min platelets: {case['min_platelets']} K/uL")
            if 'max_ast' in case:
                print(f"  Max AST: {case['max_ast']} U/L")
            if 'max_ldh' in case:
                print(f"  Max LDH: {case['max_ldh']} U/L")
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

    # Test severe preeclampsia cases
    if preeclampsia_cases:
        print("\n" + "=" * 70)
        print("TESTING SEVERE PREECLAMPSIA CASES")
        print("=" * 70)

        for i, case in enumerate(preeclampsia_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Severity markers: {case.get('severity_markers', [])}")
            if 'max_creatinine' in case:
                print(f"  Max creatinine: {case['max_creatinine']} mg/dL")
            if 'max_uric_acid' in case:
                print(f"  Max uric acid: {case['max_uric_acid']} mg/dL")
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

    # Test obstetric hemorrhage cases
    if hemorrhage_cases:
        print("\n" + "=" * 70)
        print("TESTING OBSTETRIC HEMORRHAGE CASES")
        print("=" * 70)

        for i, case in enumerate(hemorrhage_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  Initial Hgb: {case['initial_hemoglobin']} g/dL")
            print(f"  Min Hgb: {case['min_hemoglobin']} g/dL")
            print(f"  Hgb drop: {case['hemoglobin_drop']} g/dL")
            if case.get('hemorrhagic_shock'):
                print(f"  Max lactate: {case['max_lactate']} mmol/L (SHOCK)")
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

    # Test DIC cases
    if dic_cases:
        print("\n" + "=" * 70)
        print("TESTING DIC CASES")
        print("=" * 70)

        for i, case in enumerate(dic_cases[:2]):
            hadm_id = case["hadm_id"]
            print(f"\n{'-' * 70}")
            print(f"CASE {i+1}: hadm_id={hadm_id}")
            print(f"  DIC markers: {case.get('dic_markers', [])}")
            if 'max_inr' in case:
                print(f"  Max INR: {case['max_inr']}")
            if 'min_platelets' in case:
                print(f"  Min platelets: {case['min_platelets']} K/uL")
            if 'min_fibrinogen' in case:
                print(f"  Min fibrinogen: {case['min_fibrinogen']} mg/dL")
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

    # Find a severe HELLP case
    severe_case = next((c for c in hellp_cases if len(c.get('hellp_markers', [])) >= 3),
                       hellp_cases[0]) if hellp_cases else None
    if severe_case:
        hadm_id = severe_case["hadm_id"]
        print(f"\nPatient hadm_id={hadm_id} (HELLP markers: {severe_case.get('hellp_markers', [])})")

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
