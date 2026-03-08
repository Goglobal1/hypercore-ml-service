"""Build biomarker trajectories from endocrine cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names
ITEMID_TO_BIOMARKER = {
    # Glucose metabolism
    50809: "glucose", 50931: "glucose",
    50852: "hba1c",
    # Ketones/Acidosis
    50868: "anion_gap",
    50882: "bicarbonate",
    50820: "ph",
    50813: "lactate", 52442: "lactate",
    # Electrolytes
    50983: "sodium", 50824: "sodium",
    50971: "potassium", 50822: "potassium",
    50902: "chloride",
    50893: "calcium",
    50960: "magnesium",
    50970: "phosphate",
    # Osmolality
    50964: "osmolality_serum",
    51695: "osmolality_urine",
    # Thyroid
    50993: "tsh",
    50994: "t3_free",
    50995: "t4_free",
    # Renal
    50912: "creatinine", 52546: "creatinine",
    51006: "bun",
    # Cortisol
    50909: "cortisol",
}

# Normal/critical ranges for endocrine biomarkers
BIOMARKER_THRESHOLDS = {
    # Glucose
    "glucose": {"low": 70, "normal_high": 140, "elevated": 200, "critical_high": 400, "critical_low": 40, "unit": "mg/dL"},
    "hba1c": {"normal": 5.7, "prediabetes": 6.5, "diabetes": 8.0, "poor_control": 10.0, "unit": "%"},
    # Acidosis markers
    "anion_gap": {"normal": 12, "elevated": 16, "critical": 20, "unit": "mEq/L"},
    "bicarbonate": {"low": 22, "critical_low": 15, "very_critical": 10, "high": 28, "unit": "mEq/L"},
    "ph": {"low": 7.35, "critical_low": 7.20, "very_critical": 7.10, "high": 7.45, "unit": ""},
    "lactate": {"normal": 2.0, "elevated": 4.0, "critical": 6.0, "unit": "mmol/L"},
    # Electrolytes
    "sodium": {"low": 136, "critical_low": 125, "high": 145, "critical_high": 155, "unit": "mEq/L"},
    "potassium": {"low": 3.5, "critical_low": 2.5, "high": 5.0, "critical_high": 6.5, "unit": "mEq/L"},
    "chloride": {"low": 98, "high": 106, "unit": "mEq/L"},
    "calcium": {"low": 8.5, "critical_low": 7.0, "high": 10.5, "critical_high": 12.0, "unit": "mg/dL"},
    "magnesium": {"low": 1.7, "critical_low": 1.0, "high": 2.2, "unit": "mg/dL"},
    "phosphate": {"low": 2.5, "high": 4.5, "unit": "mg/dL"},
    # Osmolality
    "osmolality_serum": {"low": 280, "high": 295, "critical_high": 320, "hhs_threshold": 320, "unit": "mOsm/kg"},
    # Thyroid
    "tsh": {"low": 0.4, "high": 4.0, "suppressed": 0.1, "elevated": 10.0, "unit": "mIU/L"},
    "t3_free": {"low": 2.3, "high": 4.2, "unit": "pg/mL"},
    "t4_free": {"low": 0.8, "high": 1.8, "unit": "ng/dL"},
    # Renal
    "creatinine": {"normal": 1.2, "elevated": 2.0, "critical": 4.0, "unit": "mg/dL"},
    "bun": {"normal": 20, "elevated": 40, "critical": 80, "unit": "mg/dL"},
    # Cortisol
    "cortisol": {"low_am": 5, "high_am": 25, "low_pm": 3, "high_pm": 15, "unit": "ug/dL"},
}


def load_lab_events():
    """Load lab events from endocrine cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path, low_memory=False)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "endocrine_cohort", "admissions.csv")
    print(f"Loading admissions...")
    df = pd.read_csv(adm_path)
    print(f"Loaded {len(df):,} admissions")
    return df


def build_patient_trajectories(lab_df, admissions_df):
    """Build biomarker trajectories for each patient admission."""
    print("\nBuilding trajectories...")

    # Map itemid to biomarker name
    lab_df['biomarker'] = lab_df['itemid'].map(ITEMID_TO_BIOMARKER)

    # Parse timestamps
    lab_df['charttime'] = pd.to_datetime(lab_df['charttime'])

    # Get admission times for relative timing
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    adm_times = admissions_df.set_index('hadm_id')['admittime'].to_dict()

    # Group by hadm_id (admission)
    trajectories = {}
    admission_groups = lab_df.groupby('hadm_id')

    total = len(admission_groups)
    for i, (hadm_id, group) in enumerate(admission_groups):
        if (i + 1) % 5000 == 0:
            print(f"  Processing admission {i+1:,}/{total:,}...", end='\r')

        admit_time = adm_times.get(hadm_id)
        if admit_time is None:
            continue

        # Build trajectory for this admission
        admission_trajectory = {
            "hadm_id": int(hadm_id),
            "subject_id": int(group['subject_id'].iloc[0]),
            "admit_time": admit_time.isoformat(),
            "biomarkers": {}
        }

        # Group by biomarker
        for biomarker, bio_group in group.groupby('biomarker'):
            if pd.isna(biomarker):
                continue

            # Sort by time and extract values
            bio_group = bio_group.sort_values('charttime')

            measurements = []
            for _, row in bio_group.iterrows():
                if pd.notna(row.get('valuenum')):
                    hours_from_admit = (row['charttime'] - admit_time).total_seconds() / 3600

                    measurements.append({
                        "timestamp": row['charttime'].isoformat(),
                        "value": float(row['valuenum']),
                        "hours_from_admission": round(hours_from_admit, 2)
                    })

            if measurements:
                admission_trajectory["biomarkers"][biomarker] = measurements

        if admission_trajectory["biomarkers"]:
            trajectories[int(hadm_id)] = admission_trajectory

    print(f"\n  Built trajectories for {len(trajectories):,} admissions")
    return trajectories


def compute_trajectory_stats(trajectories):
    """Compute statistics about the trajectories."""
    stats = {
        "total_admissions": len(trajectories),
        "biomarker_coverage": defaultdict(int),
        "measurements_per_biomarker": defaultdict(list),
        "avg_measurements_per_admission": [],
    }

    for hadm_id, traj in trajectories.items():
        total_measurements = 0
        for biomarker, measurements in traj["biomarkers"].items():
            stats["biomarker_coverage"][biomarker] += 1
            stats["measurements_per_biomarker"][biomarker].append(len(measurements))
            total_measurements += len(measurements)
        stats["avg_measurements_per_admission"].append(total_measurements)

    summary = {
        "total_admissions_with_trajectories": stats["total_admissions"],
        "biomarker_stats": {}
    }

    for biomarker in stats["biomarker_coverage"]:
        counts = stats["measurements_per_biomarker"][biomarker]
        summary["biomarker_stats"][biomarker] = {
            "admissions_with_data": stats["biomarker_coverage"][biomarker],
            "coverage_pct": round(100 * stats["biomarker_coverage"][biomarker] / stats["total_admissions"], 1),
            "avg_measurements": round(sum(counts) / len(counts), 1) if counts else 0,
            "total_measurements": sum(counts),
            "thresholds": BIOMARKER_THRESHOLDS.get(biomarker, {})
        }

    if stats["avg_measurements_per_admission"]:
        summary["avg_total_measurements_per_admission"] = round(
            sum(stats["avg_measurements_per_admission"]) / len(stats["avg_measurements_per_admission"]), 1
        )

    return summary


def find_hyperglycemia_cases(trajectories, glucose_threshold=400, min_measurements=2):
    """Find cases with severe hyperglycemia (DKA/HHS risk)."""
    hyperglycemia_cases = []

    for hadm_id, traj in trajectories.items():
        glucose_data = traj["biomarkers"].get("glucose", [])

        if len(glucose_data) < min_measurements:
            continue

        values = [m["value"] for m in glucose_data]
        max_glucose = max(values)
        initial_glucose = values[0]

        if max_glucose >= glucose_threshold:
            for m in glucose_data:
                if m["value"] >= glucose_threshold:
                    hyperglycemia_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_glucose": round(initial_glucose, 1),
                        "max_glucose": round(max_glucose, 1),
                        "hours_to_critical": m["hours_from_admission"],
                        "num_measurements": len(glucose_data)
                    })
                    break

    return hyperglycemia_cases


def find_hypoglycemia_cases(trajectories, glucose_threshold=50, min_measurements=2):
    """Find cases with severe hypoglycemia."""
    hypoglycemia_cases = []

    for hadm_id, traj in trajectories.items():
        glucose_data = traj["biomarkers"].get("glucose", [])

        if len(glucose_data) < min_measurements:
            continue

        values = [m["value"] for m in glucose_data]
        min_glucose = min(values)
        initial_glucose = values[0]

        if min_glucose <= glucose_threshold:
            for m in glucose_data:
                if m["value"] <= glucose_threshold:
                    hypoglycemia_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_glucose": round(initial_glucose, 1),
                        "min_glucose": round(min_glucose, 1),
                        "hours_to_critical": m["hours_from_admission"],
                        "num_measurements": len(glucose_data)
                    })
                    break

    return hypoglycemia_cases


def find_dka_cases(trajectories, anion_gap_threshold=16, bicarb_threshold=18, min_measurements=2):
    """Find cases suggestive of DKA (high anion gap, low bicarbonate)."""
    dka_cases = []

    for hadm_id, traj in trajectories.items():
        anion_gap_data = traj["biomarkers"].get("anion_gap", [])
        bicarb_data = traj["biomarkers"].get("bicarbonate", [])

        # Need both markers
        if len(anion_gap_data) < min_measurements or len(bicarb_data) < min_measurements:
            continue

        ag_values = [m["value"] for m in anion_gap_data]
        bicarb_values = [m["value"] for m in bicarb_data]

        max_ag = max(ag_values)
        min_bicarb = min(bicarb_values)
        initial_ag = ag_values[0]
        initial_bicarb = bicarb_values[0]

        # DKA criteria: elevated anion gap AND low bicarbonate
        if max_ag >= anion_gap_threshold and min_bicarb <= bicarb_threshold:
            dka_cases.append({
                "hadm_id": hadm_id,
                "subject_id": traj["subject_id"],
                "initial_anion_gap": round(initial_ag, 1),
                "max_anion_gap": round(max_ag, 1),
                "initial_bicarbonate": round(initial_bicarb, 1),
                "min_bicarbonate": round(min_bicarb, 1),
                "num_ag_measurements": len(anion_gap_data),
                "num_bicarb_measurements": len(bicarb_data)
            })

    return dka_cases


def find_electrolyte_crisis_cases(trajectories, na_low=125, na_high=155, k_low=2.5, k_high=6.5, min_measurements=2):
    """Find cases with critical electrolyte abnormalities."""
    electrolyte_cases = []

    for hadm_id, traj in trajectories.items():
        sodium_data = traj["biomarkers"].get("sodium", [])
        potassium_data = traj["biomarkers"].get("potassium", [])

        crisis_type = None
        crisis_details = {}

        # Check sodium
        if len(sodium_data) >= min_measurements:
            na_values = [m["value"] for m in sodium_data]
            min_na = min(na_values)
            max_na = max(na_values)

            if min_na <= na_low:
                crisis_type = "severe_hyponatremia"
                crisis_details = {
                    "initial_sodium": round(na_values[0], 1),
                    "min_sodium": round(min_na, 1),
                }
            elif max_na >= na_high:
                crisis_type = "severe_hypernatremia"
                crisis_details = {
                    "initial_sodium": round(na_values[0], 1),
                    "max_sodium": round(max_na, 1),
                }

        # Check potassium
        if len(potassium_data) >= min_measurements:
            k_values = [m["value"] for m in potassium_data]
            min_k = min(k_values)
            max_k = max(k_values)

            if min_k <= k_low:
                if crisis_type:
                    crisis_type += "_and_hypokalemia"
                else:
                    crisis_type = "severe_hypokalemia"
                crisis_details["initial_potassium"] = round(k_values[0], 2)
                crisis_details["min_potassium"] = round(min_k, 2)
            elif max_k >= k_high:
                if crisis_type:
                    crisis_type += "_and_hyperkalemia"
                else:
                    crisis_type = "severe_hyperkalemia"
                crisis_details["initial_potassium"] = round(k_values[0], 2)
                crisis_details["max_potassium"] = round(max_k, 2)

        if crisis_type:
            electrolyte_cases.append({
                "hadm_id": hadm_id,
                "subject_id": traj["subject_id"],
                "crisis_type": crisis_type,
                **crisis_details
            })

    return electrolyte_cases


def format_for_tth_engine(trajectories, max_patients=None):
    """Format trajectories for Time-to-Harm engine input."""
    tth_format = []

    count = 0
    for hadm_id, traj in trajectories.items():
        if max_patients and count >= max_patients:
            break

        biomarker_trajectories = {}
        for biomarker, measurements in traj["biomarkers"].items():
            biomarker_trajectories[biomarker] = [
                {"timestamp": m["timestamp"], "value": m["value"]}
                for m in measurements
            ]

        if biomarker_trajectories:
            tth_format.append({
                "patient_id": f"MIMIC-{traj['subject_id']}",
                "hadm_id": hadm_id,
                "domain": "endocrine",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Endocrine Cohort")
    print("=" * 60)

    # Load data
    lab_df = load_lab_events()
    admissions_df = load_admissions()

    # Build trajectories
    trajectories = build_patient_trajectories(lab_df, admissions_df)

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_trajectory_stats(trajectories)

    # Find hyperglycemia cases
    print("Finding severe hyperglycemia cases...")
    hyperglycemia_cases = find_hyperglycemia_cases(trajectories)
    print(f"Found {len(hyperglycemia_cases):,} severe hyperglycemia cases")

    # Find hypoglycemia cases
    print("Finding severe hypoglycemia cases...")
    hypoglycemia_cases = find_hypoglycemia_cases(trajectories)
    print(f"Found {len(hypoglycemia_cases):,} severe hypoglycemia cases")

    # Find DKA cases
    print("Finding DKA cases...")
    dka_cases = find_dka_cases(trajectories)
    print(f"Found {len(dka_cases):,} DKA cases")

    # Find electrolyte crisis cases
    print("Finding electrolyte crisis cases...")
    electrolyte_cases = find_electrolyte_crisis_cases(trajectories)
    print(f"Found {len(electrolyte_cases):,} electrolyte crisis cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "endocrine_cohort", "trajectories")
    os.makedirs(output_dir, exist_ok=True)

    print("\nSaving outputs...")

    # Save full trajectories
    traj_file = os.path.join(output_dir, "all_trajectories.json")
    with open(traj_file, 'w') as f:
        json.dump(trajectories, f)
    print(f"  Saved all_trajectories.json ({len(trajectories):,} admissions)")

    # Save stats
    stats_file = os.path.join(output_dir, "trajectory_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved trajectory_stats.json")

    # Save TTH sample
    tth_file = os.path.join(output_dir, "tth_sample_100.json")
    with open(tth_file, 'w') as f:
        json.dump(tth_sample, f, indent=2)
    print(f"  Saved tth_sample_100.json ({len(tth_sample)} patients)")

    # Save hyperglycemia cases
    hyper_file = os.path.join(output_dir, "hyperglycemia_cases.json")
    with open(hyper_file, 'w') as f:
        json.dump(hyperglycemia_cases, f, indent=2)
    print(f"  Saved hyperglycemia_cases.json ({len(hyperglycemia_cases):,} cases)")

    # Save hypoglycemia cases
    hypo_file = os.path.join(output_dir, "hypoglycemia_cases.json")
    with open(hypo_file, 'w') as f:
        json.dump(hypoglycemia_cases, f, indent=2)
    print(f"  Saved hypoglycemia_cases.json ({len(hypoglycemia_cases):,} cases)")

    # Save DKA cases
    dka_file = os.path.join(output_dir, "dka_cases.json")
    with open(dka_file, 'w') as f:
        json.dump(dka_cases, f, indent=2)
    print(f"  Saved dka_cases.json ({len(dka_cases):,} cases)")

    # Save electrolyte crisis cases
    elec_file = os.path.join(output_dir, "electrolyte_crisis_cases.json")
    with open(elec_file, 'w') as f:
        json.dump(electrolyte_cases, f, indent=2)
    print(f"  Saved electrolyte_crisis_cases.json ({len(electrolyte_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"Severe Hyperglycemia Cases: {len(hyperglycemia_cases):,}")
    print(f"Severe Hypoglycemia Cases: {len(hypoglycemia_cases):,}")
    print(f"DKA Cases: {len(dka_cases):,}")
    print(f"Electrolyte Crisis Cases: {len(electrolyte_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:20s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, hyperglycemia_cases, hypoglycemia_cases, dka_cases, electrolyte_cases


if __name__ == "__main__":
    main()
