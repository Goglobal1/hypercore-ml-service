"""Build biomarker trajectories from neurological cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names (from lab events)
LAB_ITEMID_TO_BIOMARKER = {
    # Sodium
    50983: "sodium", 50824: "sodium",
    # Glucose
    50809: "glucose", 50931: "glucose",
    # Lactate
    50813: "lactate", 52442: "lactate",
    # Ammonia
    50866: "ammonia",
    # Osmolality
    50964: "osmolality",
    # Magnesium
    50960: "magnesium",
    # Calcium
    50893: "calcium",
    # Potassium
    50971: "potassium", 50822: "potassium",
    # Hemoglobin
    51222: "hemoglobin", 50811: "hemoglobin",
    # Platelets
    51265: "platelets",
    # INR
    51237: "inr",
}

# Map MIMIC itemids to our biomarker names (from chart events)
CHART_ITEMID_TO_BIOMARKER = {
    # GCS Components
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
    # Pupil assessments
    223907: "pupil_size_right",
    224733: "pupil_size_left",
    227121: "pupil_response_right",
    227288: "pupil_response_left",
    # Level of Consciousness
    226104: "level_of_consciousness",
}

# Normal/critical ranges for neurological biomarkers
BIOMARKER_THRESHOLDS = {
    # Labs
    "sodium": {"low": 136, "high": 145, "critical_low": 125, "critical_high": 155, "unit": "mEq/L"},
    "glucose": {"low": 70, "high": 140, "critical_low": 40, "critical_high": 400, "unit": "mg/dL"},
    "lactate": {"normal": 2.0, "elevated": 4.0, "critical": 6.0, "unit": "mmol/L"},
    "ammonia": {"normal": 50, "elevated": 80, "critical": 150, "unit": "umol/L"},
    "osmolality": {"low": 280, "high": 295, "critical_high": 320, "unit": "mOsm/kg"},
    "magnesium": {"low": 1.7, "high": 2.2, "critical_low": 1.0, "unit": "mg/dL"},
    "calcium": {"low": 8.5, "high": 10.5, "critical_low": 7.0, "unit": "mg/dL"},
    "potassium": {"low": 3.5, "high": 5.0, "critical_low": 2.5, "critical_high": 6.5, "unit": "mEq/L"},
    "hemoglobin": {"low": 12.0, "critical_low": 7.0, "unit": "g/dL"},
    "platelets": {"normal": 150, "low": 100, "critical_low": 50, "unit": "K/uL"},
    "inr": {"normal": 1.1, "elevated": 1.5, "critical": 2.5, "unit": "ratio"},
    # Chart assessments
    "gcs_eye": {"normal": 4, "impaired": 3, "severe": 1, "unit": "score"},
    "gcs_verbal": {"normal": 5, "impaired": 3, "severe": 1, "unit": "score"},
    "gcs_motor": {"normal": 6, "impaired": 4, "severe": 1, "unit": "score"},
    "pupil_size_right": {"normal_min": 2, "normal_max": 5, "dilated": 6, "unit": "mm"},
    "pupil_size_left": {"normal_min": 2, "normal_max": 5, "dilated": 6, "unit": "mm"},
}


def load_lab_events():
    """Load lab events from neurological cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "neurological_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path, low_memory=False)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_chart_events():
    """Load chart events from neurological cohort."""
    chart_path = os.path.join(OUTPUT_PATH, "neurological_cohort", "chart_events.csv")
    print(f"Loading chart events from {chart_path}...")
    df = pd.read_csv(chart_path, low_memory=False)
    print(f"Loaded {len(df):,} chart events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "neurological_cohort", "admissions.csv")
    print(f"Loading admissions...")
    df = pd.read_csv(adm_path)
    print(f"Loaded {len(df):,} admissions")
    return df


def build_patient_trajectories(lab_df, chart_df, admissions_df):
    """Build biomarker trajectories for each patient admission."""
    print("\nBuilding trajectories...")

    # Map itemid to biomarker name
    lab_df['biomarker'] = lab_df['itemid'].map(LAB_ITEMID_TO_BIOMARKER)
    chart_df['biomarker'] = chart_df['itemid'].map(CHART_ITEMID_TO_BIOMARKER)

    # Parse timestamps
    lab_df['charttime'] = pd.to_datetime(lab_df['charttime'])
    chart_df['charttime'] = pd.to_datetime(chart_df['charttime'])

    # Get admission times for relative timing
    admissions_df['admittime'] = pd.to_datetime(admissions_df['admittime'])
    adm_times = admissions_df.set_index('hadm_id')['admittime'].to_dict()

    # Combine lab and chart data
    print("  Combining lab and chart events...")

    # Standardize columns for combining
    lab_subset = lab_df[['hadm_id', 'subject_id', 'charttime', 'biomarker', 'valuenum']].copy()
    lab_subset = lab_subset.rename(columns={'valuenum': 'value'})

    chart_subset = chart_df[['hadm_id', 'subject_id', 'charttime', 'biomarker', 'valuenum']].copy()
    chart_subset = chart_subset.rename(columns={'valuenum': 'value'})

    combined_df = pd.concat([lab_subset, chart_subset], ignore_index=True)
    print(f"  Combined {len(combined_df):,} total events")

    # Group by hadm_id (admission)
    trajectories = {}
    admission_groups = combined_df.groupby('hadm_id')

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
                if pd.notna(row.get('value')):
                    hours_from_admit = (row['charttime'] - admit_time).total_seconds() / 3600

                    measurements.append({
                        "timestamp": row['charttime'].isoformat(),
                        "value": float(row['value']),
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


def find_gcs_deterioration_cases(trajectories, gcs_threshold=8, min_measurements=2):
    """Find cases with significant GCS deterioration (severe brain injury/coma)."""
    deterioration_cases = []

    for hadm_id, traj in trajectories.items():
        # Calculate total GCS from components if available
        gcs_eye = traj["biomarkers"].get("gcs_eye", [])
        gcs_verbal = traj["biomarkers"].get("gcs_verbal", [])
        gcs_motor = traj["biomarkers"].get("gcs_motor", [])

        # Need at least some GCS data
        if len(gcs_eye) < min_measurements and len(gcs_motor) < min_measurements:
            continue

        # Calculate total GCS at each timepoint (simplified - use motor as proxy if needed)
        if gcs_motor:
            values = [m["value"] for m in gcs_motor]
            min_motor = min(values)
            initial_motor = values[0]

            # Motor score <= 4 indicates severe impairment
            if min_motor <= 4:
                for m in gcs_motor:
                    if m["value"] <= 4:
                        deterioration_cases.append({
                            "hadm_id": hadm_id,
                            "subject_id": traj["subject_id"],
                            "initial_gcs_motor": round(initial_motor, 1),
                            "min_gcs_motor": round(min_motor, 1),
                            "hours_to_deterioration": m["hours_from_admission"],
                            "num_measurements": len(gcs_motor)
                        })
                        break

    return deterioration_cases


def find_sodium_abnormality_cases(trajectories, low_threshold=130, high_threshold=150, min_measurements=2):
    """Find cases with significant sodium abnormalities (hypo/hypernatremia)."""
    sodium_cases = []

    for hadm_id, traj in trajectories.items():
        sodium_data = traj["biomarkers"].get("sodium", [])

        if len(sodium_data) < min_measurements:
            continue

        values = [m["value"] for m in sodium_data]
        min_sodium = min(values)
        max_sodium = max(values)
        initial_sodium = values[0]

        # Check for hyponatremia (low sodium - cerebral edema risk)
        if min_sodium <= low_threshold:
            for m in sodium_data:
                if m["value"] <= low_threshold:
                    sodium_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "abnormality_type": "hyponatremia",
                        "initial_sodium": round(initial_sodium, 1),
                        "min_sodium": round(min_sodium, 1),
                        "max_sodium": round(max_sodium, 1),
                        "hours_to_abnormality": m["hours_from_admission"],
                        "num_measurements": len(sodium_data)
                    })
                    break
        # Check for hypernatremia (high sodium - dehydration/osmotic injury)
        elif max_sodium >= high_threshold:
            for m in sodium_data:
                if m["value"] >= high_threshold:
                    sodium_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "abnormality_type": "hypernatremia",
                        "initial_sodium": round(initial_sodium, 1),
                        "min_sodium": round(min_sodium, 1),
                        "max_sodium": round(max_sodium, 1),
                        "hours_to_abnormality": m["hours_from_admission"],
                        "num_measurements": len(sodium_data)
                    })
                    break

    return sodium_cases


def find_glucose_abnormality_cases(trajectories, low_threshold=70, high_threshold=250, min_measurements=2):
    """Find cases with significant glucose abnormalities affecting brain function."""
    glucose_cases = []

    for hadm_id, traj in trajectories.items():
        glucose_data = traj["biomarkers"].get("glucose", [])

        if len(glucose_data) < min_measurements:
            continue

        values = [m["value"] for m in glucose_data]
        min_glucose = min(values)
        max_glucose = max(values)
        initial_glucose = values[0]

        # Check for hypoglycemia (dangerous for brain)
        if min_glucose <= low_threshold:
            for m in glucose_data:
                if m["value"] <= low_threshold:
                    glucose_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "abnormality_type": "hypoglycemia",
                        "initial_glucose": round(initial_glucose, 1),
                        "min_glucose": round(min_glucose, 1),
                        "max_glucose": round(max_glucose, 1),
                        "hours_to_abnormality": m["hours_from_admission"],
                        "num_measurements": len(glucose_data)
                    })
                    break
        # Check for severe hyperglycemia
        elif max_glucose >= high_threshold:
            for m in glucose_data:
                if m["value"] >= high_threshold:
                    glucose_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "abnormality_type": "hyperglycemia",
                        "initial_glucose": round(initial_glucose, 1),
                        "min_glucose": round(min_glucose, 1),
                        "max_glucose": round(max_glucose, 1),
                        "hours_to_abnormality": m["hours_from_admission"],
                        "num_measurements": len(glucose_data)
                    })
                    break

    return glucose_cases


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
                "domain": "neurological",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Neurological Cohort")
    print("=" * 60)

    # Load data
    lab_df = load_lab_events()
    chart_df = load_chart_events()
    admissions_df = load_admissions()

    # Build trajectories
    trajectories = build_patient_trajectories(lab_df, chart_df, admissions_df)

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_trajectory_stats(trajectories)

    # Find GCS deterioration cases
    print("Finding GCS deterioration cases...")
    gcs_cases = find_gcs_deterioration_cases(trajectories)
    print(f"Found {len(gcs_cases):,} GCS deterioration cases")

    # Find sodium abnormality cases
    print("Finding sodium abnormality cases...")
    sodium_cases = find_sodium_abnormality_cases(trajectories)
    print(f"Found {len(sodium_cases):,} sodium abnormality cases")

    # Find glucose abnormality cases
    print("Finding glucose abnormality cases...")
    glucose_cases = find_glucose_abnormality_cases(trajectories)
    print(f"Found {len(glucose_cases):,} glucose abnormality cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "neurological_cohort", "trajectories")
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

    # Save GCS cases
    gcs_file = os.path.join(output_dir, "gcs_deterioration_cases.json")
    with open(gcs_file, 'w') as f:
        json.dump(gcs_cases, f, indent=2)
    print(f"  Saved gcs_deterioration_cases.json ({len(gcs_cases):,} cases)")

    # Save sodium cases
    sodium_file = os.path.join(output_dir, "sodium_abnormality_cases.json")
    with open(sodium_file, 'w') as f:
        json.dump(sodium_cases, f, indent=2)
    print(f"  Saved sodium_abnormality_cases.json ({len(sodium_cases):,} cases)")

    # Save glucose cases
    glucose_file = os.path.join(output_dir, "glucose_abnormality_cases.json")
    with open(glucose_file, 'w') as f:
        json.dump(glucose_cases, f, indent=2)
    print(f"  Saved glucose_abnormality_cases.json ({len(glucose_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"GCS Deterioration Cases: {len(gcs_cases):,}")
    print(f"Sodium Abnormality Cases: {len(sodium_cases):,}")
    print(f"Glucose Abnormality Cases: {len(glucose_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:25s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, gcs_cases, sodium_cases, glucose_cases


if __name__ == "__main__":
    main()
