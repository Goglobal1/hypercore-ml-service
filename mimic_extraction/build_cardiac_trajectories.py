"""Build biomarker trajectories from cardiac cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names
ITEMID_TO_BIOMARKER = {
    # Troponin
    51002: "troponin", 51003: "troponin", 52642: "troponin",
    # BNP
    50963: "bnp", 51921: "bnp",
    # CK-MB
    50908: "ck_mb", 51580: "ck_mb",
    # Creatinine
    50912: "creatinine", 52546: "creatinine",
    # Potassium
    50971: "potassium", 50822: "potassium",
    # Hemoglobin
    51222: "hemoglobin", 50811: "hemoglobin",
    # Platelets
    51265: "platelets",
    # INR
    51237: "inr",
    # Lactate
    50813: "lactate", 52442: "lactate", 53154: "lactate",
    # BUN
    51006: "bun",
}

# Normal/critical ranges for cardiac biomarkers
BIOMARKER_THRESHOLDS = {
    "troponin": {"normal": 0.04, "elevated": 0.1, "critical": 0.5, "unit": "ng/mL"},
    "bnp": {"normal": 100, "elevated": 400, "critical": 900, "unit": "pg/mL"},
    "ck_mb": {"normal": 5, "elevated": 10, "critical": 25, "unit": "ng/mL"},
    "creatinine": {"normal": 1.2, "elevated": 2.0, "critical": 4.0, "unit": "mg/dL"},
    "potassium": {"low": 3.5, "high": 5.0, "critical_low": 3.0, "critical_high": 6.0, "unit": "mEq/L"},
    "hemoglobin": {"low": 12.0, "critical_low": 7.0, "unit": "g/dL"},
    "platelets": {"low": 150, "critical_low": 50, "unit": "K/uL"},
    "inr": {"normal": 1.1, "elevated": 2.0, "critical": 4.0, "unit": "ratio"},
    "lactate": {"normal": 2.0, "elevated": 4.0, "critical": 6.0, "unit": "mmol/L"},
    "bun": {"normal": 20, "elevated": 40, "critical": 80, "unit": "mg/dL"},
}


def load_lab_events():
    """Load lab events from cardiac cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "cardiac_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "cardiac_cohort", "admissions.csv")
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


def find_troponin_elevations(trajectories, threshold=0.1, min_measurements=2):
    """Find cases with troponin elevation (potential MI)."""
    elevation_cases = []

    for hadm_id, traj in trajectories.items():
        troponin_data = traj["biomarkers"].get("troponin", [])

        if len(troponin_data) < min_measurements:
            continue

        values = [m["value"] for m in troponin_data]
        max_troponin = max(values)
        initial_troponin = values[0]

        if max_troponin >= threshold:
            # Find time to elevation
            for m in troponin_data:
                if m["value"] >= threshold:
                    elevation_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_troponin": round(initial_troponin, 3),
                        "max_troponin": round(max_troponin, 3),
                        "hours_to_elevation": m["hours_from_admission"],
                        "num_measurements": len(troponin_data)
                    })
                    break

    return elevation_cases


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
                "domain": "cardiac",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Cardiac Cohort")
    print("=" * 60)

    # Load data
    lab_df = load_lab_events()
    admissions_df = load_admissions()

    # Build trajectories
    trajectories = build_patient_trajectories(lab_df, admissions_df)

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_trajectory_stats(trajectories)

    # Find troponin elevations
    print("Finding troponin elevation cases...")
    elevations = find_troponin_elevations(trajectories)
    print(f"Found {len(elevations):,} troponin elevation cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "cardiac_cohort", "trajectories")
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

    # Save elevation cases
    elev_file = os.path.join(output_dir, "troponin_elevation_cases.json")
    with open(elev_file, 'w') as f:
        json.dump(elevations, f, indent=2)
    print(f"  Saved troponin_elevation_cases.json ({len(elevations):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"Troponin Elevation Cases: {len(elevations):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:12s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, elevations


if __name__ == "__main__":
    main()
