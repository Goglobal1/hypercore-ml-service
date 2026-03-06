"""Build biomarker trajectories from hepatic cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names
ITEMID_TO_BIOMARKER = {
    # ALT
    50861: "alt",
    # AST
    50878: "ast",
    # Bilirubin
    50885: "bilirubin_total",
    50883: "bilirubin_direct",
    # Albumin
    50862: "albumin",
    # Alkaline Phosphatase
    50863: "alkaline_phosphatase",
    # Ammonia
    50866: "ammonia",
    # INR
    51237: "inr",
    # PTT
    51275: "ptt",
    # Platelets
    51265: "platelets",
    # Creatinine
    50912: "creatinine", 52546: "creatinine",
    # Sodium
    50983: "sodium", 50824: "sodium",
    # Lactate
    50813: "lactate", 52442: "lactate",
}

# Normal/critical ranges for hepatic biomarkers
BIOMARKER_THRESHOLDS = {
    "alt": {"normal": 40, "elevated": 120, "critical": 1000, "unit": "U/L"},
    "ast": {"normal": 40, "elevated": 120, "critical": 1000, "unit": "U/L"},
    "bilirubin_total": {"normal": 1.2, "elevated": 3.0, "critical": 10.0, "unit": "mg/dL"},
    "bilirubin_direct": {"normal": 0.3, "elevated": 1.0, "critical": 5.0, "unit": "mg/dL"},
    "albumin": {"normal": 3.5, "low": 2.8, "critical_low": 2.0, "unit": "g/dL"},
    "alkaline_phosphatase": {"normal": 120, "elevated": 300, "critical": 500, "unit": "U/L"},
    "ammonia": {"normal": 50, "elevated": 80, "critical": 150, "unit": "umol/L"},
    "inr": {"normal": 1.1, "elevated": 1.5, "critical": 2.5, "unit": "ratio"},
    "ptt": {"normal": 35, "elevated": 50, "critical": 80, "unit": "seconds"},
    "platelets": {"normal": 150, "low": 100, "critical_low": 50, "unit": "K/uL"},
    "creatinine": {"normal": 1.2, "elevated": 2.0, "critical": 4.0, "unit": "mg/dL"},
    "sodium": {"low": 136, "high": 145, "critical_low": 125, "unit": "mEq/L"},
    "lactate": {"normal": 2.0, "elevated": 4.0, "critical": 6.0, "unit": "mmol/L"},
}


def load_lab_events():
    """Load lab events from hepatic cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "hepatic_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path, low_memory=False)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "hepatic_cohort", "admissions.csv")
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


def find_bilirubin_elevations(trajectories, threshold=3.0, min_measurements=2):
    """Find cases with significant bilirubin elevation (jaundice/liver dysfunction)."""
    elevation_cases = []

    for hadm_id, traj in trajectories.items():
        bili_data = traj["biomarkers"].get("bilirubin_total", [])

        if len(bili_data) < min_measurements:
            continue

        values = [m["value"] for m in bili_data]
        max_bili = max(values)
        initial_bili = values[0]

        if max_bili >= threshold:
            for m in bili_data:
                if m["value"] >= threshold:
                    elevation_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_bilirubin": round(initial_bili, 2),
                        "max_bilirubin": round(max_bili, 2),
                        "hours_to_elevation": m["hours_from_admission"],
                        "num_measurements": len(bili_data)
                    })
                    break

    return elevation_cases


def find_coagulopathy_cases(trajectories, inr_threshold=1.5, min_measurements=2):
    """Find cases with coagulopathy (elevated INR)."""
    coagulopathy_cases = []

    for hadm_id, traj in trajectories.items():
        inr_data = traj["biomarkers"].get("inr", [])

        if len(inr_data) < min_measurements:
            continue

        values = [m["value"] for m in inr_data]
        max_inr = max(values)
        initial_inr = values[0]

        if max_inr >= inr_threshold:
            for m in inr_data:
                if m["value"] >= inr_threshold:
                    coagulopathy_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_inr": round(initial_inr, 2),
                        "max_inr": round(max_inr, 2),
                        "hours_to_coagulopathy": m["hours_from_admission"],
                        "num_measurements": len(inr_data)
                    })
                    break

    return coagulopathy_cases


def find_hepatic_encephalopathy_risk(trajectories, ammonia_threshold=80, min_measurements=2):
    """Find cases with elevated ammonia (encephalopathy risk)."""
    encephalopathy_cases = []

    for hadm_id, traj in trajectories.items():
        ammonia_data = traj["biomarkers"].get("ammonia", [])

        if len(ammonia_data) < min_measurements:
            continue

        values = [m["value"] for m in ammonia_data]
        max_ammonia = max(values)
        initial_ammonia = values[0]

        if max_ammonia >= ammonia_threshold:
            for m in ammonia_data:
                if m["value"] >= ammonia_threshold:
                    encephalopathy_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_ammonia": round(initial_ammonia, 1),
                        "max_ammonia": round(max_ammonia, 1),
                        "hours_to_elevation": m["hours_from_admission"],
                        "num_measurements": len(ammonia_data)
                    })
                    break

    return encephalopathy_cases


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
                "domain": "hepatic",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Hepatic Cohort")
    print("=" * 60)

    # Load data
    lab_df = load_lab_events()
    admissions_df = load_admissions()

    # Build trajectories
    trajectories = build_patient_trajectories(lab_df, admissions_df)

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_trajectory_stats(trajectories)

    # Find bilirubin elevations
    print("Finding bilirubin elevation cases...")
    bilirubin_cases = find_bilirubin_elevations(trajectories)
    print(f"Found {len(bilirubin_cases):,} bilirubin elevation cases")

    # Find coagulopathy cases
    print("Finding coagulopathy cases...")
    coagulopathy_cases = find_coagulopathy_cases(trajectories)
    print(f"Found {len(coagulopathy_cases):,} coagulopathy cases")

    # Find encephalopathy risk cases
    print("Finding hepatic encephalopathy risk cases...")
    encephalopathy_cases = find_hepatic_encephalopathy_risk(trajectories)
    print(f"Found {len(encephalopathy_cases):,} encephalopathy risk cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "hepatic_cohort", "trajectories")
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

    # Save bilirubin cases
    bili_file = os.path.join(output_dir, "bilirubin_elevation_cases.json")
    with open(bili_file, 'w') as f:
        json.dump(bilirubin_cases, f, indent=2)
    print(f"  Saved bilirubin_elevation_cases.json ({len(bilirubin_cases):,} cases)")

    # Save coagulopathy cases
    coag_file = os.path.join(output_dir, "coagulopathy_cases.json")
    with open(coag_file, 'w') as f:
        json.dump(coagulopathy_cases, f, indent=2)
    print(f"  Saved coagulopathy_cases.json ({len(coagulopathy_cases):,} cases)")

    # Save encephalopathy cases
    enceph_file = os.path.join(output_dir, "encephalopathy_risk_cases.json")
    with open(enceph_file, 'w') as f:
        json.dump(encephalopathy_cases, f, indent=2)
    print(f"  Saved encephalopathy_risk_cases.json ({len(encephalopathy_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"Bilirubin Elevation Cases: {len(bilirubin_cases):,}")
    print(f"Coagulopathy Cases: {len(coagulopathy_cases):,}")
    print(f"Encephalopathy Risk Cases: {len(encephalopathy_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:20s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, bilirubin_cases, coagulopathy_cases, encephalopathy_cases


if __name__ == "__main__":
    main()
