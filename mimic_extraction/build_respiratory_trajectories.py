"""Build biomarker trajectories from respiratory cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names (lab events)
LAB_ITEMID_TO_BIOMARKER = {
    # PaO2
    50821: "pao2", 52042: "pao2",
    # PaCO2
    50818: "paco2", 52040: "paco2",
    # SpO2 (lab)
    50817: "spo2_lab",
    # pH
    50820: "ph", 52038: "ph",
    # Lactate
    50813: "lactate", 52442: "lactate",
    # Hemoglobin
    51222: "hemoglobin", 50811: "hemoglobin",
    # Bicarbonate
    50882: "bicarbonate",
}

# Map MIMIC itemids to our biomarker names (chart events)
CHART_ITEMID_TO_BIOMARKER = {
    # SpO2 (pulse ox)
    220277: "spo2",
    # Respiratory rates
    220210: "respiratory_rate",
    224688: "respiratory_rate",
    224689: "respiratory_rate",
    224690: "respiratory_rate",
    # FiO2
    223835: "fio2",
    # PEEP
    220339: "peep",
    224700: "peep",
    # Tidal volumes
    224684: "tidal_volume",
    224685: "tidal_volume",
    224686: "tidal_volume",
}

# Normal/critical ranges for respiratory biomarkers
BIOMARKER_THRESHOLDS = {
    "spo2": {"normal": 95, "low": 92, "critical_low": 88, "unit": "%"},
    "pao2": {"normal": 80, "low": 60, "critical_low": 55, "unit": "mmHg"},
    "paco2": {"normal": 45, "elevated": 50, "critical_high": 60, "unit": "mmHg"},
    "respiratory_rate": {"normal": 20, "elevated": 25, "critical_high": 35, "unit": "/min"},
    "fio2": {"normal": 0.21, "elevated": 0.4, "high": 0.6, "critical": 0.8, "unit": "fraction"},
    "ph": {"low": 7.35, "high": 7.45, "critical_low": 7.25, "critical_high": 7.55, "unit": ""},
    "lactate": {"normal": 2.0, "elevated": 4.0, "critical": 6.0, "unit": "mmol/L"},
    "peep": {"normal": 5, "elevated": 10, "high": 15, "unit": "cmH2O"},
    "tidal_volume": {"normal": 500, "low": 300, "unit": "mL"},
    "hemoglobin": {"low": 12.0, "critical_low": 7.0, "unit": "g/dL"},
    "bicarbonate": {"low": 22, "high": 29, "critical_low": 15, "unit": "mEq/L"},
}


def load_lab_events():
    """Load lab events from respiratory cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "respiratory_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path, low_memory=False)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_chart_events():
    """Load chart events from respiratory cohort."""
    chart_path = os.path.join(OUTPUT_PATH, "respiratory_cohort", "chart_events.csv")
    print(f"Loading chart events from {chart_path}...")
    df = pd.read_csv(chart_path, low_memory=False)
    print(f"Loaded {len(df):,} chart events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "respiratory_cohort", "admissions.csv")
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

    # Combine lab and chart events
    print("  Combining lab and chart events...")

    # Prepare lab events
    lab_subset = lab_df[['hadm_id', 'subject_id', 'charttime', 'valuenum', 'biomarker']].copy()
    lab_subset = lab_subset.rename(columns={'valuenum': 'value'})

    # Prepare chart events
    chart_subset = chart_df[['hadm_id', 'subject_id', 'charttime', 'valuenum', 'biomarker']].copy()
    chart_subset = chart_subset.rename(columns={'valuenum': 'value'})

    # Combine
    combined_df = pd.concat([lab_subset, chart_subset], ignore_index=True)
    combined_df = combined_df.dropna(subset=['biomarker', 'value'])
    print(f"  Combined events: {len(combined_df):,}")

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
                if pd.notna(row['value']):
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


def find_hypoxemia_cases(trajectories, spo2_threshold=92, pao2_threshold=60, min_measurements=2):
    """Find cases with hypoxemia (low oxygen)."""
    hypoxemia_cases = []

    for hadm_id, traj in trajectories.items():
        # Check SpO2
        spo2_data = traj["biomarkers"].get("spo2", [])
        pao2_data = traj["biomarkers"].get("pao2", [])

        if len(spo2_data) >= min_measurements:
            values = [m["value"] for m in spo2_data]
            min_spo2 = min(values)
            initial_spo2 = values[0]

            if min_spo2 < spo2_threshold:
                for m in spo2_data:
                    if m["value"] < spo2_threshold:
                        hypoxemia_cases.append({
                            "hadm_id": hadm_id,
                            "subject_id": traj["subject_id"],
                            "type": "spo2",
                            "initial_value": round(initial_spo2, 1),
                            "min_value": round(min_spo2, 1),
                            "hours_to_hypoxemia": m["hours_from_admission"],
                            "num_measurements": len(spo2_data)
                        })
                        break

        elif len(pao2_data) >= min_measurements:
            values = [m["value"] for m in pao2_data]
            min_pao2 = min(values)
            initial_pao2 = values[0]

            if min_pao2 < pao2_threshold:
                for m in pao2_data:
                    if m["value"] < pao2_threshold:
                        hypoxemia_cases.append({
                            "hadm_id": hadm_id,
                            "subject_id": traj["subject_id"],
                            "type": "pao2",
                            "initial_value": round(initial_pao2, 1),
                            "min_value": round(min_pao2, 1),
                            "hours_to_hypoxemia": m["hours_from_admission"],
                            "num_measurements": len(pao2_data)
                        })
                        break

    return hypoxemia_cases


def find_respiratory_distress_cases(trajectories, rr_threshold=30, min_measurements=2):
    """Find cases with respiratory distress (high respiratory rate)."""
    distress_cases = []

    for hadm_id, traj in trajectories.items():
        rr_data = traj["biomarkers"].get("respiratory_rate", [])

        if len(rr_data) < min_measurements:
            continue

        values = [m["value"] for m in rr_data]
        max_rr = max(values)
        initial_rr = values[0]

        if max_rr >= rr_threshold:
            for m in rr_data:
                if m["value"] >= rr_threshold:
                    distress_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_rr": round(initial_rr, 1),
                        "max_rr": round(max_rr, 1),
                        "hours_to_distress": m["hours_from_admission"],
                        "num_measurements": len(rr_data)
                    })
                    break

    return distress_cases


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
                "domain": "respiratory",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Respiratory Cohort")
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

    # Find hypoxemia cases
    print("Finding hypoxemia cases...")
    hypoxemia_cases = find_hypoxemia_cases(trajectories)
    print(f"Found {len(hypoxemia_cases):,} hypoxemia cases")

    # Find respiratory distress cases
    print("Finding respiratory distress cases...")
    distress_cases = find_respiratory_distress_cases(trajectories)
    print(f"Found {len(distress_cases):,} respiratory distress cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "respiratory_cohort", "trajectories")
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

    # Save hypoxemia cases
    hypox_file = os.path.join(output_dir, "hypoxemia_cases.json")
    with open(hypox_file, 'w') as f:
        json.dump(hypoxemia_cases, f, indent=2)
    print(f"  Saved hypoxemia_cases.json ({len(hypoxemia_cases):,} cases)")

    # Save distress cases
    distress_file = os.path.join(output_dir, "respiratory_distress_cases.json")
    with open(distress_file, 'w') as f:
        json.dump(distress_cases, f, indent=2)
    print(f"  Saved respiratory_distress_cases.json ({len(distress_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"Hypoxemia Cases: {len(hypoxemia_cases):,}")
    print(f"Respiratory Distress Cases: {len(distress_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:18s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, hypoxemia_cases, distress_cases


if __name__ == "__main__":
    main()
