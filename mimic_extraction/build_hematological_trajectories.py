"""Build biomarker trajectories from hematological cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names
ITEMID_TO_BIOMARKER = {
    # Complete Blood Count
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    51279: "rbc",
    51301: "wbc", 51300: "wbc",
    51265: "platelets",
    # RBC Indices
    51250: "mcv",
    51248: "mch",
    51249: "mchc",
    51277: "rdw",
    # WBC Differential
    51256: "neutrophils",
    51244: "lymphocytes", 51245: "lymphocytes",
    51254: "monocytes",
    51200: "eosinophils",
    51146: "basophils",
    # Coagulation
    51237: "inr",
    51274: "pt",
    51275: "ptt",
    51214: "fibrinogen",
    51196: "d_dimer", 50915: "d_dimer",
    # Reticulocytes
    51278: "reticulocytes",
    # Iron studies
    50952: "iron",
    50924: "ferritin",
    50998: "tibc",
    # Hemolysis markers
    50954: "ldh",
    50934: "haptoglobin",
    50884: "bilirubin_indirect",
}

# Normal/critical ranges for hematological biomarkers
BIOMARKER_THRESHOLDS = {
    # CBC
    "hemoglobin": {"low": 12.0, "critical_low": 7.0, "high": 17.0, "unit": "g/dL"},
    "hematocrit": {"low": 36.0, "critical_low": 21.0, "high": 50.0, "unit": "%"},
    "rbc": {"low": 4.0, "critical_low": 2.5, "high": 6.0, "unit": "M/uL"},
    "wbc": {"low": 4.0, "critical_low": 1.0, "high": 11.0, "critical_high": 30.0, "unit": "K/uL"},
    "platelets": {"low": 150, "critical_low": 50, "very_critical_low": 20, "high": 400, "unit": "K/uL"},
    # RBC Indices
    "mcv": {"low": 80, "high": 100, "unit": "fL"},
    "mch": {"low": 27, "high": 33, "unit": "pg"},
    "mchc": {"low": 32, "high": 36, "unit": "g/dL"},
    "rdw": {"normal": 14.5, "elevated": 16.0, "unit": "%"},
    # WBC Differential
    "neutrophils": {"low": 1.5, "critical_low": 0.5, "unit": "K/uL"},
    "lymphocytes": {"low": 1.0, "high": 4.0, "unit": "K/uL"},
    # Coagulation
    "inr": {"normal": 1.1, "elevated": 1.5, "critical": 2.5, "very_critical": 4.0, "unit": "ratio"},
    "pt": {"normal": 12.0, "elevated": 15.0, "critical": 20.0, "unit": "seconds"},
    "ptt": {"normal": 35.0, "elevated": 50.0, "critical": 80.0, "unit": "seconds"},
    "fibrinogen": {"low": 200, "critical_low": 100, "high": 400, "unit": "mg/dL"},
    "d_dimer": {"normal": 0.5, "elevated": 2.0, "critical": 10.0, "unit": "ug/mL"},
    # Iron studies
    "iron": {"low": 60, "high": 170, "unit": "ug/dL"},
    "ferritin": {"low": 30, "high": 300, "unit": "ng/mL"},
    # Hemolysis
    "ldh": {"normal": 250, "elevated": 500, "critical": 1000, "unit": "U/L"},
    "haptoglobin": {"low": 30, "very_low": 10, "unit": "mg/dL"},
}


def load_lab_events():
    """Load lab events from hematological cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "hematological_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path, low_memory=False)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "hematological_cohort", "admissions.csv")
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


def find_severe_anemia_cases(trajectories, hgb_threshold=7.0, min_measurements=2):
    """Find cases with severe anemia (critical hemoglobin)."""
    anemia_cases = []

    for hadm_id, traj in trajectories.items():
        hgb_data = traj["biomarkers"].get("hemoglobin", [])

        if len(hgb_data) < min_measurements:
            continue

        values = [m["value"] for m in hgb_data]
        min_hgb = min(values)
        initial_hgb = values[0]

        if min_hgb <= hgb_threshold:
            for m in hgb_data:
                if m["value"] <= hgb_threshold:
                    anemia_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_hemoglobin": round(initial_hgb, 1),
                        "min_hemoglobin": round(min_hgb, 1),
                        "hours_to_critical": m["hours_from_admission"],
                        "num_measurements": len(hgb_data)
                    })
                    break

    return anemia_cases


def find_thrombocytopenia_cases(trajectories, plt_threshold=50, min_measurements=2):
    """Find cases with severe thrombocytopenia (bleeding risk)."""
    thrombocytopenia_cases = []

    for hadm_id, traj in trajectories.items():
        plt_data = traj["biomarkers"].get("platelets", [])

        if len(plt_data) < min_measurements:
            continue

        values = [m["value"] for m in plt_data]
        min_plt = min(values)
        initial_plt = values[0]

        if min_plt <= plt_threshold:
            for m in plt_data:
                if m["value"] <= plt_threshold:
                    thrombocytopenia_cases.append({
                        "hadm_id": hadm_id,
                        "subject_id": traj["subject_id"],
                        "initial_platelets": round(initial_plt, 0),
                        "min_platelets": round(min_plt, 0),
                        "hours_to_critical": m["hours_from_admission"],
                        "num_measurements": len(plt_data)
                    })
                    break

    return thrombocytopenia_cases


def find_coagulopathy_cases(trajectories, inr_threshold=2.5, min_measurements=2):
    """Find cases with significant coagulopathy (elevated INR)."""
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


def find_neutropenia_cases(trajectories, anc_threshold=0.5, wbc_threshold=1.5, min_measurements=2):
    """Find cases with severe neutropenia (infection risk)."""
    neutropenia_cases = []

    for hadm_id, traj in trajectories.items():
        # Check neutrophils first
        neut_data = traj["biomarkers"].get("neutrophils", [])
        wbc_data = traj["biomarkers"].get("wbc", [])

        # Use neutrophils if available
        if len(neut_data) >= min_measurements:
            values = [m["value"] for m in neut_data]
            min_neut = min(values)
            initial_neut = values[0]

            if min_neut <= anc_threshold:
                for m in neut_data:
                    if m["value"] <= anc_threshold:
                        neutropenia_cases.append({
                            "hadm_id": hadm_id,
                            "subject_id": traj["subject_id"],
                            "marker_type": "neutrophils",
                            "initial_value": round(initial_neut, 2),
                            "min_value": round(min_neut, 2),
                            "hours_to_critical": m["hours_from_admission"],
                            "num_measurements": len(neut_data)
                        })
                        break
        # Fall back to WBC if neutrophils not available
        elif len(wbc_data) >= min_measurements:
            values = [m["value"] for m in wbc_data]
            min_wbc = min(values)
            initial_wbc = values[0]

            if min_wbc <= wbc_threshold:
                for m in wbc_data:
                    if m["value"] <= wbc_threshold:
                        neutropenia_cases.append({
                            "hadm_id": hadm_id,
                            "subject_id": traj["subject_id"],
                            "marker_type": "wbc",
                            "initial_value": round(initial_wbc, 2),
                            "min_value": round(min_wbc, 2),
                            "hours_to_critical": m["hours_from_admission"],
                            "num_measurements": len(wbc_data)
                        })
                        break

    return neutropenia_cases


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
                "domain": "hematological",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Hematological Cohort")
    print("=" * 60)

    # Load data
    lab_df = load_lab_events()
    admissions_df = load_admissions()

    # Build trajectories
    trajectories = build_patient_trajectories(lab_df, admissions_df)

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_trajectory_stats(trajectories)

    # Find severe anemia cases
    print("Finding severe anemia cases...")
    anemia_cases = find_severe_anemia_cases(trajectories)
    print(f"Found {len(anemia_cases):,} severe anemia cases")

    # Find thrombocytopenia cases
    print("Finding thrombocytopenia cases...")
    thrombocytopenia_cases = find_thrombocytopenia_cases(trajectories)
    print(f"Found {len(thrombocytopenia_cases):,} thrombocytopenia cases")

    # Find coagulopathy cases
    print("Finding coagulopathy cases...")
    coagulopathy_cases = find_coagulopathy_cases(trajectories)
    print(f"Found {len(coagulopathy_cases):,} coagulopathy cases")

    # Find neutropenia cases
    print("Finding neutropenia cases...")
    neutropenia_cases = find_neutropenia_cases(trajectories)
    print(f"Found {len(neutropenia_cases):,} neutropenia cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "hematological_cohort", "trajectories")
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

    # Save anemia cases
    anemia_file = os.path.join(output_dir, "severe_anemia_cases.json")
    with open(anemia_file, 'w') as f:
        json.dump(anemia_cases, f, indent=2)
    print(f"  Saved severe_anemia_cases.json ({len(anemia_cases):,} cases)")

    # Save thrombocytopenia cases
    thromb_file = os.path.join(output_dir, "thrombocytopenia_cases.json")
    with open(thromb_file, 'w') as f:
        json.dump(thrombocytopenia_cases, f, indent=2)
    print(f"  Saved thrombocytopenia_cases.json ({len(thrombocytopenia_cases):,} cases)")

    # Save coagulopathy cases
    coag_file = os.path.join(output_dir, "coagulopathy_cases.json")
    with open(coag_file, 'w') as f:
        json.dump(coagulopathy_cases, f, indent=2)
    print(f"  Saved coagulopathy_cases.json ({len(coagulopathy_cases):,} cases)")

    # Save neutropenia cases
    neut_file = os.path.join(output_dir, "neutropenia_cases.json")
    with open(neut_file, 'w') as f:
        json.dump(neutropenia_cases, f, indent=2)
    print(f"  Saved neutropenia_cases.json ({len(neutropenia_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"Severe Anemia Cases: {len(anemia_cases):,}")
    print(f"Thrombocytopenia Cases: {len(thrombocytopenia_cases):,}")
    print(f"Coagulopathy Cases: {len(coagulopathy_cases):,}")
    print(f"Neutropenia Cases: {len(neutropenia_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:20s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, anemia_cases, thrombocytopenia_cases, coagulopathy_cases, neutropenia_cases


if __name__ == "__main__":
    main()
