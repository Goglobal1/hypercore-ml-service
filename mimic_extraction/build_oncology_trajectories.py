"""Build biomarker trajectories from oncology cohort data."""
import pandas as pd
import json
import os
from datetime import datetime
from collections import defaultdict
from config import OUTPUT_PATH

# Map MIMIC itemids to our biomarker names
ITEMID_TO_BIOMARKER = {
    # CBC
    51301: "wbc", 51300: "wbc",
    51256: "neutrophils",
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    51265: "platelets",
    # Tumor lysis syndrome markers
    50971: "potassium", 50822: "potassium",
    50970: "phosphate",
    50893: "calcium",
    51007: "uric_acid",
    # Renal
    50912: "creatinine", 52546: "creatinine",
    51006: "bun",
    # LDH
    50954: "ldh",
    # Liver
    50861: "alt",
    50878: "ast",
    50885: "bilirubin_total",
    50862: "albumin",
    # Coagulation
    51237: "inr",
    51275: "ptt",
    51214: "fibrinogen",
    51196: "d_dimer", 50915: "d_dimer",
    # Electrolytes
    50983: "sodium", 50824: "sodium",
    50960: "magnesium",
    # Tissue perfusion
    50813: "lactate", 52442: "lactate",
}

# Normal/critical ranges for oncology biomarkers
BIOMARKER_THRESHOLDS = {
    # CBC
    "wbc": {"low": 4.0, "critical_low": 1.0, "very_critical_low": 0.5, "high": 11.0, "unit": "K/uL"},
    "neutrophils": {"low": 1.5, "critical_low": 0.5, "very_critical_low": 0.1, "unit": "K/uL"},
    "hemoglobin": {"low": 12.0, "critical_low": 7.0, "very_critical_low": 5.0, "unit": "g/dL"},
    "hematocrit": {"low": 36.0, "critical_low": 21.0, "unit": "%"},
    "platelets": {"low": 150, "critical_low": 50, "very_critical_low": 20, "unit": "K/uL"},
    # Tumor lysis syndrome
    "potassium": {"high": 5.0, "critical_high": 6.0, "very_critical_high": 7.0, "low": 3.5, "unit": "mEq/L"},
    "phosphate": {"high": 4.5, "critical_high": 6.0, "very_critical_high": 8.0, "unit": "mg/dL"},
    "calcium": {"low": 8.5, "critical_low": 7.0, "very_critical_low": 6.0, "unit": "mg/dL"},
    "uric_acid": {"high": 7.0, "critical_high": 10.0, "very_critical_high": 15.0, "unit": "mg/dL"},
    # Renal
    "creatinine": {"normal": 1.2, "elevated": 2.0, "critical": 4.0, "unit": "mg/dL"},
    "bun": {"normal": 20, "elevated": 40, "critical": 80, "unit": "mg/dL"},
    # LDH
    "ldh": {"normal": 250, "elevated": 500, "high": 1000, "very_high": 2000, "unit": "U/L"},
    # Liver
    "alt": {"normal": 40, "elevated": 120, "critical": 1000, "unit": "U/L"},
    "ast": {"normal": 40, "elevated": 120, "critical": 1000, "unit": "U/L"},
    "bilirubin_total": {"normal": 1.2, "elevated": 3.0, "critical": 10.0, "unit": "mg/dL"},
    "albumin": {"low": 3.5, "critical_low": 2.5, "unit": "g/dL"},
    # Coagulation (DIC)
    "inr": {"normal": 1.1, "elevated": 1.5, "critical": 2.5, "unit": "ratio"},
    "ptt": {"normal": 35, "elevated": 50, "critical": 80, "unit": "seconds"},
    "fibrinogen": {"low": 200, "critical_low": 100, "unit": "mg/dL"},
    "d_dimer": {"normal": 0.5, "elevated": 2.0, "critical": 10.0, "unit": "ug/mL"},
    # Electrolytes
    "sodium": {"low": 136, "critical_low": 125, "high": 145, "unit": "mEq/L"},
    "magnesium": {"low": 1.7, "critical_low": 1.0, "unit": "mg/dL"},
    # Tissue perfusion
    "lactate": {"normal": 2.0, "elevated": 4.0, "critical": 6.0, "unit": "mmol/L"},
}


def load_lab_events():
    """Load lab events from oncology cohort."""
    lab_path = os.path.join(OUTPUT_PATH, "oncology_cohort", "lab_events.csv")
    print(f"Loading lab events from {lab_path}...")
    df = pd.read_csv(lab_path, low_memory=False)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for reference times."""
    adm_path = os.path.join(OUTPUT_PATH, "oncology_cohort", "admissions.csv")
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


def find_neutropenia_cases(trajectories, anc_threshold=0.5, wbc_threshold=1.0, min_measurements=2):
    """Find cases with severe neutropenia (febrile neutropenia risk)."""
    neutropenia_cases = []

    for hadm_id, traj in trajectories.items():
        neut_data = traj["biomarkers"].get("neutrophils", [])
        wbc_data = traj["biomarkers"].get("wbc", [])

        # Check neutrophils first
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
                            "hours_to_nadir": m["hours_from_admission"],
                            "num_measurements": len(neut_data)
                        })
                        break
        # Fall back to WBC
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
                            "hours_to_nadir": m["hours_from_admission"],
                            "num_measurements": len(wbc_data)
                        })
                        break

    return neutropenia_cases


def find_tumor_lysis_cases(trajectories, k_threshold=6.0, phos_threshold=6.0, uric_threshold=10.0, min_measurements=2):
    """Find cases suggestive of tumor lysis syndrome."""
    tls_cases = []

    for hadm_id, traj in trajectories.items():
        k_data = traj["biomarkers"].get("potassium", [])
        phos_data = traj["biomarkers"].get("phosphate", [])
        uric_data = traj["biomarkers"].get("uric_acid", [])

        tls_markers = {}

        # Check potassium
        if len(k_data) >= min_measurements:
            values = [m["value"] for m in k_data]
            max_k = max(values)
            if max_k >= k_threshold:
                tls_markers["potassium"] = {"max": round(max_k, 2), "initial": round(values[0], 2)}

        # Check phosphate
        if len(phos_data) >= min_measurements:
            values = [m["value"] for m in phos_data]
            max_phos = max(values)
            if max_phos >= phos_threshold:
                tls_markers["phosphate"] = {"max": round(max_phos, 2), "initial": round(values[0], 2)}

        # Check uric acid
        if len(uric_data) >= min_measurements:
            values = [m["value"] for m in uric_data]
            max_uric = max(values)
            if max_uric >= uric_threshold:
                tls_markers["uric_acid"] = {"max": round(max_uric, 2), "initial": round(values[0], 2)}

        # TLS requires at least 2 abnormal markers
        if len(tls_markers) >= 2:
            tls_cases.append({
                "hadm_id": hadm_id,
                "subject_id": traj["subject_id"],
                "abnormal_markers": list(tls_markers.keys()),
                "marker_details": tls_markers,
                "num_criteria_met": len(tls_markers)
            })

    return tls_cases


def find_thrombocytopenia_cases(trajectories, plt_threshold=20, min_measurements=2):
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
                        "hours_to_nadir": m["hours_from_admission"],
                        "num_measurements": len(plt_data)
                    })
                    break

    return thrombocytopenia_cases


def find_severe_anemia_cases(trajectories, hgb_threshold=7.0, min_measurements=2):
    """Find cases with severe anemia."""
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
                        "hours_to_nadir": m["hours_from_admission"],
                        "num_measurements": len(hgb_data)
                    })
                    break

    return anemia_cases


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
                "domain": "oncology",
                "biomarker_trajectories": biomarker_trajectories
            })
            count += 1

    return tth_format


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Oncology Cohort")
    print("=" * 60)

    # Load data
    lab_df = load_lab_events()
    admissions_df = load_admissions()

    # Build trajectories
    trajectories = build_patient_trajectories(lab_df, admissions_df)

    # Compute stats
    print("\nComputing statistics...")
    stats = compute_trajectory_stats(trajectories)

    # Find neutropenia cases
    print("Finding neutropenia cases...")
    neutropenia_cases = find_neutropenia_cases(trajectories)
    print(f"Found {len(neutropenia_cases):,} neutropenia cases")

    # Find tumor lysis syndrome cases
    print("Finding tumor lysis syndrome cases...")
    tls_cases = find_tumor_lysis_cases(trajectories)
    print(f"Found {len(tls_cases):,} tumor lysis syndrome cases")

    # Find thrombocytopenia cases
    print("Finding severe thrombocytopenia cases...")
    thrombocytopenia_cases = find_thrombocytopenia_cases(trajectories)
    print(f"Found {len(thrombocytopenia_cases):,} severe thrombocytopenia cases")

    # Find severe anemia cases
    print("Finding severe anemia cases...")
    anemia_cases = find_severe_anemia_cases(trajectories)
    print(f"Found {len(anemia_cases):,} severe anemia cases")

    # Format for TTH engine (sample)
    print("\nFormatting for Time-to-Harm engine...")
    tth_sample = format_for_tth_engine(trajectories, max_patients=100)

    # Save outputs
    output_dir = os.path.join(OUTPUT_PATH, "oncology_cohort", "trajectories")
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

    # Save neutropenia cases
    neut_file = os.path.join(output_dir, "neutropenia_cases.json")
    with open(neut_file, 'w') as f:
        json.dump(neutropenia_cases, f, indent=2)
    print(f"  Saved neutropenia_cases.json ({len(neutropenia_cases):,} cases)")

    # Save TLS cases
    tls_file = os.path.join(output_dir, "tumor_lysis_cases.json")
    with open(tls_file, 'w') as f:
        json.dump(tls_cases, f, indent=2)
    print(f"  Saved tumor_lysis_cases.json ({len(tls_cases):,} cases)")

    # Save thrombocytopenia cases
    thromb_file = os.path.join(output_dir, "thrombocytopenia_cases.json")
    with open(thromb_file, 'w') as f:
        json.dump(thrombocytopenia_cases, f, indent=2)
    print(f"  Saved thrombocytopenia_cases.json ({len(thrombocytopenia_cases):,} cases)")

    # Save anemia cases
    anemia_file = os.path.join(output_dir, "severe_anemia_cases.json")
    with open(anemia_file, 'w') as f:
        json.dump(anemia_cases, f, indent=2)
    print(f"  Saved severe_anemia_cases.json ({len(anemia_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {stats['total_admissions_with_trajectories']:,}")
    print(f"Avg Measurements per Admission: {stats['avg_total_measurements_per_admission']}")
    print(f"Neutropenia Cases: {len(neutropenia_cases):,}")
    print(f"Tumor Lysis Syndrome Cases: {len(tls_cases):,}")
    print(f"Severe Thrombocytopenia Cases: {len(thrombocytopenia_cases):,}")
    print(f"Severe Anemia Cases: {len(anemia_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for bio, bio_stats in sorted(stats['biomarker_stats'].items(),
                                  key=lambda x: x[1]['coverage_pct'],
                                  reverse=True):
        print(f"  {bio:20s}: {bio_stats['coverage_pct']:5.1f}% ({bio_stats['admissions_with_data']:,} admissions, "
              f"avg {bio_stats['avg_measurements']:.1f} measurements)")
    print("=" * 60)

    return trajectories, stats, neutropenia_cases, tls_cases, thrombocytopenia_cases, anemia_cases


if __name__ == "__main__":
    main()
