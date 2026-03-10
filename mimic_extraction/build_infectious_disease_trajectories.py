"""Build biomarker trajectories for infectious disease cohort."""
import pandas as pd
import json
import os
from datetime import datetime
from config import OUTPUT_PATH

# Map itemids to biomarker names
ITEMID_TO_BIOMARKER = {
    # Infection/inflammation markers
    51301: "wbc", 51300: "wbc",
    51256: "neutrophils",
    51144: "bands",
    51244: "lymphocytes", 51245: "lymphocytes",
    50889: "crp",
    50976: "procalcitonin",
    # Sepsis markers
    50813: "lactate", 52442: "lactate",
    # Organ dysfunction markers
    50912: "creatinine", 52546: "creatinine",
    51006: "bun",
    50885: "bilirubin_total",
    50861: "alt",
    50878: "ast",
    51237: "inr",
    51265: "platelets",
    # Blood gas
    50820: "ph",
    50821: "pao2",
    50818: "pco2",
    50802: "base_excess",
    # Electrolytes
    50983: "sodium", 50824: "sodium",
    50971: "potassium", 50822: "potassium",
    50809: "glucose", 50931: "glucose",
    50882: "bicarbonate",
    # CBC
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    # Albumin
    50862: "albumin",
}


def load_lab_events():
    """Load lab events from extracted infectious disease cohort."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "lab_events.csv")
    print(f"Loading lab events from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for timing reference."""
    path = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "admissions.csv")
    print("Loading admissions...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} admissions")
    return df


def build_trajectories(labs, admissions):
    """Build biomarker trajectories for each admission."""
    print("\nBuilding trajectories...")

    # Create admission time lookup
    adm_times = {}
    for _, row in admissions.iterrows():
        adm_times[row['hadm_id']] = row['admittime']

    # Group labs by hadm_id
    trajectories = {}
    labs_sorted = labs.sort_values(['hadm_id', 'charttime'])

    hadm_ids = labs_sorted['hadm_id'].unique()
    total = len(hadm_ids)

    for i, hadm_id_raw in enumerate(hadm_ids):
        if i % 5000 == 0:
            print(f"  Processing admission {i:,}/{total:,}...", end='\r')

        hadm_id = int(hadm_id_raw)  # Ensure integer key
        hadm_labs = labs_sorted[labs_sorted['hadm_id'] == hadm_id_raw]
        admit_time = adm_times.get(hadm_id_raw) or adm_times.get(hadm_id)

        if pd.isna(admit_time):
            continue

        admit_dt = pd.to_datetime(admit_time)
        biomarkers = {}

        for _, lab in hadm_labs.iterrows():
            itemid = lab['itemid']
            biomarker = ITEMID_TO_BIOMARKER.get(itemid)

            if biomarker and pd.notna(lab['valuenum']):
                if biomarker not in biomarkers:
                    biomarkers[biomarker] = []

                chart_dt = pd.to_datetime(lab['charttime'])
                hours_since_admit = (chart_dt - admit_dt).total_seconds() / 3600

                biomarkers[biomarker].append({
                    "timestamp": lab['charttime'],
                    "value": float(lab['valuenum']),
                    "hours_since_admit": round(hours_since_admit, 2)
                })

        if biomarkers:
            # Get subject_id from admissions
            subj_match = admissions[admissions['hadm_id'] == hadm_id_raw]
            if len(subj_match) == 0:
                subj_match = admissions[admissions['hadm_id'] == hadm_id]
            subj = subj_match['subject_id'].iloc[0] if len(subj_match) > 0 else 0
            trajectories[str(hadm_id)] = {
                "hadm_id": hadm_id,
                "subject_id": int(subj),
                "admit_time": admit_time,
                "biomarkers": biomarkers
            }

    print(f"\n  Built trajectories for {len(trajectories):,} admissions")
    return trajectories


def compute_stats(trajectories):
    """Compute statistics across trajectories."""
    print("\nComputing statistics...")

    biomarker_counts = {}
    measurement_counts = {}

    for traj in trajectories.values():
        for biomarker, measurements in traj["biomarkers"].items():
            if biomarker not in biomarker_counts:
                biomarker_counts[biomarker] = 0
                measurement_counts[biomarker] = 0
            biomarker_counts[biomarker] += 1
            measurement_counts[biomarker] += len(measurements)

    stats = {
        "total_admissions": len(trajectories),
        "biomarker_coverage": {},
    }

    for biomarker in sorted(biomarker_counts.keys(), key=lambda x: -biomarker_counts[x]):
        count = biomarker_counts[biomarker]
        total_measurements = measurement_counts[biomarker]
        stats["biomarker_coverage"][biomarker] = {
            "admissions_with_data": count,
            "pct_coverage": round(100 * count / len(trajectories), 1),
            "avg_measurements": round(total_measurements / count, 1) if count > 0 else 0
        }

    return stats


def find_sepsis_cases(trajectories, lactate_threshold=2.0, wbc_high=12.0, wbc_low=4.0):
    """Find cases with sepsis markers (elevated lactate + abnormal WBC)."""
    print("Finding sepsis cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        lactate_data = traj["biomarkers"].get("lactate", [])
        wbc_data = traj["biomarkers"].get("wbc", [])

        if lactate_data and wbc_data:
            max_lactate = max(m["value"] for m in lactate_data)
            max_wbc = max(m["value"] for m in wbc_data)
            min_wbc = min(m["value"] for m in wbc_data)

            # Sepsis: elevated lactate + abnormal WBC (high or low)
            has_elevated_lactate = max_lactate >= lactate_threshold
            has_abnormal_wbc = max_wbc > wbc_high or min_wbc < wbc_low

            if has_elevated_lactate and has_abnormal_wbc:
                cases.append({
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "max_lactate": round(max_lactate, 1),
                    "max_wbc": round(max_wbc, 1),
                    "min_wbc": round(min_wbc, 1),
                })

    print(f"Found {len(cases):,} sepsis cases")
    return cases


def find_severe_sepsis_cases(trajectories, lactate_threshold=4.0, cr_threshold=2.0, plt_threshold=100):
    """Find severe sepsis cases (high lactate + organ dysfunction)."""
    print("Finding severe sepsis cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        lactate_data = traj["biomarkers"].get("lactate", [])
        cr_data = traj["biomarkers"].get("creatinine", [])
        plt_data = traj["biomarkers"].get("platelets", [])
        bili_data = traj["biomarkers"].get("bilirubin_total", [])

        if lactate_data:
            max_lactate = max(m["value"] for m in lactate_data)

            if max_lactate >= lactate_threshold:
                organ_dysfunction = []
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "max_lactate": round(max_lactate, 1),
                }

                # Check for organ dysfunction
                if cr_data:
                    max_cr = max(m["value"] for m in cr_data)
                    if max_cr >= cr_threshold:
                        organ_dysfunction.append("renal")
                        case_data["max_creatinine"] = round(max_cr, 2)

                if plt_data:
                    min_plt = min(m["value"] for m in plt_data)
                    if min_plt < plt_threshold:
                        organ_dysfunction.append("hematologic")
                        case_data["min_platelets"] = round(min_plt, 0)

                if bili_data:
                    max_bili = max(m["value"] for m in bili_data)
                    if max_bili > 2.0:
                        organ_dysfunction.append("hepatic")
                        case_data["max_bilirubin"] = round(max_bili, 1)

                if organ_dysfunction:
                    case_data["organ_dysfunction"] = organ_dysfunction
                    cases.append(case_data)

    print(f"Found {len(cases):,} severe sepsis cases")
    return cases


def find_leukocytosis_cases(trajectories, wbc_threshold=20.0):
    """Find cases with severe leukocytosis."""
    print("Finding leukocytosis cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        wbc_data = traj["biomarkers"].get("wbc", [])

        if wbc_data:
            initial_wbc = wbc_data[0]["value"]
            max_wbc = max(m["value"] for m in wbc_data)

            if max_wbc >= wbc_threshold:
                max_idx = [i for i, m in enumerate(wbc_data) if m["value"] == max_wbc][0]
                hours_to_peak = wbc_data[max_idx]["hours_since_admit"]

                cases.append({
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "initial_wbc": round(initial_wbc, 1),
                    "max_wbc": round(max_wbc, 1),
                    "hours_to_peak": round(hours_to_peak, 1),
                })

    print(f"Found {len(cases):,} leukocytosis cases")
    return cases


def find_leukopenia_cases(trajectories, wbc_threshold=2.0):
    """Find cases with severe leukopenia (immunocompromised)."""
    print("Finding leukopenia cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        wbc_data = traj["biomarkers"].get("wbc", [])

        if wbc_data:
            initial_wbc = wbc_data[0]["value"]
            min_wbc = min(m["value"] for m in wbc_data)

            if min_wbc <= wbc_threshold:
                min_idx = [i for i, m in enumerate(wbc_data) if m["value"] == min_wbc][0]
                hours_to_nadir = wbc_data[min_idx]["hours_since_admit"]

                cases.append({
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "initial_wbc": round(initial_wbc, 1),
                    "min_wbc": round(min_wbc, 1),
                    "hours_to_nadir": round(hours_to_nadir, 1),
                })

    print(f"Found {len(cases):,} leukopenia cases")
    return cases


def format_for_tth(trajectories, sample_size=100):
    """Format sample trajectories for Time-to-Harm engine testing."""
    print("\nFormatting for Time-to-Harm engine...")

    sample_ids = list(trajectories.keys())[:sample_size]
    tth_sample = {}

    for hadm_id in sample_ids:
        traj = trajectories[hadm_id]
        tth_sample[hadm_id] = {
            "patient_id": f"MIMIC-{traj['subject_id']}",
            "domain": "infectious_disease",
            "biomarker_trajectories": {
                biomarker: [{"timestamp": m["timestamp"], "value": m["value"]}
                           for m in measurements]
                for biomarker, measurements in traj["biomarkers"].items()
            }
        }

    return tth_sample


def main():
    """Main trajectory building pipeline."""
    print("=" * 60)
    print("Building Biomarker Trajectories from Infectious Disease Cohort")
    print("=" * 60)

    # Load data
    labs = load_lab_events()
    admissions = load_admissions()

    # Build trajectories
    trajectories = build_trajectories(labs, admissions)

    # Compute statistics
    stats = compute_stats(trajectories)
    stats["avg_measurements_per_admission"] = round(
        sum(stats["biomarker_coverage"][b]["avg_measurements"] *
            stats["biomarker_coverage"][b]["admissions_with_data"]
            for b in stats["biomarker_coverage"]) /
        max(len(trajectories), 1), 1
    )

    # Find clinical cases
    sepsis_cases = find_sepsis_cases(trajectories)
    severe_sepsis_cases = find_severe_sepsis_cases(trajectories)
    leukocytosis_cases = find_leukocytosis_cases(trajectories)
    leukopenia_cases = find_leukopenia_cases(trajectories)

    # Format sample for TTH
    tth_sample = format_for_tth(trajectories)

    # Save outputs
    print("\nSaving outputs...")
    traj_output = os.path.join(OUTPUT_PATH, "infectious_disease_cohort", "trajectories")
    os.makedirs(traj_output, exist_ok=True)

    with open(os.path.join(traj_output, "all_trajectories.json"), "w") as f:
        json.dump(trajectories, f)
    print(f"  Saved all_trajectories.json ({len(trajectories):,} admissions)")

    with open(os.path.join(traj_output, "trajectory_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print("  Saved trajectory_stats.json")

    with open(os.path.join(traj_output, "tth_sample_100.json"), "w") as f:
        json.dump(tth_sample, f, indent=2)
    print(f"  Saved tth_sample_100.json ({len(tth_sample)} patients)")

    with open(os.path.join(traj_output, "sepsis_cases.json"), "w") as f:
        json.dump(sepsis_cases, f, indent=2)
    print(f"  Saved sepsis_cases.json ({len(sepsis_cases):,} cases)")

    with open(os.path.join(traj_output, "severe_sepsis_cases.json"), "w") as f:
        json.dump(severe_sepsis_cases, f, indent=2)
    print(f"  Saved severe_sepsis_cases.json ({len(severe_sepsis_cases):,} cases)")

    with open(os.path.join(traj_output, "leukocytosis_cases.json"), "w") as f:
        json.dump(leukocytosis_cases, f, indent=2)
    print(f"  Saved leukocytosis_cases.json ({len(leukocytosis_cases):,} cases)")

    with open(os.path.join(traj_output, "leukopenia_cases.json"), "w") as f:
        json.dump(leukopenia_cases, f, indent=2)
    print(f"  Saved leukopenia_cases.json ({len(leukopenia_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {len(trajectories):,}")
    print(f"Avg Measurements per Admission: {stats['avg_measurements_per_admission']}")
    print(f"Sepsis Cases: {len(sepsis_cases):,}")
    print(f"Severe Sepsis Cases: {len(severe_sepsis_cases):,}")
    print(f"Leukocytosis Cases: {len(leukocytosis_cases):,}")
    print(f"Leukopenia Cases: {len(leukopenia_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for biomarker, cov in stats["biomarker_coverage"].items():
        print(f"  {biomarker:20}: {cov['pct_coverage']:5.1f}% ({cov['admissions_with_data']:,} admissions, avg {cov['avg_measurements']} measurements)")
    print("=" * 60)

    return trajectories, stats


if __name__ == "__main__":
    main()
