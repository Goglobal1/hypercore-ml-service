"""Build biomarker trajectories for psychiatric cohort."""
import pandas as pd
import json
import os
from datetime import datetime
from config import OUTPUT_PATH

# Map itemids to biomarker names
ITEMID_TO_BIOMARKER = {
    # Drug level monitoring
    50994: "lithium",
    51001: "valproic_acid",
    50885: "carbamazepine",  # Note: shares with bilirubin, need careful handling
    # Metabolic monitoring
    50809: "glucose", 50931: "glucose",
    50852: "hba1c",
    # Electrolytes
    50983: "sodium", 50824: "sodium",
    50971: "potassium", 50822: "potassium",
    50902: "chloride",
    50882: "bicarbonate",
    50893: "calcium",
    50960: "magnesium",
    # Renal function
    50912: "creatinine", 52546: "creatinine",
    51006: "bun",
    # Liver function
    50861: "alt",
    50878: "ast",
    50862: "albumin",
    # Thyroid
    50993: "tsh",
    # Hematologic
    51301: "wbc", 51300: "wbc",
    51256: "neutrophils",
    51265: "platelets",
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    # NMS markers
    50910: "ck",
    # Lipids
    50907: "cholesterol",
    51000: "triglycerides",
    # Toxicology
    50821: "alcohol_level",
}


def load_lab_events():
    """Load lab events from extracted psychiatric cohort."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "lab_events.csv")
    print(f"Loading lab events from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for timing reference."""
    path = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "admissions.csv")
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


def find_lithium_toxicity_cases(trajectories, lithium_threshold=1.5, cr_threshold=1.5):
    """Find cases with potential lithium toxicity."""
    print("Finding lithium toxicity cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        lithium_data = traj["biomarkers"].get("lithium", [])
        cr_data = traj["biomarkers"].get("creatinine", [])

        if lithium_data:
            max_lithium = max(m["value"] for m in lithium_data)

            if max_lithium >= lithium_threshold:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "max_lithium": round(max_lithium, 2),
                }

                # Check for renal impairment
                if cr_data:
                    max_cr = max(m["value"] for m in cr_data)
                    if max_cr >= cr_threshold:
                        case_data["max_creatinine"] = round(max_cr, 2)
                        case_data["renal_impairment"] = True

                cases.append(case_data)

    print(f"Found {len(cases):,} lithium toxicity cases")
    return cases


def find_nms_cases(trajectories, ck_threshold=1000, wbc_threshold=12.0):
    """Find cases with potential neuroleptic malignant syndrome (elevated CK + leukocytosis)."""
    print("Finding NMS cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        ck_data = traj["biomarkers"].get("ck", [])
        wbc_data = traj["biomarkers"].get("wbc", [])

        if ck_data:
            max_ck = max(m["value"] for m in ck_data)

            if max_ck >= ck_threshold:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "max_ck": round(max_ck, 0),
                }

                # Check for leukocytosis
                if wbc_data:
                    max_wbc = max(m["value"] for m in wbc_data)
                    if max_wbc >= wbc_threshold:
                        case_data["max_wbc"] = round(max_wbc, 1)
                        case_data["leukocytosis"] = True

                cases.append(case_data)

    print(f"Found {len(cases):,} NMS cases")
    return cases


def find_metabolic_syndrome_cases(trajectories, glucose_threshold=126, tg_threshold=150):
    """Find cases with metabolic abnormalities (antipsychotic side effects)."""
    print("Finding metabolic syndrome cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        glucose_data = traj["biomarkers"].get("glucose", [])
        tg_data = traj["biomarkers"].get("triglycerides", [])
        chol_data = traj["biomarkers"].get("cholesterol", [])

        abnormal_markers = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check glucose
        if glucose_data:
            max_glucose = max(m["value"] for m in glucose_data)
            if max_glucose >= glucose_threshold:
                abnormal_markers.append("hyperglycemia")
                case_data["max_glucose"] = round(max_glucose, 0)

        # Check triglycerides
        if tg_data:
            max_tg = max(m["value"] for m in tg_data)
            if max_tg >= tg_threshold:
                abnormal_markers.append("hypertriglyceridemia")
                case_data["max_triglycerides"] = round(max_tg, 0)

        # Check cholesterol
        if chol_data:
            max_chol = max(m["value"] for m in chol_data)
            if max_chol >= 240:
                abnormal_markers.append("hypercholesterolemia")
                case_data["max_cholesterol"] = round(max_chol, 0)

        # Need at least 2 metabolic abnormalities
        if len(abnormal_markers) >= 2:
            case_data["abnormal_markers"] = abnormal_markers
            cases.append(case_data)

    print(f"Found {len(cases):,} metabolic syndrome cases")
    return cases


def find_neutropenia_cases(trajectories, neutrophil_threshold=1.5, wbc_threshold=3.0):
    """Find cases with neutropenia (clozapine monitoring)."""
    print("Finding neutropenia cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        neutrophil_data = traj["biomarkers"].get("neutrophils", [])
        wbc_data = traj["biomarkers"].get("wbc", [])

        has_neutropenia = False
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check neutrophils
        if neutrophil_data:
            min_neut = min(m["value"] for m in neutrophil_data)
            if min_neut <= neutrophil_threshold:
                has_neutropenia = True
                case_data["min_neutrophils"] = round(min_neut, 2)

        # Check WBC
        if wbc_data:
            min_wbc = min(m["value"] for m in wbc_data)
            if min_wbc <= wbc_threshold:
                has_neutropenia = True
                case_data["min_wbc"] = round(min_wbc, 1)

        if has_neutropenia:
            cases.append(case_data)

    print(f"Found {len(cases):,} neutropenia cases")
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
            "domain": "psychiatric",
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
    print("Building Biomarker Trajectories from Psychiatric Cohort")
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
    lithium_cases = find_lithium_toxicity_cases(trajectories)
    nms_cases = find_nms_cases(trajectories)
    metabolic_cases = find_metabolic_syndrome_cases(trajectories)
    neutropenia_cases = find_neutropenia_cases(trajectories)

    # Format sample for TTH
    tth_sample = format_for_tth(trajectories)

    # Save outputs
    print("\nSaving outputs...")
    traj_output = os.path.join(OUTPUT_PATH, "psychiatric_cohort", "trajectories")
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

    with open(os.path.join(traj_output, "lithium_toxicity_cases.json"), "w") as f:
        json.dump(lithium_cases, f, indent=2)
    print(f"  Saved lithium_toxicity_cases.json ({len(lithium_cases):,} cases)")

    with open(os.path.join(traj_output, "nms_cases.json"), "w") as f:
        json.dump(nms_cases, f, indent=2)
    print(f"  Saved nms_cases.json ({len(nms_cases):,} cases)")

    with open(os.path.join(traj_output, "metabolic_syndrome_cases.json"), "w") as f:
        json.dump(metabolic_cases, f, indent=2)
    print(f"  Saved metabolic_syndrome_cases.json ({len(metabolic_cases):,} cases)")

    with open(os.path.join(traj_output, "neutropenia_cases.json"), "w") as f:
        json.dump(neutropenia_cases, f, indent=2)
    print(f"  Saved neutropenia_cases.json ({len(neutropenia_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {len(trajectories):,}")
    print(f"Avg Measurements per Admission: {stats['avg_measurements_per_admission']}")
    print(f"Lithium Toxicity Cases: {len(lithium_cases):,}")
    print(f"NMS Cases: {len(nms_cases):,}")
    print(f"Metabolic Syndrome Cases: {len(metabolic_cases):,}")
    print(f"Neutropenia Cases: {len(neutropenia_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for biomarker, cov in stats["biomarker_coverage"].items():
        print(f"  {biomarker:20}: {cov['pct_coverage']:5.1f}% ({cov['admissions_with_data']:,} admissions, avg {cov['avg_measurements']} measurements)")
    print("=" * 60)

    return trajectories, stats


if __name__ == "__main__":
    main()
