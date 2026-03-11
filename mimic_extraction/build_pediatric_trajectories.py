"""Build biomarker trajectories for pediatric cohort."""
import pandas as pd
import json
import os
from datetime import datetime
from config import OUTPUT_PATH

# Map itemids to biomarker names
ITEMID_TO_BIOMARKER = {
    # CBC
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    51301: "wbc", 51300: "wbc",
    51265: "platelets",
    51256: "neutrophils",
    51244: "lymphocytes", 51245: "lymphocytes",
    # Metabolic panel
    50809: "glucose", 50931: "glucose",
    50983: "sodium", 50824: "sodium",
    50971: "potassium", 50822: "potassium",
    50902: "chloride",
    50882: "bicarbonate",
    50893: "calcium",
    50960: "magnesium",
    50970: "phosphate",
    # Renal function
    50912: "creatinine", 52546: "creatinine",
    51006: "bun",
    # Liver function
    50885: "bilirubin_total",
    50883: "bilirubin_direct",
    50878: "ast",
    50861: "alt",
    50862: "albumin",
    # Blood gas
    50820: "ph",
    50821: "pao2",
    50818: "pco2",
    50802: "base_excess",
    # Infection markers
    50889: "crp",
    50976: "procalcitonin",
    50813: "lactate", 52442: "lactate",
    # Coagulation
    51237: "inr",
    51274: "pt",
    51275: "ptt",
}


def load_lab_events():
    """Load lab events from extracted pediatric cohort."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "lab_events.csv")
    print(f"Loading lab events from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for timing reference."""
    path = os.path.join(OUTPUT_PATH, "pediatric_cohort", "admissions.csv")
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


def find_neonatal_jaundice_cases(trajectories, bili_threshold=15.0):
    """Find cases with neonatal jaundice (elevated bilirubin)."""
    print("Finding neonatal jaundice cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        bili_total = traj["biomarkers"].get("bilirubin_total", [])
        bili_direct = traj["biomarkers"].get("bilirubin_direct", [])

        if bili_total:
            max_bili = max(m["value"] for m in bili_total)

            if max_bili >= bili_threshold:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "max_bilirubin_total": round(max_bili, 1),
                }

                # Check direct bilirubin
                if bili_direct:
                    max_direct = max(m["value"] for m in bili_direct)
                    case_data["max_bilirubin_direct"] = round(max_direct, 1)
                    # Direct > 20% of total suggests conjugated hyperbilirubinemia
                    if max_direct > 0.2 * max_bili:
                        case_data["conjugated"] = True

                cases.append(case_data)

    print(f"Found {len(cases):,} neonatal jaundice cases")
    return cases


def find_pediatric_sepsis_cases(trajectories, wbc_high=15.0, wbc_low=4.0, lactate_threshold=2.0):
    """Find cases with pediatric sepsis markers."""
    print("Finding pediatric sepsis cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        wbc_data = traj["biomarkers"].get("wbc", [])
        lactate_data = traj["biomarkers"].get("lactate", [])
        crp_data = traj["biomarkers"].get("crp", [])
        procalcitonin_data = traj["biomarkers"].get("procalcitonin", [])

        sepsis_markers = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check WBC
        if wbc_data:
            max_wbc = max(m["value"] for m in wbc_data)
            min_wbc = min(m["value"] for m in wbc_data)
            if max_wbc > wbc_high:
                sepsis_markers.append("leukocytosis")
                case_data["max_wbc"] = round(max_wbc, 1)
            if min_wbc < wbc_low:
                sepsis_markers.append("leukopenia")
                case_data["min_wbc"] = round(min_wbc, 1)

        # Check lactate
        if lactate_data:
            max_lactate = max(m["value"] for m in lactate_data)
            if max_lactate >= lactate_threshold:
                sepsis_markers.append("elevated_lactate")
                case_data["max_lactate"] = round(max_lactate, 1)

        # Check CRP
        if crp_data:
            max_crp = max(m["value"] for m in crp_data)
            if max_crp > 10:  # mg/L
                sepsis_markers.append("elevated_crp")
                case_data["max_crp"] = round(max_crp, 1)

        # Check procalcitonin
        if procalcitonin_data:
            max_pct = max(m["value"] for m in procalcitonin_data)
            if max_pct > 0.5:  # ng/mL
                sepsis_markers.append("elevated_procalcitonin")
                case_data["max_procalcitonin"] = round(max_pct, 2)

        # Need at least 2 sepsis markers
        if len(sepsis_markers) >= 2:
            case_data["sepsis_markers"] = sepsis_markers
            cases.append(case_data)

    print(f"Found {len(cases):,} pediatric sepsis cases")
    return cases


def find_respiratory_distress_cases(trajectories, ph_low=7.25, pco2_high=50, pao2_low=60):
    """Find cases with respiratory distress (abnormal blood gas)."""
    print("Finding respiratory distress cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        ph_data = traj["biomarkers"].get("ph", [])
        pco2_data = traj["biomarkers"].get("pco2", [])
        pao2_data = traj["biomarkers"].get("pao2", [])
        base_excess_data = traj["biomarkers"].get("base_excess", [])

        resp_markers = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check pH
        if ph_data:
            min_ph = min(m["value"] for m in ph_data)
            if min_ph < ph_low:
                resp_markers.append("acidosis")
                case_data["min_ph"] = round(min_ph, 3)

        # Check pCO2
        if pco2_data:
            max_pco2 = max(m["value"] for m in pco2_data)
            if max_pco2 > pco2_high:
                resp_markers.append("hypercapnia")
                case_data["max_pco2"] = round(max_pco2, 1)

        # Check pO2
        if pao2_data:
            min_pao2 = min(m["value"] for m in pao2_data)
            if min_pao2 < pao2_low:
                resp_markers.append("hypoxemia")
                case_data["min_pao2"] = round(min_pao2, 1)

        # Check base excess
        if base_excess_data:
            min_be = min(m["value"] for m in base_excess_data)
            if min_be < -5:
                resp_markers.append("metabolic_acidosis")
                case_data["min_base_excess"] = round(min_be, 1)

        # Need at least 1 respiratory marker
        if len(resp_markers) >= 1:
            case_data["respiratory_markers"] = resp_markers
            cases.append(case_data)

    print(f"Found {len(cases):,} respiratory distress cases")
    return cases


def find_anemia_cases(trajectories, hgb_threshold=10.0):
    """Find cases with pediatric anemia."""
    print("Finding anemia cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        hgb_data = traj["biomarkers"].get("hemoglobin", [])

        if hgb_data:
            min_hgb = min(m["value"] for m in hgb_data)

            if min_hgb < hgb_threshold:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "min_hemoglobin": round(min_hgb, 1),
                }

                # Classify severity
                if min_hgb < 7.0:
                    case_data["severity"] = "severe"
                elif min_hgb < 9.0:
                    case_data["severity"] = "moderate"
                else:
                    case_data["severity"] = "mild"

                cases.append(case_data)

    print(f"Found {len(cases):,} anemia cases")
    return cases


def find_electrolyte_imbalance_cases(trajectories):
    """Find cases with electrolyte imbalances."""
    print("Finding electrolyte imbalance cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        na_data = traj["biomarkers"].get("sodium", [])
        k_data = traj["biomarkers"].get("potassium", [])
        ca_data = traj["biomarkers"].get("calcium", [])
        glucose_data = traj["biomarkers"].get("glucose", [])

        imbalances = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check sodium
        if na_data:
            min_na = min(m["value"] for m in na_data)
            max_na = max(m["value"] for m in na_data)
            if min_na < 130:
                imbalances.append("hyponatremia")
                case_data["min_sodium"] = round(min_na, 0)
            if max_na > 150:
                imbalances.append("hypernatremia")
                case_data["max_sodium"] = round(max_na, 0)

        # Check potassium
        if k_data:
            min_k = min(m["value"] for m in k_data)
            max_k = max(m["value"] for m in k_data)
            if min_k < 3.0:
                imbalances.append("hypokalemia")
                case_data["min_potassium"] = round(min_k, 1)
            if max_k > 5.5:
                imbalances.append("hyperkalemia")
                case_data["max_potassium"] = round(max_k, 1)

        # Check calcium
        if ca_data:
            min_ca = min(m["value"] for m in ca_data)
            if min_ca < 7.5:
                imbalances.append("hypocalcemia")
                case_data["min_calcium"] = round(min_ca, 1)

        # Check glucose
        if glucose_data:
            min_glu = min(m["value"] for m in glucose_data)
            max_glu = max(m["value"] for m in glucose_data)
            if min_glu < 50:
                imbalances.append("hypoglycemia")
                case_data["min_glucose"] = round(min_glu, 0)
            if max_glu > 200:
                imbalances.append("hyperglycemia")
                case_data["max_glucose"] = round(max_glu, 0)

        # Need at least 1 imbalance
        if len(imbalances) >= 1:
            case_data["imbalances"] = imbalances
            cases.append(case_data)

    print(f"Found {len(cases):,} electrolyte imbalance cases")
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
            "domain": "pediatric",
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
    print("Building Biomarker Trajectories from Pediatric Cohort")
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
    jaundice_cases = find_neonatal_jaundice_cases(trajectories)
    sepsis_cases = find_pediatric_sepsis_cases(trajectories)
    resp_cases = find_respiratory_distress_cases(trajectories)
    anemia_cases = find_anemia_cases(trajectories)
    electrolyte_cases = find_electrolyte_imbalance_cases(trajectories)

    # Format sample for TTH
    tth_sample = format_for_tth(trajectories)

    # Save outputs
    print("\nSaving outputs...")
    traj_output = os.path.join(OUTPUT_PATH, "pediatric_cohort", "trajectories")
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

    with open(os.path.join(traj_output, "neonatal_jaundice_cases.json"), "w") as f:
        json.dump(jaundice_cases, f, indent=2)
    print(f"  Saved neonatal_jaundice_cases.json ({len(jaundice_cases):,} cases)")

    with open(os.path.join(traj_output, "pediatric_sepsis_cases.json"), "w") as f:
        json.dump(sepsis_cases, f, indent=2)
    print(f"  Saved pediatric_sepsis_cases.json ({len(sepsis_cases):,} cases)")

    with open(os.path.join(traj_output, "respiratory_distress_cases.json"), "w") as f:
        json.dump(resp_cases, f, indent=2)
    print(f"  Saved respiratory_distress_cases.json ({len(resp_cases):,} cases)")

    with open(os.path.join(traj_output, "anemia_cases.json"), "w") as f:
        json.dump(anemia_cases, f, indent=2)
    print(f"  Saved anemia_cases.json ({len(anemia_cases):,} cases)")

    with open(os.path.join(traj_output, "electrolyte_imbalance_cases.json"), "w") as f:
        json.dump(electrolyte_cases, f, indent=2)
    print(f"  Saved electrolyte_imbalance_cases.json ({len(electrolyte_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {len(trajectories):,}")
    print(f"Avg Measurements per Admission: {stats['avg_measurements_per_admission']}")
    print(f"Neonatal Jaundice Cases: {len(jaundice_cases):,}")
    print(f"Pediatric Sepsis Cases: {len(sepsis_cases):,}")
    print(f"Respiratory Distress Cases: {len(resp_cases):,}")
    print(f"Anemia Cases: {len(anemia_cases):,}")
    print(f"Electrolyte Imbalance Cases: {len(electrolyte_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for biomarker, cov in stats["biomarker_coverage"].items():
        print(f"  {biomarker:20}: {cov['pct_coverage']:5.1f}% ({cov['admissions_with_data']:,} admissions, avg {cov['avg_measurements']} measurements)")
    print("=" * 60)

    return trajectories, stats


if __name__ == "__main__":
    main()
