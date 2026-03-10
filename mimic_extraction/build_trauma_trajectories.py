"""Build biomarker trajectories for trauma cohort."""
import pandas as pd
import json
import os
from datetime import datetime
from config import OUTPUT_PATH

# Map itemids to biomarker names
ITEMID_TO_BIOMARKER = {
    # Hemorrhage/blood loss monitoring
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    # Shock/perfusion markers
    50813: "lactate", 52442: "lactate",
    50802: "base_excess",
    50820: "ph",
    # Coagulopathy
    51237: "inr",
    51274: "pt",
    51275: "ptt",
    51214: "fibrinogen",
    51265: "platelets",
    51196: "d_dimer", 50915: "d_dimer",
    # Rhabdomyolysis/muscle injury
    50912: "creatinine", 52546: "creatinine",
    50910: "ck",
    51245: "myoglobin",
    50971: "potassium", 50822: "potassium",
    # Renal function
    51006: "bun",
    # Cardiac contusion
    51003: "troponin_t", 52642: "troponin_t",
    51002: "troponin_i",
    # Massive transfusion effects
    50893: "calcium",
    50960: "magnesium",
    # General
    51301: "wbc", 51300: "wbc",
    50983: "sodium", 50824: "sodium",
    50809: "glucose", 50931: "glucose",
    50882: "bicarbonate",
}


def load_lab_events():
    """Load lab events from extracted trauma cohort."""
    path = os.path.join(OUTPUT_PATH, "trauma_cohort", "lab_events.csv")
    print(f"Loading lab events from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for timing reference."""
    path = os.path.join(OUTPUT_PATH, "trauma_cohort", "admissions.csv")
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


def find_hemorrhagic_shock_cases(trajectories, hgb_threshold=7.0, lactate_threshold=4.0, min_measurements=2):
    """Find cases with significant hemorrhage and shock markers."""
    print("Finding hemorrhagic shock cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))  # Handle "20001955.0" format
        hgb_data = traj["biomarkers"].get("hemoglobin", [])
        lactate_data = traj["biomarkers"].get("lactate", [])

        if len(hgb_data) >= min_measurements:
            initial_hgb = hgb_data[0]["value"]
            min_hgb = min(m["value"] for m in hgb_data)
            hgb_drop = initial_hgb - min_hgb

            # Check for significant drop or low hemoglobin with elevated lactate
            has_hemorrhage = min_hgb < hgb_threshold or hgb_drop > 3.0

            max_lactate = max((m["value"] for m in lactate_data), default=0)
            has_shock = max_lactate > lactate_threshold

            if has_hemorrhage and (has_shock or hgb_drop > 4.0):
                nadir_idx = [i for i, m in enumerate(hgb_data) if m["value"] == min_hgb][0]
                hours_to_nadir = hgb_data[nadir_idx]["hours_since_admit"]

                cases.append({
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "initial_hemoglobin": round(initial_hgb, 1),
                    "min_hemoglobin": round(min_hgb, 1),
                    "hemoglobin_drop": round(hgb_drop, 1),
                    "max_lactate": round(max_lactate, 1),
                    "hours_to_nadir": round(hours_to_nadir, 1)
                })

    print(f"Found {len(cases):,} hemorrhagic shock cases")
    return cases


def find_coagulopathy_cases(trajectories, inr_threshold=1.5, plt_threshold=100, fib_threshold=200):
    """Find cases with trauma-induced coagulopathy."""
    print("Finding coagulopathy cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))  # Handle "20001955.0" format
        inr_data = traj["biomarkers"].get("inr", [])
        plt_data = traj["biomarkers"].get("platelets", [])
        fib_data = traj["biomarkers"].get("fibrinogen", [])

        abnormal_count = 0
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check INR
        if inr_data:
            max_inr = max(m["value"] for m in inr_data)
            if max_inr > inr_threshold:
                abnormal_count += 1
                case_data["max_inr"] = round(max_inr, 2)

        # Check platelets
        if plt_data:
            min_plt = min(m["value"] for m in plt_data)
            if min_plt < plt_threshold:
                abnormal_count += 1
                case_data["min_platelets"] = round(min_plt, 0)

        # Check fibrinogen
        if fib_data:
            min_fib = min(m["value"] for m in fib_data)
            if min_fib < fib_threshold:
                abnormal_count += 1
                case_data["min_fibrinogen"] = round(min_fib, 0)

        # Need at least 2 abnormal markers for trauma-induced coagulopathy
        if abnormal_count >= 2:
            case_data["abnormal_markers"] = abnormal_count
            cases.append(case_data)

    print(f"Found {len(cases):,} coagulopathy cases")
    return cases


def find_rhabdomyolysis_cases(trajectories, ck_threshold=5000, cr_rise_threshold=0.5):
    """Find cases with rhabdomyolysis (elevated CK with AKI)."""
    print("Finding rhabdomyolysis cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))  # Handle "20001955.0" format
        ck_data = traj["biomarkers"].get("ck", [])
        cr_data = traj["biomarkers"].get("creatinine", [])
        k_data = traj["biomarkers"].get("potassium", [])

        if ck_data:
            max_ck = max(m["value"] for m in ck_data)
            if max_ck > ck_threshold:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "max_ck": round(max_ck, 0),
                }

                # Check for AKI
                if len(cr_data) >= 2:
                    initial_cr = cr_data[0]["value"]
                    max_cr = max(m["value"] for m in cr_data)
                    cr_rise = max_cr - initial_cr
                    if cr_rise > cr_rise_threshold:
                        case_data["initial_creatinine"] = round(initial_cr, 2)
                        case_data["max_creatinine"] = round(max_cr, 2)
                        case_data["creatinine_rise"] = round(cr_rise, 2)

                # Check for hyperkalemia
                if k_data:
                    max_k = max(m["value"] for m in k_data)
                    if max_k > 5.5:
                        case_data["max_potassium"] = round(max_k, 1)

                cases.append(case_data)

    print(f"Found {len(cases):,} rhabdomyolysis cases")
    return cases


def find_acidosis_cases(trajectories, ph_threshold=7.25, lactate_threshold=4.0, be_threshold=-6):
    """Find cases with severe metabolic acidosis (shock)."""
    print("Finding metabolic acidosis cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))  # Handle "20001955.0" format
        ph_data = traj["biomarkers"].get("ph", [])
        lactate_data = traj["biomarkers"].get("lactate", [])
        be_data = traj["biomarkers"].get("base_excess", [])
        bicarb_data = traj["biomarkers"].get("bicarbonate", [])

        abnormal_markers = []

        # Check pH
        if ph_data:
            min_ph = min(m["value"] for m in ph_data)
            if min_ph < ph_threshold:
                abnormal_markers.append("ph")

        # Check lactate
        if lactate_data:
            max_lactate = max(m["value"] for m in lactate_data)
            if max_lactate > lactate_threshold:
                abnormal_markers.append("lactate")

        # Check base excess
        if be_data:
            min_be = min(m["value"] for m in be_data)
            if min_be < be_threshold:
                abnormal_markers.append("base_excess")

        # Need at least 2 markers for severe acidosis
        if len(abnormal_markers) >= 2:
            case_data = {
                "hadm_id": hadm_id,
                "subject_id": traj["subject_id"],
                "abnormal_markers": abnormal_markers,
            }

            if ph_data:
                case_data["min_ph"] = round(min(m["value"] for m in ph_data), 3)
            if lactate_data:
                case_data["max_lactate"] = round(max(m["value"] for m in lactate_data), 1)
            if be_data:
                case_data["min_base_excess"] = round(min(m["value"] for m in be_data), 1)
            if bicarb_data:
                case_data["min_bicarbonate"] = round(min(m["value"] for m in bicarb_data), 1)

            cases.append(case_data)

    print(f"Found {len(cases):,} metabolic acidosis cases")
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
            "domain": "trauma",
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
    print("Building Biomarker Trajectories from Trauma Cohort")
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
    hemorrhage_cases = find_hemorrhagic_shock_cases(trajectories)
    coagulopathy_cases = find_coagulopathy_cases(trajectories)
    rhabdo_cases = find_rhabdomyolysis_cases(trajectories)
    acidosis_cases = find_acidosis_cases(trajectories)

    # Format sample for TTH
    tth_sample = format_for_tth(trajectories)

    # Save outputs
    print("\nSaving outputs...")
    traj_output = os.path.join(OUTPUT_PATH, "trauma_cohort", "trajectories")
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

    with open(os.path.join(traj_output, "hemorrhagic_shock_cases.json"), "w") as f:
        json.dump(hemorrhage_cases, f, indent=2)
    print(f"  Saved hemorrhagic_shock_cases.json ({len(hemorrhage_cases):,} cases)")

    with open(os.path.join(traj_output, "coagulopathy_cases.json"), "w") as f:
        json.dump(coagulopathy_cases, f, indent=2)
    print(f"  Saved coagulopathy_cases.json ({len(coagulopathy_cases):,} cases)")

    with open(os.path.join(traj_output, "rhabdomyolysis_cases.json"), "w") as f:
        json.dump(rhabdo_cases, f, indent=2)
    print(f"  Saved rhabdomyolysis_cases.json ({len(rhabdo_cases):,} cases)")

    with open(os.path.join(traj_output, "acidosis_cases.json"), "w") as f:
        json.dump(acidosis_cases, f, indent=2)
    print(f"  Saved acidosis_cases.json ({len(acidosis_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {len(trajectories):,}")
    print(f"Avg Measurements per Admission: {stats['avg_measurements_per_admission']}")
    print(f"Hemorrhagic Shock Cases: {len(hemorrhage_cases):,}")
    print(f"Coagulopathy Cases: {len(coagulopathy_cases):,}")
    print(f"Rhabdomyolysis Cases: {len(rhabdo_cases):,}")
    print(f"Metabolic Acidosis Cases: {len(acidosis_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for biomarker, cov in stats["biomarker_coverage"].items():
        print(f"  {biomarker:20}: {cov['pct_coverage']:5.1f}% ({cov['admissions_with_data']:,} admissions, avg {cov['avg_measurements']} measurements)")
    print("=" * 60)

    return trajectories, stats


if __name__ == "__main__":
    main()
