"""Build biomarker trajectories for obstetric cohort."""
import pandas as pd
import json
import os
from datetime import datetime
from config import OUTPUT_PATH

# Map itemids to biomarker names
ITEMID_TO_BIOMARKER = {
    # Hemorrhage monitoring
    51222: "hemoglobin", 50811: "hemoglobin",
    51221: "hematocrit", 50810: "hematocrit",
    # HELLP syndrome markers
    51265: "platelets",
    50878: "ast",
    50861: "alt",
    50954: "ldh",
    50885: "bilirubin_total",
    # Preeclampsia markers
    50912: "creatinine", 52546: "creatinine",
    51006: "bun",
    51007: "uric_acid",
    51492: "protein_urine",
    # Gestational diabetes
    50809: "glucose", 50931: "glucose",
    50852: "hba1c",
    # Eclampsia treatment monitoring
    50960: "magnesium",
    # Coagulation (DIC)
    51237: "inr",
    51274: "pt",
    51275: "ptt",
    51214: "fibrinogen",
    50915: "d_dimer",
    # Infection/sepsis markers
    51301: "wbc", 51300: "wbc",
    50813: "lactate", 52442: "lactate",
    # Electrolytes
    50983: "sodium", 50824: "sodium",
    50971: "potassium", 50822: "potassium",
    50893: "calcium",
    # Blood gas
    50820: "ph",
    50821: "pao2",
    50802: "base_excess",
    # Albumin
    50862: "albumin",
}


def load_lab_events():
    """Load lab events from extracted obstetric cohort."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "lab_events.csv")
    print(f"Loading lab events from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} lab events")
    return df


def load_admissions():
    """Load admissions for timing reference."""
    path = os.path.join(OUTPUT_PATH, "obstetric_cohort", "admissions.csv")
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


def find_hellp_cases(trajectories, plt_threshold=100, ast_threshold=70, ldh_threshold=600):
    """Find cases with potential HELLP syndrome (Hemolysis, Elevated Liver enzymes, Low Platelets)."""
    print("Finding HELLP syndrome cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        plt_data = traj["biomarkers"].get("platelets", [])
        ast_data = traj["biomarkers"].get("ast", [])
        ldh_data = traj["biomarkers"].get("ldh", [])

        hellp_markers = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check for low platelets
        if plt_data:
            min_plt = min(m["value"] for m in plt_data)
            if min_plt < plt_threshold:
                hellp_markers.append("low_platelets")
                case_data["min_platelets"] = round(min_plt, 0)

        # Check for elevated AST
        if ast_data:
            max_ast = max(m["value"] for m in ast_data)
            if max_ast > ast_threshold:
                hellp_markers.append("elevated_ast")
                case_data["max_ast"] = round(max_ast, 0)

        # Check for elevated LDH (hemolysis marker)
        if ldh_data:
            max_ldh = max(m["value"] for m in ldh_data)
            if max_ldh > ldh_threshold:
                hellp_markers.append("elevated_ldh")
                case_data["max_ldh"] = round(max_ldh, 0)

        # HELLP requires at least 2 of 3 criteria
        if len(hellp_markers) >= 2:
            case_data["hellp_markers"] = hellp_markers
            cases.append(case_data)

    print(f"Found {len(cases):,} HELLP syndrome cases")
    return cases


def find_severe_preeclampsia_cases(trajectories, cr_threshold=1.1, uric_threshold=6.0):
    """Find cases with severe preeclampsia markers."""
    print("Finding severe preeclampsia cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        cr_data = traj["biomarkers"].get("creatinine", [])
        uric_data = traj["biomarkers"].get("uric_acid", [])
        plt_data = traj["biomarkers"].get("platelets", [])
        ast_data = traj["biomarkers"].get("ast", [])

        severity_markers = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check for renal dysfunction
        if cr_data:
            max_cr = max(m["value"] for m in cr_data)
            if max_cr >= cr_threshold:
                severity_markers.append("renal_dysfunction")
                case_data["max_creatinine"] = round(max_cr, 2)

        # Check for elevated uric acid
        if uric_data:
            max_uric = max(m["value"] for m in uric_data)
            if max_uric >= uric_threshold:
                severity_markers.append("hyperuricemia")
                case_data["max_uric_acid"] = round(max_uric, 1)

        # Check for thrombocytopenia
        if plt_data:
            min_plt = min(m["value"] for m in plt_data)
            if min_plt < 100:
                severity_markers.append("thrombocytopenia")
                case_data["min_platelets"] = round(min_plt, 0)

        # Check for liver involvement
        if ast_data:
            max_ast = max(m["value"] for m in ast_data)
            if max_ast > 70:
                severity_markers.append("liver_involvement")
                case_data["max_ast"] = round(max_ast, 0)

        # Need at least 2 severity markers
        if len(severity_markers) >= 2:
            case_data["severity_markers"] = severity_markers
            cases.append(case_data)

    print(f"Found {len(cases):,} severe preeclampsia cases")
    return cases


def find_obstetric_hemorrhage_cases(trajectories, hgb_threshold=8.0, hgb_drop=2.0):
    """Find cases with obstetric hemorrhage (significant hemoglobin drop)."""
    print("Finding obstetric hemorrhage cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        hgb_data = traj["biomarkers"].get("hemoglobin", [])
        lactate_data = traj["biomarkers"].get("lactate", [])

        if hgb_data and len(hgb_data) >= 2:
            initial_hgb = hgb_data[0]["value"]
            min_hgb = min(m["value"] for m in hgb_data)
            hgb_drop_actual = initial_hgb - min_hgb

            # Significant hemorrhage: low Hgb or large drop
            is_hemorrhage = min_hgb < hgb_threshold or hgb_drop_actual >= hgb_drop

            if is_hemorrhage:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "initial_hemoglobin": round(initial_hgb, 1),
                    "min_hemoglobin": round(min_hgb, 1),
                    "hemoglobin_drop": round(hgb_drop_actual, 1),
                }

                # Check for shock (elevated lactate)
                if lactate_data:
                    max_lactate = max(m["value"] for m in lactate_data)
                    if max_lactate >= 2.0:
                        case_data["max_lactate"] = round(max_lactate, 1)
                        case_data["hemorrhagic_shock"] = True

                cases.append(case_data)

    print(f"Found {len(cases):,} obstetric hemorrhage cases")
    return cases


def find_dic_cases(trajectories, inr_threshold=1.5, plt_threshold=100, fib_threshold=200):
    """Find cases with DIC (disseminated intravascular coagulation)."""
    print("Finding DIC cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        inr_data = traj["biomarkers"].get("inr", [])
        plt_data = traj["biomarkers"].get("platelets", [])
        fib_data = traj["biomarkers"].get("fibrinogen", [])
        ddimer_data = traj["biomarkers"].get("d_dimer", [])

        dic_markers = []
        case_data = {
            "hadm_id": hadm_id,
            "subject_id": traj["subject_id"],
        }

        # Check for elevated INR
        if inr_data:
            max_inr = max(m["value"] for m in inr_data)
            if max_inr >= inr_threshold:
                dic_markers.append("elevated_inr")
                case_data["max_inr"] = round(max_inr, 2)

        # Check for low platelets
        if plt_data:
            min_plt = min(m["value"] for m in plt_data)
            if min_plt < plt_threshold:
                dic_markers.append("thrombocytopenia")
                case_data["min_platelets"] = round(min_plt, 0)

        # Check for low fibrinogen
        if fib_data:
            min_fib = min(m["value"] for m in fib_data)
            if min_fib < fib_threshold:
                dic_markers.append("hypofibrinogenemia")
                case_data["min_fibrinogen"] = round(min_fib, 0)

        # Check for elevated D-dimer
        if ddimer_data:
            max_ddimer = max(m["value"] for m in ddimer_data)
            if max_ddimer > 500:  # FEU ng/mL
                dic_markers.append("elevated_d_dimer")
                case_data["max_d_dimer"] = round(max_ddimer, 0)

        # DIC requires at least 2 markers
        if len(dic_markers) >= 2:
            case_data["dic_markers"] = dic_markers
            cases.append(case_data)

    print(f"Found {len(cases):,} DIC cases")
    return cases


def find_magnesium_monitoring_cases(trajectories, mg_low=1.5, mg_high=3.0):
    """Find cases with magnesium sulfate monitoring (eclampsia treatment)."""
    print("Finding magnesium monitoring cases...")
    cases = []

    for hadm_id_str, traj in trajectories.items():
        hadm_id = int(float(hadm_id_str))
        mg_data = traj["biomarkers"].get("magnesium", [])

        if mg_data and len(mg_data) >= 2:
            # Multiple magnesium measurements suggest monitoring
            values = [m["value"] for m in mg_data]
            max_mg = max(values)
            min_mg = min(values)

            # Therapeutic range is typically 4-7 mEq/L (2-3.5 mmol/L)
            # Flag if outside normal range or being actively monitored
            if max_mg >= mg_high or len(mg_data) >= 3:
                case_data = {
                    "hadm_id": hadm_id,
                    "subject_id": traj["subject_id"],
                    "mg_measurements": len(mg_data),
                    "min_magnesium": round(min_mg, 2),
                    "max_magnesium": round(max_mg, 2),
                }

                if max_mg >= 4.0:
                    case_data["therapeutic_level"] = True

                if max_mg >= 5.0:
                    case_data["toxicity_risk"] = True

                cases.append(case_data)

    print(f"Found {len(cases):,} magnesium monitoring cases")
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
            "domain": "obstetric",
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
    print("Building Biomarker Trajectories from Obstetric Cohort")
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
    hellp_cases = find_hellp_cases(trajectories)
    preeclampsia_cases = find_severe_preeclampsia_cases(trajectories)
    hemorrhage_cases = find_obstetric_hemorrhage_cases(trajectories)
    dic_cases = find_dic_cases(trajectories)
    mg_cases = find_magnesium_monitoring_cases(trajectories)

    # Format sample for TTH
    tth_sample = format_for_tth(trajectories)

    # Save outputs
    print("\nSaving outputs...")
    traj_output = os.path.join(OUTPUT_PATH, "obstetric_cohort", "trajectories")
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

    with open(os.path.join(traj_output, "hellp_cases.json"), "w") as f:
        json.dump(hellp_cases, f, indent=2)
    print(f"  Saved hellp_cases.json ({len(hellp_cases):,} cases)")

    with open(os.path.join(traj_output, "severe_preeclampsia_cases.json"), "w") as f:
        json.dump(preeclampsia_cases, f, indent=2)
    print(f"  Saved severe_preeclampsia_cases.json ({len(preeclampsia_cases):,} cases)")

    with open(os.path.join(traj_output, "obstetric_hemorrhage_cases.json"), "w") as f:
        json.dump(hemorrhage_cases, f, indent=2)
    print(f"  Saved obstetric_hemorrhage_cases.json ({len(hemorrhage_cases):,} cases)")

    with open(os.path.join(traj_output, "dic_cases.json"), "w") as f:
        json.dump(dic_cases, f, indent=2)
    print(f"  Saved dic_cases.json ({len(dic_cases):,} cases)")

    with open(os.path.join(traj_output, "magnesium_monitoring_cases.json"), "w") as f:
        json.dump(mg_cases, f, indent=2)
    print(f"  Saved magnesium_monitoring_cases.json ({len(mg_cases):,} cases)")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print(f"Total Admissions with Trajectories: {len(trajectories):,}")
    print(f"Avg Measurements per Admission: {stats['avg_measurements_per_admission']}")
    print(f"HELLP Syndrome Cases: {len(hellp_cases):,}")
    print(f"Severe Preeclampsia Cases: {len(preeclampsia_cases):,}")
    print(f"Obstetric Hemorrhage Cases: {len(hemorrhage_cases):,}")
    print(f"DIC Cases: {len(dic_cases):,}")
    print(f"Magnesium Monitoring Cases: {len(mg_cases):,}")
    print("\nBiomarker Coverage:")
    print("-" * 50)
    for biomarker, cov in stats["biomarker_coverage"].items():
        print(f"  {biomarker:20}: {cov['pct_coverage']:5.1f}% ({cov['admissions_with_data']:,} admissions, avg {cov['avg_measurements']} measurements)")
    print("=" * 60)

    return trajectories, stats


if __name__ == "__main__":
    main()
