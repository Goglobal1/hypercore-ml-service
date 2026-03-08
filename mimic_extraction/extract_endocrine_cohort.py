"""Extract endocrine patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Endocrine ICD codes (ICD-9 and ICD-10)
ENDOCRINE_ICD_CODES = [
    # ICD-10 - Diabetes Mellitus
    "E10", "E100", "E101", "E102", "E103", "E104", "E105", "E106", "E107", "E108", "E109",  # Type 1
    "E11", "E110", "E111", "E112", "E113", "E114", "E115", "E116", "E117", "E118", "E119",  # Type 2
    "E13", "E130", "E131", "E132", "E133", "E134", "E135", "E136", "E137", "E138", "E139",  # Other DM
    # ICD-10 - DKA and Hyperosmolar states
    "E101", "E111", "E131",  # With ketoacidosis
    "E100", "E110", "E130",  # With hyperosmolarity
    # ICD-10 - Hypoglycemia
    "E16", "E160", "E161", "E162",
    # ICD-10 - Thyroid disorders
    "E00", "E01", "E02", "E03",  # Hypothyroidism
    "E04", "E05", "E06", "E07",  # Hyperthyroidism and other thyroid
    "E050",  # Thyrotoxicosis with goiter
    "E051",  # Thyrotoxicosis with toxic nodule
    "E055",  # Thyroid crisis/storm
    # ICD-10 - Adrenal disorders
    "E27", "E270", "E271", "E272", "E273", "E274", "E275", "E278", "E279",
    "E240", "E241", "E242", "E243", "E244", "E248", "E249",  # Cushing's
    # ICD-10 - Pituitary disorders
    "E22", "E220", "E221", "E222", "E228", "E229",  # Hyperfunction
    "E23", "E230", "E231", "E232", "E233", "E236", "E237",  # Hypofunction
    # ICD-10 - Parathyroid/Calcium disorders
    "E20", "E200", "E201", "E208", "E209",  # Hypoparathyroidism
    "E21", "E210", "E211", "E212", "E213", "E214", "E215",  # Hyperparathyroidism
    # ICD-10 - Electrolyte disorders (endocrine-related)
    "E86", "E860", "E861", "E869",  # Volume depletion
    "E87", "E870", "E871", "E872", "E873", "E874", "E875", "E876", "E877", "E878",
    # ICD-9 - Diabetes
    "250", "2500", "2501", "2502", "2503", "2504", "2505", "2506", "2507", "2508", "2509",
    "25010", "25011", "25012", "25013",  # DKA
    "25020", "25021", "25022", "25023",  # Hyperosmolar
    # ICD-9 - Thyroid
    "240", "241", "242", "243", "244", "245", "246",
    "2420",  # Thyrotoxicosis with goiter
    # ICD-9 - Adrenal
    "255", "2550", "2551", "2552", "2553", "2554", "2555", "2556", "2558", "2559",
    # ICD-9 - Pituitary
    "253", "2530", "2531", "2532", "2533", "2534", "2535", "2536", "2537", "2538", "2539",
    # ICD-9 - Parathyroid
    "252", "2520", "2521", "2528", "2529",
    # ICD-9 - Electrolyte
    "276", "2760", "2761", "2762", "2763", "2764", "2765", "2766", "2767", "2768", "2769",
]

# Lab item IDs for endocrine biomarkers
ENDOCRINE_LAB_ITEMS = {
    # Glucose metabolism
    "glucose": [50809, 50931],
    "hba1c": [50852],  # Hemoglobin A1c
    # Ketones/Acidosis
    "anion_gap": [50868],
    "bicarbonate": [50882],
    "ph": [50820],
    "lactate": [50813, 52442],
    # Electrolytes
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "chloride": [50902],
    "calcium": [50893],
    "magnesium": [50960],
    "phosphate": [50970],
    # Osmolality
    "osmolality_serum": [50964],
    "osmolality_urine": [51695],
    # Thyroid
    "tsh": [50993],
    "t3_free": [50994],
    "t4_free": [50995],
    # Renal (for DKA/HHS monitoring)
    "creatinine": [50912, 52546],
    "bun": [51006],
    # Cortisol
    "cortisol": [50909],
}


def load_endocrine_diagnoses():
    """Load patients with endocrine diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for endocrine codes (match prefix)
    endo_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in ENDOCRINE_ICD_CODES)
    )
    endo_dx = df[endo_mask].copy()
    print(f"Endocrine diagnoses found: {len(endo_dx):,}")

    # Get unique patients and admissions
    endo_hadm_ids = endo_dx['hadm_id'].unique()
    endo_subject_ids = endo_dx['subject_id'].unique()
    print(f"Unique endocrine admissions: {len(endo_hadm_ids):,}")
    print(f"Unique endocrine patients: {len(endo_subject_ids):,}")

    return endo_dx, endo_hadm_ids, endo_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for endocrine cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for endocrine cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for endocrine cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for endocrine cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in ENDOCRINE_LAB_ITEMS.values():
        all_lab_items.extend(items)
    all_lab_items = set(all_lab_items)

    hadm_set = set(hadm_ids)
    lab_chunks = []
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(FILES["labevents"], compression='gzip', chunksize=chunk_size):
        total_rows += len(chunk)

        # Filter for our patients AND our biomarkers
        mask = (chunk['hadm_id'].isin(hadm_set)) & (chunk['itemid'].isin(all_lab_items))
        filtered = chunk[mask].copy()

        if len(filtered) > 0:
            lab_chunks.append(filtered)
            matched_rows += len(filtered)

        print(f"  Processed {total_rows:,} rows, matched {matched_rows:,}...", end='\r')

    print(f"\n  Total lab events matched: {matched_rows:,}")

    if lab_chunks:
        labs = pd.concat(lab_chunks, ignore_index=True)
        return labs
    return pd.DataFrame()


def create_cohort_summary(patients, admissions, icu_stays, labs, endo_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "endocrine",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(endo_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(ENDOCRINE_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "diabetes_type1": len(endo_dx[endo_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('E10', '25001', '25003', '25011', '25013', '25021', '25023'))]),
            "diabetes_type2": len(endo_dx[endo_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('E11', '25000', '25002', '25010', '25012', '25020', '25022'))]),
            "dka_hhs": len(endo_dx[endo_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('E101', 'E111', 'E100', 'E110', '2501', '2502'))]),
            "thyroid": len(endo_dx[endo_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('E00', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06', 'E07', '240', '241', '242', '243', '244', '245', '246'))]),
            "adrenal": len(endo_dx[endo_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('E27', 'E24', '255'))]),
            "electrolyte_disorders": len(endo_dx[endo_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('E86', 'E87', '276'))]),
        }
    }

    # Add mortality if available
    if 'hospital_expire_flag' in admissions.columns:
        mortality = admissions['hospital_expire_flag'].mean() * 100
        summary["outcomes"]["hospital_mortality_pct"] = round(mortality, 2)

    if 'deathtime' in admissions.columns:
        deaths = admissions['deathtime'].notna().sum()
        summary["outcomes"]["total_deaths"] = int(deaths)

    return summary


def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("MIMIC-IV Endocrine Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    endo_output = os.path.join(OUTPUT_PATH, "endocrine_cohort")
    os.makedirs(endo_output, exist_ok=True)

    # Step 1: Find endocrine patients
    endo_dx, hadm_ids, subject_ids = load_endocrine_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, endo_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(endo_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(endo_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(endo_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(endo_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    endo_dx.to_csv(os.path.join(endo_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(endo_dx):,} rows)")

    with open(os.path.join(endo_output, "cohort_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved cohort_summary.json")

    print("\n" + "=" * 60)
    print("COHORT SUMMARY")
    print("=" * 60)
    print(f"Total Patients:     {summary['total_patients']:,}")
    print(f"Total Admissions:   {summary['total_admissions']:,}")
    print(f"Total ICU Stays:    {summary['total_icu_stays']:,}")
    print(f"Total Lab Events:   {summary['total_lab_events']:,}")
    if 'hospital_mortality_pct' in summary['outcomes']:
        print(f"Hospital Mortality: {summary['outcomes']['hospital_mortality_pct']:.1f}%")
    print("\nDiagnosis Categories:")
    for cat, count in summary['diagnosis_categories'].items():
        print(f"  {cat}: {count:,}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
