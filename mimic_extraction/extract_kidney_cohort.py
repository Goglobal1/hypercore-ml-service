"""Extract kidney/AKI patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Kidney/AKI ICD codes (ICD-9 and ICD-10)
KIDNEY_ICD_CODES = [
    # ICD-10 - Acute Kidney Injury
    "N17", "N170", "N171", "N172", "N178", "N179",
    # ICD-10 - Chronic Kidney Disease
    "N18", "N181", "N182", "N183", "N184", "N185", "N186", "N189",
    # ICD-10 - Kidney failure unspecified
    "N19",
    # ICD-10 - Other kidney disorders
    "N14", "N15", "N16",
    # ICD-10 - Dialysis status
    "Z99.2", "Z992",
    # ICD-9 - Acute Kidney Failure
    "584", "5840", "5845", "5846", "5847", "5848", "5849",
    # ICD-9 - Chronic Kidney Disease
    "585", "5851", "5852", "5853", "5854", "5855", "5856", "5859",
    # ICD-9 - Kidney failure unspecified
    "586",
    # ICD-9 - Renal dialysis status
    "V451", "V4511", "V4512",
]

# Key lab item IDs for kidney biomarkers
KIDNEY_LAB_ITEMS = {
    "creatinine": [50912, 52546],
    "bun": [51006, 52647],
    "potassium": [50971, 50822],
    "sodium": [50983, 50824],
    "bicarbonate": [50882],
    "chloride": [50902],
    "gfr": [50920, 52026],
    "phosphate": [50970],
    "calcium": [50893],
    "albumin": [50862],
    "hemoglobin": [51222, 50811],
    "platelets": [51265],
}


def load_kidney_diagnoses():
    """Load patients with kidney diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for kidney codes (match prefix)
    kidney_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in KIDNEY_ICD_CODES)
    )
    kidney_dx = df[kidney_mask].copy()
    print(f"Kidney diagnoses found: {len(kidney_dx):,}")

    # Get unique patients and admissions
    kidney_hadm_ids = kidney_dx['hadm_id'].unique()
    kidney_subject_ids = kidney_dx['subject_id'].unique()
    print(f"Unique kidney admissions: {len(kidney_hadm_ids):,}")
    print(f"Unique kidney patients: {len(kidney_subject_ids):,}")

    return kidney_dx, kidney_hadm_ids, kidney_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for kidney cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for kidney cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for kidney cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for kidney cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in KIDNEY_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, kidney_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "kidney",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(kidney_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(KIDNEY_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "aki": len(kidney_dx[kidney_dx['icd_code'].str.upper().str.startswith(('N17', '584'))]),
            "ckd": len(kidney_dx[kidney_dx['icd_code'].str.upper().str.startswith(('N18', '585'))]),
            "esrd_dialysis": len(kidney_dx[kidney_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('Z992', 'V451'))]),
            "other_kidney": len(kidney_dx[kidney_dx['icd_code'].str.upper().str.startswith(('N14', 'N15', 'N16', 'N19', '586'))]),
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
    print("MIMIC-IV Kidney/AKI Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    kidney_output = os.path.join(OUTPUT_PATH, "kidney_cohort")
    os.makedirs(kidney_output, exist_ok=True)

    # Step 1: Find kidney patients
    kidney_dx, hadm_ids, subject_ids = load_kidney_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, kidney_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(kidney_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(kidney_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(kidney_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(kidney_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    kidney_dx.to_csv(os.path.join(kidney_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(kidney_dx):,} rows)")

    with open(os.path.join(kidney_output, "cohort_summary.json"), "w") as f:
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
