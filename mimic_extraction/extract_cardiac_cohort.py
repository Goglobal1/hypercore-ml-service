"""Extract cardiac patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Cardiac ICD codes (ICD-9 and ICD-10)
CARDIAC_ICD_CODES = [
    # ICD-10 - Acute MI
    "I21", "I210", "I211", "I212", "I213", "I214", "I219",
    "I22", "I220", "I221", "I228", "I229",
    # ICD-10 - Heart Failure
    "I50", "I501", "I502", "I5020", "I5021", "I5022", "I5023",
    "I503", "I5030", "I5031", "I5032", "I5033",
    "I504", "I5040", "I5041", "I5042", "I5043", "I509",
    # ICD-10 - Cardiomyopathy
    "I42", "I420", "I421", "I422", "I423", "I424", "I425", "I426", "I427", "I428", "I429",
    # ICD-10 - Arrhythmias
    "I47", "I48", "I49",
    # ICD-10 - Cardiac arrest
    "I46", "I460", "I461", "I469",
    # ICD-9 - Acute MI
    "410", "4100", "4101", "4102", "4103", "4104", "4105", "4106", "4107", "4108", "4109",
    "411", "4110", "4111", "4118",
    # ICD-9 - Heart Failure
    "428", "4280", "4281", "42820", "42821", "42822", "42823",
    "42830", "42831", "42832", "42833", "42840", "42841", "42842", "42843", "4289",
    # ICD-9 - Cardiomyopathy
    "425", "4250", "4251", "4252", "4253", "4254", "4255", "4257", "4258", "4259",
    # ICD-9 - Cardiac arrest
    "4275",
]

# Key lab item IDs for cardiac biomarkers
CARDIAC_LAB_ITEMS = {
    "troponin": [51002, 51003, 52642],  # Troponin I, T
    "bnp": [50963, 51921],  # NTproBNP
    "ck_mb": [50908, 51580],  # CK-MB
    "creatinine": [50912, 52546],
    "potassium": [50971, 50822],
    "hemoglobin": [51222, 50811],
    "platelets": [51265],
    "inr": [51237],
    "lactate": [50813, 52442, 53154],
    "bun": [51006],
}


def load_cardiac_diagnoses():
    """Load patients with cardiac diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for cardiac codes (match prefix)
    cardiac_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().startswith(code) for code in CARDIAC_ICD_CODES)
    )
    cardiac_dx = df[cardiac_mask].copy()
    print(f"Cardiac diagnoses found: {len(cardiac_dx):,}")

    # Get unique patients and admissions
    cardiac_hadm_ids = cardiac_dx['hadm_id'].unique()
    cardiac_subject_ids = cardiac_dx['subject_id'].unique()
    print(f"Unique cardiac admissions: {len(cardiac_hadm_ids):,}")
    print(f"Unique cardiac patients: {len(cardiac_subject_ids):,}")

    return cardiac_dx, cardiac_hadm_ids, cardiac_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for cardiac cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for cardiac cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for cardiac cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for cardiac cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in CARDIAC_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, cardiac_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "cardiac",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(cardiac_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(CARDIAC_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "ami": len(cardiac_dx[cardiac_dx['icd_code'].str.startswith(('I21', 'I22', '410', '411'))]),
            "heart_failure": len(cardiac_dx[cardiac_dx['icd_code'].str.startswith(('I50', '428'))]),
            "cardiomyopathy": len(cardiac_dx[cardiac_dx['icd_code'].str.startswith(('I42', '425'))]),
            "arrhythmia": len(cardiac_dx[cardiac_dx['icd_code'].str.startswith(('I47', 'I48', 'I49'))]),
            "cardiac_arrest": len(cardiac_dx[cardiac_dx['icd_code'].str.startswith(('I46', '4275'))]),
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
    print("MIMIC-IV Cardiac Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    cardiac_output = os.path.join(OUTPUT_PATH, "cardiac_cohort")
    os.makedirs(cardiac_output, exist_ok=True)

    # Step 1: Find cardiac patients
    cardiac_dx, hadm_ids, subject_ids = load_cardiac_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, cardiac_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(cardiac_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(cardiac_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(cardiac_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(cardiac_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    cardiac_dx.to_csv(os.path.join(cardiac_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(cardiac_dx):,} rows)")

    with open(os.path.join(cardiac_output, "cohort_summary.json"), "w") as f:
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
