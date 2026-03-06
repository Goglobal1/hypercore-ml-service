"""Extract hepatic (liver) patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Hepatic ICD codes (ICD-9 and ICD-10)
HEPATIC_ICD_CODES = [
    # ICD-10 - Liver failure
    "K72", "K720", "K7200", "K7201", "K721", "K7210", "K7211", "K729", "K7290", "K7291",
    # ICD-10 - Cirrhosis
    "K74", "K740", "K741", "K742", "K743", "K744", "K745", "K746",
    # ICD-10 - Alcoholic liver disease
    "K70", "K700", "K701", "K7010", "K7011", "K702", "K703", "K7030", "K7031",
    "K704", "K7040", "K7041", "K709",
    # ICD-10 - Hepatic fibrosis/sclerosis
    "K74", "K740", "K741", "K742",
    # ICD-10 - Hepatorenal syndrome
    "K767",
    # ICD-10 - Portal hypertension
    "K766",
    # ICD-10 - Hepatic encephalopathy
    "K7282",
    # ICD-10 - Toxic liver disease
    "K71", "K710", "K711", "K712", "K713", "K714", "K715", "K716", "K717", "K718", "K719",
    # ICD-9 - Liver failure/hepatic coma
    "570", "5722", "5723", "5724", "5728",
    # ICD-9 - Cirrhosis
    "5712", "5715", "5716",
    # ICD-9 - Alcoholic liver disease
    "5710", "5711", "5712", "5713",
    # ICD-9 - Portal hypertension
    "5723",
    # ICD-9 - Hepatorenal syndrome
    "5724",
]

# Lab item IDs for hepatic biomarkers
HEPATIC_LAB_ITEMS = {
    "alt": [50861],  # Alanine Aminotransferase
    "ast": [50878],  # Aspartate Aminotransferase
    "bilirubin_total": [50885],  # Bilirubin, Total
    "bilirubin_direct": [50883],  # Bilirubin, Direct
    "albumin": [50862],  # Albumin
    "alkaline_phosphatase": [50863],  # Alkaline Phosphatase
    "ammonia": [50866],  # Ammonia
    "inr": [51237],  # INR
    "ptt": [51275],  # PTT
    "platelets": [51265],  # Platelets (for cirrhosis)
    "creatinine": [50912, 52546],  # Creatinine (hepatorenal)
    "sodium": [50983, 50824],  # Sodium
    "lactate": [50813, 52442],  # Lactate
}


def load_hepatic_diagnoses():
    """Load patients with hepatic diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for hepatic codes (match prefix)
    hepatic_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in HEPATIC_ICD_CODES)
    )
    hepatic_dx = df[hepatic_mask].copy()
    print(f"Hepatic diagnoses found: {len(hepatic_dx):,}")

    # Get unique patients and admissions
    hepatic_hadm_ids = hepatic_dx['hadm_id'].unique()
    hepatic_subject_ids = hepatic_dx['subject_id'].unique()
    print(f"Unique hepatic admissions: {len(hepatic_hadm_ids):,}")
    print(f"Unique hepatic patients: {len(hepatic_subject_ids):,}")

    return hepatic_dx, hepatic_hadm_ids, hepatic_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for hepatic cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for hepatic cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for hepatic cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for hepatic cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in HEPATIC_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, hepatic_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "hepatic",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(hepatic_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(HEPATIC_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "liver_failure": len(hepatic_dx[hepatic_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('K72', '570', '5722'))]),
            "cirrhosis": len(hepatic_dx[hepatic_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('K74', '5712', '5715', '5716'))]),
            "alcoholic_liver": len(hepatic_dx[hepatic_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('K70', '5710', '5711', '5713'))]),
            "hepatorenal": len(hepatic_dx[hepatic_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('K767', '5724'))]),
            "toxic_liver": len(hepatic_dx[hepatic_dx['icd_code'].str.upper().str.replace('.', '').str.startswith('K71')]),
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
    print("MIMIC-IV Hepatic Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    hepatic_output = os.path.join(OUTPUT_PATH, "hepatic_cohort")
    os.makedirs(hepatic_output, exist_ok=True)

    # Step 1: Find hepatic patients
    hepatic_dx, hadm_ids, subject_ids = load_hepatic_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, hepatic_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(hepatic_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(hepatic_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(hepatic_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(hepatic_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    hepatic_dx.to_csv(os.path.join(hepatic_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(hepatic_dx):,} rows)")

    with open(os.path.join(hepatic_output, "cohort_summary.json"), "w") as f:
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
