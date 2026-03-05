"""Extract sepsis patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, ICD_CODES, OUTPUT_PATH

# Sepsis ICD codes (ICD-9 and ICD-10)
SEPSIS_ICD_CODES = [
    # ICD-10
    "A40", "A400", "A401", "A403", "A408", "A409",  # Streptococcal sepsis
    "A41", "A410", "A411", "A412", "A413", "A414", "A4150", "A4151", "A4152", "A4153", "A4159",
    "A418", "A4181", "A4189", "A419",  # Other sepsis
    "R6520", "R6521",  # Severe sepsis
    # ICD-9
    "99591", "99592",  # Sepsis, severe sepsis
    "78552",  # Septic shock
    "0380", "0381", "0382", "0383", "03810", "03811", "03812", "03819",
    "03840", "03841", "03842", "03843", "03844", "03849", "0388", "0389",  # Septicemia
]

# Key lab item IDs for sepsis biomarkers (from our mapping)
SEPSIS_LAB_ITEMS = {
    "lactate": [50813, 52442, 53154],
    "wbc": [51300, 51301],
    "creatinine": [50912, 52546],
    "bilirubin": [50885],
    "platelets": [51265],
    "inr": [51237],
    "crp": [50889, 51652],
    "hemoglobin": [51222, 50811],
    "potassium": [50971, 50822],
    "bun": [51006],
    "albumin": [50862],
}


def load_sepsis_diagnoses():
    """Load patients with sepsis diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for sepsis codes (match prefix)
    sepsis_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().startswith(code) for code in SEPSIS_ICD_CODES)
    )
    sepsis_dx = df[sepsis_mask].copy()
    print(f"Sepsis diagnoses found: {len(sepsis_dx):,}")

    # Get unique patients and admissions
    sepsis_hadm_ids = sepsis_dx['hadm_id'].unique()
    sepsis_subject_ids = sepsis_dx['subject_id'].unique()
    print(f"Unique sepsis admissions: {len(sepsis_hadm_ids):,}")
    print(f"Unique sepsis patients: {len(sepsis_subject_ids):,}")

    return sepsis_dx, sepsis_hadm_ids, sepsis_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for sepsis cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for sepsis cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for sepsis cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for sepsis cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in SEPSIS_LAB_ITEMS.values():
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


def load_microbiology(hadm_ids):
    """Load microbiology events for sepsis cohort."""
    print("\nLoading microbiology events...")
    df = pd.read_csv(FILES["microbiologyevents"], compression='gzip')
    micro = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Microbiology events loaded: {len(micro):,}")
    return micro


def create_cohort_summary(patients, admissions, icu_stays, labs, micro):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "sepsis",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_micro_events": len(micro),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(SEPSIS_LAB_ITEMS.keys()),
    }

    # Add mortality if available
    if 'hospital_expire_flag' in admissions.columns:
        mortality = admissions['hospital_expire_flag'].mean() * 100
        summary["outcomes"]["hospital_mortality_pct"] = round(mortality, 2)

    # Add ICU mortality if available
    if 'deathtime' in admissions.columns:
        icu_deaths = admissions['deathtime'].notna().sum()
        summary["outcomes"]["total_deaths"] = int(icu_deaths)

    return summary


def main():
    """Main extraction pipeline."""
    print("=" * 60)
    print("MIMIC-IV Sepsis Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    sepsis_output = os.path.join(OUTPUT_PATH, "sepsis_cohort")
    os.makedirs(sepsis_output, exist_ok=True)

    # Step 1: Find sepsis patients
    sepsis_dx, hadm_ids, subject_ids = load_sepsis_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Load microbiology
    micro = load_microbiology(hadm_ids)

    # Step 7: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, micro)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(sepsis_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(sepsis_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(sepsis_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(sepsis_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    micro.to_csv(os.path.join(sepsis_output, "microbiology.csv"), index=False)
    print(f"  Saved microbiology.csv ({len(micro):,} rows)")

    sepsis_dx.to_csv(os.path.join(sepsis_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(sepsis_dx):,} rows)")

    with open(os.path.join(sepsis_output, "cohort_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved cohort_summary.json")

    print("\n" + "=" * 60)
    print("COHORT SUMMARY")
    print("=" * 60)
    print(f"Total Patients:     {summary['total_patients']:,}")
    print(f"Total Admissions:   {summary['total_admissions']:,}")
    print(f"Total ICU Stays:    {summary['total_icu_stays']:,}")
    print(f"Total Lab Events:   {summary['total_lab_events']:,}")
    print(f"Total Micro Events: {summary['total_micro_events']:,}")
    if 'hospital_mortality_pct' in summary['outcomes']:
        print(f"Hospital Mortality: {summary['outcomes']['hospital_mortality_pct']:.1f}%")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
