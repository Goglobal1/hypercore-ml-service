"""Extract oncology (cancer) patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Oncology ICD codes (ICD-9 and ICD-10)
ONCOLOGY_ICD_CODES = [
    # ICD-10 - Malignant neoplasms
    # Lip, oral cavity, pharynx
    "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14",
    # Digestive organs
    "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
    # Respiratory and intrathoracic
    "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39",
    # Bone and cartilage
    "C40", "C41",
    # Melanoma and skin
    "C43", "C44",
    # Soft tissue
    "C45", "C46", "C47", "C48", "C49",
    # Breast
    "C50",
    # Female genital organs
    "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
    # Male genital organs
    "C60", "C61", "C62", "C63",
    # Urinary tract
    "C64", "C65", "C66", "C67", "C68",
    # Eye, brain, CNS
    "C69", "C70", "C71", "C72",
    # Thyroid and other endocrine
    "C73", "C74", "C75",
    # Ill-defined and secondary
    "C76", "C77", "C78", "C79", "C80",
    # Lymphoid, hematopoietic (leukemia/lymphoma)
    "C81", "C82", "C83", "C84", "C85", "C86",  # Lymphomas
    "C88",  # Immunoproliferative
    "C90",  # Multiple myeloma
    "C91", "C92", "C93", "C94", "C95", "C96",  # Leukemias
    # Chemotherapy complications
    "T45", "T451",  # Antineoplastic drug poisoning
    "D70",  # Neutropenia (often chemo-induced)
    "E883",  # Tumor lysis syndrome
    # ICD-9 - Malignant neoplasms (140-208)
    "140", "141", "142", "143", "144", "145", "146", "147", "148", "149",  # Lip, oral, pharynx
    "150", "151", "152", "153", "154", "155", "156", "157", "158", "159",  # Digestive
    "160", "161", "162", "163", "164", "165",  # Respiratory
    "170", "171", "172", "173", "174", "175", "176",  # Bone, skin, breast
    "179", "180", "181", "182", "183", "184",  # Female genital
    "185", "186", "187", "188", "189",  # Male genital, urinary
    "190", "191", "192", "193", "194",  # Eye, brain, thyroid
    "195", "196", "197", "198", "199",  # Ill-defined, secondary
    "200", "201", "202", "203", "204", "205", "206", "207", "208",  # Lymphatic/hematopoietic
    # ICD-9 - Chemo complications
    "2880",  # Neutropenia
    "9931",  # Antineoplastic drug toxicity
]

# Lab item IDs for oncology biomarkers
ONCOLOGY_LAB_ITEMS = {
    # CBC (bone marrow suppression monitoring)
    "wbc": [51301, 51300],
    "neutrophils": [51256],  # Absolute neutrophil count
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    "platelets": [51265],
    # Tumor lysis syndrome markers
    "potassium": [50971, 50822],
    "phosphate": [50970],
    "calcium": [50893],
    "uric_acid": [51007],
    # Renal function
    "creatinine": [50912, 52546],
    "bun": [51006],
    # LDH (tumor burden)
    "ldh": [50954],
    # Liver function (metastases, drug toxicity)
    "alt": [50861],
    "ast": [50878],
    "bilirubin_total": [50885],
    "albumin": [50862],
    # Coagulation (DIC risk)
    "inr": [51237],
    "ptt": [51275],
    "fibrinogen": [51214],
    "d_dimer": [51196, 50915],
    # Electrolytes
    "sodium": [50983, 50824],
    "magnesium": [50960],
    # Tissue perfusion
    "lactate": [50813, 52442],
}


def load_oncology_diagnoses():
    """Load patients with oncology diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for oncology codes (match prefix)
    onco_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in ONCOLOGY_ICD_CODES)
    )
    onco_dx = df[onco_mask].copy()
    print(f"Oncology diagnoses found: {len(onco_dx):,}")

    # Get unique patients and admissions
    onco_hadm_ids = onco_dx['hadm_id'].unique()
    onco_subject_ids = onco_dx['subject_id'].unique()
    print(f"Unique oncology admissions: {len(onco_hadm_ids):,}")
    print(f"Unique oncology patients: {len(onco_subject_ids):,}")

    return onco_dx, onco_hadm_ids, onco_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for oncology cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for oncology cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for oncology cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for oncology cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in ONCOLOGY_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, onco_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "oncology",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(onco_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(ONCOLOGY_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "solid_tumors": len(onco_dx[onco_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple([f"C{i:02d}" for i in range(0, 76)] + [str(i) for i in range(140, 196)]))]),
            "lymphoma": len(onco_dx[onco_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('C81', 'C82', 'C83', 'C84', 'C85', 'C86', '200', '201', '202'))]),
            "leukemia": len(onco_dx[onco_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('C91', 'C92', 'C93', 'C94', 'C95', '204', '205', '206', '207', '208'))]),
            "myeloma": len(onco_dx[onco_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('C90', '203'))]),
            "metastatic": len(onco_dx[onco_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('C77', 'C78', 'C79', '196', '197', '198'))]),
            "neutropenia": len(onco_dx[onco_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('D70', '2880'))]),
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
    print("MIMIC-IV Oncology Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    onco_output = os.path.join(OUTPUT_PATH, "oncology_cohort")
    os.makedirs(onco_output, exist_ok=True)

    # Step 1: Find oncology patients
    onco_dx, hadm_ids, subject_ids = load_oncology_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, onco_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(onco_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(onco_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(onco_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(onco_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    onco_dx.to_csv(os.path.join(onco_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(onco_dx):,} rows)")

    with open(os.path.join(onco_output, "cohort_summary.json"), "w") as f:
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
