"""Extract neurological patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Neurological ICD codes (ICD-9 and ICD-10)
NEUROLOGICAL_ICD_CODES = [
    # ICD-10 - Stroke/Cerebrovascular
    "I63", "I630", "I631", "I632", "I633", "I634", "I635", "I636", "I638", "I639",
    "I64",  # Stroke, not specified
    "I60", "I600", "I601", "I602", "I603", "I604", "I605", "I606", "I607", "I608", "I609",  # SAH
    "I61", "I610", "I611", "I612", "I613", "I614", "I615", "I616", "I618", "I619",  # ICH
    "I62", "I620", "I621", "I629",  # Other nontraumatic ICH
    # ICD-10 - Traumatic Brain Injury
    "S06", "S060", "S061", "S062", "S063", "S064", "S065", "S066", "S067", "S068", "S069",
    # ICD-10 - Seizures/Epilepsy
    "G40", "G400", "G401", "G402", "G403", "G404", "G405", "G408", "G409",
    "G41", "G410", "G411", "G412", "G418", "G419",  # Status epilepticus
    # ICD-10 - Encephalopathy
    "G93", "G930", "G931", "G932", "G933", "G934", "G935", "G936", "G937", "G938", "G939",
    "G92",  # Toxic encephalopathy
    # ICD-10 - Coma
    "R40", "R402",
    # ICD-9 - Stroke
    "433", "4330", "4331", "4332", "4333", "4338", "4339",
    "434", "4340", "4341", "4349",
    "436",  # Acute CVA
    # ICD-9 - Intracranial hemorrhage
    "430",  # SAH
    "431",  # ICH
    "432", "4320", "4321", "4329",
    # ICD-9 - TBI
    "850", "851", "852", "853", "854",
    # ICD-9 - Seizures
    "345", "3450", "3451", "3452", "3453", "3454", "3455", "3456", "3457", "3458", "3459",
    # ICD-9 - Encephalopathy
    "348", "3481", "3482", "3483", "3484", "3485", "3488", "3489",
    # ICD-9 - Coma
    "780", "78001", "78003",
]

# Lab item IDs for neurological biomarkers
NEUROLOGICAL_LAB_ITEMS = {
    "sodium": [50983, 50824],  # Critical for neuro function
    "glucose": [50809, 50931],  # Hypoglycemia affects brain
    "lactate": [50813, 52442],  # Tissue perfusion
    "ammonia": [50866],  # Encephalopathy
    "osmolality": [50964],  # Cerebral edema risk
    "magnesium": [50960],  # Seizure threshold
    "calcium": [50893],  # Neuronal function
    "potassium": [50971, 50822],  # Cardiac/neuro function
    "hemoglobin": [51222, 50811],  # Oxygen carrying
    "platelets": [51265],  # Bleeding risk
    "inr": [51237],  # Coagulation
}

# ICU chart item IDs for neurological assessments
NEUROLOGICAL_CHART_ITEMS = {
    "gcs_eye": [220739],  # GCS - Eye Opening
    "gcs_verbal": [223900],  # GCS - Verbal Response
    "gcs_motor": [223901],  # GCS - Motor Response
    "pupil_size_right": [223907],  # Pupil Size Right
    "pupil_size_left": [224733],  # Pupil Size Left
    "pupil_response_right": [227121],  # Pupil Response Right
    "pupil_response_left": [227288],  # Pupil Response Left
    "level_of_consciousness": [226104],  # Level of Consciousness
}


def load_neurological_diagnoses():
    """Load patients with neurological diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for neurological codes (match prefix)
    neuro_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in NEUROLOGICAL_ICD_CODES)
    )
    neuro_dx = df[neuro_mask].copy()
    print(f"Neurological diagnoses found: {len(neuro_dx):,}")

    # Get unique patients and admissions
    neuro_hadm_ids = neuro_dx['hadm_id'].unique()
    neuro_subject_ids = neuro_dx['subject_id'].unique()
    print(f"Unique neurological admissions: {len(neuro_hadm_ids):,}")
    print(f"Unique neurological patients: {len(neuro_subject_ids):,}")

    return neuro_dx, neuro_hadm_ids, neuro_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for neurological cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for neurological cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for neurological cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for neurological cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in NEUROLOGICAL_LAB_ITEMS.values():
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


def load_chart_events(hadm_ids, chunk_size=5_000_000):
    """Load ICU chart events for neurological assessments."""
    print("\nLoading chart events (this may take a while)...")

    # Flatten chart item IDs
    all_chart_items = []
    for items in NEUROLOGICAL_CHART_ITEMS.values():
        all_chart_items.extend(items)
    all_chart_items = set(all_chart_items)

    hadm_set = set(hadm_ids)
    chart_chunks = []
    total_rows = 0
    matched_rows = 0

    for chunk in pd.read_csv(FILES["chartevents"], compression='gzip', chunksize=chunk_size):
        total_rows += len(chunk)

        # Filter for our patients AND our biomarkers
        mask = (chunk['hadm_id'].isin(hadm_set)) & (chunk['itemid'].isin(all_chart_items))
        filtered = chunk[mask].copy()

        if len(filtered) > 0:
            chart_chunks.append(filtered)
            matched_rows += len(filtered)

        print(f"  Processed {total_rows:,} rows, matched {matched_rows:,}...", end='\r')

    print(f"\n  Total chart events matched: {matched_rows:,}")

    if chart_chunks:
        charts = pd.concat(chart_chunks, ignore_index=True)
        return charts
    return pd.DataFrame()


def create_cohort_summary(patients, admissions, icu_stays, labs, charts, neuro_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "neurological",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_chart_events": len(charts),
        "total_diagnoses": len(neuro_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(NEUROLOGICAL_LAB_ITEMS.keys()),
        "chart_assessments_available": list(NEUROLOGICAL_CHART_ITEMS.keys()),
        "diagnosis_categories": {
            "stroke_ischemic": len(neuro_dx[neuro_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('I63', '433', '434', '436'))]),
            "hemorrhage_ich": len(neuro_dx[neuro_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('I61', '431'))]),
            "hemorrhage_sah": len(neuro_dx[neuro_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('I60', '430'))]),
            "tbi": len(neuro_dx[neuro_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('S06', '850', '851', '852', '853', '854'))]),
            "seizures": len(neuro_dx[neuro_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('G40', 'G41', '345'))]),
            "encephalopathy": len(neuro_dx[neuro_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('G93', 'G92', '348'))]),
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
    print("MIMIC-IV Neurological Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    neuro_output = os.path.join(OUTPUT_PATH, "neurological_cohort")
    os.makedirs(neuro_output, exist_ok=True)

    # Step 1: Find neurological patients
    neuro_dx, hadm_ids, subject_ids = load_neurological_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Load chart events (GCS, pupils, etc.)
    charts = load_chart_events(hadm_ids)

    # Step 7: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, charts, neuro_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(neuro_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(neuro_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(neuro_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(neuro_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    charts.to_csv(os.path.join(neuro_output, "chart_events.csv"), index=False)
    print(f"  Saved chart_events.csv ({len(charts):,} rows)")

    neuro_dx.to_csv(os.path.join(neuro_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(neuro_dx):,} rows)")

    with open(os.path.join(neuro_output, "cohort_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved cohort_summary.json")

    print("\n" + "=" * 60)
    print("COHORT SUMMARY")
    print("=" * 60)
    print(f"Total Patients:     {summary['total_patients']:,}")
    print(f"Total Admissions:   {summary['total_admissions']:,}")
    print(f"Total ICU Stays:    {summary['total_icu_stays']:,}")
    print(f"Total Lab Events:   {summary['total_lab_events']:,}")
    print(f"Total Chart Events: {summary['total_chart_events']:,}")
    if 'hospital_mortality_pct' in summary['outcomes']:
        print(f"Hospital Mortality: {summary['outcomes']['hospital_mortality_pct']:.1f}%")
    print("\nDiagnosis Categories:")
    for cat, count in summary['diagnosis_categories'].items():
        print(f"  {cat}: {count:,}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
