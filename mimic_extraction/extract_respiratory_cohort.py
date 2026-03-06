"""Extract respiratory patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Respiratory ICD codes (ICD-9 and ICD-10)
RESPIRATORY_ICD_CODES = [
    # ICD-10 - Respiratory failure
    "J96", "J960", "J9600", "J9601", "J9602",
    "J961", "J9610", "J9611", "J9612",
    "J962", "J9620", "J9621", "J9622",
    "J969", "J9690", "J9691", "J9692",
    # ICD-10 - ARDS
    "J80",
    # ICD-10 - Other respiratory conditions
    "J95", "J9500", "J9501", "J9502", "J9503", "J9504",  # Ventilator-associated
    "J98", "J984",  # Other respiratory disorders
    # ICD-9 - Respiratory failure
    "518", "51881", "51882", "51883", "51884",
    "5185",  # Pulmonary insufficiency
    # ICD-9 - ARDS
    "5188",  # Other diseases of lung
    # ICD-9 - Ventilator-associated
    "9672",  # Ventilator dependence
]

# Lab item IDs for respiratory biomarkers (from labevents)
RESPIRATORY_LAB_ITEMS = {
    "pao2": [50821, 52042],  # pO2
    "paco2": [50818, 52040],  # pCO2
    "spo2_lab": [50817],  # Oxygen Saturation (lab)
    "ph": [50820, 52038],  # pH arterial
    "lactate": [50813, 52442],  # Lactate (tissue oxygenation marker)
    "hemoglobin": [51222, 50811],  # Hemoglobin (oxygen carrying)
    "bicarbonate": [50882],  # Bicarbonate (acid-base)
}

# ICU chart item IDs for respiratory vitals (from chartevents)
RESPIRATORY_CHART_ITEMS = {
    "spo2": [220277],  # O2 saturation pulseoxymetry
    "respiratory_rate": [220210, 224688, 224689, 224690],  # Respiratory rates
    "fio2": [223835],  # Inspired O2 Fraction
    "peep": [220339, 224700],  # PEEP
    "tidal_volume": [224684, 224685, 224686],  # Tidal volumes
}


def load_respiratory_diagnoses():
    """Load patients with respiratory diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for respiratory codes (match prefix)
    respiratory_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in RESPIRATORY_ICD_CODES)
    )
    respiratory_dx = df[respiratory_mask].copy()
    print(f"Respiratory diagnoses found: {len(respiratory_dx):,}")

    # Get unique patients and admissions
    respiratory_hadm_ids = respiratory_dx['hadm_id'].unique()
    respiratory_subject_ids = respiratory_dx['subject_id'].unique()
    print(f"Unique respiratory admissions: {len(respiratory_hadm_ids):,}")
    print(f"Unique respiratory patients: {len(respiratory_subject_ids):,}")

    return respiratory_dx, respiratory_hadm_ids, respiratory_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for respiratory cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for respiratory cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for respiratory cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for respiratory cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in RESPIRATORY_LAB_ITEMS.values():
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
    """Load ICU chart events for respiratory vitals."""
    print("\nLoading chart events (this may take a while)...")

    # Flatten chart item IDs
    all_chart_items = []
    for items in RESPIRATORY_CHART_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, charts, respiratory_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "respiratory",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_chart_events": len(charts),
        "total_diagnoses": len(respiratory_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(RESPIRATORY_LAB_ITEMS.keys()),
        "chart_vitals_available": list(RESPIRATORY_CHART_ITEMS.keys()),
        "diagnosis_categories": {
            "respiratory_failure": len(respiratory_dx[respiratory_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('J96', '518'))]),
            "ards": len(respiratory_dx[respiratory_dx['icd_code'].str.upper().str.startswith('J80')]),
            "ventilator_associated": len(respiratory_dx[respiratory_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(('J95', '9672'))]),
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
    print("MIMIC-IV Respiratory Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    respiratory_output = os.path.join(OUTPUT_PATH, "respiratory_cohort")
    os.makedirs(respiratory_output, exist_ok=True)

    # Step 1: Find respiratory patients
    respiratory_dx, hadm_ids, subject_ids = load_respiratory_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Load chart events (ICU vitals)
    charts = load_chart_events(hadm_ids)

    # Step 7: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, charts, respiratory_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(respiratory_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(respiratory_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(respiratory_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(respiratory_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    charts.to_csv(os.path.join(respiratory_output, "chart_events.csv"), index=False)
    print(f"  Saved chart_events.csv ({len(charts):,} rows)")

    respiratory_dx.to_csv(os.path.join(respiratory_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(respiratory_dx):,} rows)")

    with open(os.path.join(respiratory_output, "cohort_summary.json"), "w") as f:
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
