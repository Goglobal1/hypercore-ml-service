"""Extract psychiatric patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Psychiatric ICD codes (ICD-9 and ICD-10)
PSYCHIATRIC_ICD_CODES = [
    # ICD-10 - Mental and behavioral disorders (F00-F99)
    # Organic mental disorders
    "F00", "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F09",
    # Substance use disorders
    "F10",  # Alcohol
    "F11",  # Opioids
    "F12",  # Cannabis
    "F13",  # Sedatives
    "F14",  # Cocaine
    "F15",  # Stimulants
    "F16",  # Hallucinogens
    "F17",  # Tobacco
    "F18",  # Inhalants
    "F19",  # Multiple/other substances
    # Schizophrenia and psychotic disorders
    "F20", "F21", "F22", "F23", "F24", "F25", "F28", "F29",
    # Mood disorders
    "F30",  # Manic episode
    "F31",  # Bipolar disorder
    "F32",  # Depressive episode
    "F33",  # Recurrent depression
    "F34",  # Persistent mood disorders
    "F38", "F39",
    # Anxiety disorders
    "F40",  # Phobic anxiety
    "F41",  # Other anxiety disorders
    "F42",  # OCD
    "F43",  # Reaction to stress/adjustment disorders
    "F44",  # Dissociative disorders
    "F45",  # Somatoform disorders
    "F48",  # Other neurotic disorders
    # Behavioral syndromes
    "F50",  # Eating disorders
    "F51",  # Sleep disorders
    "F52",  # Sexual dysfunction
    "F53",  # Puerperal mental disorders
    "F54", "F55", "F59",
    # Personality disorders
    "F60", "F61", "F62", "F63", "F64", "F65", "F66", "F68", "F69",
    # Intellectual disabilities
    "F70", "F71", "F72", "F73", "F78", "F79",
    # Developmental disorders
    "F80", "F81", "F82", "F83", "F84", "F88", "F89",
    # Behavioral/emotional disorders (childhood)
    "F90",  # ADHD
    "F91", "F92", "F93", "F94", "F95", "F98",
    # Unspecified mental disorder
    "F99",
    # Suicidal ideation/self-harm
    "R45.851",  # Suicidal ideation
    "T14.91",  # Suicide attempt
    # ICD-9 - Mental disorders (290-319)
    # Organic psychoses
    "290", "291", "292", "293", "294",
    # Other psychoses
    "295",  # Schizophrenia
    "296",  # Affective psychoses (bipolar, depression)
    "297",  # Paranoid states
    "298",  # Other nonorganic psychoses
    "299",  # Psychoses of childhood
    # Neurotic disorders
    "300",  # Anxiety, dissociative, somatoform
    # Personality disorders
    "301",
    # Sexual deviations
    "302",
    # Alcohol dependence
    "303",
    # Drug dependence
    "304",
    # Nondependent abuse
    "305",
    # Physiological malfunction
    "306",
    # Special symptoms
    "307",  # Including eating disorders
    # Acute reaction to stress
    "308",
    # Adjustment reaction
    "309",
    # Specific nonpsychotic disorders
    "310", "311", "312", "313", "314", "315", "316",
    # Intellectual disabilities
    "317", "318", "319",
    # Suicide/self-harm (E-codes)
    "E950", "E951", "E952", "E953", "E954", "E955", "E956", "E957", "E958", "E959",
]

# Lab item IDs for psychiatric biomarkers
PSYCHIATRIC_LAB_ITEMS = {
    # Drug level monitoring
    "lithium": [50994],
    "valproic_acid": [51001],
    "carbamazepine": [50885],
    # Metabolic monitoring (antipsychotic side effects)
    "glucose": [50809, 50931],
    "hba1c": [50852],
    # Electrolytes (lithium toxicity)
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "chloride": [50902],
    "bicarbonate": [50882],
    "calcium": [50893],
    "magnesium": [50960],
    # Renal function (lithium monitoring)
    "creatinine": [50912, 52546],
    "bun": [51006],
    # Liver function (valproate/carbamazepine monitoring)
    "alt": [50861],
    "ast": [50878],
    "albumin": [50862],
    # Thyroid function (lithium effects)
    "tsh": [50993],
    # Hematologic (clozapine monitoring)
    "wbc": [51301, 51300],
    "neutrophils": [51256],
    "platelets": [51265],
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    # NMS markers
    "ck": [50910],  # Creatine kinase
    # Lipid panel (metabolic syndrome)
    "cholesterol": [50907],
    "triglycerides": [51000],
    # Toxicology
    "alcohol_level": [50821],
}


def load_psychiatric_diagnoses():
    """Load patients with psychiatric diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for psychiatric codes (match prefix)
    psych_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in PSYCHIATRIC_ICD_CODES)
    )
    psych_dx = df[psych_mask].copy()
    print(f"Psychiatric diagnoses found: {len(psych_dx):,}")

    # Get unique patients and admissions
    psych_hadm_ids = psych_dx['hadm_id'].unique()
    psych_subject_ids = psych_dx['subject_id'].unique()
    print(f"Unique psychiatric admissions: {len(psych_hadm_ids):,}")
    print(f"Unique psychiatric patients: {len(psych_subject_ids):,}")

    return psych_dx, psych_hadm_ids, psych_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for psychiatric cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for psychiatric cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for psychiatric cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for psychiatric cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in PSYCHIATRIC_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, psych_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "psychiatric",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(psych_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(PSYCHIATRIC_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "substance_use": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["F1" + str(i) for i in range(10)] + ["291", "292", "303", "304", "305"]))]),
            "schizophrenia_psychosis": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["F2" + str(i) for i in range(10)] + ["295", "297", "298"]))]),
            "mood_disorders": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["F3" + str(i) for i in range(10)] + ["296"]))]),
            "anxiety_disorders": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["F4" + str(i) for i in range(10)] + ["300"]))]),
            "eating_disorders": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("F50", "307"))]),
            "personality_disorders": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("F60", "F61", "301"))]),
            "adhd": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("F90", "314"))]),
            "intellectual_disability": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["F7" + str(i) for i in range(10)] + ["317", "318", "319"]))]),
            "suicide_self_harm": len(psych_dx[psych_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("R45851", "T1491", "E95"))]),
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
    print("MIMIC-IV Psychiatric Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    psych_output = os.path.join(OUTPUT_PATH, "psychiatric_cohort")
    os.makedirs(psych_output, exist_ok=True)

    # Step 1: Find psychiatric patients
    psych_dx, hadm_ids, subject_ids = load_psychiatric_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, psych_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(psych_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(psych_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(psych_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(psych_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    psych_dx.to_csv(os.path.join(psych_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(psych_dx):,} rows)")

    with open(os.path.join(psych_output, "cohort_summary.json"), "w") as f:
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
