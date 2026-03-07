"""Extract hematological (blood disorder) patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Hematological ICD codes (ICD-9 and ICD-10)
HEMATOLOGICAL_ICD_CODES = [
    # ICD-10 - Anemia
    "D50", "D500", "D501", "D508", "D509",  # Iron deficiency anemia
    "D51", "D510", "D511", "D512", "D513", "D518", "D519",  # Vitamin B12 deficiency
    "D52", "D520", "D521", "D528", "D529",  # Folate deficiency
    "D53", "D530", "D531", "D532", "D538", "D539",  # Other nutritional anemia
    "D55", "D56", "D57", "D58", "D59",  # Hemolytic anemias
    "D60", "D61", "D62", "D63", "D64",  # Aplastic and other anemias
    # ICD-10 - Coagulation defects
    "D65",  # DIC
    "D66",  # Hemophilia A
    "D67",  # Hemophilia B
    "D68", "D680", "D681", "D682", "D683", "D684", "D688", "D689",  # Other coagulation defects
    # ICD-10 - Thrombocytopenia and platelet disorders
    "D69", "D690", "D691", "D692", "D693", "D694", "D695", "D696", "D698", "D699",
    # ICD-10 - Neutropenia and WBC disorders
    "D70", "D700", "D701", "D702", "D703", "D704", "D708", "D709",  # Neutropenia
    "D71", "D72",  # Other WBC disorders
    # ICD-10 - Pancytopenia
    "D61", "D610", "D611", "D612", "D613", "D618", "D619",
    # ICD-9 - Anemia
    "280", "2800", "2801", "2808", "2809",  # Iron deficiency
    "281", "2810", "2811", "2812", "2813", "2814", "2818", "2819",  # Other deficiency
    "282", "2820", "2821", "2822", "2823", "2824", "2825", "2826", "2827", "2828", "2829",  # Hereditary hemolytic
    "283", "2830", "2831", "2832", "2839",  # Acquired hemolytic
    "284", "2840", "2841", "2842", "2848", "2849",  # Aplastic
    "285", "2850", "2851", "2852", "2858", "2859",  # Other anemia
    # ICD-9 - Coagulation
    "286", "2860", "2861", "2862", "2863", "2864", "2865", "2866", "2867", "2869",
    # ICD-9 - Thrombocytopenia
    "287", "2870", "2871", "2872", "2873", "2874", "2875", "2878", "2879",
    # ICD-9 - Neutropenia/WBC disorders
    "288", "2880", "2881", "2882", "2883", "2884", "2888", "2889",
]

# Lab item IDs for hematological biomarkers
HEMATOLOGICAL_LAB_ITEMS = {
    # Complete Blood Count
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    "rbc": [51279],  # Red blood cell count
    "wbc": [51301, 51300],  # White blood cell count
    "platelets": [51265],
    # RBC Indices
    "mcv": [51250],  # Mean Corpuscular Volume
    "mch": [51248],  # Mean Corpuscular Hemoglobin
    "mchc": [51249],  # Mean Corpuscular Hemoglobin Concentration
    "rdw": [51277],  # Red Cell Distribution Width
    # WBC Differential
    "neutrophils": [51256],  # Absolute neutrophil count
    "lymphocytes": [51244, 51245],
    "monocytes": [51254],
    "eosinophils": [51200],
    "basophils": [51146],
    # Coagulation
    "inr": [51237],
    "pt": [51274],  # Prothrombin Time
    "ptt": [51275],  # Partial Thromboplastin Time
    "fibrinogen": [51214],
    "d_dimer": [51196, 50915],
    # Reticulocytes
    "reticulocytes": [51278],
    # Iron studies
    "iron": [50952],
    "ferritin": [50924],
    "tibc": [50998],  # Total Iron Binding Capacity
    # Hemolysis markers
    "ldh": [50954],  # Lactate Dehydrogenase
    "haptoglobin": [50934],
    "bilirubin_indirect": [50884],
}


def load_hematological_diagnoses():
    """Load patients with hematological diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for hematological codes (match prefix)
    heme_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in HEMATOLOGICAL_ICD_CODES)
    )
    heme_dx = df[heme_mask].copy()
    print(f"Hematological diagnoses found: {len(heme_dx):,}")

    # Get unique patients and admissions
    heme_hadm_ids = heme_dx['hadm_id'].unique()
    heme_subject_ids = heme_dx['subject_id'].unique()
    print(f"Unique hematological admissions: {len(heme_hadm_ids):,}")
    print(f"Unique hematological patients: {len(heme_subject_ids):,}")

    return heme_dx, heme_hadm_ids, heme_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for hematological cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for hematological cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for hematological cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for hematological cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in HEMATOLOGICAL_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, heme_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "hematological",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(heme_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(HEMATOLOGICAL_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "anemia": len(heme_dx[heme_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('D50', 'D51', 'D52', 'D53', 'D55', 'D56', 'D57', 'D58', 'D59', 'D60', 'D61', 'D62', 'D63', 'D64',
                 '280', '281', '282', '283', '284', '285'))]),
            "coagulation_disorder": len(heme_dx[heme_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('D65', 'D66', 'D67', 'D68', '286'))]),
            "thrombocytopenia": len(heme_dx[heme_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('D69', '287'))]),
            "neutropenia_wbc": len(heme_dx[heme_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ('D70', 'D71', 'D72', '288'))]),
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
    print("MIMIC-IV Hematological Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    heme_output = os.path.join(OUTPUT_PATH, "hematological_cohort")
    os.makedirs(heme_output, exist_ok=True)

    # Step 1: Find hematological patients
    heme_dx, hadm_ids, subject_ids = load_hematological_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, heme_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(heme_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(heme_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(heme_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(heme_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    heme_dx.to_csv(os.path.join(heme_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(heme_dx):,} rows)")

    with open(os.path.join(heme_output, "cohort_summary.json"), "w") as f:
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
