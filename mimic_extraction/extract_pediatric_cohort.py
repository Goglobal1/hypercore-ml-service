"""Extract pediatric patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Pediatric ICD codes (ICD-9 and ICD-10)
# Note: We also filter by age, but include pediatric-specific diagnoses
PEDIATRIC_ICD_CODES = [
    # ICD-10 - Conditions originating in perinatal period (P00-P96)
    "P00", "P01", "P02", "P03", "P04", "P05", "P07", "P08",
    "P10", "P11", "P12", "P13", "P14", "P15",
    "P20", "P21", "P22", "P23", "P24", "P25", "P26", "P27", "P28", "P29",
    "P35", "P36", "P37", "P38", "P39",
    "P50", "P51", "P52", "P53", "P54", "P55", "P56", "P57", "P58", "P59",
    "P60", "P61",
    "P70", "P71", "P72", "P74", "P76", "P77", "P78",
    "P80", "P81", "P83", "P84",
    "P90", "P91", "P92", "P93", "P94", "P95", "P96",
    # Congenital malformations (Q00-Q99)
    "Q00", "Q01", "Q02", "Q03", "Q04", "Q05", "Q06", "Q07",
    "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q17", "Q18",
    "Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28",
    "Q30", "Q31", "Q32", "Q33", "Q34", "Q35", "Q36", "Q37",
    "Q38", "Q39", "Q40", "Q41", "Q42", "Q43", "Q44", "Q45",
    "Q50", "Q51", "Q52", "Q53", "Q54", "Q55", "Q56",
    "Q60", "Q61", "Q62", "Q63", "Q64",
    "Q65", "Q66", "Q67", "Q68", "Q69", "Q70", "Q71", "Q72", "Q73", "Q74",
    "Q75", "Q76", "Q77", "Q78", "Q79",
    "Q80", "Q81", "Q82", "Q83", "Q84", "Q85", "Q86", "Q87",
    "Q89", "Q90", "Q91", "Q92", "Q93", "Q95", "Q96", "Q97", "Q98", "Q99",
    # Pediatric developmental disorders
    "F80", "F81", "F82", "F83", "F84",  # Developmental disorders
    "F90",  # ADHD
    "F91", "F92", "F93", "F94", "F95", "F98",  # Childhood behavioral disorders
    # Pediatric respiratory
    "J20", "J21",  # Bronchitis, bronchiolitis
    "J45",  # Asthma
    # Pediatric infections
    "A08",  # Viral gastroenteritis
    "B05", "B06",  # Measles, rubella
    "B26",  # Mumps
    # Childhood epilepsy
    "G40",
    # ICD-9 - Perinatal conditions (760-779)
    "760", "761", "762", "763", "764", "765", "766", "767", "768", "769",
    "770", "771", "772", "773", "774", "775", "776", "777", "778", "779",
    # ICD-9 - Congenital anomalies (740-759)
    "740", "741", "742", "743", "744", "745", "746", "747", "748", "749",
    "750", "751", "752", "753", "754", "755", "756", "757", "758", "759",
    # V/Z codes for pediatric care
    "V20", "V29", "V30", "V31", "V32", "V33", "V34", "V35", "V36", "V37", "V38", "V39",
    "Z00.1",  # Newborn health examination
    "Z38",  # Liveborn infants
    "Z76.1", "Z76.2",  # Child health supervision
]

# Lab item IDs for pediatric biomarkers
PEDIATRIC_LAB_ITEMS = {
    # CBC
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    "wbc": [51301, 51300],
    "platelets": [51265],
    "neutrophils": [51256],
    "lymphocytes": [51244, 51245],
    # Metabolic panel
    "glucose": [50809, 50931],
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "chloride": [50902],
    "bicarbonate": [50882],
    "calcium": [50893],
    "magnesium": [50960],
    "phosphate": [50970],
    # Renal function
    "creatinine": [50912, 52546],
    "bun": [51006],
    # Liver function
    "bilirubin_total": [50885],
    "bilirubin_direct": [50883],
    "ast": [50878],
    "alt": [50861],
    "albumin": [50862],
    # Blood gas
    "ph": [50820],
    "pao2": [50821],
    "pco2": [50818],
    "base_excess": [50802],
    # Infection markers
    "crp": [50889],
    "procalcitonin": [50976],
    "lactate": [50813, 52442],
    # Coagulation
    "inr": [51237],
    "pt": [51274],
    "ptt": [51275],
}


def load_pediatric_patients():
    """Load pediatric patients based on age and diagnoses."""
    print("Loading patients...")
    patients = pd.read_csv(FILES["patients"], compression='gzip')
    print(f"Total patients: {len(patients):,}")

    # Calculate age from anchor_year and anchor_age
    # Pediatric = age < 18 at any point during data collection
    # anchor_age is the age at anchor_year
    if 'anchor_age' in patients.columns:
        # Keep patients who were under 18 at anchor year
        # (they could have been pediatric during some admissions)
        pediatric_by_age = patients[patients['anchor_age'] < 18].copy()
        print(f"Patients with anchor_age < 18: {len(pediatric_by_age):,}")
    else:
        pediatric_by_age = pd.DataFrame()

    return patients, pediatric_by_age


def load_pediatric_diagnoses():
    """Load patients with pediatric diagnoses."""
    print("\nLoading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for pediatric codes (match prefix)
    peds_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in PEDIATRIC_ICD_CODES)
    )
    peds_dx = df[peds_mask].copy()
    print(f"Pediatric diagnoses found: {len(peds_dx):,}")

    # Get unique patients and admissions
    peds_hadm_ids = peds_dx['hadm_id'].unique()
    peds_subject_ids = peds_dx['subject_id'].unique()
    print(f"Unique pediatric admissions (by diagnosis): {len(peds_hadm_ids):,}")
    print(f"Unique pediatric patients (by diagnosis): {len(peds_subject_ids):,}")

    return peds_dx, peds_hadm_ids, peds_subject_ids


def combine_pediatric_cohort(patients, pediatric_by_age, peds_dx_subject_ids):
    """Combine age-based and diagnosis-based pediatric identification."""
    print("\nCombining pediatric cohort...")

    # Get subject IDs from both methods
    age_based_ids = set(pediatric_by_age['subject_id'].unique()) if len(pediatric_by_age) > 0 else set()
    dx_based_ids = set(peds_dx_subject_ids)

    # Union of both
    all_peds_ids = age_based_ids | dx_based_ids
    print(f"Pediatric by age: {len(age_based_ids):,}")
    print(f"Pediatric by diagnosis: {len(dx_based_ids):,}")
    print(f"Combined unique pediatric patients: {len(all_peds_ids):,}")

    return all_peds_ids


def load_admissions(subject_ids):
    """Load admission details for pediatric cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for pediatric cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for pediatric cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in PEDIATRIC_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, peds_dx, pediatric_subject_ids):
    """Create summary statistics for the cohort."""
    # Filter patients to pediatric cohort
    peds_patients = patients[patients['subject_id'].isin(pediatric_subject_ids)]

    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "pediatric",
        "total_patients": len(peds_patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(peds_dx),
        "demographics": {
            "gender_distribution": peds_patients['gender'].value_counts().to_dict() if 'gender' in peds_patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(PEDIATRIC_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "perinatal_conditions": len(peds_dx[peds_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["P" + str(i) for i in range(10)] + [str(i) for i in range(760, 780)]))]),
            "congenital_malformations": len(peds_dx[peds_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["Q" + str(i) for i in range(100)] + [str(i) for i in range(740, 760)]))]),
            "developmental_disorders": len(peds_dx[peds_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("F80", "F81", "F82", "F83", "F84"))]),
            "adhd_behavioral": len(peds_dx[peds_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("F90", "F91", "F92", "F93", "F94", "F95", "F98"))]),
            "respiratory": len(peds_dx[peds_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("J20", "J21", "J45"))]),
            "newborn_care": len(peds_dx[peds_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("Z38", "V3"))]),
        }
    }

    # Age distribution if available
    if 'anchor_age' in peds_patients.columns:
        age_dist = peds_patients['anchor_age'].describe().to_dict()
        summary["demographics"]["age_statistics"] = {k: round(v, 1) for k, v in age_dist.items()}

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
    print("MIMIC-IV Pediatric Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    peds_output = os.path.join(OUTPUT_PATH, "pediatric_cohort")
    os.makedirs(peds_output, exist_ok=True)

    # Step 1: Find pediatric patients by age
    patients, pediatric_by_age = load_pediatric_patients()

    # Step 2: Find pediatric diagnoses
    peds_dx, dx_hadm_ids, dx_subject_ids = load_pediatric_diagnoses()

    # Step 3: Combine cohort
    pediatric_subject_ids = combine_pediatric_cohort(patients, pediatric_by_age, dx_subject_ids)

    # Step 4: Load admissions for pediatric patients
    admissions = load_admissions(pediatric_subject_ids)
    hadm_ids = admissions['hadm_id'].unique()

    # Step 5: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 6: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 7: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, peds_dx, pediatric_subject_ids)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    peds_patients = patients[patients['subject_id'].isin(pediatric_subject_ids)]
    peds_patients.to_csv(os.path.join(peds_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(peds_patients):,} rows)")

    admissions.to_csv(os.path.join(peds_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(peds_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(peds_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    peds_dx.to_csv(os.path.join(peds_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(peds_dx):,} rows)")

    with open(os.path.join(peds_output, "cohort_summary.json"), "w") as f:
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
