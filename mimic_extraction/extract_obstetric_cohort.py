"""Extract obstetric patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Obstetric ICD codes (ICD-9 and ICD-10)
OBSTETRIC_ICD_CODES = [
    # ICD-10 - Pregnancy, childbirth and the puerperium (O00-O9A)
    # Ectopic and molar pregnancy
    "O00", "O01", "O02", "O03", "O04", "O05", "O06", "O07", "O08",
    # Supervision of high-risk pregnancy
    "O09",
    # Edema, proteinuria, hypertensive disorders
    "O10",  # Pre-existing hypertension
    "O11",  # Pre-existing hypertension with superimposed preeclampsia
    "O12",  # Gestational edema/proteinuria
    "O13",  # Gestational hypertension
    "O14",  # Preeclampsia
    "O15",  # Eclampsia
    "O16",  # Unspecified maternal hypertension
    # Other maternal disorders
    "O20", "O21", "O22", "O23", "O24", "O25", "O26", "O28", "O29",
    # Maternal care related to fetus
    "O30", "O31", "O32", "O33", "O34", "O35", "O36", "O37", "O38", "O39",
    # Complications of labor and delivery
    "O40", "O41", "O42", "O43", "O44", "O45", "O46", "O47", "O48",
    # Labor and delivery complications
    "O60", "O61", "O62", "O63", "O64", "O65", "O66", "O67", "O68", "O69",
    # Labor and delivery complications continued
    "O70", "O71", "O72", "O73", "O74", "O75", "O76", "O77",
    # Encounter for delivery
    "O80", "O82",
    # Complications of puerperium
    "O85",  # Puerperal sepsis
    "O86",  # Other puerperal infections
    "O87",  # Venous complications in puerperium
    "O88",  # Obstetric embolism
    "O89",  # Complications of anesthesia during puerperium
    "O90",  # Complications of puerperium NEC
    "O91",  # Infections of breast associated with pregnancy
    "O92",  # Other disorders of breast associated with pregnancy
    # Other obstetric conditions
    "O94", "O95", "O96", "O97", "O98", "O99", "O9A",
    # ICD-9 - Complications of pregnancy, childbirth, puerperium (630-679)
    # Ectopic and molar pregnancy
    "630", "631", "632", "633", "634", "635", "636", "637", "638", "639",
    # Complications mainly related to pregnancy
    "640",  # Hemorrhage in early pregnancy
    "641",  # Antepartum hemorrhage
    "642",  # Hypertension complicating pregnancy (includes preeclampsia/eclampsia)
    "643",  # Excessive vomiting in pregnancy
    "644",  # Early or threatened labor
    "645",  # Late pregnancy
    "646",  # Other complications of pregnancy
    "647",  # Infectious diseases in pregnancy
    "648",  # Other current conditions in pregnancy
    "649",  # Other conditions complicating pregnancy
    # Normal delivery and other indications
    "650", "651", "652", "653", "654", "655", "656", "657", "658", "659",
    # Complications of labor and delivery
    "660", "661", "662", "663", "664", "665", "666", "667", "668", "669",
    # Complications of puerperium
    "670", "671", "672", "673", "674", "675", "676",
    # Other maternal and fetal complications
    "677", "678", "679",
    # V codes for pregnancy
    "V22", "V23", "V24", "V27", "V28",
    # Z codes for pregnancy (ICD-10)
    "Z33", "Z34", "Z36", "Z37", "Z39",
]

# Lab item IDs for obstetric biomarkers
OBSTETRIC_LAB_ITEMS = {
    # Hemorrhage monitoring
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    # HELLP syndrome markers
    "platelets": [51265],
    "ast": [50878],
    "alt": [50861],
    "ldh": [50954],
    "bilirubin_total": [50885],
    # Preeclampsia markers
    "creatinine": [50912, 52546],
    "bun": [51006],
    "uric_acid": [51007],
    "protein_urine": [51492],
    # Gestational diabetes
    "glucose": [50809, 50931],
    "hba1c": [50852],
    # Eclampsia treatment monitoring
    "magnesium": [50960],
    # Coagulation (DIC)
    "inr": [51237],
    "pt": [51274],
    "ptt": [51275],
    "fibrinogen": [51214],
    "d_dimer": [50915],
    # Infection/sepsis markers
    "wbc": [51301, 51300],
    "lactate": [50813, 52442],
    # Electrolytes
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "calcium": [50893],
    # Blood gas (severe cases)
    "ph": [50820],
    "pao2": [50821],
    "base_excess": [50802],
    # Albumin (edema assessment)
    "albumin": [50862],
}


def load_obstetric_diagnoses():
    """Load patients with obstetric diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for obstetric codes (match prefix)
    obs_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in OBSTETRIC_ICD_CODES)
    )
    obs_dx = df[obs_mask].copy()
    print(f"Obstetric diagnoses found: {len(obs_dx):,}")

    # Get unique patients and admissions
    obs_hadm_ids = obs_dx['hadm_id'].unique()
    obs_subject_ids = obs_dx['subject_id'].unique()
    print(f"Unique obstetric admissions: {len(obs_hadm_ids):,}")
    print(f"Unique obstetric patients: {len(obs_subject_ids):,}")

    return obs_dx, obs_hadm_ids, obs_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for obstetric cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for obstetric cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for obstetric cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for obstetric cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in OBSTETRIC_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, obs_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "obstetric",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(obs_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(OBSTETRIC_LAB_ITEMS.keys()),
        "diagnosis_categories": {
            "hypertensive_disorders": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O10", "O11", "O12", "O13", "O14", "O15", "O16", "642"))]),
            "preeclampsia_eclampsia": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O14", "O15"))]),
            "hemorrhage": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O20", "O44", "O45", "O46", "O67", "O72", "640", "641", "666"))]),
            "gestational_diabetes": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O24", "648"))]),
            "puerperal_sepsis": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O85", "O86", "670"))]),
            "obstetric_embolism": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O88", "673"))]),
            "labor_delivery_complications": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple([f"O6{i}" for i in range(10)] + [f"O7{i}" for i in range(8)] +
                      [f"66{i}" for i in range(10)]))]),
            "ectopic_molar_pregnancy": len(obs_dx[obs_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("O00", "O01", "O02", "630", "631", "632", "633"))]),
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
    print("MIMIC-IV Obstetric Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    obs_output = os.path.join(OUTPUT_PATH, "obstetric_cohort")
    os.makedirs(obs_output, exist_ok=True)

    # Step 1: Find obstetric patients
    obs_dx, hadm_ids, subject_ids = load_obstetric_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, obs_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(obs_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(obs_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(obs_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(obs_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    obs_dx.to_csv(os.path.join(obs_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(obs_dx):,} rows)")

    with open(os.path.join(obs_output, "cohort_summary.json"), "w") as f:
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
