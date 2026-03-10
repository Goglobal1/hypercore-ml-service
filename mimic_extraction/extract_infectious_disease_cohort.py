"""Extract infectious disease patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Infectious disease ICD codes (ICD-9 and ICD-10)
INFECTIOUS_DISEASE_ICD_CODES = [
    # ICD-10 - Certain infectious and parasitic diseases (A00-B99)
    # Intestinal infectious diseases
    "A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09",
    # Tuberculosis
    "A15", "A16", "A17", "A18", "A19",
    # Bacterial diseases
    "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28",
    # Bacterial zoonoses
    "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39",
    # Sepsis/Septicemia
    "A40", "A41",  # Streptococcal/Other sepsis
    # Other bacterial infections
    "A42", "A43", "A44", "A46", "A48", "A49",
    # STIs
    "A50", "A51", "A52", "A53", "A54", "A55", "A56", "A57", "A58", "A59",
    "A60", "A63", "A64",
    # Spirochetal diseases
    "A65", "A66", "A67", "A68", "A69",
    # Chlamydial/Rickettsial
    "A70", "A71", "A74", "A75", "A77", "A78", "A79",
    # Viral CNS infections
    "A80", "A81", "A82", "A83", "A84", "A85", "A86", "A87", "A88", "A89",
    # Viral fevers
    "A90", "A91", "A92", "A93", "A94", "A95", "A96", "A98", "A99",
    # Viral infections (skin/mucous)
    "B00", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09",
    # Viral hepatitis
    "B15", "B16", "B17", "B18", "B19",
    # HIV
    "B20", "B21", "B22", "B23", "B24",
    # Other viral diseases
    "B25", "B26", "B27", "B30", "B33", "B34",
    # Mycoses (fungal)
    "B35", "B36", "B37", "B38", "B39", "B40", "B41", "B42", "B43", "B44",
    "B45", "B46", "B47", "B48", "B49",
    # Protozoal diseases
    "B50", "B51", "B52", "B53", "B54", "B55", "B56", "B57", "B58", "B59",
    "B60", "B64",
    # Helminthiases
    "B65", "B66", "B67", "B68", "B69", "B70", "B71", "B72", "B73", "B74",
    "B75", "B76", "B77", "B78", "B79", "B80", "B81", "B82", "B83",
    # Parasitic diseases
    "B85", "B86", "B87", "B88", "B89",
    # Other infectious diseases
    "B90", "B91", "B92", "B94", "B95", "B96", "B97", "B99",
    # Pneumonia
    "J09", "J10", "J11",  # Influenza
    "J12", "J13", "J14", "J15", "J16", "J17", "J18",  # Pneumonia
    # COVID-19
    "U07",
    # Meningitis
    "G00", "G01", "G02", "G03",
    # Endocarditis
    "I33",
    # UTI
    "N390",
    # Skin/soft tissue infections
    "L00", "L01", "L02", "L03", "L04", "L05", "L08",
    # Osteomyelitis
    "M86",
    # ICD-9 - Infectious and parasitic diseases (001-139)
    "001", "002", "003", "004", "005", "006", "007", "008", "009",
    "010", "011", "012", "013", "014", "015", "016", "017", "018",
    "020", "021", "022", "023", "024", "025", "026", "027",
    "030", "031", "032", "033", "034", "035", "036", "037", "038", "039",
    "040", "041", "042",  # HIV/AIDS
    "045", "046", "047", "048", "049",
    "050", "051", "052", "053", "054", "055", "056", "057", "058", "059",
    "060", "061", "062", "063", "064", "065", "066",
    "070", "071", "072", "073", "074", "075", "076", "077", "078", "079",
    "080", "081", "082", "083", "084", "085", "086", "087", "088",
    "090", "091", "092", "093", "094", "095", "096", "097", "098", "099",
    "100", "101", "102", "103", "104",
    "110", "111", "112", "114", "115", "116", "117", "118",
    "120", "121", "122", "123", "124", "125", "126", "127", "128", "129",
    "130", "131", "132", "133", "134", "135", "136", "137", "138", "139",
    # ICD-9 Septicemia
    "995.91", "995.92",  # Sepsis, severe sepsis
    # ICD-9 Pneumonia
    "480", "481", "482", "483", "484", "485", "486", "487", "488",
    # ICD-9 Meningitis
    "320", "321", "322",
    # ICD-9 UTI
    "599.0",
    # ICD-9 Cellulitis
    "681", "682",
]

# Lab item IDs for infectious disease biomarkers
INFECTIOUS_DISEASE_LAB_ITEMS = {
    # Infection/inflammation markers
    "wbc": [51301, 51300],
    "neutrophils": [51256],
    "bands": [51144],  # Band neutrophils (left shift)
    "lymphocytes": [51244, 51245],
    "crp": [50889],  # C-reactive protein
    "procalcitonin": [50976],
    # Sepsis markers
    "lactate": [50813, 52442],
    # Organ dysfunction markers
    "creatinine": [50912, 52546],
    "bun": [51006],
    "bilirubin_total": [50885],
    "alt": [50861],
    "ast": [50878],
    "inr": [51237],
    "platelets": [51265],
    # Blood gas (sepsis severity)
    "ph": [50820],
    "pao2": [50821],
    "pco2": [50818],
    "base_excess": [50802],
    # Electrolytes
    "sodium": [50983, 50824],
    "potassium": [50971, 50822],
    "glucose": [50809, 50931],
    "bicarbonate": [50882],
    # CBC
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    # Albumin (nutrition/severity)
    "albumin": [50862],
}


def load_infectious_disease_diagnoses():
    """Load patients with infectious disease diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for infectious disease codes (match prefix)
    id_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in INFECTIOUS_DISEASE_ICD_CODES)
    )
    id_dx = df[id_mask].copy()
    print(f"Infectious disease diagnoses found: {len(id_dx):,}")

    # Get unique patients and admissions
    id_hadm_ids = id_dx['hadm_id'].unique()
    id_subject_ids = id_dx['subject_id'].unique()
    print(f"Unique infectious disease admissions: {len(id_hadm_ids):,}")
    print(f"Unique infectious disease patients: {len(id_subject_ids):,}")

    return id_dx, id_hadm_ids, id_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for infectious disease cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for infectious disease cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for infectious disease cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for infectious disease cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in INFECTIOUS_DISEASE_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, id_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "infectious_disease",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(id_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(INFECTIOUS_DISEASE_LAB_ITEMS.keys()),
        "infection_categories": {
            "sepsis": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("A40", "A41", "99591", "99592", "038"))]),
            "pneumonia": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["J1" + str(i) for i in range(2, 9)] + [str(i) for i in range(480, 489)]))]),
            "urinary_tract": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("N390", "5990"))]),
            "skin_soft_tissue": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("L00", "L01", "L02", "L03", "L04", "L05", "L08", "681", "682"))]),
            "meningitis": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("G00", "G01", "G02", "G03", "320", "321", "322"))]),
            "endocarditis": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("I33",))]),
            "hiv": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("B20", "B21", "B22", "B23", "B24", "042"))]),
            "viral_hepatitis": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("B15", "B16", "B17", "B18", "B19", "070"))]),
            "tuberculosis": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("A15", "A16", "A17", "A18", "A19", "010", "011", "012", "013", "014", "015", "016", "017", "018"))]),
            "fungal": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["B3" + str(i) for i in range(5, 10)] + ["B4" + str(i) for i in range(10)] +
                      [str(i) for i in range(110, 119)]))]),
            "covid19": len(id_dx[id_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(("U07",))]),
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
    print("MIMIC-IV Infectious Disease Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    id_output = os.path.join(OUTPUT_PATH, "infectious_disease_cohort")
    os.makedirs(id_output, exist_ok=True)

    # Step 1: Find infectious disease patients
    id_dx, hadm_ids, subject_ids = load_infectious_disease_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, id_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(id_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(id_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(id_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(id_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    id_dx.to_csv(os.path.join(id_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(id_dx):,} rows)")

    with open(os.path.join(id_output, "cohort_summary.json"), "w") as f:
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
    print("\nInfection Categories:")
    for cat, count in summary['infection_categories'].items():
        print(f"  {cat}: {count:,}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
