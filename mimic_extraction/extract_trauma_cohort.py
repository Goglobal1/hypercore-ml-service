"""Extract trauma patient cohorts from MIMIC-IV."""
import pandas as pd
import json
import os
from datetime import datetime
from config import FILES, OUTPUT_PATH

# Trauma ICD codes (ICD-9 and ICD-10)
TRAUMA_ICD_CODES = [
    # ICD-10 - Injuries (S00-S99)
    # Head injuries
    "S00", "S01", "S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09",
    # Neck injuries
    "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S19",
    # Thorax injuries
    "S20", "S21", "S22", "S23", "S24", "S25", "S26", "S27", "S28", "S29",
    # Abdomen/pelvis injuries
    "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37", "S38", "S39",
    # Upper limb injuries
    "S40", "S41", "S42", "S43", "S44", "S45", "S46", "S47", "S48", "S49",
    "S50", "S51", "S52", "S53", "S54", "S55", "S56", "S57", "S58", "S59",
    "S60", "S61", "S62", "S63", "S64", "S65", "S66", "S67", "S68", "S69",
    # Lower limb injuries
    "S70", "S71", "S72", "S73", "S74", "S75", "S76", "S77", "S78", "S79",
    "S80", "S81", "S82", "S83", "S84", "S85", "S86", "S87", "S88", "S89",
    "S90", "S91", "S92", "S93", "S94", "S95", "S96", "S97", "S98", "S99",
    # Multiple body regions (T00-T07)
    "T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07",
    # Burns (T20-T32)
    "T20", "T21", "T22", "T23", "T24", "T25", "T26", "T27", "T28", "T29",
    "T30", "T31", "T32",
    # Frostbite (T33-T35)
    "T33", "T34", "T35",
    # Complications of trauma (T79)
    "T79",  # Includes traumatic shock, embolism, compartment syndrome
    # Hemorrhagic shock
    "R57", "T794",
    # ICD-9 - Injuries (800-959)
    # Fracture of skull
    "800", "801", "802", "803", "804",
    # Fracture of spine/trunk
    "805", "806", "807", "808", "809",
    # Fracture of upper limb
    "810", "811", "812", "813", "814", "815", "816", "817", "818", "819",
    # Fracture of lower limb
    "820", "821", "822", "823", "824", "825", "826", "827", "828", "829",
    # Dislocation
    "830", "831", "832", "833", "834", "835", "836", "837", "838", "839",
    # Sprains and strains
    "840", "841", "842", "843", "844", "845", "846", "847", "848",
    # Intracranial injury
    "850", "851", "852", "853", "854",
    # Internal injury thorax/abdomen/pelvis
    "860", "861", "862", "863", "864", "865", "866", "867", "868", "869",
    # Open wound
    "870", "871", "872", "873", "874", "875", "876", "877", "878", "879",
    "880", "881", "882", "883", "884", "885", "886", "887",
    "890", "891", "892", "893", "894", "895", "896", "897",
    # Blood vessel injury
    "900", "901", "902", "903", "904",
    # Late effects and complications
    "905", "906", "907", "908", "909",
    # Superficial injury
    "910", "911", "912", "913", "914", "915", "916", "917", "918", "919",
    # Contusion
    "920", "921", "922", "923", "924",
    # Crushing injury
    "925", "926", "927", "928", "929",
    # Burns
    "940", "941", "942", "943", "944", "945", "946", "947", "948", "949",
    # Injury to nerves/spinal cord
    "950", "951", "952", "953", "954", "955", "956", "957",
    # Certain traumatic complications
    "958",  # Traumatic shock, embolism, compartment syndrome
    # Injury other/unspecified
    "959",
]

# Lab item IDs for trauma biomarkers
TRAUMA_LAB_ITEMS = {
    # Hemorrhage/blood loss monitoring
    "hemoglobin": [51222, 50811],
    "hematocrit": [51221, 50810],
    # Shock/perfusion markers
    "lactate": [50813, 52442],
    "base_excess": [50802],
    "ph": [50820],
    # Coagulopathy (trauma triad of death)
    "inr": [51237],
    "pt": [51274],
    "ptt": [51275],
    "fibrinogen": [51214],
    "platelets": [51265],
    "d_dimer": [51196, 50915],
    # Rhabdomyolysis/muscle injury
    "creatinine": [50912, 52546],
    "ck": [50910],  # Creatine kinase
    "myoglobin": [51245],
    "potassium": [50971, 50822],
    # Renal function (AKI from shock/rhabdo)
    "bun": [51006],
    # Cardiac contusion
    "troponin_t": [51003, 52642],
    "troponin_i": [51002],
    # Massive transfusion effects
    "calcium": [50893],
    "magnesium": [50960],
    # General
    "wbc": [51301, 51300],
    "sodium": [50983, 50824],
    "glucose": [50809, 50931],
    "bicarbonate": [50882],
}


def load_trauma_diagnoses():
    """Load patients with trauma diagnoses."""
    print("Loading diagnoses_icd...")
    df = pd.read_csv(FILES["diagnoses_icd"], compression='gzip')
    print(f"Total diagnoses: {len(df):,}")

    # Filter for trauma codes (match prefix)
    trauma_mask = df['icd_code'].apply(
        lambda x: any(str(x).upper().replace('.', '').startswith(code.replace('.', ''))
                     for code in TRAUMA_ICD_CODES)
    )
    trauma_dx = df[trauma_mask].copy()
    print(f"Trauma diagnoses found: {len(trauma_dx):,}")

    # Get unique patients and admissions
    trauma_hadm_ids = trauma_dx['hadm_id'].unique()
    trauma_subject_ids = trauma_dx['subject_id'].unique()
    print(f"Unique trauma admissions: {len(trauma_hadm_ids):,}")
    print(f"Unique trauma patients: {len(trauma_subject_ids):,}")

    return trauma_dx, trauma_hadm_ids, trauma_subject_ids


def load_patient_demographics(subject_ids):
    """Load patient demographics for trauma cohort."""
    print("\nLoading patient demographics...")
    df = pd.read_csv(FILES["patients"], compression='gzip')
    patients = df[df['subject_id'].isin(subject_ids)].copy()
    print(f"Patients loaded: {len(patients):,}")
    return patients


def load_admissions(hadm_ids):
    """Load admission details for trauma cohort."""
    print("\nLoading admissions...")
    df = pd.read_csv(FILES["admissions"], compression='gzip')
    admissions = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"Admissions loaded: {len(admissions):,}")
    return admissions


def load_icu_stays(hadm_ids):
    """Load ICU stays for trauma cohort."""
    print("\nLoading ICU stays...")
    df = pd.read_csv(FILES["icustays"], compression='gzip')
    icu_stays = df[df['hadm_id'].isin(hadm_ids)].copy()
    print(f"ICU stays loaded: {len(icu_stays):,}")
    return icu_stays


def load_lab_events(hadm_ids, chunk_size=1_000_000):
    """Load lab events for trauma cohort (chunked for memory)."""
    print("\nLoading lab events (this may take a while)...")

    # Flatten lab item IDs
    all_lab_items = []
    for items in TRAUMA_LAB_ITEMS.values():
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


def create_cohort_summary(patients, admissions, icu_stays, labs, trauma_dx):
    """Create summary statistics for the cohort."""
    summary = {
        "extraction_date": datetime.now().isoformat(),
        "cohort": "trauma",
        "total_patients": len(patients),
        "total_admissions": len(admissions),
        "total_icu_stays": len(icu_stays),
        "total_lab_events": len(labs),
        "total_diagnoses": len(trauma_dx),
        "demographics": {
            "gender_distribution": patients['gender'].value_counts().to_dict() if 'gender' in patients.columns else {},
        },
        "outcomes": {},
        "lab_biomarkers_available": list(TRAUMA_LAB_ITEMS.keys()),
        "injury_categories": {
            "head_injury": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple([f"S0{i}" for i in range(10)] + [str(i) for i in range(800, 805)] + ["850", "851", "852", "853", "854"]))]),
            "thorax_injury": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["S2" + str(i) for i in range(10)] + ["860", "861", "862"]))]),
            "abdominal_injury": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["S3" + str(i) for i in range(10)] + ["863", "864", "865", "866", "867", "868", "869"]))]),
            "spine_injury": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["S12", "S13", "S14", "S22", "S23", "S24", "S32", "S33", "S34"] + ["805", "806"]))]),
            "fractures": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["S42", "S52", "S62", "S72", "S82", "S92"] + [str(i) for i in range(810, 830)]))]),
            "burns": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["T2" + str(i) for i in range(10)] + ["T30", "T31", "T32"] + [str(i) for i in range(940, 950)]))]),
            "multiple_trauma": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                tuple(["T00", "T01", "T02", "T03", "T04", "T05", "T06", "T07"]))]),
            "traumatic_shock": len(trauma_dx[trauma_dx['icd_code'].str.upper().str.replace('.', '').str.startswith(
                ("T79", "R57", "958"))]),
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
    print("MIMIC-IV Trauma Cohort Extraction")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    trauma_output = os.path.join(OUTPUT_PATH, "trauma_cohort")
    os.makedirs(trauma_output, exist_ok=True)

    # Step 1: Find trauma patients
    trauma_dx, hadm_ids, subject_ids = load_trauma_diagnoses()

    # Step 2: Load demographics
    patients = load_patient_demographics(subject_ids)

    # Step 3: Load admissions
    admissions = load_admissions(hadm_ids)

    # Step 4: Load ICU stays
    icu_stays = load_icu_stays(hadm_ids)

    # Step 5: Load lab events
    labs = load_lab_events(hadm_ids)

    # Step 6: Create summary
    summary = create_cohort_summary(patients, admissions, icu_stays, labs, trauma_dx)

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving cohort data...")

    patients.to_csv(os.path.join(trauma_output, "patients.csv"), index=False)
    print(f"  Saved patients.csv ({len(patients):,} rows)")

    admissions.to_csv(os.path.join(trauma_output, "admissions.csv"), index=False)
    print(f"  Saved admissions.csv ({len(admissions):,} rows)")

    icu_stays.to_csv(os.path.join(trauma_output, "icu_stays.csv"), index=False)
    print(f"  Saved icu_stays.csv ({len(icu_stays):,} rows)")

    labs.to_csv(os.path.join(trauma_output, "lab_events.csv"), index=False)
    print(f"  Saved lab_events.csv ({len(labs):,} rows)")

    trauma_dx.to_csv(os.path.join(trauma_output, "diagnoses.csv"), index=False)
    print(f"  Saved diagnoses.csv ({len(trauma_dx):,} rows)")

    with open(os.path.join(trauma_output, "cohort_summary.json"), "w") as f:
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
    print("\nInjury Categories:")
    for cat, count in summary['injury_categories'].items():
        print(f"  {cat}: {count:,}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()
