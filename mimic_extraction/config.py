"""MIMIC-IV data paths and configuration."""
import os

# MIMIC-IV data location
MIMIC_BASE_PATH = r"C:\Users\letsa\Downloads\mimic-iv-3.1\mimic-iv-3.1"

# Key files
HOSP_PATH = os.path.join(MIMIC_BASE_PATH, "hosp")
ICU_PATH = os.path.join(MIMIC_BASE_PATH, "icu")

# Specific files we need
FILES = {
    "patients": os.path.join(HOSP_PATH, "patients.csv.gz"),
    "admissions": os.path.join(HOSP_PATH, "admissions.csv.gz"),
    "labevents": os.path.join(HOSP_PATH, "labevents.csv.gz"),
    "d_labitems": os.path.join(HOSP_PATH, "d_labitems.csv.gz"),
    "diagnoses_icd": os.path.join(HOSP_PATH, "diagnoses_icd.csv.gz"),
    "icustays": os.path.join(ICU_PATH, "icustays.csv.gz"),
    "chartevents": os.path.join(ICU_PATH, "chartevents.csv.gz"),
    "d_items": os.path.join(ICU_PATH, "d_items.csv.gz"),
}

# ICD codes for conditions we want to extract
ICD_CODES = {
    "sepsis": ["A40", "A41", "R65.2", "99591", "99592", "78552"],
    "cardiac": ["I21", "I22", "I50", "410", "428"],
    "aki": ["N17", "584"],
    "respiratory_failure": ["J96", "518.81", "518.82"],
}

# Lab item IDs for key biomarkers (from d_labitems)
# These will be populated after reading d_labitems
LAB_ITEMS = {}

# Output directory for extracted patterns
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pattern_library")
