"""
HyperCore 24-Endpoint Definitions
=================================

Each endpoint has:
- Name and category
- Associated biomarkers with normal ranges
- Scoring logic
- Status thresholds (normal, elevated, critical)

Categories:
- organ_system (10 endpoints)
- blood_circulation (4 endpoints)
- metabolism_hormones (4 endpoints)
- immune_inflammation (3 endpoints)
- cellular_genetic (3 endpoints)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


ENDPOINT_DEFINITIONS = {

    # ══════════════════════════════════════════════════════════════════════════
    # ORGAN SYSTEMS (10 Endpoints)
    # ══════════════════════════════════════════════════════════════════════════

    "cardiac": {
        "category": "organ_system",
        "description": "Heart function and cardiac markers",
        "biomarkers": {
            "heart_rate": {"unit": "bpm", "normal_range": [60, 100], "critical_low": 40, "critical_high": 150},
            "troponin": {"unit": "ng/mL", "normal_range": [0, 0.04], "critical_high": 0.1},
            "bnp": {"unit": "pg/mL", "normal_range": [0, 100], "critical_high": 400},
            "ck_mb": {"unit": "ng/mL", "normal_range": [0, 5], "critical_high": 10},
            "pro_bnp": {"unit": "pg/mL", "normal_range": [0, 300], "critical_high": 900},
        },
        "diseases_detected": ["heart_failure", "myocardial_infarction", "arrhythmia", "cardiomyopathy"],
        "weight": 1.2,
    },

    "renal": {
        "category": "organ_system",
        "description": "Kidney function and filtration",
        "biomarkers": {
            "creatinine": {"unit": "mg/dL", "normal_range": [0.6, 1.2], "critical_high": 2.0},
            "bun": {"unit": "mg/dL", "normal_range": [7, 20], "critical_high": 40},
            "egfr": {"unit": "mL/min", "normal_range": [90, 120], "critical_low": 30},
            "urine_albumin": {"unit": "mg/L", "normal_range": [0, 30], "critical_high": 300},
            "cystatin_c": {"unit": "mg/L", "normal_range": [0.5, 1.0], "critical_high": 1.5},
            "urine_output": {"unit": "mL/hr", "normal_range": [30, 100], "critical_low": 20},
            "potassium": {"unit": "mEq/L", "normal_range": [3.5, 5.0], "critical_low": 2.5, "critical_high": 6.5},
        },
        "diseases_detected": ["chronic_kidney_disease", "acute_kidney_injury", "glomerulonephritis", "nephropathy"],
        "weight": 1.2,
    },

    "respiratory": {
        "category": "organ_system",
        "description": "Lung function and oxygenation",
        "biomarkers": {
            "spo2": {"unit": "%", "normal_range": [95, 100], "critical_low": 90},
            "respiratory_rate": {"unit": "breaths/min", "normal_range": [12, 20], "critical_high": 30},
            "fio2": {"unit": "%", "normal_range": [21, 21], "critical_high": 50},
            "pao2": {"unit": "mmHg", "normal_range": [80, 100], "critical_low": 60},
            "paco2": {"unit": "mmHg", "normal_range": [35, 45], "critical_high": 50, "critical_low": 30},
            "ph_blood": {"unit": "", "normal_range": [7.35, 7.45], "critical_low": 7.2, "critical_high": 7.5},
        },
        "diseases_detected": ["copd", "pneumonia", "ards", "pulmonary_fibrosis", "asthma", "respiratory_failure"],
        "weight": 1.2,
    },

    "hepatic": {
        "category": "organ_system",
        "description": "Liver function and hepatobiliary markers",
        "biomarkers": {
            "ast": {"unit": "IU/L", "normal_range": [10, 40], "critical_high": 200},
            "alt": {"unit": "IU/L", "normal_range": [7, 56], "critical_high": 200},
            "alp": {"unit": "IU/L", "normal_range": [44, 147], "critical_high": 300},
            "ggt": {"unit": "IU/L", "normal_range": [9, 48], "critical_high": 100},
            "bilirubin": {"unit": "mg/dL", "normal_range": [0.1, 1.2], "critical_high": 3.0},
            "albumin": {"unit": "g/dL", "normal_range": [3.5, 5.0], "critical_low": 2.5},
            "inr": {"unit": "", "normal_range": [0.8, 1.2], "critical_high": 2.0},
        },
        "diseases_detected": ["nafld", "masld", "cirrhosis", "hepatitis", "cholestasis", "liver_failure"],
        "weight": 1.1,
    },

    "neurological": {
        "category": "organ_system",
        "description": "Brain and nervous system markers",
        "biomarkers": {
            "neurofilament_light": {"unit": "pg/mL", "normal_range": [0, 10], "critical_high": 50},
            "s100b": {"unit": "ug/L", "normal_range": [0, 0.1], "critical_high": 0.5},
            "nse": {"unit": "ug/L", "normal_range": [0, 12.5], "critical_high": 25},
            "tau_protein": {"unit": "pg/mL", "normal_range": [0, 300], "critical_high": 500},
            "gcs_score": {"unit": "", "normal_range": [15, 15], "critical_low": 8},
        },
        "diseases_detected": ["alzheimers", "parkinsons", "multiple_sclerosis", "stroke", "tbi", "encephalopathy"],
        "weight": 1.3,
    },

    "gastrointestinal": {
        "category": "organ_system",
        "description": "GI tract function and integrity",
        "biomarkers": {
            "calprotectin": {"unit": "ug/g", "normal_range": [0, 50], "critical_high": 200},
            "lactoferrin": {"unit": "ug/g", "normal_range": [0, 7.25], "critical_high": 20},
            "zonulin": {"unit": "ng/mL", "normal_range": [0, 30], "critical_high": 60},
            "lipase": {"unit": "U/L", "normal_range": [0, 160], "critical_high": 500},
            "amylase": {"unit": "U/L", "normal_range": [28, 100], "critical_high": 300},
        },
        "diseases_detected": ["ibd", "crohns", "celiac", "pancreatitis", "leaky_gut", "gi_bleeding"],
        "weight": 1.0,
    },

    "musculoskeletal": {
        "category": "organ_system",
        "description": "Bone, muscle, and joint markers",
        "biomarkers": {
            "ck": {"unit": "U/L", "normal_range": [22, 198], "critical_high": 1000},
            "myoglobin": {"unit": "ng/mL", "normal_range": [0, 85], "critical_high": 500},
            "bone_alp": {"unit": "ug/L", "normal_range": [6, 20], "critical_high": 40},
            "calcium": {"unit": "mg/dL", "normal_range": [8.5, 10.5], "critical_low": 7.0, "critical_high": 12.0},
            "vitamin_d": {"unit": "ng/mL", "normal_range": [30, 100], "critical_low": 10},
            "uric_acid": {"unit": "mg/dL", "normal_range": [2.5, 7.0], "critical_high": 10.0},
            "phosphorus": {"unit": "mg/dL", "normal_range": [2.5, 4.5], "critical_low": 1.0, "critical_high": 6.0},
        },
        "diseases_detected": ["osteoporosis", "rheumatoid_arthritis", "gout", "rhabdomyolysis", "myopathy"],
        "weight": 0.9,
    },

    "dermatological": {
        "category": "organ_system",
        "description": "Skin and wound healing markers",
        "biomarkers": {
            "collagen_markers": {"unit": "score", "normal_range": [0, 1], "critical_high": 3},
            "wound_healing_score": {"unit": "score", "normal_range": [0, 1], "critical_high": 3},
            "skin_inflammation": {"unit": "score", "normal_range": [0, 1], "critical_high": 3},
        },
        "diseases_detected": ["psoriasis", "eczema", "wound_healing_impairment", "skin_cancer_risk"],
        "weight": 0.7,
    },

    "ophthalmologic": {
        "category": "organ_system",
        "description": "Eye and vision markers",
        "biomarkers": {
            "iop": {"unit": "mmHg", "normal_range": [10, 21], "critical_high": 30},
            "retinal_thickness": {"unit": "um", "normal_range": [200, 300], "critical_high": 400},
            "macular_markers": {"unit": "score", "normal_range": [0, 1], "critical_high": 3},
        },
        "diseases_detected": ["glaucoma", "macular_degeneration", "diabetic_retinopathy", "cataracts"],
        "weight": 0.7,
    },

    "reproductive": {
        "category": "organ_system",
        "description": "Reproductive and hormonal markers",
        "biomarkers": {
            "fsh": {"unit": "mIU/mL", "normal_range": [1.5, 12.4], "varies_by": "sex_age"},
            "lh": {"unit": "mIU/mL", "normal_range": [1.7, 8.6], "varies_by": "sex_age"},
            "estrogen": {"unit": "pg/mL", "normal_range": [15, 350], "varies_by": "sex_age"},
            "testosterone": {"unit": "ng/dL", "normal_range": [300, 1000], "varies_by": "sex"},
            "amh": {"unit": "ng/mL", "normal_range": [1.0, 3.5], "varies_by": "age"},
            "psa": {"unit": "ng/mL", "normal_range": [0, 4.0], "critical_high": 10.0},
        },
        "diseases_detected": ["pcos", "menopause", "infertility", "prostate_cancer", "hypogonadism"],
        "weight": 0.8,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # BLOOD & CIRCULATION (4 Endpoints)
    # ══════════════════════════════════════════════════════════════════════════

    "hematologic": {
        "category": "blood_circulation",
        "description": "Blood cell counts and morphology",
        "biomarkers": {
            "wbc": {"unit": "K/uL", "normal_range": [4.5, 11.0], "critical_low": 2.0, "critical_high": 20.0},
            "rbc": {"unit": "M/uL", "normal_range": [4.5, 5.5], "critical_low": 3.0, "critical_high": 6.5},
            "hemoglobin": {"unit": "g/dL", "normal_range": [12, 17], "critical_low": 7.0, "critical_high": 20.0},
            "hematocrit": {"unit": "%", "normal_range": [36, 50], "critical_low": 25, "critical_high": 60},
            "platelets": {"unit": "K/uL", "normal_range": [150, 400], "critical_low": 50, "critical_high": 1000},
            "mcv": {"unit": "fL", "normal_range": [80, 100], "critical_low": 60, "critical_high": 120},
            "mch": {"unit": "pg", "normal_range": [27, 33], "critical_low": 20, "critical_high": 40},
            "rdw": {"unit": "%", "normal_range": [11.5, 14.5], "critical_high": 20},
        },
        "diseases_detected": ["anemia", "leukemia", "thrombocytopenia", "polycythemia", "myelodysplastic_syndrome"],
        "weight": 1.1,
    },

    "hemodynamic": {
        "category": "blood_circulation",
        "description": "Blood pressure and circulation",
        "biomarkers": {
            "sbp": {"unit": "mmHg", "normal_range": [90, 120], "critical_low": 80, "critical_high": 180},
            "dbp": {"unit": "mmHg", "normal_range": [60, 80], "critical_low": 50, "critical_high": 120},
            "map": {"unit": "mmHg", "normal_range": [70, 100], "critical_low": 60, "critical_high": 130},
            "shock_index": {"unit": "", "normal_range": [0.5, 0.7], "critical_high": 1.0},
            "pulse_pressure": {"unit": "mmHg", "normal_range": [30, 50], "critical_low": 20, "critical_high": 80},
        },
        "diseases_detected": ["hypertension", "hypotension", "shock", "heart_failure", "vascular_disease"],
        "weight": 1.2,
    },

    "coagulation": {
        "category": "blood_circulation",
        "description": "Blood clotting and fibrinolysis",
        "biomarkers": {
            "pt": {"unit": "seconds", "normal_range": [11, 13.5], "critical_high": 20},
            "ptt": {"unit": "seconds", "normal_range": [25, 35], "critical_high": 50},
            "inr": {"unit": "", "normal_range": [0.8, 1.2], "critical_high": 3.0},
            "d_dimer": {"unit": "ng/mL", "normal_range": [0, 500], "critical_high": 2000},
            "fibrinogen": {"unit": "mg/dL", "normal_range": [200, 400], "critical_low": 100, "critical_high": 700},
            "vwf": {"unit": "%", "normal_range": [50, 150], "critical_low": 30, "critical_high": 300},
            "antithrombin": {"unit": "%", "normal_range": [80, 120], "critical_low": 50},
        },
        "diseases_detected": ["dvt", "pulmonary_embolism", "dic", "hemophilia", "hypercoagulable_state"],
        "weight": 1.1,
    },

    "vascular_endothelial": {
        "category": "blood_circulation",
        "description": "Blood vessel and endothelial function",
        "biomarkers": {
            "endothelin": {"unit": "pg/mL", "normal_range": [0, 2.5], "critical_high": 5.0},
            "vcam": {"unit": "ng/mL", "normal_range": [0, 1000], "critical_high": 2000},
            "icam": {"unit": "ng/mL", "normal_range": [0, 300], "critical_high": 600},
            "nitric_oxide": {"unit": "uM", "normal_range": [20, 40], "critical_low": 10},
            "homocysteine": {"unit": "umol/L", "normal_range": [5, 15], "critical_high": 30},
        },
        "diseases_detected": ["atherosclerosis", "vasculitis", "endothelial_dysfunction", "raynauds"],
        "weight": 1.0,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # METABOLISM & HORMONES (4 Endpoints)
    # ══════════════════════════════════════════════════════════════════════════

    "metabolic": {
        "category": "metabolism_hormones",
        "description": "Energy metabolism and glucose regulation",
        "biomarkers": {
            "glucose": {"unit": "mg/dL", "normal_range": [70, 99], "critical_low": 50, "critical_high": 250},
            "lactate": {"unit": "mmol/L", "normal_range": [0.5, 2.0], "critical_high": 4.0},
            "ph": {"unit": "", "normal_range": [7.35, 7.45], "critical_low": 7.2, "critical_high": 7.5},
            "bicarbonate": {"unit": "mEq/L", "normal_range": [22, 28], "critical_low": 15, "critical_high": 35},
            "ketones": {"unit": "mmol/L", "normal_range": [0, 0.5], "critical_high": 3.0},
            "anion_gap": {"unit": "mEq/L", "normal_range": [8, 12], "critical_high": 20},
        },
        "diseases_detected": ["diabetes", "metabolic_acidosis", "dka", "lactic_acidosis", "metabolic_syndrome"],
        "weight": 1.2,
    },

    "endocrine": {
        "category": "metabolism_hormones",
        "description": "Hormone regulation and glandular function",
        "biomarkers": {
            "insulin": {"unit": "uIU/mL", "normal_range": [2.6, 24.9], "critical_high": 50},
            "hba1c": {"unit": "%", "normal_range": [4.0, 5.6], "critical_high": 9.0},
            "homa_ir": {"unit": "", "normal_range": [0, 2.5], "critical_high": 5.0},
            "tsh": {"unit": "mIU/L", "normal_range": [0.4, 4.0], "critical_low": 0.1, "critical_high": 10.0},
            "t3": {"unit": "ng/dL", "normal_range": [80, 200], "critical_low": 40, "critical_high": 300},
            "t4": {"unit": "ug/dL", "normal_range": [4.5, 12], "critical_low": 2.0, "critical_high": 20.0},
            "cortisol": {"unit": "ug/dL", "normal_range": [6, 23], "critical_low": 3, "critical_high": 50},
            "acth": {"unit": "pg/mL", "normal_range": [7, 63], "critical_high": 150},
            "growth_hormone": {"unit": "ng/mL", "normal_range": [0, 5], "critical_high": 20},
        },
        "diseases_detected": ["insulin_resistance", "hypothyroidism", "hyperthyroidism", "addisons", "cushings", "diabetes"],
        "weight": 1.1,
    },

    "lipid_atherogenic": {
        "category": "metabolism_hormones",
        "description": "Lipid metabolism and cardiovascular risk",
        "biomarkers": {
            "total_cholesterol": {"unit": "mg/dL", "normal_range": [0, 200], "critical_high": 300},
            "ldl": {"unit": "mg/dL", "normal_range": [0, 100], "critical_high": 190},
            "hdl": {"unit": "mg/dL", "normal_range": [40, 100], "critical_low": 30},
            "triglycerides": {"unit": "mg/dL", "normal_range": [0, 150], "critical_high": 500},
            "apob": {"unit": "mg/dL", "normal_range": [0, 100], "critical_high": 150},
            "lp_a": {"unit": "nmol/L", "normal_range": [0, 75], "critical_high": 125},
            "oxidized_ldl": {"unit": "U/L", "normal_range": [0, 60], "critical_high": 100},
            "sdldl": {"unit": "mg/dL", "normal_range": [0, 30], "critical_high": 50},
        },
        "diseases_detected": ["dyslipidemia", "atherosclerosis", "familial_hyperlipidemia", "metabolic_syndrome"],
        "weight": 1.0,
    },

    "nutritional": {
        "category": "metabolism_hormones",
        "description": "Vitamins, minerals, and nutritional status",
        "biomarkers": {
            "vitamin_d": {"unit": "ng/mL", "normal_range": [30, 100], "critical_low": 10, "critical_high": 150},
            "vitamin_b12": {"unit": "pg/mL", "normal_range": [200, 900], "critical_low": 150},
            "folate": {"unit": "ng/mL", "normal_range": [3, 17], "critical_low": 2},
            "ferritin": {"unit": "ng/mL", "normal_range": [20, 250], "critical_low": 10, "critical_high": 500},
            "iron": {"unit": "ug/dL", "normal_range": [60, 170], "critical_low": 30},
            "tibc": {"unit": "ug/dL", "normal_range": [250, 400], "critical_high": 500},
            "zinc": {"unit": "ug/dL", "normal_range": [60, 120], "critical_low": 40},
            "magnesium": {"unit": "mg/dL", "normal_range": [1.7, 2.2], "critical_low": 1.0, "critical_high": 3.0},
            "selenium": {"unit": "ug/L", "normal_range": [70, 150], "critical_low": 40},
        },
        "diseases_detected": ["vitamin_deficiency", "malnutrition", "iron_deficiency", "anemia", "osteomalacia"],
        "weight": 0.9,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # IMMUNE & INFLAMMATION (3 Endpoints)
    # ══════════════════════════════════════════════════════════════════════════

    "inflammatory": {
        "category": "immune_inflammation",
        "description": "Inflammation and acute phase response",
        "biomarkers": {
            "crp": {"unit": "mg/L", "normal_range": [0, 3], "critical_high": 10},
            "esr": {"unit": "mm/hr", "normal_range": [0, 20], "critical_high": 50},
            "il_6": {"unit": "pg/mL", "normal_range": [0, 7], "critical_high": 20},
            "tnf_alpha": {"unit": "pg/mL", "normal_range": [0, 8.1], "critical_high": 20},
            "procalcitonin": {"unit": "ng/mL", "normal_range": [0, 0.1], "critical_high": 2.0},
            "ferritin": {"unit": "ng/mL", "normal_range": [20, 250], "critical_high": 1000},
            "il_1b": {"unit": "pg/mL", "normal_range": [0, 5], "critical_high": 15},
            "temperature": {"unit": "C", "normal_range": [36.1, 37.2], "critical_low": 35.0, "critical_high": 39.0},
        },
        "diseases_detected": ["sepsis", "chronic_inflammation", "cytokine_storm", "autoimmune_flare", "infection"],
        "weight": 1.3,
    },

    "immune_autoimmune": {
        "category": "immune_inflammation",
        "description": "Immune function and autoimmunity markers",
        "biomarkers": {
            "ana": {"unit": "titer", "normal_range": [0, 1], "critical_high": 3},
            "rf": {"unit": "IU/mL", "normal_range": [0, 14], "critical_high": 50},
            "anti_ccp": {"unit": "U/mL", "normal_range": [0, 20], "critical_high": 60},
            "c3": {"unit": "mg/dL", "normal_range": [90, 180], "critical_low": 60},
            "c4": {"unit": "mg/dL", "normal_range": [10, 40], "critical_low": 8},
            "igg": {"unit": "mg/dL", "normal_range": [700, 1600], "critical_low": 400},
            "igm": {"unit": "mg/dL", "normal_range": [40, 230], "critical_low": 25},
            "iga": {"unit": "mg/dL", "normal_range": [70, 400], "critical_low": 40},
            "cd4_count": {"unit": "cells/uL", "normal_range": [500, 1500], "critical_low": 200},
        },
        "diseases_detected": ["lupus", "rheumatoid_arthritis", "sjogrens", "ms", "hiv", "immunodeficiency"],
        "weight": 1.1,
    },

    "infectious_pathogenic": {
        "category": "immune_inflammation",
        "description": "Infection and pathogen markers",
        "biomarkers": {
            "wbc_differential": {"unit": "score", "normal_range": [0, 1], "critical_high": 3},
            "neutrophils": {"unit": "%", "normal_range": [40, 70], "critical_low": 20, "critical_high": 85},
            "lymphocytes": {"unit": "%", "normal_range": [20, 40], "critical_low": 10, "critical_high": 60},
            "bands": {"unit": "%", "normal_range": [0, 5], "critical_high": 15},
            "procalcitonin": {"unit": "ng/mL", "normal_range": [0, 0.1], "critical_high": 2.0},
            "lactate": {"unit": "mmol/L", "normal_range": [0.5, 2.0], "critical_high": 4.0},
        },
        "diseases_detected": ["bacterial_infection", "viral_infection", "fungal_infection", "sepsis", "pneumonia"],
        "weight": 1.2,
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CELLULAR & GENETIC (3 Endpoints)
    # ══════════════════════════════════════════════════════════════════════════

    "oncologic_tumor": {
        "category": "cellular_genetic",
        "description": "Cancer and tumor markers",
        "biomarkers": {
            "cea": {"unit": "ng/mL", "normal_range": [0, 3], "critical_high": 10},
            "ca_125": {"unit": "U/mL", "normal_range": [0, 35], "critical_high": 100},
            "ca_19_9": {"unit": "U/mL", "normal_range": [0, 37], "critical_high": 100},
            "psa": {"unit": "ng/mL", "normal_range": [0, 4], "critical_high": 10},
            "afp": {"unit": "ng/mL", "normal_range": [0, 10], "critical_high": 50},
            "ldh": {"unit": "U/L", "normal_range": [140, 280], "critical_high": 500},
            "ctdna": {"unit": "copies/mL", "normal_range": [0, 0], "critical_high": 1},
        },
        "diseases_detected": ["cancer_early_detection", "tumor_progression", "metastasis", "treatment_response"],
        "weight": 1.2,
    },

    "genetic_genomic": {
        "category": "cellular_genetic",
        "description": "Genetic variants and pharmacogenomics",
        "biomarkers": {
            "mthfr_status": {"unit": "variant", "normal_range": ["normal"], "variants": ["C677T", "A1298C"]},
            "apoe_status": {"unit": "variant", "normal_range": ["e3/e3"], "variants": ["e4/e4", "e3/e4"]},
            "brca_status": {"unit": "variant", "normal_range": ["negative"], "variants": ["BRCA1+", "BRCA2+"]},
            "factor_v_leiden": {"unit": "variant", "normal_range": ["negative"], "variants": ["heterozygous", "homozygous"]},
            "cyp2c19_status": {"unit": "variant", "normal_range": ["normal_metabolizer"], "variants": ["poor", "rapid"]},
        },
        "diseases_detected": ["genetic_disease_risk", "drug_metabolism", "hereditary_conditions", "cancer_risk"],
        "weight": 0.8,
    },

    "microbiome_gut_axis": {
        "category": "cellular_genetic",
        "description": "Gut microbiome and gut-organ axes",
        "biomarkers": {
            "microbiome_diversity": {"unit": "score", "normal_range": [70, 100], "critical_low": 40},
            "lps": {"unit": "EU/mL", "normal_range": [0, 0.1], "critical_high": 0.5},
            "scfa": {"unit": "mmol/L", "normal_range": [50, 100], "critical_low": 30},
            "zonulin": {"unit": "ng/mL", "normal_range": [0, 30], "critical_high": 60},
            "firmicutes_bacteroidetes": {"unit": "ratio", "normal_range": [0.5, 2.0], "critical_high": 4.0},
            "akkermansia": {"unit": "%", "normal_range": [1, 5], "critical_low": 0.5},
        },
        "diseases_detected": ["dysbiosis", "leaky_gut", "gut_brain_axis", "gut_liver_axis", "sibo", "ibs"],
        "weight": 1.0,
    },
}


# Endpoint categories for grouping
ENDPOINT_CATEGORIES = {
    "organ_system": [
        "cardiac", "renal", "respiratory", "hepatic", "neurological",
        "gastrointestinal", "musculoskeletal", "dermatological", "ophthalmologic", "reproductive"
    ],
    "blood_circulation": [
        "hematologic", "hemodynamic", "coagulation", "vascular_endothelial"
    ],
    "metabolism_hormones": [
        "metabolic", "endocrine", "lipid_atherogenic", "nutritional"
    ],
    "immune_inflammation": [
        "inflammatory", "immune_autoimmune", "infectious_pathogenic"
    ],
    "cellular_genetic": [
        "oncologic_tumor", "genetic_genomic", "microbiome_gut_axis"
    ],
}


# All endpoint names
ALL_ENDPOINTS = list(ENDPOINT_DEFINITIONS.keys())


class EndpointScorer:
    """Score individual endpoints based on biomarker values."""

    def __init__(self):
        self.endpoints = ENDPOINT_DEFINITIONS

    def score_endpoint(
        self,
        endpoint_name: str,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score a single endpoint with whatever biomarkers are available.

        Returns:
            dict with score, status, flagged biomarkers, and trajectory
        """
        if endpoint_name not in self.endpoints:
            return {"error": f"Unknown endpoint: {endpoint_name}"}

        endpoint_def = self.endpoints[endpoint_name]
        biomarker_defs = endpoint_def["biomarkers"]

        scores = []
        flagged = []
        has_data = False

        for biomarker_name, biomarker_def in biomarker_defs.items():
            # Check for biomarker in patient data (case-insensitive)
            value = self._get_biomarker_value(patient_data, biomarker_name)

            if value is not None:
                has_data = True
                score, is_flagged = self._score_biomarker(value, biomarker_def)
                scores.append(score)
                if is_flagged:
                    flagged.append(biomarker_name)

        # Calculate endpoint score
        if scores:
            endpoint_score = np.mean(scores)
            max_score = max(scores)
        else:
            endpoint_score = 0.0
            max_score = 0.0

        # Determine status
        status = self._determine_status(endpoint_score, max_score)

        # Determine trajectory (requires historical data)
        trajectory = self._determine_trajectory(patient_data, endpoint_name)

        return {
            "endpoint": endpoint_name,
            "category": endpoint_def["category"],
            "score": round(endpoint_score, 3),
            "max_score": round(max_score, 3),
            "status": status,
            "biomarkers_flagged": flagged,
            "biomarkers_available": len(scores),
            "biomarkers_total": len(biomarker_defs),
            "has_data": has_data,
            "trajectory": trajectory,
            "diseases_possible": endpoint_def["diseases_detected"] if status in ["elevated", "critical"] else [],
            "weight": endpoint_def.get("weight", 1.0),
        }

    def score_all_endpoints(
        self,
        patient_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Score all 24 endpoints."""
        results = {}
        for endpoint_name in self.endpoints:
            results[endpoint_name] = self.score_endpoint(endpoint_name, patient_data)
        return results

    def _get_biomarker_value(
        self,
        patient_data: Dict[str, Any],
        biomarker_name: str
    ) -> Optional[float]:
        """Get biomarker value from patient data (case-insensitive)."""
        # Direct match
        if biomarker_name in patient_data:
            val = patient_data[biomarker_name]
            if val is not None and val != "" and not (isinstance(val, float) and np.isnan(val)):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return None

        # Case-insensitive match
        for key, val in patient_data.items():
            if key.lower() == biomarker_name.lower():
                if val is not None and val != "" and not (isinstance(val, float) and np.isnan(val)):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        return None

        return None

    def _score_biomarker(
        self,
        value: float,
        biomarker_def: Dict[str, Any]
    ) -> Tuple[float, bool]:
        """
        Score a single biomarker value.
        Returns (score 0-1, is_flagged boolean)
        """
        normal_range = biomarker_def.get("normal_range", [0, 100])
        critical_low = biomarker_def.get("critical_low")
        critical_high = biomarker_def.get("critical_high")

        # Handle non-numeric normal ranges (e.g., genetic variants)
        if not isinstance(normal_range[0], (int, float)):
            return (0.0, False)

        low, high = normal_range

        # Check critical values first
        if critical_high is not None and value >= critical_high:
            return (1.0, True)
        if critical_low is not None and value <= critical_low:
            return (1.0, True)

        # Check if in normal range
        if low <= value <= high:
            return (0.0, False)

        # Calculate distance from normal
        if value < low:
            # Below normal
            if critical_low is not None:
                distance = (low - value) / (low - critical_low)
            else:
                distance = (low - value) / low if low > 0 else 0.5
            score = min(distance, 1.0)
        else:
            # Above normal
            if critical_high is not None:
                distance = (value - high) / (critical_high - high)
            else:
                distance = (value - high) / high if high > 0 else 0.5
            score = min(distance, 1.0)

        is_flagged = score > 0.3
        return (score, is_flagged)

    def _determine_status(self, avg_score: float, max_score: float) -> str:
        """Determine endpoint status based on scores."""
        if max_score >= 0.8:
            return "critical"
        elif max_score >= 0.5 or avg_score >= 0.4:
            return "elevated"
        elif avg_score >= 0.2:
            return "borderline"
        else:
            return "normal"

    def _determine_trajectory(
        self,
        patient_data: Dict[str, Any],
        endpoint_name: str
    ) -> str:
        """Determine trajectory (requires historical data)."""
        # This would use trajectory analysis if historical data is available
        # For now, return based on current severity
        score = patient_data.get(f"{endpoint_name}_trajectory_score")
        if score is not None:
            if score > 0.5:
                return "worsening"
            elif score > 0.2:
                return "concerning"
            elif score < -0.2:
                return "improving"
        return "stable"
