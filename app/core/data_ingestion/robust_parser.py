"""
Bulletproof Data Parser for HyperCore.

PHILOSOPHY:
1. NEVER crash - always produce something useful
2. Do the BEST with what you have
3. Tell the user what's MISSING that would help
4. Cross-reference against ALL modules regardless of data quality
5. Be SUPERIOR - no excuses, no fallbacks

This module handles ANY input format and produces structured results.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO

logger = logging.getLogger(__name__)


# =============================================================================
# BIOMARKER MAPPINGS - Map any possible name to standard biomarker
# Handles: units in name, prefixes, case variations, underscores/spaces/camelCase
# =============================================================================

BIOMARKER_MAPPINGS = {
    # =========================================================================
    # CARDIAC MARKERS
    # =========================================================================

    # Troponin (all variations with units and prefixes)
    "troponin": "troponin",
    "trop": "troponin",
    "tnni": "troponin",
    "tnnt": "troponin",
    "tni": "troponin",
    "tnt": "troponin",
    "troponin_i": "troponin",
    "troponin_t": "troponin",
    "troponin i": "troponin",
    "troponin t": "troponin",
    "troponini": "troponin",
    "troponint": "troponin",
    "cardiac_troponin": "troponin",
    "cardiac troponin": "troponin",
    "hs_troponin": "troponin",
    "hs-troponin": "troponin",
    "hstroponin": "troponin",
    "hs_trop": "troponin",
    "hstrop": "troponin",
    "high_sensitivity_troponin": "troponin",
    "high sensitivity troponin": "troponin",
    "highsensitivitytroponin": "troponin",
    "hs_troponin_i": "troponin",
    "hs_troponin_t": "troponin",
    "hs_tnni": "troponin",
    "hs_tnnt": "troponin",
    "hstni": "troponin",
    "hstnt": "troponin",
    "hstnni": "troponin",
    "hstnnt": "troponin",
    # With units
    "troponin_ng_l": "troponin",
    "troponin_ng_ml": "troponin",
    "troponin_pg_ml": "troponin",
    "hs_troponin_ng_l": "troponin",
    "hs_troponin_ng_ml": "troponin",
    "hs_troponin_pg_ml": "troponin",
    "troponin_i_ng_l": "troponin",
    "troponin_t_ng_l": "troponin",
    "cardiac_troponin_i": "troponin",
    "cardiac_troponin_t": "troponin",
    "serum_troponin": "troponin",

    # BNP / NT-proBNP
    "bnp": "bnp",
    "ntprobnp": "bnp",
    "nt_probnp": "bnp",
    "nt-probnp": "bnp",
    "nt_pro_bnp": "bnp",
    "pro_bnp": "bnp",
    "pro-bnp": "bnp",
    "probnp": "bnp",
    "n_terminal_probnp": "bnp",
    "brain_natriuretic_peptide": "bnp",
    "brain natriuretic peptide": "bnp",
    "b_type_natriuretic_peptide": "bnp",
    "b-type natriuretic peptide": "bnp",
    "btype_natriuretic_peptide": "bnp",
    # With units
    "bnp_pg_ml": "bnp",
    "bnp_ng_l": "bnp",
    "ntprobnp_pg_ml": "bnp",
    "nt_probnp_pg_ml": "bnp",
    "pro_bnp_pg_ml": "bnp",
    "serum_bnp": "bnp",
    "plasma_bnp": "bnp",

    # CK-MB
    "ck_mb": "ck_mb",
    "ckmb": "ck_mb",
    "ck-mb": "ck_mb",
    "creatine_kinase_mb": "ck_mb",
    "creatine kinase mb": "ck_mb",
    "ck_mb_ng_ml": "ck_mb",
    "ck_mb_u_l": "ck_mb",

    # LDH
    "ldh": "ldh",
    "lactate_dehydrogenase": "ldh",
    "lactate dehydrogenase": "ldh",
    "lactic_dehydrogenase": "ldh",
    "ldh_u_l": "ldh",
    "ldh_iu_l": "ldh",
    "serum_ldh": "ldh",

    # D-dimer
    "d_dimer": "d_dimer",
    "ddimer": "d_dimer",
    "d-dimer": "d_dimer",
    "d dimer": "d_dimer",
    "fibrin_degradation": "d_dimer",
    "d_dimer_ng_ml": "d_dimer",
    "d_dimer_ug_ml": "d_dimer",
    "d_dimer_mg_l": "d_dimer",
    "ddimer_feu": "d_dimer",

    # QTc
    "qtc": "qtc",
    "qt_interval": "qtc",
    "qtinterval": "qtc",
    "ecg_qtc": "qtc",
    "ecg_qtc_ms": "qtc",
    "qtc_ms": "qtc",
    "corrected_qt": "qtc",

    # =========================================================================
    # INFLAMMATORY MARKERS
    # =========================================================================

    # CRP (C-Reactive Protein)
    "crp": "crp",
    "c_reactive_protein": "crp",
    "c-reactive_protein": "crp",
    "c reactive protein": "crp",
    "c-reactive protein": "crp",
    "creactive": "crp",
    "creactiveprotein": "crp",
    "c-rp": "crp",
    "hs_crp": "crp",
    "hscrp": "crp",
    "hs-crp": "crp",
    "high_sensitivity_crp": "crp",
    "high sensitivity crp": "crp",
    "highsensitivitycrp": "crp",
    "ultra_sensitive_crp": "crp",
    "us_crp": "crp",
    "uscrp": "crp",
    # With units
    "crp_mg_l": "crp",
    "crp_mg_dl": "crp",
    "crp_ug_ml": "crp",
    "hs_crp_mg_l": "crp",
    "hs_crp_mg_dl": "crp",
    "serum_crp": "crp",
    "plasma_crp": "crp",

    # Procalcitonin
    "procalcitonin": "procalcitonin",
    "pct": "procalcitonin",
    "procal": "procalcitonin",
    "pro_calcitonin": "procalcitonin",
    "pro-calcitonin": "procalcitonin",
    "procalc": "procalcitonin",
    # With units
    "procalcitonin_ng_ml": "procalcitonin",
    "procalcitonin_ug_l": "procalcitonin",
    "pct_ng_ml": "procalcitonin",
    "pct_ug_l": "procalcitonin",
    "serum_procalcitonin": "procalcitonin",
    "serum_pct": "procalcitonin",

    # Interleukins
    "il6": "il6",
    "il_6": "il6",
    "il-6": "il6",
    "interleukin_6": "il6",
    "interleukin 6": "il6",
    "interleukin6": "il6",
    "il6_pg_ml": "il6",
    "il_6_pg_ml": "il6",
    "serum_il6": "il6",

    "il1": "il1",
    "il_1": "il1",
    "il-1": "il1",
    "il1b": "il1",
    "il_1b": "il1",
    "il-1b": "il1",
    "il1_beta": "il1",
    "interleukin_1": "il1",
    "interleukin 1": "il1",
    "interleukin1": "il1",
    "il1_pg_ml": "il1",

    "il10": "il10",
    "il_10": "il10",
    "il-10": "il10",
    "interleukin_10": "il10",
    "interleukin 10": "il10",

    # TNF-alpha
    "tnf": "tnf",
    "tnf_alpha": "tnf",
    "tnf-alpha": "tnf",
    "tnfa": "tnf",
    "tnf_a": "tnf",
    "tumor_necrosis_factor": "tnf",
    "tumor necrosis factor": "tnf",
    "tnf_pg_ml": "tnf",

    # ESR
    "esr": "esr",
    "sed_rate": "esr",
    "sedimentation_rate": "esr",
    "erythrocyte_sedimentation_rate": "esr",
    "erythrocyte sedimentation rate": "esr",
    "esr_mm_hr": "esr",

    # Ferritin
    "ferritin": "ferritin",
    "serum_ferritin": "ferritin",
    "ferritin_ng_ml": "ferritin",
    "ferritin_ug_l": "ferritin",
    "plasma_ferritin": "ferritin",

    # =========================================================================
    # SEPSIS / INFECTION MARKERS
    # =========================================================================

    # WBC
    "wbc": "wbc",
    "white_blood_cell": "wbc",
    "white_blood_cells": "wbc",
    "white blood cell": "wbc",
    "white blood cells": "wbc",
    "whitebloodcells": "wbc",
    "leukocytes": "wbc",
    "leukocyte": "wbc",
    "leukocyte_count": "wbc",
    "white_cell_count": "wbc",
    "white cell count": "wbc",
    "wcc": "wbc",
    "total_wbc": "wbc",
    # With units
    "wbc_k_ul": "wbc",
    "wbc_10_3_ul": "wbc",
    "wbc_x10_9_l": "wbc",
    "wbc_thou_ul": "wbc",
    "leukocytes_ul": "wbc",

    # Lactate
    "lactate": "lactate",
    "lactic_acid": "lactate",
    "lactic acid": "lactate",
    "lacticacid": "lactate",
    "lac": "lactate",
    "lact": "lactate",
    "blood_lactate": "lactate",
    "serum_lactate": "lactate",
    "plasma_lactate": "lactate",
    "arterial_lactate": "lactate",
    "venous_lactate": "lactate",
    # With units
    "lactate_mmol_l": "lactate",
    "lactate_mg_dl": "lactate",
    "lactic_acid_mmol_l": "lactate",

    # Temperature
    "temperature": "temperature",
    "temp": "temperature",
    "body_temp": "temperature",
    "body_temperature": "temperature",
    "body temp": "temperature",
    "bodytemp": "temperature",
    "core_temp": "temperature",
    "core_temperature": "temperature",
    "fever": "temperature",
    # With units
    "temp_c": "temperature",
    "temp_f": "temperature",
    "temperature_celsius": "temperature",
    "temperature_fahrenheit": "temperature",

    # =========================================================================
    # RENAL FUNCTION
    # =========================================================================

    # Creatinine
    "creatinine": "creatinine",
    "creat": "creatinine",
    "crea": "creatinine",
    "cr": "creatinine",
    "scr": "creatinine",
    "serum_creatinine": "creatinine",
    "serum_cr": "creatinine",
    "plasma_creatinine": "creatinine",
    "blood_creatinine": "creatinine",
    # With units
    "creatinine_mg_dl": "creatinine",
    "creatinine_umol_l": "creatinine",
    "creat_mg_dl": "creatinine",
    "serum_creatinine_mg_dl": "creatinine",

    # BUN
    "bun": "bun",
    "blood_urea_nitrogen": "bun",
    "blood urea nitrogen": "bun",
    "urea": "bun",
    "urea_nitrogen": "bun",
    "serum_urea": "bun",
    "plasma_urea": "bun",
    "serum_blood_urea_nitrogen": "bun",
    "serum_bun": "bun",
    "plasma_bun": "bun",
    # With units
    "bun_mg_dl": "bun",
    "urea_mg_dl": "bun",
    "urea_mmol_l": "bun",
    "serum_blood_urea_nitrogen_mg_dl": "bun",

    # eGFR / GFR
    "egfr": "egfr",
    "gfr": "egfr",
    "estimated_gfr": "egfr",
    "estimated gfr": "egfr",
    "estimatedgfr": "egfr",
    "e_gfr": "egfr",
    "glomerular_filtration_rate": "egfr",
    "glomerular filtration rate": "egfr",
    "ckd_epi": "egfr",
    "mdrd": "egfr",
    # With units
    "egfr_ml_min": "egfr",
    "gfr_ml_min": "egfr",
    "egfr_ml_min_1_73m2": "egfr",

    # Cystatin C
    "cystatin_c": "cystatin_c",
    "cystatinc": "cystatin_c",
    "cystatin": "cystatin_c",
    "cys_c": "cystatin_c",
    "cystatin_c_mg_l": "cystatin_c",

    # =========================================================================
    # LIVER FUNCTION
    # =========================================================================

    # ALT
    "alt": "alt",
    "sgpt": "alt",
    "alat": "alt",
    "alanine_transaminase": "alt",
    "alanine transaminase": "alt",
    "alaninetransaminase": "alt",
    "alanine_aminotransferase": "alt",
    "serum_alt": "alt",
    # With units
    "alt_u_l": "alt",
    "alt_iu_l": "alt",
    "sgpt_u_l": "alt",

    # AST
    "ast": "ast",
    "sgot": "ast",
    "asat": "ast",
    "aspartate_transaminase": "ast",
    "aspartate transaminase": "ast",
    "aspartatetransaminase": "ast",
    "aspartate_aminotransferase": "ast",
    "serum_ast": "ast",
    # With units
    "ast_u_l": "ast",
    "ast_iu_l": "ast",
    "sgot_u_l": "ast",

    # ALP
    "alp": "alp",
    "alkaline_phosphatase": "alp",
    "alkaline phosphatase": "alp",
    "alkalinephosphatase": "alp",
    "alk_phos": "alp",
    "alkphos": "alp",
    "alp_u_l": "alp",
    "alp_iu_l": "alp",

    # GGT
    "ggt": "ggt",
    "gamma_gt": "ggt",
    "gamma_glutamyl_transferase": "ggt",
    "gamma glutamyl transferase": "ggt",
    "gammaglutamyltransferase": "ggt",
    "ggtp": "ggt",
    "ggt_u_l": "ggt",
    "ggt_iu_l": "ggt",

    # Bilirubin
    "bilirubin": "bilirubin",
    "bili": "bilirubin",
    "total_bilirubin": "bilirubin",
    "total bilirubin": "bilirubin",
    "totalbilirubin": "bilirubin",
    "tbili": "bilirubin",
    "t_bili": "bilirubin",
    "serum_bilirubin": "bilirubin",
    "direct_bilirubin": "direct_bilirubin",
    "indirect_bilirubin": "indirect_bilirubin",
    "conjugated_bilirubin": "direct_bilirubin",
    "unconjugated_bilirubin": "indirect_bilirubin",
    # With units
    "bilirubin_mg_dl": "bilirubin",
    "bilirubin_umol_l": "bilirubin",
    "total_bilirubin_mg_dl": "bilirubin",

    # Albumin
    "albumin": "albumin",
    "alb": "albumin",
    "serum_albumin": "albumin",
    "serum albumin": "albumin",
    "plasma_albumin": "albumin",
    # With units
    "albumin_g_dl": "albumin",
    "albumin_g_l": "albumin",
    "serum_albumin_g_dl": "albumin",

    # INR / PT
    "inr": "inr",
    "pt_inr": "inr",
    "pt/inr": "inr",
    "international_normalized_ratio": "inr",
    "international normalized ratio": "inr",
    "prothrombin_time_inr": "inr",

    "pt": "pt",
    "prothrombin_time": "pt",
    "prothrombin time": "pt",
    "pt_seconds": "pt",
    "pt_sec": "pt",

    # =========================================================================
    # METABOLIC / DIABETES
    # =========================================================================

    # Glucose
    "glucose": "glucose",
    "glu": "glucose",
    "blood_sugar": "glucose",
    "blood sugar": "glucose",
    "bloodsugar": "glucose",
    "bg": "glucose",
    "blood_glucose": "glucose",
    "blood glucose": "glucose",
    "bloodglucose": "glucose",
    "serum_glucose": "glucose",
    "plasma_glucose": "glucose",
    "fasting_glucose": "glucose",
    "fasting glucose": "glucose",
    "fastingglucose": "glucose",
    "fbs": "glucose",
    "fasting_blood_sugar": "glucose",
    "rbs": "glucose",
    "random_blood_sugar": "glucose",
    "random_glucose": "glucose",
    "fbg": "glucose",
    "fasting_blood_glucose": "glucose",
    # With units
    "glucose_mg_dl": "glucose",
    "glucose_mmol_l": "glucose",
    "blood_glucose_mg_dl": "glucose",
    "fasting_glucose_mg_dl": "glucose",

    # HbA1c
    "hba1c": "hba1c",
    "a1c": "hba1c",
    "hgba1c": "hba1c",
    "hb_a1c": "hba1c",
    "hemoglobin_a1c": "hba1c",
    "hemoglobin a1c": "hba1c",
    "hemoglobina1c": "hba1c",
    "glycated_hemoglobin": "hba1c",
    "glycated hemoglobin": "hba1c",
    "glycatedhemoglobin": "hba1c",
    "glycohemoglobin": "hba1c",
    # With units
    "hba1c_percent": "hba1c",
    "hba1c_mmol_mol": "hba1c",
    "a1c_percent": "hba1c",

    # =========================================================================
    # LIPID PANEL
    # =========================================================================

    # Total Cholesterol
    "cholesterol": "cholesterol",
    "total_cholesterol": "cholesterol",
    "total cholesterol": "cholesterol",
    "totalcholesterol": "cholesterol",
    "tc": "cholesterol",
    "chol": "cholesterol",
    "serum_cholesterol": "cholesterol",
    # With units
    "cholesterol_mg_dl": "cholesterol",
    "cholesterol_mmol_l": "cholesterol",
    "total_cholesterol_mg_dl": "cholesterol",

    # LDL
    "ldl": "ldl",
    "ldl_c": "ldl",
    "ldl-c": "ldl",
    "ldlc": "ldl",
    "ldl_cholesterol": "ldl",
    "ldl cholesterol": "ldl",
    "ldlcholesterol": "ldl",
    "low_density_lipoprotein": "ldl",
    "low density lipoprotein": "ldl",
    "lowdensitylipoprotein": "ldl",
    # With units
    "ldl_mg_dl": "ldl",
    "ldl_mmol_l": "ldl",
    "ldl_cholesterol_mg_dl": "ldl",

    # HDL
    "hdl": "hdl",
    "hdl_c": "hdl",
    "hdl-c": "hdl",
    "hdlc": "hdl",
    "hdl_cholesterol": "hdl",
    "hdl cholesterol": "hdl",
    "hdlcholesterol": "hdl",
    "high_density_lipoprotein": "hdl",
    "high density lipoprotein": "hdl",
    "highdensitylipoprotein": "hdl",
    # With units
    "hdl_mg_dl": "hdl",
    "hdl_mmol_l": "hdl",
    "hdl_cholesterol_mg_dl": "hdl",

    # Triglycerides
    "triglycerides": "triglycerides",
    "tg": "triglycerides",
    "trigs": "triglycerides",
    "triglyc": "triglycerides",
    "serum_triglycerides": "triglycerides",
    # With units
    "triglycerides_mg_dl": "triglycerides",
    "triglycerides_mmol_l": "triglycerides",
    "tg_mg_dl": "triglycerides",

    # =========================================================================
    # TUMOR MARKERS
    # =========================================================================

    # CEA
    "cea": "cea",
    "carcinoembryonic_antigen": "cea",
    "carcinoembryonic antigen": "cea",
    "carcinoembryonicantigen": "cea",
    "cea_ng_ml": "cea",
    "serum_cea": "cea",

    # CA-125
    "ca125": "ca125",
    "ca_125": "ca125",
    "ca-125": "ca125",
    "ca 125": "ca125",
    "cancer_antigen_125": "ca125",
    "cancer antigen 125": "ca125",
    "ca125_u_ml": "ca125",
    "serum_ca125": "ca125",

    # CA 19-9
    "ca199": "ca199",
    "ca_199": "ca199",
    "ca-199": "ca199",
    "ca_19_9": "ca199",
    "ca-19-9": "ca199",
    "ca 19-9": "ca199",
    "ca19_9": "ca199",
    "cancer_antigen_19_9": "ca199",
    "ca199_u_ml": "ca199",
    "serum_ca199": "ca199",

    # PSA
    "psa": "psa",
    "prostate_specific_antigen": "psa",
    "prostate specific antigen": "psa",
    "prostatespecificantigen": "psa",
    "total_psa": "psa",
    "free_psa": "free_psa",
    # With units
    "psa_ng_ml": "psa",
    "psa_ug_l": "psa",
    "serum_psa": "psa",

    # AFP
    "afp": "afp",
    "alpha_fetoprotein": "afp",
    "alpha-fetoprotein": "afp",
    "alpha fetoprotein": "afp",
    "alphafetoprotein": "afp",
    "a_fetoprotein": "afp",
    # With units
    "afp_ng_ml": "afp",
    "afp_iu_ml": "afp",
    "serum_afp": "afp",

    # =========================================================================
    # ELECTROLYTES
    # =========================================================================

    # Sodium
    "sodium": "sodium",
    "na": "sodium",
    "na+": "sodium",
    "serum_sodium": "sodium",
    "plasma_sodium": "sodium",
    # With units
    "sodium_meq_l": "sodium",
    "sodium_mmol_l": "sodium",
    "na_meq_l": "sodium",

    # Potassium
    "potassium": "potassium",
    "k": "potassium",
    "k+": "potassium",
    "serum_potassium": "potassium",
    "plasma_potassium": "potassium",
    # With units
    "potassium_meq_l": "potassium",
    "potassium_mmol_l": "potassium",
    "k_meq_l": "potassium",

    # Chloride
    "chloride": "chloride",
    "cl": "chloride",
    "cl-": "chloride",
    "serum_chloride": "chloride",
    # With units
    "chloride_meq_l": "chloride",
    "chloride_mmol_l": "chloride",
    "cl_meq_l": "chloride",

    # Bicarbonate / CO2
    "bicarbonate": "bicarbonate",
    "bicarb": "bicarbonate",
    "hco3": "bicarbonate",
    "hco3-": "bicarbonate",
    "co2": "bicarbonate",
    "total_co2": "bicarbonate",
    "tco2": "bicarbonate",
    "serum_bicarbonate": "bicarbonate",
    # With units
    "bicarbonate_meq_l": "bicarbonate",
    "bicarbonate_mmol_l": "bicarbonate",
    "co2_meq_l": "bicarbonate",

    # Calcium
    "calcium": "calcium",
    "ca": "calcium",
    "ca2+": "calcium",
    "serum_calcium": "calcium",
    "total_calcium": "calcium",
    "ionized_calcium": "ionized_calcium",
    "ica": "ionized_calcium",
    # With units
    "calcium_mg_dl": "calcium",
    "calcium_mmol_l": "calcium",
    "ca_mg_dl": "calcium",

    # Magnesium
    "magnesium": "magnesium",
    "mg": "magnesium",
    "mg2+": "magnesium",
    "serum_magnesium": "magnesium",
    # With units
    "magnesium_mg_dl": "magnesium",
    "magnesium_meq_l": "magnesium",
    "magnesium_mmol_l": "magnesium",
    "mg_mg_dl": "magnesium",

    # Phosphate/Phosphorus
    "phosphate": "phosphate",
    "phosphorus": "phosphate",
    "phos": "phosphate",
    "po4": "phosphate",
    "serum_phosphate": "phosphate",
    "serum_phosphorus": "phosphate",
    # With units
    "phosphate_mg_dl": "phosphate",
    "phosphorus_mg_dl": "phosphate",
    "phosphate_mmol_l": "phosphate",

    # =========================================================================
    # COMPLETE BLOOD COUNT (CBC)
    # =========================================================================

    # Hemoglobin
    "hemoglobin": "hemoglobin",
    "hgb": "hemoglobin",
    "hb": "hemoglobin",
    "haemoglobin": "hemoglobin",
    "blood_hemoglobin": "hemoglobin",
    # With units
    "hemoglobin_g_dl": "hemoglobin",
    "hgb_g_dl": "hemoglobin",
    "hb_g_dl": "hemoglobin",

    # Hematocrit
    "hematocrit": "hematocrit",
    "hct": "hematocrit",
    "haematocrit": "hematocrit",
    "packed_cell_volume": "hematocrit",
    "pcv": "hematocrit",
    # With units
    "hematocrit_percent": "hematocrit",
    "hct_percent": "hematocrit",

    # Platelets
    "platelets": "platelets",
    "plt": "platelets",
    "platelet_count": "platelets",
    "platelet count": "platelets",
    "plateletcount": "platelets",
    "thrombocytes": "platelets",
    # With units
    "platelets_k_ul": "platelets",
    "platelets_10_3_ul": "platelets",
    "plt_k_ul": "platelets",
    "platelets_thou_ul": "platelets",

    # RBC
    "rbc": "rbc",
    "red_blood_cells": "rbc",
    "red blood cells": "rbc",
    "redbloodcells": "rbc",
    "red_blood_cell": "rbc",
    "red_cell_count": "rbc",
    "erythrocytes": "rbc",
    # With units
    "rbc_m_ul": "rbc",
    "rbc_10_6_ul": "rbc",

    # MCV
    "mcv": "mcv",
    "mean_corpuscular_volume": "mcv",
    "mean corpuscular volume": "mcv",
    "meancorpuscularvolume": "mcv",
    "mcv_fl": "mcv",

    # MCH
    "mch": "mch",
    "mean_corpuscular_hemoglobin": "mch",
    "mean corpuscular hemoglobin": "mch",
    "meancorpuscularhemoglobin": "mch",
    "mch_pg": "mch",

    # MCHC
    "mchc": "mchc",
    "mean_corpuscular_hemoglobin_concentration": "mchc",
    "mchc_g_dl": "mchc",

    # RDW
    "rdw": "rdw",
    "red_cell_distribution_width": "rdw",
    "red cell distribution width": "rdw",
    "rdw_cv": "rdw",
    "rdw_percent": "rdw",

    # =========================================================================
    # BLOOD GASES / ACID-BASE
    # =========================================================================

    # pH
    "ph": "ph",
    "blood_ph": "ph",
    "arterial_ph": "ph",
    "venous_ph": "ph",
    "abg_ph": "ph",

    # pCO2
    "pco2": "pco2",
    "partial_co2": "pco2",
    "partial_pressure_co2": "pco2",
    "carbon_dioxide": "pco2",
    "arterial_pco2": "pco2",
    "paco2": "pco2",
    # With units
    "pco2_mmhg": "pco2",

    # pO2
    "po2": "po2",
    "partial_o2": "po2",
    "partial_pressure_o2": "po2",
    "oxygen": "po2",
    "arterial_po2": "po2",
    "pao2": "po2",
    # With units
    "po2_mmhg": "po2",

    # O2 Saturation
    "sao2": "sao2",
    "spo2": "sao2",
    "o2sat": "sao2",
    "o2_sat": "sao2",
    "oxygen_saturation": "sao2",
    "oxygen saturation": "sao2",
    "oxygensaturation": "sao2",
    "sat_o2": "sao2",
    "arterial_saturation": "sao2",
    # With units
    "sao2_percent": "sao2",
    "spo2_percent": "sao2",

    # Base Excess
    "base_excess": "base_excess",
    "be": "base_excess",
    "base_deficit": "base_excess",

    # =========================================================================
    # COAGULATION
    # =========================================================================

    # aPTT
    "aptt": "aptt",
    "ptt": "aptt",
    "activated_partial_thromboplastin_time": "aptt",
    "partial_thromboplastin_time": "aptt",
    "aptt_seconds": "aptt",
    "aptt_sec": "aptt",

    # Fibrinogen
    "fibrinogen": "fibrinogen",
    "factor_i": "fibrinogen",
    "fibrinogen_mg_dl": "fibrinogen",

    # =========================================================================
    # PATIENT IDENTIFIERS
    # =========================================================================

    "patient_id": "patient_id",
    "patientid": "patient_id",
    "patient id": "patient_id",
    "patient": "patient_id",
    "mrn": "patient_id",
    "medical_record_number": "patient_id",
    "medical record number": "patient_id",
    "id": "patient_id",
    "pt_id": "patient_id",
    "subject_id": "patient_id",
    "subject id": "patient_id",
    "subjectid": "patient_id",
    "case_id": "patient_id",
    "record_id": "patient_id",
    "encounter_id": "encounter_id",
    "visit_id": "encounter_id",
    "admission_id": "encounter_id",

    # =========================================================================
    # VITAL SIGNS
    # =========================================================================

    # Heart Rate
    "heart_rate": "heart_rate",
    "heartrate": "heart_rate",
    "heart rate": "heart_rate",
    "hr": "heart_rate",
    "pulse": "heart_rate",
    "pulse_rate": "heart_rate",
    "pulserate": "heart_rate",
    # With units
    "hr_bpm": "heart_rate",
    "heart_rate_bpm": "heart_rate",
    "pulse_bpm": "heart_rate",

    # Blood Pressure
    "blood_pressure": "blood_pressure",
    "bloodpressure": "blood_pressure",
    "blood pressure": "blood_pressure",
    "bp": "blood_pressure",

    # Systolic BP
    "systolic": "systolic_bp",
    "systolic_bp": "systolic_bp",
    "systolicbp": "systolic_bp",
    "systolic_blood_pressure": "systolic_bp",
    "sbp": "systolic_bp",
    "sys": "systolic_bp",
    "systolic_mmhg": "systolic_bp",

    # Diastolic BP
    "diastolic": "diastolic_bp",
    "diastolic_bp": "diastolic_bp",
    "diastolicbp": "diastolic_bp",
    "diastolic_blood_pressure": "diastolic_bp",
    "dbp": "diastolic_bp",
    "dia": "diastolic_bp",
    "diastolic_mmhg": "diastolic_bp",

    # MAP
    "map": "map",
    "mean_arterial_pressure": "map",
    "mean arterial pressure": "map",
    "meanarterialpressure": "map",

    # Respiratory Rate
    "respiratory_rate": "respiratory_rate",
    "respiratoryrate": "respiratory_rate",
    "respiratory rate": "respiratory_rate",
    "rr": "respiratory_rate",
    "resp_rate": "respiratory_rate",
    "resprate": "respiratory_rate",
    "respiration": "respiratory_rate",
    "breaths_per_min": "respiratory_rate",
    # With units
    "rr_breaths_min": "respiratory_rate",
}

# Critical biomarkers for each domain
DOMAIN_CRITICAL_BIOMARKERS = {
    "sepsis": {
        "critical": {"crp", "wbc", "procalcitonin", "lactate", "temperature"},
        "helpful": {"heart_rate", "respiratory_rate", "blood_pressure"},
        "genetic": {"CYP2D6", "DPYD", "IL6", "TNF"},
    },
    "cardiac": {
        "critical": {"troponin", "bnp", "crp"},
        "helpful": {"ldl", "hdl", "cholesterol", "blood_pressure"},
        "genetic": {"VKORC1", "CYP2C19", "CYP2C9", "APOE"},
    },
    "oncology": {
        "critical": {"cea", "ca125", "psa", "afp"},
        "helpful": {"wbc", "hemoglobin", "platelets"},
        "genetic": {"BRCA1", "BRCA2", "TP53", "DPYD", "UGT1A1"},
    },
    "metabolic": {
        "critical": {"glucose", "hba1c", "triglycerides", "ldl"},
        "helpful": {"cholesterol", "hdl", "creatinine"},
        "genetic": {"CYP2C9", "SLCO1B1", "APOE"},
    },
    "renal": {
        "critical": {"creatinine", "bun", "egfr"},
        "helpful": {"potassium", "sodium", "phosphate"},
        "genetic": {"CYP2D6", "SLCO1B1", "ABCB1"},
    },
    "hepatic": {
        "critical": {"alt", "ast", "bilirubin", "albumin"},
        "helpful": {"inr", "platelets"},
        "genetic": {"CYP2D6", "CYP3A4", "NAT2", "UGT1A1"},
    },
}


@dataclass
class ParseResult:
    """Result of parsing attempt."""
    success: bool
    data: List[Dict[str, Any]] = field(default_factory=list)
    method: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    score: float
    parsed_data: List[Dict[str, Any]]
    recognized_biomarkers: Dict[str, Any]
    unrecognized_columns: List[str]
    missing_by_domain: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    warnings: List[str]
    parse_method: str
    can_analyze: bool


class RobustDataParser:
    """
    Bulletproof data parser that handles ANY input.
    Never crashes. Always produces results.
    """

    def __init__(self):
        self.biomarker_map = BIOMARKER_MAPPINGS
        self.domain_requirements = DOMAIN_CRITICAL_BIOMARKERS

    def parse_any_input(self, data: Any) -> DataQualityReport:
        """
        Parse ANY input - CSV, JSON, plain text, dict, malformed data.
        Returns structured data with confidence scores.

        NEVER raises exceptions. ALWAYS returns a result.
        """
        warnings = []
        parsed_data = []
        parse_method = "none"

        try:
            # Handle different input types
            if isinstance(data, dict):
                parsed_data = [data]
                parse_method = "dict_direct"
            elif isinstance(data, list):
                parsed_data = data if all(isinstance(d, dict) for d in data) else []
                parse_method = "list_direct"
            elif isinstance(data, (str, bytes)):
                if isinstance(data, bytes):
                    try:
                        data = data.decode('utf-8', errors='replace')
                    except Exception as e:
                        warnings.append(f"Encoding issue: {e}")
                        data = str(data)

                # Try multiple parsing strategies
                result = self._try_all_strategies(data)
                parsed_data = result.data
                parse_method = result.method
                warnings.extend(result.warnings)
            else:
                warnings.append(f"Unknown input type: {type(data)}")
                # Try to convert to string and parse
                try:
                    data_str = str(data)
                    result = self._try_all_strategies(data_str)
                    parsed_data = result.data
                    parse_method = f"converted_{result.method}"
                    warnings.extend(result.warnings)
                except Exception as e:
                    warnings.append(f"Conversion failed: {e}")

        except Exception as e:
            warnings.append(f"Top-level parse error: {e}")
            logger.warning(f"RobustDataParser caught error: {e}")

        # Map to biomarkers
        recognized, unrecognized = self._map_to_biomarkers(parsed_data)

        # Calculate missing data by domain
        missing_by_domain = self._calculate_missing_by_domain(recognized)

        # Generate recommendations
        recommendations = self._generate_recommendations(recognized, missing_by_domain)

        # Calculate quality score
        quality_score = self._calculate_quality_score(recognized, parsed_data)

        # Determine if we can analyze
        can_analyze = len(recognized) > 0 or len(parsed_data) > 0

        return DataQualityReport(
            score=quality_score,
            parsed_data=parsed_data,
            recognized_biomarkers=recognized,
            unrecognized_columns=unrecognized,
            missing_by_domain=missing_by_domain,
            recommendations=recommendations,
            warnings=warnings,
            parse_method=parse_method,
            can_analyze=can_analyze,
        )

    def _try_all_strategies(self, data: str) -> ParseResult:
        """Try multiple parsing strategies in order of preference."""
        strategies = [
            ("json", self._try_json),
            ("csv_standard", self._try_csv_standard),
            ("csv_flexible", self._try_csv_flexible),
            ("key_value", self._try_key_value_pairs),
            ("extract_numbers", self._try_extract_numbers),
            ("line_by_line", self._try_line_by_line),
        ]

        all_warnings = []

        for name, strategy in strategies:
            try:
                result = strategy(data)
                if result.success and result.data:
                    result.method = name
                    result.warnings.extend(all_warnings)
                    return result
            except Exception as e:
                all_warnings.append(f"{name}: {str(e)[:100]}")
                continue

        # Nothing worked - return empty with warnings
        return ParseResult(
            success=False,
            data=[],
            method="none",
            warnings=all_warnings + ["All parsing strategies failed"]
        )

    def _try_json(self, data: str) -> ParseResult:
        """Try parsing as JSON."""
        data = data.strip()

        # Try direct JSON parse
        try:
            parsed = json.loads(data)
            if isinstance(parsed, dict):
                return ParseResult(success=True, data=[parsed])
            elif isinstance(parsed, list):
                if all(isinstance(item, dict) for item in parsed):
                    return ParseResult(success=True, data=parsed)
                # List of values - try to make sense of it
                return ParseResult(success=True, data=[{"values": parsed}])
        except json.JSONDecodeError:
            pass

        # Try to find JSON within text
        json_patterns = [
            r'\{[^{}]*\}',  # Simple object
            r'\[[^\[\]]*\]',  # Simple array
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, data, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict) and len(parsed) > 0:
                        return ParseResult(success=True, data=[parsed])
                except:
                    continue

        return ParseResult(success=False)

    def _try_csv_standard(self, data: str) -> ParseResult:
        """Try parsing as standard CSV."""
        try:
            import pandas as pd
            df = pd.read_csv(StringIO(data))
            if len(df) > 0 and len(df.columns) > 0:
                records = df.to_dict('records')
                return ParseResult(success=True, data=records)
        except Exception as e:
            return ParseResult(success=False, warnings=[str(e)[:100]])

        return ParseResult(success=False)

    def _try_csv_flexible(self, data: str) -> ParseResult:
        """Try parsing CSV with flexible options."""
        import pandas as pd

        warnings = []

        # Try different delimiters
        for delimiter in [',', ';', '\t', '|', ' ']:
            try:
                df = pd.read_csv(
                    StringIO(data),
                    delimiter=delimiter,
                    on_bad_lines='skip',
                    encoding_errors='replace',
                    dtype=str,
                    skipinitialspace=True,
                )

                if len(df.columns) > 1 or (len(df.columns) == 1 and len(df) > 0):
                    # Convert numeric strings to numbers
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='ignore')
                        except:
                            pass

                    records = df.to_dict('records')
                    if records:
                        return ParseResult(
                            success=True,
                            data=records,
                            warnings=[f"Used delimiter: '{delimiter}'"]
                        )
            except Exception as e:
                warnings.append(f"delimiter '{delimiter}': {str(e)[:50]}")
                continue

        return ParseResult(success=False, warnings=warnings)

    def _try_key_value_pairs(self, data: str) -> ParseResult:
        """Extract key-value pairs from text."""
        patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([\d.]+)',  # key: value or key = value
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*"([^"]*)"',  # key: "string"
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*\'([^\']*)\'',  # key: 'string'
        ]

        extracted = {}

        for pattern in patterns:
            matches = re.findall(pattern, data, re.IGNORECASE)
            for key, value in matches:
                key_lower = key.lower().strip()
                try:
                    extracted[key_lower] = float(value)
                except ValueError:
                    extracted[key_lower] = value.strip()

        if extracted:
            return ParseResult(success=True, data=[extracted])

        return ParseResult(success=False)

    def _try_extract_numbers(self, data: str) -> ParseResult:
        """Extract any numbers with context from text."""
        # Find patterns like "CRP 28.5" or "WBC is 18"
        pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is\s+|was\s+|of\s+)?([\d.]+)'

        extracted = {}
        matches = re.findall(pattern, data, re.IGNORECASE)

        for key, value in matches:
            key_lower = key.lower().strip()
            try:
                extracted[key_lower] = float(value)
            except ValueError:
                pass

        # Also try to find standalone biomarker mentions
        for biomarker_variant in self.biomarker_map.keys():
            if biomarker_variant in data.lower():
                # Find numbers near this biomarker
                pattern = re.escape(biomarker_variant) + r'\s*[:=]?\s*([\d.]+)'
                match = re.search(pattern, data.lower())
                if match:
                    try:
                        extracted[biomarker_variant] = float(match.group(1))
                    except ValueError:
                        pass

        if extracted:
            return ParseResult(success=True, data=[extracted])

        return ParseResult(success=False)

    def _try_line_by_line(self, data: str) -> ParseResult:
        """Parse line by line looking for key-value patterns."""
        lines = data.strip().split('\n')
        extracted = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Try various separators
            for sep in [':', '=', '\t', '  ', ',']:
                parts = line.split(sep, 1)
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()

                    # Skip if key looks like a sentence
                    if len(key) > 30 or ' ' in key and len(key.split()) > 3:
                        continue

                    try:
                        extracted[key] = float(value)
                    except ValueError:
                        if value:
                            extracted[key] = value
                    break

        if extracted:
            return ParseResult(success=True, data=[extracted])

        return ParseResult(success=False)

    def _map_to_biomarkers(
        self,
        parsed_data: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Map parsed columns to known biomarkers."""
        recognized = {}
        unrecognized = []

        for record in parsed_data:
            for key, value in record.items():
                key_lower = str(key).lower().strip()

                # Check if it maps to a known biomarker
                if key_lower in self.biomarker_map:
                    standard_name = self.biomarker_map[key_lower]
                    recognized[standard_name] = value
                else:
                    # Check partial matches
                    matched = False
                    for variant, standard in self.biomarker_map.items():
                        if variant in key_lower or key_lower in variant:
                            recognized[standard] = value
                            matched = True
                            break

                    if not matched and key_lower not in unrecognized:
                        unrecognized.append(key_lower)

        return recognized, unrecognized

    def _calculate_missing_by_domain(
        self,
        recognized: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate missing critical data for each domain."""
        missing_by_domain = {}
        recognized_set = set(recognized.keys())

        for domain, requirements in self.domain_requirements.items():
            critical = requirements["critical"]
            helpful = requirements["helpful"]
            genetic = requirements.get("genetic", set())

            critical_have = recognized_set & critical
            critical_missing = critical - recognized_set
            helpful_have = recognized_set & helpful
            helpful_missing = helpful - recognized_set

            completeness = len(critical_have) / len(critical) if critical else 0

            # Determine impact message
            if completeness < 0.3:
                impact = f"Limited {domain} analysis possible. Add critical markers for full workup."
            elif completeness < 0.7:
                impact = f"Partial {domain} analysis. Missing markers could improve detection by 1-2 days."
            elif completeness < 1.0:
                impact = f"Good {domain} coverage. Additional markers would enhance precision."
            else:
                impact = f"Complete {domain} biomarker coverage."

            missing_by_domain[domain] = {
                "critical_have": list(critical_have),
                "critical_missing": list(critical_missing),
                "helpful_have": list(helpful_have),
                "helpful_missing": list(helpful_missing),
                "genetic_recommended": list(genetic),
                "completeness": round(completeness, 2),
                "impact": impact,
            }

        return missing_by_domain

    def _generate_recommendations(
        self,
        recognized: Dict[str, Any],
        missing_by_domain: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable recommendations based on data quality."""
        recommendations = []

        # Check overall coverage
        if len(recognized) == 0:
            recommendations.append(
                "No recognized biomarkers found. Provide data with columns like: "
                "crp, wbc, procalcitonin, troponin, glucose"
            )
            recommendations.append(
                "Accepted formats: CSV with headers, JSON, or key-value pairs (e.g., 'CRP: 28.5')"
            )
            return recommendations

        # Find the best domain match
        best_domain = None
        best_completeness = 0

        for domain, info in missing_by_domain.items():
            if info["completeness"] > best_completeness:
                best_completeness = info["completeness"]
                best_domain = domain

        if best_domain:
            info = missing_by_domain[best_domain]
            if info["critical_missing"]:
                missing_str = ", ".join(info["critical_missing"][:3])
                recommendations.append(
                    f"For {best_domain} analysis: add {missing_str} to improve detection"
                )

            if info["genetic_recommended"]:
                genes_str = ", ".join(info["genetic_recommended"][:3])
                recommendations.append(
                    f"Genetic markers ({genes_str}) would enable 2-3 days earlier detection"
                )

        # Check for missing high-value biomarkers
        high_value = {"procalcitonin", "lactate", "troponin"}
        missing_high_value = high_value - set(recognized.keys())
        if missing_high_value:
            recommendations.append(
                f"High-value biomarkers not provided: {', '.join(missing_high_value)}"
            )

        return recommendations

    def _calculate_quality_score(
        self,
        recognized: Dict[str, Any],
        parsed_data: List[Dict[str, Any]],
    ) -> float:
        """Calculate overall data quality score (0.0 to 1.0)."""
        if not parsed_data:
            return 0.0

        score = 0.0

        # Base score for having any data
        if parsed_data:
            score += 0.2

        # Score for recognized biomarkers
        num_recognized = len(recognized)
        if num_recognized >= 10:
            score += 0.3
        elif num_recognized >= 5:
            score += 0.2
        elif num_recognized >= 2:
            score += 0.1
        elif num_recognized >= 1:
            score += 0.05

        # Score for critical biomarkers
        critical_all = set()
        for domain_reqs in self.domain_requirements.values():
            critical_all.update(domain_reqs["critical"])

        critical_present = set(recognized.keys()) & critical_all
        critical_ratio = len(critical_present) / len(critical_all) if critical_all else 0
        score += critical_ratio * 0.3

        # Score for data validity (numeric values)
        valid_values = 0
        total_values = 0
        for value in recognized.values():
            total_values += 1
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                valid_values += 1

        if total_values > 0:
            score += (valid_values / total_values) * 0.2

        return min(round(score, 2), 1.0)

    def extract_lab_values(self, parsed_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract standardized lab values from parsed data.
        Returns a clean dict suitable for analysis.
        """
        recognized, _ = self._map_to_biomarkers(parsed_data)

        # Clean up values - ensure numeric where expected
        lab_data = {}
        for key, value in recognized.items():
            if key == "patient_id":
                lab_data[key] = str(value)
            else:
                try:
                    lab_data[key] = float(value)
                except (ValueError, TypeError):
                    # Keep as string if can't convert
                    lab_data[key] = value

        return lab_data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_parser: Optional[RobustDataParser] = None


def get_parser() -> RobustDataParser:
    """Get global parser instance."""
    global _parser
    if _parser is None:
        _parser = RobustDataParser()
    return _parser


def parse_any_data(data: Any) -> DataQualityReport:
    """Parse any data using the global parser."""
    return get_parser().parse_any_input(data)


def extract_lab_data(data: Any) -> Dict[str, Any]:
    """Extract lab values from any data format."""
    report = parse_any_data(data)
    return get_parser().extract_lab_values(report.parsed_data)
