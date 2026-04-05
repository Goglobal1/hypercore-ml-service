"""
Universal Data Ingestion Layer
Accepts any data format, normalizes it, maps to 24 endpoints
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class DataType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class MappedColumn:
    original_name: str
    normalized_name: str
    data_type: DataType
    endpoints: List[str]
    confidence: float
    unit: Optional[str]
    reference_range: Optional[Tuple[float, float]]


# Master mapping of all possible biomarker names to endpoints
BIOMARKER_MAPPINGS = {
    'cardiac': {
        'markers': [
            'troponin', 'troponin_i', 'troponin_t', 'trop', 'tnni', 'tnnt',
            'bnp', 'nt_probnp', 'ntprobnp', 'pro_bnp', 'brain_natriuretic',
            'ck_mb', 'ckmb', 'creatine_kinase_mb',
            'ldh', 'lactate_dehydrogenase',
            'myoglobin',
            'heart_rate', 'hr', 'pulse', 'bpm',
            'bp_systolic', 'systolic', 'sbp', 'sys_bp',
            'bp_diastolic', 'diastolic', 'dbp', 'dia_bp',
            'blood_pressure', 'bp',
            'map', 'mean_arterial_pressure',
            'ejection_fraction', 'ef', 'lvef',
            'ecg', 'ekg', 'electrocardiogram',
            'qt_interval', 'qtc', 'pr_interval', 'qrs'
        ],
        'reference_ranges': {
            'troponin': (0, 0.04),
            'bnp': (0, 100),
            'heart_rate': (60, 100),
            'bp_systolic': (90, 140),
            'bp_diastolic': (60, 90)
        }
    },
    'renal': {
        'markers': [
            'creatinine', 'creat', 'cr', 'serum_creatinine',
            'egfr', 'gfr', 'estimated_gfr', 'glomerular_filtration',
            'bun', 'blood_urea_nitrogen', 'urea', 'urea_nitrogen',
            'cystatin', 'cystatin_c',
            'urine_output', 'uop', 'urinary_output',
            'urine_protein', 'proteinuria', 'albuminuria',
            'urine_creatinine', 'ucr',
            'uacr', 'albumin_creatinine_ratio',
            'sodium', 'na', 'serum_sodium',
            'potassium', 'k', 'serum_potassium',
            'chloride', 'cl',
            'bicarbonate', 'hco3', 'bicarb', 'co2'
        ],
        'reference_ranges': {
            'creatinine': (0.6, 1.2),
            'egfr': (90, 120),
            'bun': (7, 20),
            'sodium': (136, 145),
            'potassium': (3.5, 5.0)
        }
    },
    'hepatic': {
        'markers': [
            'alt', 'alanine_aminotransferase', 'sgpt',
            'ast', 'aspartate_aminotransferase', 'sgot',
            'alp', 'alkaline_phosphatase', 'alk_phos',
            'ggt', 'gamma_glutamyl', 'gamma_gt',
            'bilirubin', 'bili', 'total_bilirubin', 'tbili',
            'direct_bilirubin', 'dbili', 'conjugated_bilirubin',
            'indirect_bilirubin', 'ibil',
            'albumin', 'alb', 'serum_albumin',
            'total_protein', 'tp', 'protein',
            'inr', 'international_normalized_ratio',
            'pt', 'prothrombin_time',
            'ammonia', 'nh3'
        ],
        'reference_ranges': {
            'alt': (7, 56),
            'ast': (10, 40),
            'bilirubin': (0.1, 1.2),
            'albumin': (3.5, 5.0),
            'inr': (0.8, 1.2)
        }
    },
    'respiratory': {
        'markers': [
            'spo2', 'oxygen_saturation', 'o2_sat', 'sao2', 'pulse_ox',
            'pao2', 'partial_pressure_oxygen', 'arterial_oxygen',
            'paco2', 'partial_pressure_co2', 'arterial_co2',
            'fio2', 'fraction_inspired_oxygen', 'oxygen_percentage',
            'respiratory_rate', 'rr', 'resp_rate', 'breaths_per_min',
            'pf_ratio', 'p_f_ratio', 'pao2_fio2',
            'peak_flow', 'pef', 'peak_expiratory_flow',
            'fev1', 'forced_expiratory_volume',
            'fvc', 'forced_vital_capacity',
            'tidal_volume', 'tv', 'vt',
            'minute_ventilation', 'mv',
            'peep', 'positive_end_expiratory'
        ],
        'reference_ranges': {
            'spo2': (95, 100),
            'pao2': (80, 100),
            'paco2': (35, 45),
            'respiratory_rate': (12, 20)
        }
    },
    'sepsis': {
        'markers': [
            'procalcitonin', 'pct', 'pro_calcitonin',
            'lactate', 'lactic_acid', 'serum_lactate',
            'wbc', 'white_blood_cell', 'leukocytes', 'white_count',
            'temperature', 'temp', 'body_temp', 'fever',
            'crp', 'c_reactive_protein', 'c_reactive',
            'esr', 'sed_rate', 'erythrocyte_sedimentation',
            'il6', 'interleukin_6', 'interleukin6',
            'il1', 'interleukin_1',
            'tnf', 'tnf_alpha', 'tumor_necrosis_factor',
            'bands', 'band_neutrophils', 'immature_neutrophils',
            'blood_culture', 'culture_positive',
            'sofa', 'sofa_score',
            'qsofa', 'quick_sofa'
        ],
        'reference_ranges': {
            'procalcitonin': (0, 0.5),
            'lactate': (0.5, 2.0),
            'wbc': (4.5, 11.0),
            'temperature': (97.0, 99.0),
            'crp': (0, 10)
        }
    },
    'metabolic': {
        'markers': [
            'glucose', 'blood_glucose', 'blood_sugar', 'bs', 'sugar',
            'fasting_glucose', 'fbg', 'fbs',
            'hba1c', 'a1c', 'glycated_hemoglobin', 'hemoglobin_a1c',
            'insulin', 'serum_insulin', 'fasting_insulin',
            'c_peptide', 'cpeptide',
            'triglycerides', 'tg', 'trigs',
            'cholesterol', 'total_cholesterol', 'tc',
            'hdl', 'hdl_cholesterol', 'good_cholesterol',
            'ldl', 'ldl_cholesterol', 'bad_cholesterol',
            'vldl',
            'bmi', 'body_mass_index',
            'weight', 'body_weight', 'wt',
            'height', 'ht',
            'waist', 'waist_circumference',
            'leptin', 'adiponectin', 'ghrelin'
        ],
        'reference_ranges': {
            'glucose': (70, 100),
            'hba1c': (4.0, 5.6),
            'triglycerides': (0, 150),
            'cholesterol': (0, 200),
            'bmi': (18.5, 24.9)
        }
    },
    'endocrine': {
        'markers': [
            'tsh', 'thyroid_stimulating', 'thyrotropin',
            't3', 'triiodothyronine', 'free_t3', 'ft3',
            't4', 'thyroxine', 'free_t4', 'ft4',
            'cortisol', 'serum_cortisol', 'am_cortisol',
            'acth', 'adrenocorticotropic',
            'aldosterone',
            'renin', 'plasma_renin',
            'parathyroid', 'pth', 'parathyroid_hormone',
            'vitamin_d', 'vit_d', '25_oh_d', 'calcidiol',
            'calcium', 'ca', 'serum_calcium',
            'phosphorus', 'phos', 'phosphate',
            'magnesium', 'mg', 'serum_magnesium',
            'growth_hormone', 'gh', 'hgh',
            'igf1', 'igf_1', 'insulin_like_growth',
            'prolactin', 'prl',
            'testosterone', 'free_testosterone',
            'estrogen', 'estradiol', 'e2',
            'progesterone',
            'fsh', 'follicle_stimulating',
            'lh', 'luteinizing_hormone'
        ],
        'reference_ranges': {
            'tsh': (0.4, 4.0),
            'cortisol': (6, 23),
            'calcium': (8.5, 10.5),
            'vitamin_d': (30, 100)
        }
    },
    'hematologic': {
        'markers': [
            'hemoglobin', 'hgb', 'hb',
            'hematocrit', 'hct',
            'rbc', 'red_blood_cell', 'erythrocytes', 'red_count',
            'mcv', 'mean_corpuscular_volume',
            'mch', 'mean_corpuscular_hemoglobin',
            'mchc', 'mean_corpuscular_hemoglobin_concentration',
            'rdw', 'red_distribution_width',
            'platelets', 'plt', 'platelet_count', 'thrombocytes',
            'mpv', 'mean_platelet_volume',
            'reticulocytes', 'retic', 'retic_count',
            'iron', 'serum_iron', 'fe',
            'tibc', 'total_iron_binding',
            'ferritin', 'serum_ferritin',
            'transferrin', 'transferrin_saturation',
            'b12', 'vitamin_b12', 'cobalamin',
            'folate', 'folic_acid',
            'epo', 'erythropoietin'
        ],
        'reference_ranges': {
            'hemoglobin': (12, 17),
            'hematocrit': (36, 50),
            'platelets': (150, 400),
            'ferritin': (20, 200)
        }
    },
    'coagulation': {
        'markers': [
            'pt', 'prothrombin_time',
            'inr', 'international_normalized_ratio',
            'ptt', 'partial_thromboplastin', 'aptt',
            'fibrinogen', 'factor_i',
            'd_dimer', 'ddimer', 'd-dimer',
            'fsp', 'fibrin_split', 'fdp',
            'antithrombin', 'at3', 'antithrombin_iii',
            'protein_c', 'prot_c',
            'protein_s', 'prot_s',
            'factor_v', 'factor_viii', 'factor_x',
            'von_willebrand', 'vwf',
            'bleeding_time', 'bt',
            'clotting_time', 'ct'
        ],
        'reference_ranges': {
            'pt': (11, 13.5),
            'inr': (0.8, 1.2),
            'ptt': (25, 35),
            'd_dimer': (0, 0.5),
            'fibrinogen': (200, 400)
        }
    },
    'inflammatory': {
        'markers': [
            'crp', 'c_reactive_protein', 'c_reactive',
            'hs_crp', 'high_sensitivity_crp',
            'esr', 'sed_rate', 'erythrocyte_sedimentation',
            'ferritin', 'serum_ferritin',
            'il1', 'interleukin_1', 'il_1',
            'il6', 'interleukin_6', 'il_6',
            'il8', 'interleukin_8', 'il_8',
            'il10', 'interleukin_10', 'il_10',
            'tnf', 'tnf_alpha', 'tumor_necrosis_factor',
            'interferon', 'ifn', 'ifn_gamma',
            'complement_c3', 'c3',
            'complement_c4', 'c4',
            'ana', 'antinuclear_antibody',
            'rf', 'rheumatoid_factor',
            'anti_ccp', 'ccp'
        ],
        'reference_ranges': {
            'crp': (0, 10),
            'esr': (0, 20),
            'ferritin': (20, 200)
        }
    },
    'immune': {
        'markers': [
            'wbc', 'white_blood_cell', 'leukocytes',
            'lymphocytes', 'lymph', 'lymph_percent',
            'neutrophils', 'neut', 'neut_percent', 'pmn',
            'monocytes', 'mono', 'mono_percent',
            'eosinophils', 'eos', 'eos_percent',
            'basophils', 'baso', 'baso_percent',
            'cd4', 'cd4_count', 'helper_t',
            'cd8', 'cd8_count', 'cytotoxic_t',
            'cd4_cd8_ratio', 'cd4_ratio',
            'nk_cells', 'natural_killer',
            'b_cells', 'cd19',
            'igg', 'immunoglobulin_g',
            'iga', 'immunoglobulin_a',
            'igm', 'immunoglobulin_m',
            'ige', 'immunoglobulin_e'
        ],
        'reference_ranges': {
            'wbc': (4.5, 11.0),
            'lymphocytes': (20, 40),
            'neutrophils': (40, 70),
            'cd4': (500, 1500)
        }
    },
    'neurologic': {
        'markers': [
            'gcs', 'glasgow_coma', 'coma_scale',
            'mental_status', 'consciousness', 'loc',
            'pupils', 'pupil_response', 'pupillary',
            'neuro_exam',
            'motor_score', 'motor_function',
            'sensory_score', 'sensory_function',
            'reflexes', 'dtr', 'deep_tendon',
            'nss', 's100b', 's100_b',
            'nse', 'neuron_specific_enolase',
            'csf_protein', 'csf_glucose',
            'opening_pressure', 'csf_pressure',
            'mmse', 'mini_mental',
            'moca', 'montreal_cognitive',
            'nihss', 'stroke_scale'
        ],
        'reference_ranges': {
            'gcs': (15, 15),
            'pupils': (2, 4)
        }
    },
    'fluid_balance': {
        'markers': [
            'sodium', 'na', 'serum_sodium',
            'potassium', 'k', 'serum_potassium',
            'chloride', 'cl',
            'bicarbonate', 'hco3', 'bicarb',
            'osmolality', 'serum_osmolality', 'osm',
            'urine_osmolality', 'uosm',
            'specific_gravity', 'urine_sg',
            'weight', 'daily_weight',
            'input', 'fluid_input', 'intake',
            'output', 'fluid_output', 'urine_output',
            'net_fluid', 'fluid_balance', 'i_o',
            'edema', 'peripheral_edema',
            'cvp', 'central_venous_pressure',
            'jvp', 'jugular_venous'
        ],
        'reference_ranges': {
            'sodium': (136, 145),
            'potassium': (3.5, 5.0),
            'osmolality': (275, 295)
        }
    },
    'acid_base': {
        'markers': [
            'ph', 'blood_ph', 'arterial_ph',
            'pco2', 'paco2', 'partial_co2',
            'hco3', 'bicarbonate', 'bicarb',
            'base_excess', 'be', 'base_deficit',
            'anion_gap', 'ag',
            'lactate', 'lactic_acid',
            'ketones', 'beta_hydroxybutyrate', 'bhb'
        ],
        'reference_ranges': {
            'ph': (7.35, 7.45),
            'paco2': (35, 45),
            'hco3': (22, 26),
            'anion_gap': (8, 12)
        }
    },
    'perfusion': {
        'markers': [
            'lactate', 'lactic_acid', 'serum_lactate',
            'map', 'mean_arterial_pressure',
            'bp_systolic', 'systolic', 'sbp',
            'bp_diastolic', 'diastolic', 'dbp',
            'capillary_refill', 'cap_refill', 'crt',
            'urine_output', 'uop',
            'cardiac_output', 'co', 'cardiac_index', 'ci',
            'svr', 'systemic_vascular_resistance',
            'svo2', 'mixed_venous_o2',
            'central_venous_o2', 'scvo2',
            'pulse_pressure', 'pp',
            'shock_index', 'si'
        ],
        'reference_ranges': {
            'lactate': (0.5, 2.0),
            'map': (70, 100),
            'urine_output': (0.5, 1.0)
        }
    },
    'nutritional': {
        'markers': [
            'albumin', 'alb', 'serum_albumin',
            'prealbumin', 'transthyretin',
            'transferrin',
            'total_protein', 'tp',
            'bmi', 'body_mass_index',
            'weight', 'body_weight',
            'weight_change', 'weight_loss', 'weight_gain',
            'nitrogen_balance',
            'calorie_intake', 'kcal',
            'protein_intake',
            'vitamin_a', 'retinol',
            'vitamin_c', 'ascorbic_acid',
            'vitamin_e', 'tocopherol',
            'zinc', 'zn',
            'selenium', 'se',
            'copper', 'cu'
        ],
        'reference_ranges': {
            'albumin': (3.5, 5.0),
            'prealbumin': (15, 35),
            'bmi': (18.5, 24.9)
        }
    },
    'vitals': {
        'markers': [
            'heart_rate', 'hr', 'pulse', 'bpm',
            'bp_systolic', 'systolic', 'sbp',
            'bp_diastolic', 'diastolic', 'dbp',
            'blood_pressure', 'bp',
            'temperature', 'temp', 'body_temp',
            'respiratory_rate', 'rr', 'resp_rate',
            'spo2', 'oxygen_saturation', 'o2_sat',
            'pain_score', 'pain', 'pain_level',
            'weight', 'wt',
            'height', 'ht'
        ],
        'reference_ranges': {
            'heart_rate': (60, 100),
            'bp_systolic': (90, 140),
            'bp_diastolic': (60, 90),
            'temperature': (97.0, 99.0),
            'respiratory_rate': (12, 20),
            'spo2': (95, 100)
        }
    },
    'medication': {
        'markers': [
            'medication_count', 'med_count', 'num_meds',
            'high_risk_meds', 'high_alert_meds',
            'anticoagulant', 'anticoagulation',
            'insulin', 'insulin_dose',
            'opioid', 'opioid_dose', 'morphine_equivalent',
            'vasopressor', 'pressor', 'levophed', 'norepinephrine',
            'sedation', 'sedation_score', 'rass',
            'antibiotic', 'antibiotic_days',
            'steroid', 'corticosteroid', 'prednisone_equivalent',
            'polypharmacy',
            'drug_interaction', 'ddi'
        ],
        'reference_ranges': {}
    },
    'infection': {
        'markers': [
            'wbc', 'white_blood_cell', 'leukocytes',
            'temperature', 'temp', 'fever',
            'culture_positive', 'blood_culture', 'culture',
            'antibiotic_days', 'abx_days',
            'procalcitonin', 'pct',
            'bands', 'band_neutrophils',
            'wound_culture', 'urine_culture', 'sputum_culture',
            'mrsa', 'vre', 'esbl', 'cre',
            'isolation', 'contact_precautions',
            'central_line_days', 'cld',
            'foley_days', 'catheter_days',
            'ventilator_days', 'vent_days'
        ],
        'reference_ranges': {
            'wbc': (4.5, 11.0),
            'temperature': (97.0, 99.0),
            'procalcitonin': (0, 0.5)
        }
    },
    'mobility': {
        'markers': [
            'mobility_score', 'mobility', 'ambulation',
            'fall_risk', 'morse_fall', 'fall_score',
            'bed_days', 'bedrest', 'immobility',
            'activity_level', 'adl', 'adl_score',
            'physical_therapy', 'pt_sessions',
            'braden_score', 'braden',
            'restraints', 'restraint_use',
            'delirium', 'cam', 'cam_icu'
        ],
        'reference_ranges': {}
    },
    'skin': {
        'markers': [
            'wound_count', 'wounds', 'num_wounds',
            'pressure_ulcer', 'pressure_injury', 'bedsore',
            'skin_integrity', 'skin_breakdown',
            'wound_stage', 'ulcer_stage',
            'wound_size', 'wound_area',
            'wound_drainage', 'exudate',
            'wound_infection', 'wound_culture',
            'braden_score', 'braden',
            'skin_turgor', 'turgor'
        ],
        'reference_ranges': {}
    },
    'psychosocial': {
        'markers': [
            'anxiety_score', 'anxiety', 'gad7', 'gad_7',
            'depression_score', 'depression', 'phq9', 'phq_9',
            'delirium_screen', 'cam', 'cam_icu', 'confusion',
            'suicide_risk', 'suicide_screen', 'si',
            'substance_use', 'alcohol', 'drug_screen',
            'pain_score', 'pain_level',
            'sleep', 'sleep_quality', 'insomnia',
            'stress', 'stress_level',
            'social_support', 'caregiver'
        ],
        'reference_ranges': {}
    },
    'genetic': {
        'markers': [
            'brca', 'brca1', 'brca2',
            'apoe', 'apoe4',
            'factor_v_leiden', 'fvl',
            'mthfr',
            'cyp2c19', 'cyp2d6', 'cyp3a4',
            'hla', 'hla_b27', 'hla_dr4',
            'kras', 'braf', 'egfr',
            'her2', 'her2_neu',
            'pd_l1', 'pdl1',
            'microsatellite', 'msi',
            'tmb', 'tumor_mutation_burden',
            'pharmacogenomic', 'pgx'
        ],
        'reference_ranges': {}
    },
    'oncology': {
        'markers': [
            'cea', 'carcinoembryonic',
            'ca125', 'ca_125',
            'ca19_9', 'ca_19_9',
            'psa', 'prostate_specific',
            'afp', 'alpha_fetoprotein',
            'hcg', 'beta_hcg',
            'ldh', 'lactate_dehydrogenase',
            'tumor_size', 'tumor_volume',
            'tumor_stage', 'tnm',
            'metastasis', 'mets',
            'ecog', 'performance_status',
            'wbc', 'anc', 'absolute_neutrophil'
        ],
        'reference_ranges': {}
    }
}


class UniversalIngestion:
    """
    Ingests ANY patient data and maps it to the 24-endpoint system.
    """

    def __init__(self):
        self.mappings = BIOMARKER_MAPPINGS
        self._build_reverse_index()

    def _build_reverse_index(self):
        """Build a reverse index: marker_name -> [endpoints]"""
        self.marker_to_endpoints = {}
        for endpoint, config in self.mappings.items():
            for marker in config['markers']:
                if marker not in self.marker_to_endpoints:
                    self.marker_to_endpoints[marker] = []
                self.marker_to_endpoints[marker].append(endpoint)

    def _normalize_column_name(self, name: str) -> str:
        """Normalize column name for matching."""
        normalized = name.lower()
        normalized = re.sub(r'[\s\-\.\/\\]+', '_', normalized)
        normalized = re.sub(r'[^\w]', '', normalized)
        for remove in ['patient_', 'pt_', '_value', '_result', '_level', '_count']:
            normalized = normalized.replace(remove, '')
        return normalized

    def ingest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Ingest a dataframe and map all columns to endpoints.
        NOW WITH BIOMARKER INFERENCE for unknown columns.
        """
        from .biomarker_inference import get_inference_engine

        columns = list(data.columns)
        mapped_columns = []
        endpoint_data = {endpoint: {} for endpoint in self.mappings.keys()}
        unmapped = []
        inferred_columns = []

        # Get inference engine
        inference_engine = get_inference_engine()

        for col in columns:
            # Skip ID columns
            if any(x in col.lower() for x in ['patient_id', 'id', 'mrn', 'encounter', 'sample_id', 'label']):
                continue

            normalized = self._normalize_column_name(col)
            matched = False

            # Try to match to endpoints by name
            for marker, endpoints in self.marker_to_endpoints.items():
                # Require exact match for short markers (prevents 'k' matching 'biomarker1')
                # For longer markers, allow substring match if marker is significant portion
                is_match = False
                if len(marker) <= 2:
                    # Short markers (k, na, etc.) require exact match
                    is_match = (marker == normalized)
                else:
                    # Longer markers can be substring matches
                    is_match = (marker in normalized or normalized in marker)

                if is_match:
                    for endpoint in endpoints:
                        endpoint_data[endpoint][col] = data[col].tolist()
                        mapped_columns.append({
                            'original': col,
                            'normalized': normalized,
                            'matched_marker': marker,
                            'endpoints': endpoints,
                            'method': 'name_match'
                        })
                    matched = True
                    break

            # If no name match, try INFERENCE
            if not matched:
                inference_result = inference_engine.infer_column(col, data[col].tolist())

                if inference_result['inferred_biomarker'] and inference_result['confidence'] > 0.4:
                    # Use the inferred biomarker
                    inferred_marker = inference_result['inferred_biomarker']
                    inferred_endpoints = inference_result['endpoints']

                    for endpoint in inferred_endpoints:
                        if endpoint in endpoint_data:
                            endpoint_data[endpoint][col] = data[col].tolist()

                    mapped_columns.append({
                        'original': col,
                        'normalized': normalized,
                        'matched_marker': inferred_marker,
                        'endpoints': inferred_endpoints,
                        'method': 'inference',
                        'confidence': inference_result['confidence']
                    })
                    inferred_columns.append({
                        'column': col,
                        'inferred_as': inferred_marker,
                        'confidence': inference_result['confidence'],
                        'endpoints': inferred_endpoints
                    })
                    matched = True

            if not matched:
                unmapped.append(col)

        endpoints_available = [ep for ep, data_dict in endpoint_data.items() if data_dict]

        # Build reference_ranges for each endpoint that has data
        reference_ranges = {}
        for ep in endpoints_available:
            if ep in BIOMARKER_MAPPINGS:
                reference_ranges[ep] = BIOMARKER_MAPPINGS[ep]

        return {
            'success': True,
            'columns_mapped': mapped_columns,
            'endpoints_available': endpoints_available,
            'endpoint_count': len(endpoints_available),
            'endpoint_data': {ep: data_dict for ep, data_dict in endpoint_data.items() if data_dict},
            'reference_ranges': reference_ranges,
            'unmapped_columns': unmapped,
            'inferred_columns': inferred_columns,
            'patient_count': len(data),
            'total_columns': len(data.columns),
            'message': f"Mapped {len(mapped_columns)} columns to {len(endpoints_available)} endpoints "
                       f"({len(inferred_columns)} by inference)"
        }
