"""
Long-to-Wide Data Transformation Module
========================================

Automatically detects and transforms long-format lab data to wide format
for HyperCore's /discover endpoint.

Long format (typical from HL7/EHR):
    patient_id, date, analyte, value
    P001, 2024-01-15, creatinine, 1.1
    P001, 2024-01-15, glucose, 95

Wide format (HyperCore expected):
    patient_id, creatinine, glucose, date
    P001, 1.1, 95, 2024-01-15
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ANALYTE NAME NORMALIZATION
# ============================================================================

ANALYTE_NORMALIZATION = {
    # Kidney Function
    'creatinine': ['creatinine', 'creat', 'cr', 'serum creatinine', 'creatinine, serum', 'creatinine serum'],
    'bun': ['bun', 'blood urea nitrogen', 'urea nitrogen', 'urea', 'urea nitrogen, serum'],
    'egfr': ['egfr', 'gfr', 'estimated gfr', 'glomerular filtration rate', 'egfr (ckd-epi)'],

    # Electrolytes
    'potassium': ['potassium', 'k', 'k+', 'serum potassium', 'potassium, serum'],
    'sodium': ['sodium', 'na', 'na+', 'serum sodium', 'sodium, serum'],
    'chloride': ['chloride', 'cl', 'cl-', 'chloride, serum'],
    'bicarbonate': ['bicarbonate', 'hco3', 'co2', 'carbon dioxide', 'co2, total'],
    'calcium': ['calcium', 'ca', 'ca++', 'serum calcium', 'calcium, serum', 'calcium, total'],
    'magnesium': ['magnesium', 'mg', 'mg++', 'magnesium, serum'],
    'phosphorus': ['phosphorus', 'phosphate', 'phos', 'p', 'phosphorus, serum'],

    # Liver Function
    'ast': ['ast', 'sgot', 'aspartate aminotransferase', 'aspartate transaminase', 'ast (sgot)'],
    'alt': ['alt', 'sgpt', 'alanine aminotransferase', 'alanine transaminase', 'alt (sgpt)'],
    'alp': ['alp', 'alkaline phosphatase', 'alk phos', 'alkaline phos'],
    'bilirubin': ['bilirubin', 'total bilirubin', 'tbili', 'bili', 'bilirubin, total'],
    'bilirubin_direct': ['direct bilirubin', 'dbili', 'conjugated bilirubin', 'bilirubin, direct'],
    'albumin': ['albumin', 'alb', 'serum albumin', 'albumin, serum'],
    'protein_total': ['total protein', 'protein', 'tp', 'protein, total'],
    'ggt': ['ggt', 'gamma-glutamyl transferase', 'gamma gt', 'ggtp'],

    # Complete Blood Count
    'wbc': ['wbc', 'white blood cells', 'white blood cell count', 'leukocytes', 'white blood cell'],
    'rbc': ['rbc', 'red blood cells', 'red blood cell count', 'erythrocytes', 'red blood cell'],
    'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
    'hematocrit': ['hematocrit', 'hct'],
    'platelets': ['platelets', 'plt', 'platelet count', 'thrombocytes', 'platelet'],
    'mcv': ['mcv', 'mean corpuscular volume'],
    'mch': ['mch', 'mean corpuscular hemoglobin'],
    'mchc': ['mchc', 'mean corpuscular hemoglobin concentration'],
    'rdw': ['rdw', 'red cell distribution width'],
    'neutrophils': ['neutrophils', 'neut', 'neutrophil', 'neutrophils %', 'neutrophils, absolute'],
    'lymphocytes': ['lymphocytes', 'lymph', 'lymphocyte', 'lymphocytes %'],
    'monocytes': ['monocytes', 'mono', 'monocyte', 'monocytes %'],
    'eosinophils': ['eosinophils', 'eos', 'eosinophil', 'eosinophils %'],
    'basophils': ['basophils', 'baso', 'basophil', 'basophils %'],

    # Cardiac
    'troponin': ['troponin', 'troponin i', 'troponin t', 'tni', 'tnt', 'hs-troponin', 'troponin-i', 'troponin-t'],
    'bnp': ['bnp', 'b-type natriuretic peptide', 'nt-probnp', 'pro-bnp', 'brain natriuretic peptide'],
    'ck': ['ck', 'creatine kinase', 'cpk'],
    'ck_mb': ['ck-mb', 'ckmb', 'creatine kinase mb', 'ck mb'],
    'ldh': ['ldh', 'lactate dehydrogenase'],

    # Metabolic
    'glucose': ['glucose', 'blood glucose', 'serum glucose', 'fasting glucose', 'random glucose', 'glucose, serum'],
    'hba1c': ['hba1c', 'a1c', 'hemoglobin a1c', 'glycated hemoglobin', 'glycohemoglobin'],
    'cholesterol': ['cholesterol', 'total cholesterol', 'cholesterol, total'],
    'ldl': ['ldl', 'ldl cholesterol', 'ldl-c', 'ldl cholesterol, calculated'],
    'hdl': ['hdl', 'hdl cholesterol', 'hdl-c'],
    'triglycerides': ['triglycerides', 'trig', 'tg'],
    'vldl': ['vldl', 'vldl cholesterol'],

    # Inflammatory
    'crp': ['crp', 'c-reactive protein', 'hs-crp', 'c reactive protein'],
    'esr': ['esr', 'sed rate', 'erythrocyte sedimentation rate'],
    'procalcitonin': ['procalcitonin', 'pct'],
    'ferritin': ['ferritin', 'ferritin, serum'],
    'lactate': ['lactate', 'lactic acid', 'lactate, plasma'],

    # Coagulation
    'pt': ['pt', 'prothrombin time'],
    'inr': ['inr', 'international normalized ratio'],
    'ptt': ['ptt', 'aptt', 'partial thromboplastin time', 'activated partial thromboplastin time'],
    'fibrinogen': ['fibrinogen', 'fib'],
    'd_dimer': ['d-dimer', 'ddimer', 'd dimer', 'd-dimer, quantitative'],

    # Thyroid
    'tsh': ['tsh', 'thyroid stimulating hormone'],
    't4': ['t4', 'thyroxine', 'free t4', 't4 free', 't4, free'],
    't3': ['t3', 'triiodothyronine', 'free t3', 't3 free', 't3, free'],

    # Iron Studies
    'iron': ['iron', 'serum iron', 'iron, serum'],
    'tibc': ['tibc', 'total iron binding capacity'],
    'transferrin': ['transferrin', 'transferrin, serum'],
    'transferrin_sat': ['transferrin saturation', 'iron saturation', 'tsat'],

    # Urinalysis
    'urine_protein': ['urine protein', 'protein urine', 'proteinuria', 'protein, urine'],
    'urine_glucose': ['urine glucose', 'glucose urine', 'glucosuria', 'glucose, urine'],
    'urine_blood': ['urine blood', 'blood urine', 'hematuria', 'blood, urine'],
    'urine_ph': ['urine ph', 'ph urine', 'ph, urine'],
    'specific_gravity': ['specific gravity', 'urine specific gravity', 'sp gravity'],

    # Vitamins
    'vitamin_d': ['vitamin d', 'vit d', '25-hydroxy vitamin d', 'vitamin d, 25-hydroxy'],
    'vitamin_b12': ['vitamin b12', 'b12', 'cobalamin'],
    'folate': ['folate', 'folic acid'],

    # Vitals (if included in lab feed)
    'heart_rate': ['heart rate', 'hr', 'pulse', 'pulse rate'],
    'blood_pressure_systolic': ['systolic', 'sbp', 'systolic bp', 'blood pressure systolic', 'bp systolic'],
    'blood_pressure_diastolic': ['diastolic', 'dbp', 'diastolic bp', 'blood pressure diastolic', 'bp diastolic'],
    'temperature': ['temperature', 'temp', 'body temperature'],
    'respiratory_rate': ['respiratory rate', 'rr', 'resp rate', 'respirations'],
    'spo2': ['spo2', 'oxygen saturation', 'o2 sat', 'pulse ox', 'o2 saturation'],
    'weight': ['weight', 'wt', 'body weight'],
    'height': ['height', 'ht'],
    'bmi': ['bmi', 'body mass index'],
}


def normalize_analyte_name(analyte: str) -> str:
    """
    Convert various lab test names to standardized column names.

    Examples:
        'Creatinine, Serum' -> 'creatinine'
        'Blood Urea Nitrogen' -> 'bun'
        'AST (SGOT)' -> 'ast'
    """
    if not analyte:
        return 'unknown'

    analyte_lower = analyte.lower().strip()

    # Remove common suffixes/prefixes
    analyte_lower = analyte_lower.replace(', serum', '').replace(', plasma', '')
    analyte_lower = analyte_lower.replace('serum ', '').replace('plasma ', '')
    analyte_lower = analyte_lower.strip()

    for normalized_name, variations in ANALYTE_NORMALIZATION.items():
        if analyte_lower in variations:
            return normalized_name
        # Also check if any variation is contained in the analyte name
        for var in variations:
            if var == analyte_lower or (len(var) > 3 and var in analyte_lower):
                return normalized_name

    # If not found, clean up the name and use as-is
    cleaned = analyte_lower.replace(' ', '_').replace('-', '_').replace(',', '').replace('(', '').replace(')', '')
    return cleaned


# ============================================================================
# FORMAT DETECTION
# ============================================================================

def get_patient_id_column(df: pd.DataFrame) -> Optional[str]:
    """Find the patient ID column in a DataFrame."""
    patient_id_candidates = [
        'patient_id', 'patientid', 'patient', 'id', 'mrn',
        'subject_id', 'subjectid', 'pt_id', 'patient_name',
        'name', 'hadm_id', 'encounter_id', 'visit_id'
    ]

    columns_lower = {col.lower(): col for col in df.columns}

    for candidate in patient_id_candidates:
        if candidate in columns_lower:
            return columns_lower[candidate]

    return None


def detect_data_format(df: pd.DataFrame) -> str:
    """
    Detect if DataFrame is in LONG or WIDE format.

    LONG format indicators:
    - Has columns like 'analyte', 'test_name', 'lab_name', 'component'
    - Has a 'value' or 'result' column
    - Same patient_id appears in many rows
    - Few columns (typically 5-15)
    - Many rows relative to unique patients

    WIDE format indicators:
    - Has columns named after biomarkers (creatinine, glucose, potassium, etc.)
    - Each patient_id appears once (or once per date)
    - Many columns (typically 15+)
    - Rows approx equal to unique patients

    Returns: 'long', 'wide', or 'unknown'
    """
    if df.empty:
        return 'unknown'

    columns_lower = [col.lower() for col in df.columns]

    # Find patient ID column for ratio calculations
    patient_col = get_patient_id_column(df)

    # LONG format indicators
    long_analyte_cols = [
        'analyte', 'test_name', 'test', 'lab_name', 'lab',
        'component', 'component_name', 'observation', 'loinc_name',
        'result_name', 'test_code', 'orderable', 'lab_test'
    ]
    long_value_cols = [
        'value', 'result', 'result_value', 'numeric_value',
        'observation_value', 'lab_value', 'test_result'
    ]

    has_analyte_col = any(col in columns_lower for col in long_analyte_cols)
    has_value_col = any(col in columns_lower for col in long_value_cols)

    # Calculate row ratio (many rows per patient = long format)
    if patient_col:
        unique_patients = df[patient_col].nunique()
        row_ratio = len(df) / max(unique_patients, 1)
        high_row_ratio = row_ratio > 3
    else:
        high_row_ratio = False

    # WIDE format indicators - check for known biomarker columns
    known_biomarkers = list(ANALYTE_NORMALIZATION.keys())
    biomarker_col_count = sum(1 for col in columns_lower if col in known_biomarkers)
    has_biomarker_columns = biomarker_col_count >= 3

    # Reasonable row ratio for wide (close to 1:1 patient to row)
    if patient_col:
        reasonable_row_ratio = row_ratio <= 2
    else:
        reasonable_row_ratio = True

    # Score calculation
    long_score = sum([has_analyte_col, has_value_col, high_row_ratio])
    wide_score = sum([has_biomarker_columns, reasonable_row_ratio])

    # Decision logic
    if long_score >= 2 and long_score > wide_score:
        return 'long'
    elif wide_score >= 2 or biomarker_col_count >= 5:
        return 'wide'
    elif has_analyte_col and has_value_col:
        return 'long'  # Strong indicators even without high ratio
    else:
        return 'unknown'


# ============================================================================
# COLUMN IDENTIFICATION
# ============================================================================

def identify_long_format_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Identify the role of each column in long-format data.

    Returns dict with:
    - patient_id_col: Column identifying the patient
    - date_col: Column with test date
    - analyte_col: Column with test/analyte name
    - value_col: Column with the result value
    - unit_col: Column with units (optional)
    - reference_col: Column with reference range (optional)
    - abnormal_col: Column with abnormal flag (optional)
    """
    columns_lower = {col.lower(): col for col in df.columns}
    result = {}

    # Patient ID column
    patient_id_candidates = [
        'patient_id', 'patientid', 'mrn', 'patient_name',
        'subject_id', 'id', 'patient', 'pt_id', 'name'
    ]
    for candidate in patient_id_candidates:
        if candidate in columns_lower:
            result['patient_id_col'] = columns_lower[candidate]
            break

    # Date column
    date_candidates = [
        'date', 'performed_on', 'test_date', 'collection_date',
        'result_date', 'observation_date', 'specimen_date',
        'collected_on', 'lab_date', 'datetime', 'collected_date',
        'order_date', 'report_date'
    ]
    for candidate in date_candidates:
        if candidate in columns_lower:
            result['date_col'] = columns_lower[candidate]
            break

    # Analyte/Test name column
    analyte_candidates = [
        'analyte', 'test_name', 'test', 'lab_name', 'component',
        'component_name', 'observation', 'loinc_name', 'orderable',
        'result_name', 'lab', 'test_code', 'lab_test'
    ]
    for candidate in analyte_candidates:
        if candidate in columns_lower:
            result['analyte_col'] = columns_lower[candidate]
            break

    # Value column
    value_candidates = [
        'value', 'result', 'result_value', 'numeric_value',
        'observation_value', 'lab_value', 'test_result',
        'result_numeric', 'numeric_result'
    ]
    for candidate in value_candidates:
        if candidate in columns_lower:
            result['value_col'] = columns_lower[candidate]
            break

    # Unit column (optional)
    unit_candidates = ['unit', 'units', 'uom', 'unit_of_measure']
    for candidate in unit_candidates:
        if candidate in columns_lower:
            result['unit_col'] = columns_lower[candidate]
            break

    # Reference range column (optional)
    ref_candidates = [
        'reference_range', 'reference', 'ref_range', 'normal_range',
        'reference_low', 'reference_high', 'ref_low', 'ref_high'
    ]
    for candidate in ref_candidates:
        if candidate in columns_lower:
            result['reference_col'] = columns_lower[candidate]
            break

    # Abnormal flag column (optional)
    abnormal_candidates = [
        'abnormality', 'abnormal', 'flag', 'abnormal_flag',
        'result_flag', 'interpretation'
    ]
    for candidate in abnormal_candidates:
        if candidate in columns_lower:
            result['abnormal_col'] = columns_lower[candidate]
            break

    return result


# ============================================================================
# PIVOT TRANSFORMATION
# ============================================================================

def transform_long_to_wide(
    df: pd.DataFrame,
    columns: Dict[str, Optional[str]] = None,
    include_previous: bool = True
) -> pd.DataFrame:
    """
    Transform long-format lab data to wide format.

    Args:
        df: Input DataFrame in long format
        columns: Dict mapping column roles (from identify_long_format_columns)
        include_previous: If True, also create _previous columns for trend analysis

    Returns:
        DataFrame in wide format, one row per patient-date combination
    """
    if columns is None:
        columns = identify_long_format_columns(df)

    # Extract column names
    patient_col = columns.get('patient_id_col')
    date_col = columns.get('date_col')
    analyte_col = columns.get('analyte_col')
    value_col = columns.get('value_col')

    if not all([patient_col, analyte_col, value_col]):
        raise ValueError(
            f"Missing required columns. Found: patient_id={patient_col}, "
            f"analyte={analyte_col}, value={value_col}"
        )

    logger.info(f"Transforming long-to-wide: patient={patient_col}, date={date_col}, "
                f"analyte={analyte_col}, value={value_col}")

    # Make a copy
    df_work = df.copy()

    # Normalize analyte names
    df_work['_normalized_analyte'] = df_work[analyte_col].apply(normalize_analyte_name)

    # Convert value to numeric where possible
    df_work['_numeric_value'] = pd.to_numeric(df_work[value_col], errors='coerce')

    # If no date column, create a dummy one (single timepoint)
    if not date_col:
        df_work['_date'] = pd.Timestamp('2024-01-01')
    else:
        # Parse dates
        df_work['_date'] = pd.to_datetime(df_work[date_col], errors='coerce')
        # Fill NaT with a default date
        df_work['_date'] = df_work['_date'].fillna(pd.Timestamp('2024-01-01'))

    # Remove rows with null analyte or value
    df_work = df_work.dropna(subset=['_normalized_analyte', '_numeric_value'])

    if df_work.empty:
        logger.warning("No valid data after cleaning")
        return pd.DataFrame()

    # Group by patient and date, then pivot
    try:
        pivot_df = df_work.pivot_table(
            index=[patient_col, '_date'],
            columns='_normalized_analyte',
            values='_numeric_value',
            aggfunc='first'  # If multiple values for same test on same day, take first
        ).reset_index()
    except Exception as e:
        logger.error(f"Pivot failed: {e}")
        raise

    # Rename columns
    pivot_df = pivot_df.rename(columns={patient_col: 'patient_id', '_date': 'date'})

    # Sort by patient and date
    pivot_df = pivot_df.sort_values(['patient_id', 'date']).reset_index(drop=True)

    # Create visit-level patient IDs
    pivot_df['visit_number'] = pivot_df.groupby('patient_id').cumcount() + 1
    pivot_df['original_patient_id'] = pivot_df['patient_id']
    pivot_df['patient_visit_id'] = (
        pivot_df['patient_id'].astype(str).str.replace(' ', '_') +
        '_v' + pivot_df['visit_number'].astype(str)
    )

    # Add _previous columns for trend analysis
    if include_previous:
        biomarker_cols = [col for col in pivot_df.columns
                          if col not in ['patient_id', 'date', 'visit_number',
                                        'patient_visit_id', 'original_patient_id']]

        for col in biomarker_cols:
            pivot_df[f'{col}_previous'] = pivot_df.groupby('original_patient_id')[col].shift(1)

    logger.info(f"Transformed {len(df)} long rows to {len(pivot_df)} wide rows "
                f"({pivot_df['original_patient_id'].nunique()} patients)")

    return pivot_df


# ============================================================================
# DEDUPLICATION
# ============================================================================

def deduplicate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows before analysis.

    Handles:
    - Explicit 'duplicate_of' column marking duplicates
    - True duplicates (same patient + date + all values)
    """
    if df.empty:
        return df

    original_len = len(df)

    # Check for explicit duplicate_of column
    duplicate_col = None
    for col in df.columns:
        if col.lower() == 'duplicate_of':
            duplicate_col = col
            break

    if duplicate_col:
        # Keep only rows where duplicate_of is empty/null
        mask = df[duplicate_col].isna() | (df[duplicate_col].astype(str).str.strip() == '')
        df = df[mask].copy()
        logger.info(f"Removed {original_len - len(df)} rows marked as duplicates")
        return df

    # Otherwise, deduplicate by patient + date + key values
    patient_col = get_patient_id_column(df)

    # Find date column
    date_col = None
    date_candidates = ['date', 'performed_on', 'test_date', 'collection_date']
    for col in df.columns:
        if col.lower() in date_candidates:
            date_col = col
            break

    if patient_col and date_col:
        df = df.drop_duplicates(subset=[patient_col, date_col], keep='first')
    elif patient_col:
        # If no date, keep first row per patient for wide format
        # But don't deduplicate if this looks like long format (analyte column exists)
        analyte_cols = ['analyte', 'test_name', 'test', 'lab_name', 'component']
        has_analyte = any(col.lower() in analyte_cols for col in df.columns)

        if not has_analyte:
            df = df.drop_duplicates(subset=[patient_col], keep='first')

    if len(df) < original_len:
        logger.info(f"Removed {original_len - len(df)} duplicate rows")

    return df


# ============================================================================
# MAIN TRANSFORMATION ENTRY POINT
# ============================================================================

def auto_transform_data(
    df: pd.DataFrame,
    force_transform: bool = False
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Automatically detect and transform data format if needed.

    Args:
        df: Input DataFrame
        force_transform: If True, transform even if format is detected as wide

    Returns:
        Tuple of (transformed_df, metadata_dict)
    """
    original_rows = len(df)
    original_cols = len(df.columns)

    # Deduplicate first (for wide format data with duplicate_of column)
    df = deduplicate_data(df)
    rows_after_dedup = len(df)

    # Detect format
    data_format = detect_data_format(df)
    logger.info(f"Detected data format: {data_format}")

    metadata = {
        'original_format': data_format,
        'transformed': False,
        'original_rows': original_rows,
        'original_columns': original_cols,
        'rows_after_dedup': rows_after_dedup,
        'duplicates_removed': original_rows - rows_after_dedup,
        'result_rows': rows_after_dedup,
        'unique_patients': None,
        'visits_detected': None,
        'columns_identified': None
    }

    if data_format == 'long' or force_transform:
        # Transform long to wide
        columns = identify_long_format_columns(df)
        metadata['columns_identified'] = columns

        try:
            wide_df = transform_long_to_wide(df, columns, include_previous=True)

            if not wide_df.empty:
                # Use visit-level IDs for analysis
                wide_df['patient_id'] = wide_df['patient_visit_id']

                metadata['transformed'] = True
                metadata['result_rows'] = len(wide_df)
                metadata['unique_patients'] = wide_df['original_patient_id'].nunique()
                metadata['visits_detected'] = len(wide_df)
                metadata['biomarker_columns'] = [
                    col for col in wide_df.columns
                    if col not in ['patient_id', 'date', 'visit_number',
                                   'patient_visit_id', 'original_patient_id']
                    and not col.endswith('_previous')
                ]

                return wide_df, metadata
            else:
                logger.warning("Transformation produced empty result, returning original")
                return df, metadata

        except Exception as e:
            logger.error(f"Transformation failed: {e}")
            metadata['error'] = str(e)
            return df, metadata

    # Data is already wide or unknown - return as-is
    patient_col = get_patient_id_column(df)
    if patient_col:
        metadata['unique_patients'] = df[patient_col].nunique()

    return df, metadata
