"""
Disease Detection Module with Proper Validation
================================================

CRITICAL FIX: Diseases are only detected when biomarker values are ACTUALLY ABNORMAL.
This eliminates false positives like "93% AKI" when creatinine is 1.04 (normal).

The Problem (Before):
    Input: creatinine=1.04, BUN=11, eGFR=89 (ALL NORMAL)
    Output: Acute Kidney Injury 93% confidence [WRONG]

The Fix (After):
    Input: creatinine=1.04, BUN=11, eGFR=89 (ALL NORMAL)
    Output: No kidney disease detected [CORRECT]
"""

from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# REFERENCE RANGES FOR ALL BIOMARKERS
# =============================================================================

REFERENCE_RANGES = {
    # Kidney Function
    'creatinine': {'low': 0.6, 'high': 1.3, 'unit': 'mg/dL', 'critical_high': 2.0, 'critical_low': 0.3},
    'bun': {'low': 7, 'high': 20, 'unit': 'mg/dL', 'critical_high': 40},
    'egfr': {'low': 60, 'high': 999, 'unit': 'mL/min/1.73m2', 'critical_low': 15},

    # Liver Function
    'ast': {'low': 0, 'high': 40, 'unit': 'IU/L', 'critical_high': 120},
    'alt': {'low': 0, 'high': 44, 'unit': 'IU/L', 'critical_high': 132},
    'alkaline_phosphatase': {'low': 44, 'high': 147, 'unit': 'IU/L', 'critical_high': 500},
    'alp': {'low': 44, 'high': 147, 'unit': 'IU/L', 'critical_high': 500},
    'bilirubin': {'low': 0.1, 'high': 1.2, 'unit': 'mg/dL', 'critical_high': 3.0},
    'albumin': {'low': 3.5, 'high': 5.0, 'unit': 'g/dL', 'critical_low': 2.5},
    'ggt': {'low': 0, 'high': 60, 'unit': 'IU/L', 'critical_high': 180},

    # Metabolic
    'glucose': {'low': 70, 'high': 99, 'unit': 'mg/dL', 'prediabetic': 100, 'diabetic': 126, 'critical_low': 50, 'critical_high': 400},
    'hba1c': {'low': 4.0, 'high': 5.6, 'unit': '%', 'prediabetic': 5.7, 'diabetic': 6.5},

    # Electrolytes
    'sodium': {'low': 136, 'high': 145, 'unit': 'mmol/L', 'critical_low': 125, 'critical_high': 155},
    'potassium': {'low': 3.5, 'high': 5.0, 'unit': 'mEq/L', 'critical_low': 2.5, 'critical_high': 6.5},
    'chloride': {'low': 96, 'high': 106, 'unit': 'mmol/L'},
    'calcium': {'low': 8.5, 'high': 10.5, 'unit': 'mg/dL', 'critical_low': 7.0, 'critical_high': 12.0},
    'magnesium': {'low': 1.7, 'high': 2.2, 'unit': 'mg/dL'},
    'phosphorus': {'low': 2.5, 'high': 4.5, 'unit': 'mg/dL'},
    'bicarbonate': {'low': 22, 'high': 28, 'unit': 'mEq/L'},

    # Complete Blood Count
    'wbc': {'low': 4.5, 'high': 11.0, 'unit': 'K/uL', 'critical_low': 2.0, 'critical_high': 30.0},
    'hemoglobin': {'low': 12.0, 'high': 17.5, 'unit': 'g/dL', 'critical_low': 7.0},
    'hematocrit': {'low': 36, 'high': 50, 'unit': '%', 'critical_low': 21},
    'platelets': {'low': 150, 'high': 400, 'unit': 'K/uL', 'critical_low': 50, 'critical_high': 1000},
    'mcv': {'low': 80, 'high': 100, 'unit': 'fL'},
    'mch': {'low': 27, 'high': 33, 'unit': 'pg'},
    'mchc': {'low': 32, 'high': 36, 'unit': 'g/dL'},
    'rdw': {'low': 11.5, 'high': 14.5, 'unit': '%'},

    # Cardiac
    'troponin': {'low': 0, 'high': 0.04, 'unit': 'ng/mL', 'critical_high': 0.1},
    'bnp': {'low': 0, 'high': 100, 'unit': 'pg/mL', 'critical_high': 500},
    'ck': {'low': 30, 'high': 200, 'unit': 'IU/L'},
    'ck_mb': {'low': 0, 'high': 5, 'unit': 'ng/mL'},
    'ldh': {'low': 140, 'high': 280, 'unit': 'IU/L'},

    # Inflammatory
    'crp': {'low': 0, 'high': 3.0, 'unit': 'mg/L', 'critical_high': 10.0},
    'lactate': {'low': 0.5, 'high': 2.0, 'unit': 'mmol/L', 'critical_high': 4.0},
    'procalcitonin': {'low': 0, 'high': 0.1, 'unit': 'ng/mL', 'critical_high': 2.0},
    'esr': {'low': 0, 'high': 20, 'unit': 'mm/hr'},
    'ferritin': {'low': 12, 'high': 300, 'unit': 'ng/mL'},

    # Coagulation
    'pt': {'low': 11, 'high': 13.5, 'unit': 'seconds'},
    'inr': {'low': 0.9, 'high': 1.1, 'unit': 'ratio', 'therapeutic_low': 2.0, 'therapeutic_high': 3.0},
    'ptt': {'low': 25, 'high': 35, 'unit': 'seconds'},
    'fibrinogen': {'low': 200, 'high': 400, 'unit': 'mg/dL'},
    'd_dimer': {'low': 0, 'high': 0.5, 'unit': 'ug/mL'},

    # Thyroid
    'tsh': {'low': 0.4, 'high': 4.0, 'unit': 'mIU/L'},
    't4': {'low': 0.8, 'high': 1.8, 'unit': 'ng/dL'},
    't3': {'low': 80, 'high': 200, 'unit': 'ng/dL'},

    # Lipids
    'cholesterol': {'low': 0, 'high': 200, 'unit': 'mg/dL'},
    'ldl': {'low': 0, 'high': 100, 'unit': 'mg/dL'},
    'hdl': {'low': 40, 'high': 999, 'unit': 'mg/dL'},
    'triglycerides': {'low': 0, 'high': 150, 'unit': 'mg/dL'},

    # Iron Studies
    'iron': {'low': 60, 'high': 170, 'unit': 'ug/dL'},
    'tibc': {'low': 250, 'high': 370, 'unit': 'ug/dL'},
    'transferrin_sat': {'low': 20, 'high': 50, 'unit': '%'},
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_value(patient_data: dict, possible_names: list) -> Optional[float]:
    """
    Get a numeric value from patient data, trying multiple column name variations.
    Handles: creatinine, creatinine_value, Creatinine, CREATININE, etc.
    """
    if not patient_data:
        return None

    for name in possible_names:
        # Try exact match
        if name in patient_data:
            val = patient_data[name]
            if val is not None and val != '' and str(val).lower() not in ('nan', 'none', 'null'):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue

        # Try with _value suffix
        val_key = f'{name}_value'
        if val_key in patient_data:
            val = patient_data[val_key]
            if val is not None and val != '' and str(val).lower() not in ('nan', 'none', 'null'):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue

        # Try case-insensitive
        for key in patient_data.keys():
            key_lower = str(key).lower().replace('_value', '').replace('_result', '')
            if key_lower == name.lower():
                val = patient_data[key]
                if val is not None and val != '' and str(val).lower() not in ('nan', 'none', 'null'):
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue

    return None


def is_abnormal(value: float, marker: str) -> bool:
    """Check if a value is outside normal reference range."""
    if value is None:
        return False

    ref = REFERENCE_RANGES.get(marker.lower())
    if not ref:
        return False

    return value < ref['low'] or value > ref['high']


def get_abnormality_level(value: float, marker: str) -> str:
    """
    Returns: 'normal', 'borderline', 'abnormal', 'critical'
    """
    if value is None:
        return 'unknown'

    ref = REFERENCE_RANGES.get(marker.lower())
    if not ref:
        return 'unknown'

    low = ref['low']
    high = ref['high']
    critical_low = ref.get('critical_low', low * 0.5)
    critical_high = ref.get('critical_high', high * 2)

    # Check normal range
    if low <= value <= high:
        return 'normal'

    # Check critical range
    if value < critical_low or value > critical_high:
        return 'critical'

    # Check how far outside normal (borderline vs abnormal)
    if value < low:
        pct_below = ((low - value) / max(low, 0.01)) * 100
        return 'abnormal' if pct_below > 15 else 'borderline'
    else:
        pct_above = ((value - high) / max(high, 0.01)) * 100
        return 'abnormal' if pct_above > 15 else 'borderline'


# =============================================================================
# DISEASE DETECTION WITH VALIDATION
# =============================================================================

def detect_acute_kidney_injury(patient_data: dict) -> dict:
    """
    Detect AKI ONLY when kidney markers are ACTUALLY ABNORMAL.

    KDIGO Criteria for AKI:
    - Creatinine increase >=0.3 mg/dL within 48h, OR
    - Creatinine increase >=1.5x baseline within 7 days, OR
    - Urine output <0.5 mL/kg/h for 6 hours

    For single-timepoint (no baseline), use:
    - Creatinine > 1.3 mg/dL (above normal), OR
    - eGFR < 60 mL/min
    """

    result = {
        'disease': 'Acute Kidney Injury',
        'icd10': 'N17.9',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    # Get values
    creatinine = get_value(patient_data, ['creatinine', 'cr', 'serum_creatinine'])
    creatinine_prev = get_value(patient_data, ['creatinine_previous', 'creatinine_baseline', 'baseline_creatinine'])
    bun = get_value(patient_data, ['bun', 'blood_urea_nitrogen', 'urea'])
    egfr = get_value(patient_data, ['egfr', 'gfr', 'estimated_gfr'])

    # =========================================================================
    # CRITICAL VALIDATION: Check if values are actually abnormal
    # =========================================================================

    creatinine_abnormal = False
    egfr_abnormal = False
    bun_abnormal = False

    if creatinine is not None:
        creatinine_abnormal = creatinine > 1.3  # Above normal range

    if egfr is not None:
        egfr_abnormal = egfr < 60  # Below normal

    if bun is not None:
        bun_abnormal = bun > 20  # Above normal range

    # =========================================================================
    # IF ALL KIDNEY MARKERS ARE NORMAL -> NO AKI
    # =========================================================================

    if not creatinine_abnormal and not egfr_abnormal and not bun_abnormal:
        result['confidence'] = 0.0
        result['detected'] = False
        evidence = []
        if creatinine is not None:
            evidence.append(f'Creatinine {creatinine} mg/dL - within normal range (0.6-1.3)')
        if egfr is not None:
            evidence.append(f'eGFR {egfr} mL/min - normal kidney function (>=60)')
        if bun is not None:
            evidence.append(f'BUN {bun} mg/dL - within normal range (7-20)')
        evidence.append('No evidence of acute kidney injury')
        result['evidence'] = evidence
        return result

    # =========================================================================
    # CALCULATE CONFIDENCE BASED ON SEVERITY OF ABNORMALITY
    # =========================================================================

    confidence = 0.0

    # Creatinine scoring
    if creatinine is not None:
        if creatinine > 4.0:
            confidence += 0.40
            result['evidence'].append(f'Creatinine severely elevated: {creatinine} mg/dL')
        elif creatinine > 2.5:
            confidence += 0.30
            result['evidence'].append(f'Creatinine significantly elevated: {creatinine} mg/dL')
        elif creatinine > 1.8:
            confidence += 0.20
            result['evidence'].append(f'Creatinine elevated: {creatinine} mg/dL')
        elif creatinine > 1.3:
            confidence += 0.10
            result['evidence'].append(f'Creatinine mildly elevated: {creatinine} mg/dL')

    # eGFR scoring
    if egfr is not None:
        if egfr < 15:
            confidence += 0.40
            result['evidence'].append(f'eGFR severely reduced: {egfr} mL/min (kidney failure)')
        elif egfr < 30:
            confidence += 0.30
            result['evidence'].append(f'eGFR significantly reduced: {egfr} mL/min')
        elif egfr < 45:
            confidence += 0.20
            result['evidence'].append(f'eGFR moderately reduced: {egfr} mL/min')
        elif egfr < 60:
            confidence += 0.10
            result['evidence'].append(f'eGFR mildly reduced: {egfr} mL/min')

    # BUN scoring (supportive, not primary)
    if bun_abnormal and (creatinine_abnormal or egfr_abnormal):
        if bun > 40:
            confidence += 0.10
            result['evidence'].append(f'BUN elevated: {bun} mg/dL')
        elif bun > 25:
            confidence += 0.05

    # Acute change scoring
    if creatinine is not None and creatinine_prev is not None and creatinine_prev > 0:
        absolute_change = creatinine - creatinine_prev
        relative_change = creatinine / creatinine_prev

        if absolute_change >= 0.3:
            confidence += 0.20
            result['evidence'].append(f'Acute creatinine rise: +{absolute_change:.2f} mg/dL from baseline')

        if relative_change >= 1.5:
            confidence += 0.15
            result['evidence'].append(f'Creatinine {relative_change:.1f}x baseline')

    result['confidence'] = min(confidence, 0.95)
    result['detected'] = confidence >= 0.25  # Require at least 25% confidence

    return result


def detect_chronic_kidney_disease(patient_data: dict) -> dict:
    """
    Detect CKD based on KDIGO criteria.

    CKD Definition:
    - eGFR < 60 mL/min/1.73m2 for >=3 months, OR
    - Markers of kidney damage (albuminuria, structural abnormalities)
    """

    result = {
        'disease': 'Chronic Kidney Disease',
        'icd10': 'N18.9',
        'confidence': 0.0,
        'evidence': [],
        'stage': None,
        'detected': False
    }

    egfr = get_value(patient_data, ['egfr', 'gfr', 'estimated_gfr'])

    # =========================================================================
    # CRITICAL: CKD REQUIRES eGFR < 60
    # =========================================================================

    if egfr is None:
        result['evidence'] = ['eGFR not available - cannot assess CKD']
        return result

    if egfr >= 60:
        result['confidence'] = 0.0
        result['detected'] = False
        result['evidence'] = [f'eGFR {egfr} mL/min indicates normal kidney function (>=60)']
        return result

    # =========================================================================
    # STAGE AND CONFIDENCE BASED ON eGFR
    # =========================================================================

    if egfr >= 45:
        result['stage'] = 'Stage 3a (mild-moderate)'
        result['confidence'] = 0.50
        result['icd10'] = 'N18.31'
    elif egfr >= 30:
        result['stage'] = 'Stage 3b (moderate-severe)'
        result['confidence'] = 0.70
        result['icd10'] = 'N18.32'
    elif egfr >= 15:
        result['stage'] = 'Stage 4 (severe)'
        result['confidence'] = 0.85
        result['icd10'] = 'N18.4'
    else:
        result['stage'] = 'Stage 5 (kidney failure)'
        result['confidence'] = 0.95
        result['icd10'] = 'N18.5'

    result['evidence'] = [f'eGFR {egfr} mL/min - {result["stage"]}']
    result['detected'] = True

    return result


def detect_diabetes(patient_data: dict) -> dict:
    """
    Detect diabetes and pre-diabetes based on glucose or HbA1c.

    Pre-diabetes criteria:
    - Fasting glucose 100-125 mg/dL, OR
    - HbA1c 5.7-6.4%

    Diabetes criteria:
    - Fasting glucose >=126 mg/dL, OR
    - HbA1c >=6.5%
    """

    result = {
        'disease': 'Pre-diabetes',
        'icd10': 'R73.03',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    glucose = get_value(patient_data, ['glucose', 'fasting_glucose', 'blood_glucose'])
    hba1c = get_value(patient_data, ['hba1c', 'a1c', 'hemoglobin_a1c', 'glycated_hemoglobin'])

    # Check glucose
    if glucose is not None:
        if glucose >= 126:
            result['disease'] = 'Type 2 Diabetes Mellitus'
            result['icd10'] = 'E11.9'
            result['confidence'] = 0.85
            result['evidence'].append(f'Glucose {glucose} mg/dL - diabetic range (>=126)')
            result['detected'] = True
        elif glucose >= 100:
            result['confidence'] = max(result['confidence'], 0.70)
            result['evidence'].append(f'Glucose {glucose} mg/dL - pre-diabetic range (100-125)')
            result['detected'] = True
        else:
            result['evidence'].append(f'Glucose {glucose} mg/dL - normal (<100)')

    # Check HbA1c
    if hba1c is not None:
        if hba1c >= 6.5:
            result['disease'] = 'Type 2 Diabetes Mellitus'
            result['icd10'] = 'E11.9'
            result['confidence'] = 0.90
            result['evidence'].append(f'HbA1c {hba1c}% - diabetic range (>=6.5%)')
            result['detected'] = True
        elif hba1c >= 5.7:
            result['confidence'] = max(result['confidence'], 0.75)
            result['evidence'].append(f'HbA1c {hba1c}% - pre-diabetic range (5.7-6.4%)')
            result['detected'] = True

    if not result['detected']:
        result['evidence'] = ['Glucose and HbA1c within normal limits']

    return result


def detect_liver_disease(patient_data: dict) -> dict:
    """
    Detect elevated liver enzymes and liver disease.
    """

    result = {
        'disease': 'Elevated Liver Enzymes',
        'icd10': 'R74.0',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    ast = get_value(patient_data, ['ast', 'sgot', 'ast_sgot', 'aspartate_aminotransferase'])
    alt = get_value(patient_data, ['alt', 'sgpt', 'alt_sgpt', 'alanine_aminotransferase'])
    alp = get_value(patient_data, ['alkaline_phosphatase', 'alk_phos', 'alp'])
    bilirubin = get_value(patient_data, ['bilirubin', 'total_bilirubin', 'bilirubin_total'])
    albumin = get_value(patient_data, ['albumin', 'alb'])

    abnormal_count = 0

    if ast is not None and ast > 40:
        abnormal_count += 1
        result['evidence'].append(f'AST elevated: {ast} IU/L (normal 0-40)')

    if alt is not None and alt > 44:
        abnormal_count += 1
        result['evidence'].append(f'ALT elevated: {alt} IU/L (normal 0-44)')

    if alp is not None and alp > 147:
        abnormal_count += 1
        result['evidence'].append(f'Alkaline Phosphatase elevated: {alp} IU/L (normal 44-147)')

    if bilirubin is not None and bilirubin > 1.2:
        abnormal_count += 1
        result['evidence'].append(f'Bilirubin elevated: {bilirubin} mg/dL (normal 0-1.2)')

    if albumin is not None and albumin < 3.5:
        abnormal_count += 1
        result['evidence'].append(f'Albumin low: {albumin} g/dL (normal 3.5-5.0)')

    if abnormal_count == 0:
        result['evidence'] = ['Liver enzymes within normal limits']
        return result

    result['detected'] = True
    result['confidence'] = min(0.3 + (abnormal_count * 0.15), 0.90)

    return result


def detect_anemia(patient_data: dict) -> dict:
    """
    Detect anemia based on hemoglobin/hematocrit levels.
    """

    result = {
        'disease': 'Anemia',
        'icd10': 'D64.9',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    hemoglobin = get_value(patient_data, ['hemoglobin', 'hgb', 'hb'])
    hematocrit = get_value(patient_data, ['hematocrit', 'hct'])

    if hemoglobin is not None:
        if hemoglobin < 7.0:
            result['confidence'] = 0.95
            result['evidence'].append(f'Hemoglobin severely low: {hemoglobin} g/dL (critical <7)')
            result['detected'] = True
        elif hemoglobin < 10.0:
            result['confidence'] = 0.80
            result['evidence'].append(f'Hemoglobin low: {hemoglobin} g/dL (moderate anemia)')
            result['detected'] = True
        elif hemoglobin < 12.0:
            result['confidence'] = 0.60
            result['evidence'].append(f'Hemoglobin mildly low: {hemoglobin} g/dL (mild anemia)')
            result['detected'] = True

    if hematocrit is not None and hematocrit < 36:
        if not result['detected']:
            result['confidence'] = 0.50
            result['evidence'].append(f'Hematocrit low: {hematocrit}% (normal 36-50%)')
            result['detected'] = True
        else:
            result['evidence'].append(f'Hematocrit confirms anemia: {hematocrit}%')

    if not result['detected']:
        result['evidence'] = ['Hemoglobin and hematocrit within normal limits']

    return result


def detect_hyperkalemia(patient_data: dict) -> dict:
    """
    Detect hyperkalemia (elevated potassium).
    """

    result = {
        'disease': 'Hyperkalemia',
        'icd10': 'E87.5',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    potassium = get_value(patient_data, ['potassium', 'k', 'serum_potassium'])

    if potassium is None:
        result['evidence'] = ['Potassium not available']
        return result

    if potassium > 6.5:
        result['confidence'] = 0.95
        result['evidence'].append(f'Potassium critically elevated: {potassium} mEq/L (critical >6.5)')
        result['detected'] = True
    elif potassium > 5.5:
        result['confidence'] = 0.80
        result['evidence'].append(f'Potassium elevated: {potassium} mEq/L (high 5.0-5.5)')
        result['detected'] = True
    elif potassium > 5.0:
        result['confidence'] = 0.50
        result['evidence'].append(f'Potassium borderline elevated: {potassium} mEq/L (upper limit 5.0)')
        result['detected'] = True
    else:
        result['evidence'] = [f'Potassium {potassium} mEq/L - within normal range (3.5-5.0)']

    return result


def detect_sepsis(patient_data: dict) -> dict:
    """
    Detect sepsis based on inflammatory markers and clinical criteria.
    """

    result = {
        'disease': 'Sepsis',
        'icd10': 'A41.9',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    wbc = get_value(patient_data, ['wbc', 'white_blood_cells', 'leukocytes'])
    lactate = get_value(patient_data, ['lactate', 'lactic_acid'])
    procalcitonin = get_value(patient_data, ['procalcitonin', 'pct'])
    crp = get_value(patient_data, ['crp', 'c_reactive_protein'])
    temperature = get_value(patient_data, ['temperature', 'temp', 'body_temperature'])

    score = 0

    if wbc is not None:
        if wbc > 12 or wbc < 4:
            score += 1
            direction = 'elevated' if wbc > 12 else 'low'
            result['evidence'].append(f'WBC {direction}: {wbc} K/uL')

    if lactate is not None and lactate > 2.0:
        score += 2  # Lactate elevation is more specific
        result['evidence'].append(f'Lactate elevated: {lactate} mmol/L (>2.0 suggests tissue hypoperfusion)')

    if procalcitonin is not None and procalcitonin > 0.5:
        score += 2
        result['evidence'].append(f'Procalcitonin elevated: {procalcitonin} ng/mL (suggests bacterial infection)')

    if crp is not None and crp > 10:
        score += 1
        result['evidence'].append(f'CRP elevated: {crp} mg/L')

    if temperature is not None:
        if temperature > 38.3 or temperature < 36:
            score += 1
            status = 'Fever' if temperature > 38.3 else 'Hypothermia'
            result['evidence'].append(f'{status}: {temperature}C')

    if score >= 3:
        result['confidence'] = min(0.4 + (score * 0.1), 0.90)
        result['detected'] = True
    elif score > 0:
        result['evidence'] = ['Some inflammatory markers elevated but insufficient for sepsis diagnosis']

    return result


def detect_cardiac_injury(patient_data: dict) -> dict:
    """
    Detect cardiac injury based on troponin and BNP.
    """

    result = {
        'disease': 'Acute Cardiac Injury',
        'icd10': 'I21.9',
        'confidence': 0.0,
        'evidence': [],
        'detected': False
    }

    troponin = get_value(patient_data, ['troponin', 'troponin_i', 'troponin_t', 'tni', 'tnt'])
    bnp = get_value(patient_data, ['bnp', 'nt_probnp', 'brain_natriuretic_peptide'])

    if troponin is not None and troponin > 0.04:
        if troponin > 0.5:
            result['confidence'] = 0.90
            result['evidence'].append(f'Troponin significantly elevated: {troponin} ng/mL (suggests MI)')
        elif troponin > 0.1:
            result['confidence'] = 0.75
            result['evidence'].append(f'Troponin elevated: {troponin} ng/mL')
        else:
            result['confidence'] = 0.50
            result['evidence'].append(f'Troponin mildly elevated: {troponin} ng/mL')
        result['detected'] = True

    if bnp is not None and bnp > 100:
        if bnp > 500:
            if not result['detected']:
                result['disease'] = 'Heart Failure'
                result['icd10'] = 'I50.9'
                result['confidence'] = 0.80
            result['evidence'].append(f'BNP significantly elevated: {bnp} pg/mL (suggests heart failure)')
            result['detected'] = True
        else:
            result['evidence'].append(f'BNP mildly elevated: {bnp} pg/mL')

    if not result['detected']:
        result['evidence'] = ['Cardiac markers within normal limits']

    return result


# =============================================================================
# MASTER DISEASE DETECTION FUNCTION
# =============================================================================

def detect_all_diseases(patient_data: dict) -> List[Dict]:
    """
    Run all disease detectors and return only those that are actually detected.
    """

    detectors = [
        detect_acute_kidney_injury,
        detect_chronic_kidney_disease,
        detect_diabetes,
        detect_liver_disease,
        detect_anemia,
        detect_hyperkalemia,
        detect_sepsis,
        detect_cardiac_injury,
    ]

    results = []

    for detector in detectors:
        try:
            result = detector(patient_data)
            if result.get('detected', False) and result.get('confidence', 0) > 0:
                results.append(result)
        except Exception as e:
            logger.error(f"Error in {detector.__name__}: {e}")
            continue

    # Sort by confidence (highest first)
    results.sort(key=lambda x: x.get('confidence', 0), reverse=True)

    return results


# =============================================================================
# ABNORMAL VALUES DETECTION
# =============================================================================

def detect_abnormal_values(patient_data: dict) -> List[Dict]:
    """
    Detect all abnormal lab values with proper status classification.
    """

    abnormal_values = []

    # Map of possible column names for each marker
    marker_aliases = {
        'creatinine': ['creatinine', 'cr', 'serum_creatinine'],
        'bun': ['bun', 'blood_urea_nitrogen', 'urea'],
        'egfr': ['egfr', 'gfr', 'estimated_gfr'],
        'glucose': ['glucose', 'fasting_glucose', 'blood_glucose'],
        'hba1c': ['hba1c', 'a1c', 'hemoglobin_a1c'],
        'sodium': ['sodium', 'na', 'serum_sodium'],
        'potassium': ['potassium', 'k', 'serum_potassium'],
        'chloride': ['chloride', 'cl'],
        'calcium': ['calcium', 'ca'],
        'magnesium': ['magnesium', 'mg'],
        'ast': ['ast', 'sgot'],
        'alt': ['alt', 'sgpt'],
        'alkaline_phosphatase': ['alkaline_phosphatase', 'alk_phos', 'alp'],
        'bilirubin': ['bilirubin', 'total_bilirubin'],
        'albumin': ['albumin', 'alb'],
        'wbc': ['wbc', 'white_blood_cells'],
        'hemoglobin': ['hemoglobin', 'hgb', 'hb'],
        'hematocrit': ['hematocrit', 'hct'],
        'platelets': ['platelets', 'plt'],
        'troponin': ['troponin', 'troponin_i', 'troponin_t'],
        'bnp': ['bnp', 'nt_probnp'],
        'lactate': ['lactate', 'lactic_acid'],
        'procalcitonin': ['procalcitonin', 'pct'],
        'crp': ['crp', 'c_reactive_protein'],
    }

    for marker, aliases in marker_aliases.items():
        value = get_value(patient_data, aliases)

        if value is None:
            continue

        ref = REFERENCE_RANGES.get(marker)
        if not ref:
            continue

        level = get_abnormality_level(value, marker)

        if level in ['borderline', 'abnormal', 'critical']:
            direction = 'high' if value > ref['high'] else 'low'

            abnormal_values.append({
                'marker': marker,
                'value': round(value, 2),
                'unit': ref.get('unit', ''),
                'reference': f"{ref['low']}-{ref['high']}",
                'status': level,
                'direction': direction
            })

    return abnormal_values


# =============================================================================
# CONVERGENCE DETECTION
# =============================================================================

SYSTEM_MAPPING = {
    'renal': ['creatinine', 'bun', 'egfr', 'potassium'],
    'hepatic': ['ast', 'alt', 'alkaline_phosphatase', 'bilirubin', 'albumin', 'ggt'],
    'cardiac': ['troponin', 'bnp', 'ck_mb', 'ck'],
    'metabolic': ['glucose', 'hba1c', 'sodium', 'potassium', 'calcium', 'bicarbonate'],
    'hematologic': ['wbc', 'hemoglobin', 'hematocrit', 'platelets'],
    'inflammatory': ['crp', 'lactate', 'procalcitonin', 'esr', 'ferritin']
}


def detect_convergence(patient_data: dict, abnormal_values: List[Dict], conditions: List[Dict]) -> Dict:
    """
    Detect multi-system convergence ONLY when multiple systems are actually abnormal.
    """

    result = {
        'type': 'none',
        'score': 0.0,
        'systems_involved': [],
        'explanation': ''
    }

    # Find which systems have abnormal markers
    affected_systems = set()

    for abnormal in abnormal_values:
        marker = abnormal['marker'].lower()
        for system, markers in SYSTEM_MAPPING.items():
            if marker in markers:
                affected_systems.add(system)

    # =========================================================================
    # CONVERGENCE REQUIRES 2+ SYSTEMS TO BE ABNORMAL
    # =========================================================================

    if len(affected_systems) < 2:
        result['type'] = 'none'
        result['score'] = 0.0
        result['explanation'] = 'No multi-system involvement detected'
        return result

    # Calculate severity based on number of systems and severity of abnormalities
    system_count = len(affected_systems)
    critical_count = sum(1 for a in abnormal_values if a.get('status') == 'critical')

    if system_count >= 4 or critical_count >= 2:
        result['type'] = 'critical'
        result['score'] = 0.85
    elif system_count >= 3 or critical_count >= 1:
        result['type'] = 'severe'
        result['score'] = 0.70
    elif system_count >= 2:
        result['type'] = 'moderate'
        result['score'] = 0.50

    result['systems_involved'] = list(affected_systems)
    result['explanation'] = f"Abnormalities detected in {', '.join(affected_systems)} systems"

    return result


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_patient_validated(patient_data: dict) -> Dict[str, Any]:
    """
    Analyze a single patient with proper validation.
    Returns only conditions that are actually detected based on abnormal values.
    """

    # Detect abnormal values first
    abnormal_values = detect_abnormal_values(patient_data)

    # Detect diseases (only if values are actually abnormal)
    conditions = detect_all_diseases(patient_data)

    # Detect convergence (only if 2+ systems involved)
    convergence = detect_convergence(patient_data, abnormal_values, conditions)

    # Calculate clinical state
    if convergence['type'] == 'critical' or any(c['confidence'] > 0.85 for c in conditions):
        clinical_state = 'S3'
        state_label = 'CRITICAL'
    elif convergence['type'] == 'severe' or any(c['confidence'] > 0.70 for c in conditions):
        clinical_state = 'S2'
        state_label = 'ESCALATING'
    elif len(abnormal_values) > 0 or len(conditions) > 0:
        clinical_state = 'S1'
        state_label = 'WATCH'
    else:
        clinical_state = 'S0'
        state_label = 'STABLE'

    # Calculate risk score
    risk_score = 0.0
    for condition in conditions:
        risk_score += condition.get('confidence', 0) * 0.3
    for abnormal in abnormal_values:
        if abnormal['status'] == 'critical':
            risk_score += 0.15
        elif abnormal['status'] == 'abnormal':
            risk_score += 0.08
        elif abnormal['status'] == 'borderline':
            risk_score += 0.03
    risk_score += convergence.get('score', 0) * 0.3
    risk_score = min(risk_score, 1.0)

    return {
        'patient_id': patient_data.get('patient_id', 'Unknown'),
        'clinical_state': clinical_state,
        'state_label': state_label,
        'risk_score': round(risk_score, 3),
        'conditions': conditions,
        'abnormal_values': abnormal_values,
        'convergence': convergence
    }
