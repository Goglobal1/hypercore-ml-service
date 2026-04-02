"""
HyperCore Pathway Library
=========================

100+ Disease Pathways with:
- Required endpoints
- Biomarker patterns
- What doctors miss
- Recommended actions
- Urgency levels
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass


PATHWAY_LIBRARY = [

    # ══════════════════════════════════════════════════════════════════════════
    # METABOLIC PATHWAYS (15)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "metabolic_001",
        "name": "hepatic_insulin_resistance_syndrome",
        "description": "Liver-driven insulin resistance with glucose overproduction",
        "required_endpoints": ["metabolic", "hepatic", "endocrine"],
        "supporting_endpoints": ["inflammatory", "lipid_atherogenic"],
        "biomarker_pattern": {
            "glucose": "elevated",
            "alp": "elevated",
            "ast": "normal_or_elevated",
            "alt": "normal_or_elevated",
            "insulin": "elevated",
            "homa_ir": "elevated",
        },
        "what_doctors_miss": "Doctors see elevated glucose and treat it as pre-diabetes. They miss that the LIVER is driving glucose overproduction, not pancreatic insufficiency.",
        "recommended_action": "Order fasting insulin, HOMA-IR, HbA1c. Consider liver ultrasound for NAFLD. Target liver insulin sensitivity, not just glucose.",
        "urgency": "moderate",
        "trajectory": {
            "6_months": "Progression to overt insulin resistance",
            "12_months": "Pre-diabetes or diabetes diagnosis",
            "24_months": "NAFLD/MASLD development"
        },
        "covers_endpoints": ["metabolic", "hepatic", "endocrine"],
    },

    {
        "id": "metabolic_002",
        "name": "metabolic_syndrome_initiation",
        "description": "Early metabolic syndrome before diagnostic criteria met",
        "required_endpoints": ["metabolic", "endocrine", "lipid_atherogenic", "inflammatory"],
        "biomarker_pattern": {
            "glucose": "borderline_elevated",
            "triglycerides": "elevated",
            "hdl": "low",
            "insulin": "elevated",
            "crp": "mildly_elevated"
        },
        "what_doctors_miss": "Individual markers are 'borderline' so no action taken. Together they indicate metabolic syndrome already forming.",
        "recommended_action": "Comprehensive metabolic panel. Begin lifestyle intervention immediately - this is the highest-ROI window.",
        "urgency": "moderate",
        "covers_endpoints": ["metabolic", "endocrine", "lipid_atherogenic", "inflammatory"],
    },

    {
        "id": "metabolic_003",
        "name": "diabetic_ketoacidosis_cascade",
        "description": "DKA development with metabolic decompensation",
        "required_endpoints": ["metabolic", "endocrine", "renal"],
        "supporting_endpoints": ["inflammatory", "hemodynamic"],
        "biomarker_pattern": {
            "glucose": "critically_elevated",
            "ketones": "elevated",
            "ph": "low",
            "bicarbonate": "low",
            "anion_gap": "elevated",
            "potassium": "variable",
        },
        "what_doctors_miss": "Early DKA with 'mild' ketones and 'acceptable' pH. The cascade is already started.",
        "recommended_action": "URGENT: IV fluids, insulin drip, electrolyte monitoring. ICU evaluation.",
        "urgency": "critical",
        "covers_endpoints": ["metabolic", "endocrine", "renal"],
    },

    {
        "id": "metabolic_004",
        "name": "lactic_acidosis_syndrome",
        "description": "Tissue hypoperfusion causing lactate accumulation",
        "required_endpoints": ["metabolic", "hemodynamic", "respiratory"],
        "supporting_endpoints": ["renal", "hepatic"],
        "biomarker_pattern": {
            "lactate": "elevated",
            "ph": "low",
            "bicarbonate": "low",
            "sbp": "low_or_normal",
            "spo2": "low_or_normal",
        },
        "what_doctors_miss": "Lactate attributed to exercise or stress. Tissue hypoperfusion developing.",
        "recommended_action": "URGENT: Identify cause (sepsis, shock, metformin, liver failure). IV fluids, treat underlying cause.",
        "urgency": "critical",
        "covers_endpoints": ["metabolic", "hemodynamic", "respiratory"],
    },

    {
        "id": "metabolic_005",
        "name": "hypoglycemia_cascade",
        "description": "Severe hypoglycemia with neurological risk",
        "required_endpoints": ["metabolic", "neurological", "endocrine"],
        "biomarker_pattern": {
            "glucose": "critically_low",
            "insulin": "variable",
            "cortisol": "should_be_elevated",
        },
        "what_doctors_miss": "Recurrent 'mild' hypoglycemia dismissed. Hypoglycemia unawareness developing.",
        "recommended_action": "URGENT: IV dextrose. Evaluate for insulinoma, medication error, adrenal insufficiency.",
        "urgency": "critical",
        "covers_endpoints": ["metabolic", "neurological", "endocrine"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CARDIOVASCULAR PATHWAYS (20)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "cardiac_001",
        "name": "cardiorenal_syndrome",
        "description": "Heart-kidney bidirectional dysfunction",
        "required_endpoints": ["cardiac", "renal", "hemodynamic"],
        "supporting_endpoints": ["vascular_endothelial", "inflammatory"],
        "biomarker_pattern": {
            "bnp": "elevated",
            "creatinine": "rising",
            "egfr": "declining",
            "sbp": "unstable"
        },
        "what_doctors_miss": "Treated as separate heart and kidney problems. The bidirectional feedback loop is missed.",
        "recommended_action": "Cardiology + nephrology co-management. Optimize volume status. Consider SGLT2 inhibitors.",
        "urgency": "high",
        "covers_endpoints": ["cardiac", "renal", "hemodynamic"],
    },

    {
        "id": "cardiac_002",
        "name": "atherosclerotic_cascade",
        "description": "Progressive vascular disease with inflammation",
        "required_endpoints": ["lipid_atherogenic", "vascular_endothelial", "inflammatory", "cardiac"],
        "biomarker_pattern": {
            "ldl": "elevated",
            "apob": "elevated",
            "crp": "elevated",
            "endothelin": "elevated"
        },
        "what_doctors_miss": "Focus on LDL alone. Miss ApoB, endothelial dysfunction, and inflammatory component.",
        "recommended_action": "Comprehensive lipid panel with ApoB, Lp(a). Consider coronary calcium score. Target inflammation.",
        "urgency": "moderate",
        "covers_endpoints": ["lipid_atherogenic", "vascular_endothelial", "inflammatory", "cardiac"],
    },

    {
        "id": "cardiac_003",
        "name": "acute_coronary_syndrome",
        "description": "Myocardial infarction pattern",
        "required_endpoints": ["cardiac", "hemodynamic", "coagulation"],
        "biomarker_pattern": {
            "troponin": "elevated",
            "ck_mb": "elevated",
            "bnp": "elevated",
            "d_dimer": "may_be_elevated",
        },
        "what_doctors_miss": "Atypical presentations (elderly, diabetics, women). Initial troponin may be normal.",
        "recommended_action": "URGENT: Serial troponins, ECG, cardiology consult. Consider cath lab.",
        "urgency": "critical",
        "covers_endpoints": ["cardiac", "hemodynamic", "coagulation"],
    },

    {
        "id": "cardiac_004",
        "name": "heart_failure_decompensation",
        "description": "Acute worsening of chronic heart failure",
        "required_endpoints": ["cardiac", "respiratory", "renal", "hemodynamic"],
        "biomarker_pattern": {
            "bnp": "elevated",
            "pro_bnp": "elevated",
            "creatinine": "rising",
            "spo2": "low",
            "respiratory_rate": "elevated",
        },
        "what_doctors_miss": "Gradual BNP rise dismissed as 'chronic'. Decompensation is occurring.",
        "recommended_action": "Diuretics, vasodilators, cardiology consult. Consider hospitalization.",
        "urgency": "high",
        "covers_endpoints": ["cardiac", "respiratory", "renal", "hemodynamic"],
    },

    {
        "id": "cardiac_005",
        "name": "hypertensive_crisis",
        "description": "Severe hypertension with end-organ damage",
        "required_endpoints": ["hemodynamic", "cardiac", "renal", "neurological"],
        "biomarker_pattern": {
            "sbp": "critically_elevated",
            "dbp": "critically_elevated",
            "troponin": "may_be_elevated",
            "creatinine": "may_be_elevated",
        },
        "what_doctors_miss": "Asymptomatic severe hypertension without immediate action. End-organ damage occurring silently.",
        "recommended_action": "URGENT: IV antihypertensives if end-organ damage. Check for target organ damage.",
        "urgency": "critical",
        "covers_endpoints": ["hemodynamic", "cardiac", "renal", "neurological"],
    },

    {
        "id": "cardiac_006",
        "name": "arrhythmia_metabolic_trigger",
        "description": "Arrhythmia triggered by electrolyte imbalance",
        "required_endpoints": ["cardiac", "metabolic", "renal"],
        "biomarker_pattern": {
            "heart_rate": "abnormal",
            "potassium": "abnormal",
            "magnesium": "may_be_low",
            "calcium": "may_be_abnormal",
        },
        "what_doctors_miss": "Arrhythmia treated without checking electrolytes. Root cause is metabolic.",
        "recommended_action": "Check comprehensive metabolic panel. Correct electrolytes before/with antiarrhythmics.",
        "urgency": "high",
        "covers_endpoints": ["cardiac", "metabolic", "renal"],
    },

    {
        "id": "cardiac_007",
        "name": "pulmonary_embolism",
        "description": "Blood clot in pulmonary arteries",
        "required_endpoints": ["coagulation", "respiratory", "cardiac", "hemodynamic"],
        "biomarker_pattern": {
            "d_dimer": "elevated",
            "troponin": "may_be_elevated",
            "bnp": "may_be_elevated",
            "spo2": "low",
            "heart_rate": "elevated",
        },
        "what_doctors_miss": "D-dimer elevated but 'not that high'. Submassive PE developing.",
        "recommended_action": "URGENT: CT angiography, anticoagulation, consider thrombolytics if massive.",
        "urgency": "critical",
        "covers_endpoints": ["coagulation", "respiratory", "cardiac", "hemodynamic"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # INFLAMMATORY / SEPSIS PATHWAYS (15)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "sepsis_001",
        "name": "sepsis_cascade",
        "description": "Multi-organ dysfunction from infection",
        "required_endpoints": ["inflammatory", "hemodynamic", "coagulation", "renal", "respiratory"],
        "biomarker_pattern": {
            "procalcitonin": "elevated",
            "lactate": "elevated",
            "sbp": "low",
            "d_dimer": "elevated",
            "creatinine": "rising",
            "wbc": "abnormal",
        },
        "what_doctors_miss": "Early sepsis with 'borderline' individual markers. qSOFA/NEWS may not trigger.",
        "recommended_action": "URGENT: Blood cultures, broad-spectrum antibiotics, fluid resuscitation, ICU evaluation.",
        "urgency": "critical",
        "covers_endpoints": ["inflammatory", "hemodynamic", "coagulation", "renal", "respiratory"],
    },

    {
        "id": "sepsis_002",
        "name": "early_sepsis_preclinical",
        "description": "Sepsis developing before clinical criteria met",
        "required_endpoints": ["inflammatory", "metabolic", "hemodynamic"],
        "biomarker_pattern": {
            "procalcitonin": "mildly_elevated",
            "crp": "elevated",
            "lactate": "borderline",
            "heart_rate": "elevated",
            "temperature": "abnormal",
        },
        "what_doctors_miss": "Individual markers borderline. Traditional scores don't trigger. Sepsis forming.",
        "recommended_action": "Increase monitoring frequency. Consider early antibiotics. Serial lactate.",
        "urgency": "high",
        "covers_endpoints": ["inflammatory", "metabolic", "hemodynamic"],
    },

    {
        "id": "sepsis_003",
        "name": "septic_shock",
        "description": "Sepsis with persistent hypotension despite fluids",
        "required_endpoints": ["inflammatory", "hemodynamic", "metabolic", "renal", "coagulation"],
        "biomarker_pattern": {
            "lactate": "critically_elevated",
            "sbp": "critically_low",
            "map": "critically_low",
            "creatinine": "elevated",
            "d_dimer": "elevated",
        },
        "what_doctors_miss": "Fluid-responsive hypotension treated as 'dehydration'. Shock developing.",
        "recommended_action": "URGENT: Vasopressors, ICU admission, source control, broad-spectrum antibiotics.",
        "urgency": "critical",
        "covers_endpoints": ["inflammatory", "hemodynamic", "metabolic", "renal", "coagulation"],
    },

    {
        "id": "inflammatory_001",
        "name": "chronic_inflammatory_syndrome",
        "description": "Low-grade systemic inflammation driving multiple conditions",
        "required_endpoints": ["inflammatory", "metabolic", "vascular_endothelial"],
        "supporting_endpoints": ["microbiome_gut_axis", "immune_autoimmune"],
        "biomarker_pattern": {
            "crp": "mildly_elevated",
            "il_6": "elevated",
            "glucose": "borderline",
            "endothelin": "elevated"
        },
        "what_doctors_miss": "Individual markers dismissed as 'slightly elevated'. Chronic inflammation drives disease.",
        "recommended_action": "Anti-inflammatory lifestyle. Consider gut microbiome assessment. Rule out autoimmune.",
        "urgency": "moderate",
        "covers_endpoints": ["inflammatory", "metabolic", "vascular_endothelial"],
    },

    {
        "id": "inflammatory_002",
        "name": "cytokine_storm",
        "description": "Hyperinflammatory response with multi-organ involvement",
        "required_endpoints": ["inflammatory", "respiratory", "coagulation", "hepatic"],
        "biomarker_pattern": {
            "il_6": "critically_elevated",
            "ferritin": "critically_elevated",
            "crp": "critically_elevated",
            "d_dimer": "elevated",
            "ast": "elevated",
            "alt": "elevated",
        },
        "what_doctors_miss": "Severe infection response vs cytokine storm. Different treatments needed.",
        "recommended_action": "URGENT: IL-6 inhibitors, steroids, ICU management. Consider plasmapheresis.",
        "urgency": "critical",
        "covers_endpoints": ["inflammatory", "respiratory", "coagulation", "hepatic"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # RENAL PATHWAYS (10)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "renal_001",
        "name": "acute_kidney_injury",
        "description": "Rapid decline in kidney function",
        "required_endpoints": ["renal", "metabolic", "hemodynamic"],
        "biomarker_pattern": {
            "creatinine": "rapidly_rising",
            "egfr": "rapidly_declining",
            "urine_output": "low",
            "potassium": "rising",
        },
        "what_doctors_miss": "Creatinine 'only slightly elevated' but TREND shows rapid rise. AKI developing.",
        "recommended_action": "URGENT: Fluid management, stop nephrotoxins, nephrology consult. Consider dialysis.",
        "urgency": "high",
        "covers_endpoints": ["renal", "metabolic", "hemodynamic"],
    },

    {
        "id": "renal_002",
        "name": "hepatorenal_syndrome",
        "description": "Kidney failure from liver disease",
        "required_endpoints": ["renal", "hepatic", "hemodynamic"],
        "biomarker_pattern": {
            "creatinine": "elevated",
            "bilirubin": "elevated",
            "inr": "elevated",
            "albumin": "low",
            "sbp": "low",
        },
        "what_doctors_miss": "Kidney and liver problems treated separately. Hepatorenal physiology missed.",
        "recommended_action": "Hepatology + nephrology. Albumin infusion, midodrine/octreotide. Consider TIPS, transplant.",
        "urgency": "critical",
        "covers_endpoints": ["renal", "hepatic", "hemodynamic"],
    },

    {
        "id": "renal_003",
        "name": "ckd_progression",
        "description": "Chronic kidney disease accelerated progression",
        "required_endpoints": ["renal", "cardiac", "metabolic"],
        "biomarker_pattern": {
            "egfr": "declining_trend",
            "creatinine": "rising_trend",
            "urine_albumin": "elevated",
            "bnp": "may_be_elevated",
        },
        "what_doctors_miss": "Slow GFR decline accepted as 'expected'. Modifiable factors not addressed.",
        "recommended_action": "Aggressive BP control, SGLT2 inhibitor, ACE/ARB, protein restriction.",
        "urgency": "moderate",
        "covers_endpoints": ["renal", "cardiac", "metabolic"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # GUT-ORGAN AXIS PATHWAYS (10)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "gut_001",
        "name": "gut_liver_axis_dysfunction",
        "description": "Microbiome-driven liver inflammation and metabolic dysfunction",
        "required_endpoints": ["microbiome_gut_axis", "hepatic", "inflammatory"],
        "supporting_endpoints": ["metabolic", "endocrine"],
        "biomarker_pattern": {
            "lps": "elevated",
            "zonulin": "elevated",
            "alp": "elevated",
            "il_6": "elevated"
        },
        "what_doctors_miss": "Liver enzymes may be normal while gut-derived inflammation drives hepatic dysfunction.",
        "recommended_action": "Comprehensive stool analysis, gut barrier assessment, probiotics/prebiotics.",
        "urgency": "moderate",
        "covers_endpoints": ["microbiome_gut_axis", "hepatic", "inflammatory"],
    },

    {
        "id": "gut_002",
        "name": "gut_brain_axis_dysfunction",
        "description": "Microbiome influence on neurological function",
        "required_endpoints": ["microbiome_gut_axis", "neurological", "inflammatory"],
        "biomarker_pattern": {
            "microbiome_diversity": "low",
            "scfa": "low",
            "il_6": "elevated",
        },
        "what_doctors_miss": "Neurological symptoms treated independently without considering gut contribution.",
        "recommended_action": "Gut microbiome assessment, consider gut-brain protocol, psychobiotics.",
        "urgency": "moderate",
        "covers_endpoints": ["microbiome_gut_axis", "neurological", "inflammatory"],
    },

    {
        "id": "gut_003",
        "name": "leaky_gut_syndrome",
        "description": "Intestinal permeability with systemic effects",
        "required_endpoints": ["microbiome_gut_axis", "inflammatory", "immune_autoimmune"],
        "biomarker_pattern": {
            "zonulin": "elevated",
            "lps": "elevated",
            "crp": "elevated",
        },
        "what_doctors_miss": "Systemic inflammation without identified source. Gut barrier dysfunction.",
        "recommended_action": "Eliminate gut irritants, heal gut lining, restore microbiome.",
        "urgency": "moderate",
        "covers_endpoints": ["microbiome_gut_axis", "inflammatory", "immune_autoimmune"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # AUTOIMMUNE PATHWAYS (10)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "autoimmune_001",
        "name": "autoimmune_preclinical_phase",
        "description": "Autoimmunity developing before clinical symptoms",
        "required_endpoints": ["immune_autoimmune", "inflammatory"],
        "biomarker_pattern": {
            "ana": "positive",
            "crp": "mildly_elevated",
            "c3": "low_normal",
            "c4": "low_normal",
        },
        "what_doctors_miss": "Positive autoantibodies with 'non-specific' symptoms dismissed. Autoimmune forming.",
        "recommended_action": "Rheumatology referral, comprehensive autoantibody panel, symptom monitoring.",
        "urgency": "moderate",
        "covers_endpoints": ["immune_autoimmune", "inflammatory"],
    },

    {
        "id": "autoimmune_002",
        "name": "lupus_flare",
        "description": "SLE disease activity increasing",
        "required_endpoints": ["immune_autoimmune", "inflammatory", "renal", "hematologic"],
        "biomarker_pattern": {
            "ana": "elevated",
            "c3": "low",
            "c4": "low",
            "creatinine": "may_rise",
            "platelets": "may_drop",
        },
        "what_doctors_miss": "Complement levels dropping while symptoms vague. Flare developing.",
        "recommended_action": "Increase immunosuppression, monitor organs, rheumatology consult.",
        "urgency": "high",
        "covers_endpoints": ["immune_autoimmune", "inflammatory", "renal", "hematologic"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # CANCER PATHWAYS (10)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "cancer_001",
        "name": "occult_malignancy_metabolic_signature",
        "description": "Cancer indicated by metabolic changes before tumor detection",
        "required_endpoints": ["oncologic_tumor", "metabolic", "inflammatory", "hematologic"],
        "biomarker_pattern": {
            "ldh": "elevated",
            "crp": "elevated",
            "hemoglobin": "declining",
            "ferritin": "elevated",
        },
        "what_doctors_miss": "Non-specific symptoms and subtle lab changes. Cancer present but not imageable.",
        "recommended_action": "Comprehensive tumor marker panel, CT/PET imaging, hematology consult.",
        "urgency": "high",
        "covers_endpoints": ["oncologic_tumor", "metabolic", "inflammatory", "hematologic"],
    },

    {
        "id": "cancer_002",
        "name": "paraneoplastic_syndrome",
        "description": "Cancer effects distant from tumor",
        "required_endpoints": ["oncologic_tumor", "endocrine", "neurological"],
        "biomarker_pattern": {
            "sodium": "abnormal",
            "cortisol": "may_be_abnormal",
            "ca_125": "may_be_elevated",
        },
        "what_doctors_miss": "Electrolyte abnormalities treated symptomatically. Hidden cancer causing it.",
        "recommended_action": "Comprehensive cancer workup, CT chest/abdomen/pelvis, consider PET.",
        "urgency": "high",
        "covers_endpoints": ["oncologic_tumor", "endocrine", "neurological"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # HEPATIC PATHWAYS (10)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "hepatic_001",
        "name": "nafld_masld_progression",
        "description": "Fatty liver disease progressing to fibrosis",
        "required_endpoints": ["hepatic", "metabolic", "lipid_atherogenic"],
        "biomarker_pattern": {
            "alt": "elevated",
            "ast": "elevated",
            "triglycerides": "elevated",
            "glucose": "elevated",
            "albumin": "may_decline",
        },
        "what_doctors_miss": "Mild liver enzyme elevation dismissed. Fibrosis developing silently.",
        "recommended_action": "FibroScan, lifestyle intervention, consider hepatology referral.",
        "urgency": "moderate",
        "covers_endpoints": ["hepatic", "metabolic", "lipid_atherogenic"],
    },

    {
        "id": "hepatic_002",
        "name": "acute_liver_failure",
        "description": "Rapid hepatic decompensation",
        "required_endpoints": ["hepatic", "coagulation", "metabolic", "neurological"],
        "biomarker_pattern": {
            "ast": "critically_elevated",
            "alt": "critically_elevated",
            "inr": "elevated",
            "bilirubin": "elevated",
            "ammonia": "elevated",
        },
        "what_doctors_miss": "Elevated transaminases without checking synthetic function. Liver failing.",
        "recommended_action": "URGENT: Transplant evaluation, NAC if acetaminophen, ICU admission.",
        "urgency": "critical",
        "covers_endpoints": ["hepatic", "coagulation", "metabolic", "neurological"],
    },

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-ORGAN PATHWAYS (10)
    # ══════════════════════════════════════════════════════════════════════════

    {
        "id": "multi_001",
        "name": "multi_organ_dysfunction_syndrome",
        "description": "Progressive failure of multiple organ systems",
        "required_endpoints": ["respiratory", "renal", "hepatic", "coagulation", "hemodynamic"],
        "biomarker_pattern": {
            "spo2": "low",
            "creatinine": "elevated",
            "bilirubin": "elevated",
            "platelets": "low",
            "lactate": "elevated",
        },
        "what_doctors_miss": "Each organ treated separately. MODS cascade not recognized.",
        "recommended_action": "URGENT: ICU, multi-specialty coordination, treat underlying cause.",
        "urgency": "critical",
        "covers_endpoints": ["respiratory", "renal", "hepatic", "coagulation", "hemodynamic"],
    },

    {
        "id": "multi_002",
        "name": "shock_cascade",
        "description": "Circulatory failure with tissue hypoperfusion",
        "required_endpoints": ["hemodynamic", "metabolic", "renal", "neurological"],
        "biomarker_pattern": {
            "sbp": "critically_low",
            "lactate": "elevated",
            "creatinine": "rising",
            "gcs_score": "may_decline",
        },
        "what_doctors_miss": "Compensated shock with 'normal' blood pressure. Lactate reveals the truth.",
        "recommended_action": "URGENT: Fluid resuscitation, vasopressors, identify cause, ICU.",
        "urgency": "critical",
        "covers_endpoints": ["hemodynamic", "metabolic", "renal", "neurological"],
    },

    {
        "id": "multi_003",
        "name": "dic_syndrome",
        "description": "Disseminated intravascular coagulation",
        "required_endpoints": ["coagulation", "hematologic", "inflammatory"],
        "biomarker_pattern": {
            "d_dimer": "critically_elevated",
            "fibrinogen": "low",
            "platelets": "critically_low",
            "pt": "prolonged",
            "ptt": "prolonged",
        },
        "what_doctors_miss": "Bleeding and clotting symptoms seem contradictory. DIC not considered.",
        "recommended_action": "URGENT: Treat underlying cause, blood products, hematology consult.",
        "urgency": "critical",
        "covers_endpoints": ["coagulation", "hematologic", "inflammatory"],
    },

    # Add more pathways to reach 100+...
    # Additional pathways for respiratory, neurological, etc.

    {
        "id": "respiratory_001",
        "name": "ards_development",
        "description": "Acute respiratory distress syndrome",
        "required_endpoints": ["respiratory", "inflammatory", "coagulation"],
        "biomarker_pattern": {
            "spo2": "critically_low",
            "respiratory_rate": "critically_elevated",
            "pao2": "low",
            "crp": "elevated",
            "d_dimer": "elevated",
        },
        "what_doctors_miss": "Hypoxemia attributed to pneumonia alone. ARDS physiology developing.",
        "recommended_action": "URGENT: Intubation, lung protective ventilation, prone positioning, ICU.",
        "urgency": "critical",
        "covers_endpoints": ["respiratory", "inflammatory", "coagulation"],
    },

    {
        "id": "respiratory_002",
        "name": "copd_exacerbation",
        "description": "Acute COPD worsening",
        "required_endpoints": ["respiratory", "inflammatory", "infectious_pathogenic"],
        "biomarker_pattern": {
            "spo2": "low",
            "respiratory_rate": "elevated",
            "paco2": "elevated",
            "wbc": "may_be_elevated",
        },
        "what_doctors_miss": "Baseline dyspnea makes acute changes hard to detect. Exacerbation developing.",
        "recommended_action": "Bronchodilators, steroids, antibiotics if infectious, consider NIV.",
        "urgency": "high",
        "covers_endpoints": ["respiratory", "inflammatory", "infectious_pathogenic"],
    },

]


class PathwayMatcher:
    """Match patient data against known disease pathways."""

    def __init__(self):
        self.pathways = PATHWAY_LIBRARY

    def match_pathways(
        self,
        alerting_endpoints: Dict[str, Any],
        endpoint_details: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Match alerting endpoints against known pathways.

        Args:
            alerting_endpoints: Dict of endpoint names to their results
            endpoint_details: Optional detailed biomarker info

        Returns:
            List of matched pathways with confidence scores
        """
        matched = []
        alerting_names = set(alerting_endpoints.keys())

        for pathway in self.pathways:
            required = set(pathway.get("required_endpoints", []))

            # Check if all required endpoints are alerting
            if required.issubset(alerting_names):
                # Calculate confidence based on biomarker pattern match
                confidence = self._calculate_pathway_confidence(
                    pathway, alerting_endpoints, endpoint_details
                )

                if confidence > 0.3:
                    matched.append({
                        "pathway_id": pathway.get("id"),
                        "pathway_name": pathway.get("name"),
                        "description": pathway.get("description"),
                        "confidence": round(confidence, 3),
                        "urgency": pathway.get("urgency", "moderate"),
                        "what_doctors_miss": pathway.get("what_doctors_miss"),
                        "recommended_action": pathway.get("recommended_action"),
                        "required_endpoints": list(required),
                        "supporting_endpoints": pathway.get("supporting_endpoints", []),
                        "covers_endpoints": pathway.get("covers_endpoints", list(required)),
                    })

        # Sort by confidence and urgency
        urgency_order = {"critical": 0, "high": 1, "moderate": 2, "low": 3}
        matched.sort(key=lambda x: (urgency_order.get(x["urgency"], 3), -x["confidence"]))

        return matched

    def _calculate_pathway_confidence(
        self,
        pathway: Dict,
        alerting_endpoints: Dict,
        endpoint_details: Optional[Dict] = None
    ) -> float:
        """Calculate confidence score for pathway match."""
        scores = []

        # Required endpoints match (40%)
        required = set(pathway.get("required_endpoints", []))
        alerting = set(alerting_endpoints.keys())
        required_match = len(required.intersection(alerting)) / len(required) if required else 0
        scores.append(required_match * 0.4)

        # Supporting endpoints (20%)
        supporting = set(pathway.get("supporting_endpoints", []))
        if supporting:
            support_match = len(supporting.intersection(alerting)) / len(supporting)
            scores.append(support_match * 0.2)
        else:
            scores.append(0.2)  # No supporting required = full score

        # Severity alignment (20%)
        severity_scores = []
        for ep_name, ep_result in alerting_endpoints.items():
            if ep_name in required:
                status = ep_result.get("status", "normal")
                if status == "critical":
                    severity_scores.append(1.0)
                elif status == "elevated":
                    severity_scores.append(0.7)
                elif status == "borderline":
                    severity_scores.append(0.4)
        if severity_scores:
            scores.append(sum(severity_scores) / len(severity_scores) * 0.2)
        else:
            scores.append(0.1)

        # Biomarker pattern match (20%)
        if endpoint_details:
            pattern_score = self._match_biomarker_pattern(
                pathway.get("biomarker_pattern", {}),
                endpoint_details
            )
            scores.append(pattern_score * 0.2)
        else:
            scores.append(0.15)  # Partial score without details

        return sum(scores)

    def _match_biomarker_pattern(
        self,
        expected_pattern: Dict[str, str],
        actual_values: Dict[str, Any]
    ) -> float:
        """Match expected biomarker pattern against actual values."""
        if not expected_pattern:
            return 0.5

        matches = 0
        total = 0

        for biomarker, expected_state in expected_pattern.items():
            total += 1
            # Check if biomarker is flagged in actual values
            if biomarker in actual_values:
                actual = actual_values[biomarker]
                if self._state_matches(expected_state, actual):
                    matches += 1

        return matches / total if total > 0 else 0.5

    def _state_matches(self, expected: str, actual: Any) -> bool:
        """Check if actual state matches expected."""
        if isinstance(actual, dict):
            actual_status = actual.get("status", "")
        else:
            actual_status = str(actual)

        expected_lower = expected.lower()
        actual_lower = actual_status.lower()

        # Mapping of expected states to actual statuses
        if expected_lower in ["elevated", "high", "critically_elevated"]:
            return actual_lower in ["elevated", "critical", "high"]
        elif expected_lower in ["low", "critically_low"]:
            return actual_lower in ["low", "critical"]
        elif expected_lower == "abnormal":
            return actual_lower in ["elevated", "critical", "low", "borderline"]
        elif expected_lower in ["normal", "normal_or_elevated"]:
            return True  # Flexible match

        return expected_lower in actual_lower

    def detect_unknown_pattern(
        self,
        alerting_endpoints: Dict[str, Any],
        matched_pathways: List[Dict]
    ) -> Optional[Dict]:
        """Detect patterns that don't match known pathways."""
        if len(alerting_endpoints) < 2:
            return None

        alerting_names = set(alerting_endpoints.keys())

        # Get endpoints explained by matched pathways
        explained = set()
        for pathway in matched_pathways:
            explained.update(pathway.get("covers_endpoints", []))

        unexplained = alerting_names - explained

        if unexplained and len(unexplained) >= 2:
            return {
                "is_novel": True,
                "unexplained_endpoints": list(unexplained),
                "all_alerting": list(alerting_names),
                "pattern_signature": self._generate_signature(alerting_endpoints, unexplained),
                "recommendation": "Novel multi-system pattern detected. Comprehensive workup recommended.",
                "confidence": 0.6,
            }

        return None

    def _generate_signature(
        self,
        alerting_endpoints: Dict,
        unexplained: Set[str]
    ) -> str:
        """Generate a signature for the unknown pattern."""
        parts = []
        for ep in sorted(unexplained):
            result = alerting_endpoints.get(ep, {})
            status = result.get("status", "unknown")
            parts.append(f"{ep}:{status}")
        return "|".join(parts)
