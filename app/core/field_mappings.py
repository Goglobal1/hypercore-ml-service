# field_mappings.py
# Location: app/core/field_mappings.py
"""
Field name aliases and endpoint schema definitions.
Allows flexible input formats to be normalized to standard names.
"""

# Standard field names with all acceptable aliases
FIELD_ALIASES = {
    # Patient identification
    "patient_id": [
        "patient_id", "patientid", "patient", "subject_id", "subjectid",
        "subject", "id", "pt_id", "ptid", "participant_id", "participantid",
        "record_id", "recordid", "mrn", "medical_record_number"
    ],

    # Time/Date fields
    "time": [
        "time", "date", "timestamp", "datetime", "visit_date", "visitdate",
        "observation_date", "obs_date", "collection_date", "sample_date",
        "event_date", "eventdate", "day", "timepoint", "visit"
    ],

    # Outcome/Label fields
    "outcome": [
        "outcome", "label", "target", "response", "result", "event",
        "endpoint", "primary_endpoint", "y", "class", "diagnosis",
        "status", "survived", "death", "mortality", "responder"
    ],

    # Treatment fields
    "treatment": [
        "treatment", "treatment_arm", "arm", "drug", "intervention",
        "therapy", "medication", "treatment_group", "group", "cohort",
        "assigned_treatment", "tx", "rx"
    ],

    # Demographics
    "age": [
        "age", "patient_age", "years", "age_years", "age_at_enrollment",
        "enrollment_age", "baseline_age"
    ],
    "sex": [
        "sex", "gender", "male", "female", "m_f", "patient_sex"
    ],
    "weight": [
        "weight", "weight_kg", "wt", "body_weight", "mass"
    ],
    "height": [
        "height", "height_cm", "ht", "stature"
    ],
    "bmi": [
        "bmi", "body_mass_index", "bodymassindex"
    ],

    # Common lab values
    "crp": [
        "crp", "c_reactive_protein", "creactiveprotein", "crp_baseline",
        "crp_value", "c-reactive_protein"
    ],
    "il6": [
        "il6", "il-6", "interleukin6", "interleukin_6", "il6_baseline",
        "il6_value", "interleukin-6"
    ],
    "tnf": [
        "tnf", "tnfa", "tnf_alpha", "tnf-alpha", "tnfalpha",
        "tumor_necrosis_factor", "tnfa_baseline"
    ],
    "wbc": [
        "wbc", "white_blood_cells", "whitebloodcells", "leukocytes",
        "white_count", "wbc_count"
    ],
    "hemoglobin": [
        "hemoglobin", "hgb", "hb", "haemoglobin"
    ],
    "creatinine": [
        "creatinine", "creat", "cr", "serum_creatinine"
    ],
    "glucose": [
        "glucose", "blood_glucose", "bg", "fasting_glucose", "fbg"
    ],
    "hba1c": [
        "hba1c", "a1c", "glycated_hemoglobin", "hemoglobin_a1c"
    ],
    "lactate": [
        "lactate", "lactic_acid", "serum_lactate"
    ],

    # Vital signs
    "heart_rate": [
        "heart_rate", "hr", "pulse", "heartrate", "pulse_rate"
    ],
    "blood_pressure_systolic": [
        "sbp", "systolic", "systolic_bp", "blood_pressure_systolic"
    ],
    "blood_pressure_diastolic": [
        "dbp", "diastolic", "diastolic_bp", "blood_pressure_diastolic"
    ],
    "temperature": [
        "temperature", "temp", "body_temp", "body_temperature"
    ],
    "respiratory_rate": [
        "respiratory_rate", "resp_rate", "rr", "resprate", "breathing_rate"
    ],
    "oxygen_saturation": [
        "spo2", "o2_sat", "oxygen_saturation", "o2sat", "saturation"
    ],

    # Clinical scores
    "das28": [
        "das28", "das_28", "das28_score", "disease_activity_score"
    ],
    "acr20": [
        "acr20", "acr_20", "acr20_response", "week_12_acr20"
    ],
    "acr50": [
        "acr50", "acr_50", "acr50_response"
    ],
    "acr70": [
        "acr70", "acr_70", "acr70_response"
    ],

    # Site/Location
    "site": [
        "site", "site_id", "location", "center", "centre", "clinic",
        "hospital", "facility", "study_site"
    ],
    "region": [
        "region", "geographic_region", "area", "zone", "territory"
    ],

    # Case counts (for surveillance)
    "cases": [
        "cases", "case_count", "case_counts", "n_cases", "num_cases",
        "positive_cases", "confirmed_cases", "infections"
    ],
    "deaths": [
        "deaths", "death_count", "mortality_count", "fatalities"
    ],
    "population": [
        "population", "pop", "total_population", "pop_size"
    ]
}

# Endpoint-specific schema definitions
ENDPOINT_SCHEMAS = {
    "analyze": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": False,
        "optional_fields": ["patient_id_column", "time_column"]
    },
    "trial_rescue": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": True,
        "optional_fields": []
    },
    "responder_prediction": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": True,
        "optional_fields": []
    },
    "early_risk_discovery": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": False,
        "optional_fields": ["patient_id_column", "time_column", "outcome_type"]
    },
    "confounder_detection": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": True,
        "optional_fields": []
    },
    "outbreak_detection": {
        "needs_csv": True,
        "needs_label_column": False,
        "needs_treatment_column": False,
        "required_fields": ["region_column", "time_column", "case_count_column"]
    },
    "multi_omic_fusion": {
        "needs_csv": False,
        "custom_format": "multi_omic",
        "required_arrays": ["immune", "metabolic", "microbiome"]
    },
    "digital_twin_simulation": {
        "needs_csv": False,
        "custom_format": "baseline_profile",
        "required_fields": ["baseline_profile"]
    },
    "population_risk": {
        "needs_csv": False,
        "custom_format": "population_risk",
        "required_fields": ["analyses", "region"]
    },
    "synthetic_cohort": {
        "needs_csv": False,
        "custom_format": "distributions",
        "required_fields": ["real_data_distributions", "n_subjects"]
    },
    "fluview_ingest": {
        "needs_csv": False,
        "custom_format": "fluview",
        "required_fields": ["fluview_json"]
    },
    "medication_interaction": {
        "needs_csv": False,
        "custom_format": "medications",
        "required_fields": ["medications"]
    },
    "forecast_timeline": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": False,
        "optional_fields": []
    },
    "patient_report": {
        "needs_csv": False,
        "custom_format": "patient_report",
        "required_fields": ["executive_summary"]
    },
    "emerging_phenotype": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": False,
        "optional_fields": []
    },
    "create_digital_twin": {
        "needs_csv": False,
        "custom_format": "digital_twin_storage",
        "required_fields": ["patient_id", "dataset_id", "analysis_id", "csv_content"]
    },
    "root_cause_sim": {
        "needs_csv": False,
        "custom_format": "root_cause",
        "required_fields": ["condition"]
    },
    "predictive_modeling": {
        "needs_csv": True,
        "needs_label_column": True,
        "needs_treatment_column": False,
        "optional_fields": ["model_type"]
    }
}

# Data type detection patterns
DATA_TYPE_PATTERNS = {
    "clinical_trial": {
        "required_columns": ["treatment", "outcome"],
        "optional_columns": ["patient_id", "site"],
        "applicable_endpoints": [
            "analyze", "trial_rescue", "responder_prediction",
            "confounder_detection", "emerging_phenotype"
        ]
    },
    "patient_monitoring": {
        "required_columns": ["patient_id", "time"],
        "optional_columns": ["heart_rate", "temperature", "wbc"],
        "applicable_endpoints": [
            "analyze", "early_risk_discovery", "forecast_timeline",
            "digital_twin_simulation"
        ]
    },
    "surveillance": {
        "required_columns": ["region", "cases"],
        "optional_columns": ["deaths", "population", "time"],
        "applicable_endpoints": [
            "outbreak_detection", "population_risk"
        ]
    },
    "multi_omic": {
        "required_columns": [],
        "required_keys": ["immune", "metabolic", "microbiome"],
        "applicable_endpoints": ["multi_omic_fusion"]
    },
    "medication_list": {
        "required_keys": ["medications"],
        "applicable_endpoints": ["medication_interaction"]
    }
}
