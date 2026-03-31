"""
HyperCore Comprehensive Cross-Dataset Validation
=================================================
Tests the hybrid scoring system across 5 diverse clinical scenarios
to validate consistency, cross-domain convergence, and edge case handling.
"""

import requests
import json
import time
from typing import Dict, List, Tuple

API_BASE = "https://hypercore-ml-service-production.up.railway.app"

# ============================================================
# DATASET 1: SEPSIS PROGRESSION (10 patients)
# Mix of progressors and non-progressors
# ============================================================
DATASET_1_SEPSIS = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
SEP001,2025-01-01T00:00:00Z,78,14,125,36.8,98,0.9,1.0,7.5,250
SEP001,2025-01-01T04:00:00Z,85,16,120,37.2,97,1.0,1.2,8.5,240
SEP001,2025-01-01T08:00:00Z,98,20,110,38.1,95,1.4,2.0,12.0,180
SEP001,2025-01-01T12:00:00Z,115,26,95,38.8,91,2.2,3.5,18.0,120
SEP002,2025-01-01T00:00:00Z,72,12,130,36.5,99,0.8,0.8,6.0,280
SEP002,2025-01-01T04:00:00Z,74,13,128,36.6,99,0.8,0.9,6.2,275
SEP002,2025-01-01T08:00:00Z,76,14,126,36.7,98,0.9,0.9,6.5,270
SEP002,2025-01-01T12:00:00Z,78,14,125,36.8,98,0.9,1.0,7.0,265
SEP003,2025-01-01T00:00:00Z,80,15,120,37.0,97,1.0,1.1,8.0,220
SEP003,2025-01-01T04:00:00Z,92,18,108,37.8,94,1.5,1.8,11.0,170
SEP003,2025-01-01T08:00:00Z,108,24,92,38.5,90,2.4,3.2,16.0,110
SEP003,2025-01-01T12:00:00Z,125,30,78,39.2,86,3.5,5.5,22.0,65
SEP004,2025-01-01T00:00:00Z,68,11,135,36.4,99,0.7,0.7,5.5,300
SEP004,2025-01-01T04:00:00Z,70,12,133,36.5,99,0.7,0.7,5.8,295
SEP004,2025-01-01T08:00:00Z,72,12,132,36.5,99,0.8,0.8,6.0,290
SEP004,2025-01-01T12:00:00Z,74,13,130,36.6,99,0.8,0.8,6.2,285
SEP005,2025-01-01T00:00:00Z,82,16,115,37.2,96,1.2,1.3,9.0,200
SEP005,2025-01-01T04:00:00Z,95,20,100,38.0,93,1.8,2.2,13.0,150
SEP005,2025-01-01T08:00:00Z,110,25,88,38.8,89,2.6,3.8,17.5,95
SEP005,2025-01-01T12:00:00Z,128,32,72,39.5,84,3.8,6.0,24.0,55
SEP006,2025-01-01T00:00:00Z,75,14,125,36.7,98,0.9,0.9,7.0,260
SEP006,2025-01-01T04:00:00Z,78,15,122,36.9,97,1.0,1.0,7.5,250
SEP006,2025-01-01T08:00:00Z,82,16,118,37.2,96,1.1,1.2,8.5,235
SEP006,2025-01-01T12:00:00Z,88,18,112,37.6,95,1.3,1.5,10.0,210
SEP007,2025-01-01T00:00:00Z,85,17,110,37.5,95,1.3,1.4,10.5,190
SEP007,2025-01-01T04:00:00Z,100,22,95,38.2,91,2.0,2.5,14.5,140
SEP007,2025-01-01T08:00:00Z,118,28,82,39.0,87,2.8,4.2,19.0,90
SEP007,2025-01-01T12:00:00Z,135,35,68,39.8,82,4.0,6.8,26.0,45
SEP008,2025-01-01T00:00:00Z,70,13,128,36.6,98,0.8,0.8,6.5,270
SEP008,2025-01-01T04:00:00Z,72,13,126,36.7,98,0.9,0.9,6.8,265
SEP008,2025-01-01T08:00:00Z,75,14,124,36.8,98,0.9,0.9,7.2,258
SEP008,2025-01-01T12:00:00Z,78,15,122,37.0,97,1.0,1.0,7.5,250
SEP009,2025-01-01T00:00:00Z,88,18,105,37.8,94,1.5,1.6,11.0,175
SEP009,2025-01-01T04:00:00Z,102,23,90,38.5,90,2.2,2.8,15.0,125
SEP009,2025-01-01T08:00:00Z,120,29,76,39.2,85,3.2,4.5,20.0,75
SEP009,2025-01-01T12:00:00Z,140,38,62,40.0,80,4.5,7.5,28.0,35
SEP010,2025-01-01T00:00:00Z,76,14,122,36.8,97,1.0,1.0,7.8,245
SEP010,2025-01-01T04:00:00Z,80,15,118,37.0,96,1.1,1.1,8.2,235
SEP010,2025-01-01T08:00:00Z,85,17,114,37.3,95,1.2,1.3,9.0,220
SEP010,2025-01-01T12:00:00Z,92,19,108,37.8,94,1.4,1.6,10.5,200"""

# Expected: SEP003, SEP005, SEP007, SEP009 should be high risk (severe progression)
# SEP001, SEP006, SEP010 should be moderate risk (mild progression)
# SEP002, SEP004, SEP008 should be low risk (stable)

# ============================================================
# DATASET 2: CARDIAC DETERIORATION (8 patients)
# Focused on hemodynamic instability
# ============================================================
DATASET_2_CARDIAC = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,dbp,map,temperature,spo2,creatinine,lactate,bnp
CARD001,2025-01-01T00:00:00Z,72,14,130,80,97,36.6,98,0.9,0.9,50
CARD001,2025-01-01T04:00:00Z,85,16,115,72,86,36.8,96,1.0,1.2,150
CARD001,2025-01-01T08:00:00Z,105,20,95,60,72,37.0,93,1.3,1.8,450
CARD001,2025-01-01T12:00:00Z,125,26,78,48,58,37.2,89,1.8,2.8,1200
CARD002,2025-01-01T00:00:00Z,68,12,140,85,103,36.5,99,0.8,0.7,30
CARD002,2025-01-01T04:00:00Z,70,12,138,84,102,36.5,99,0.8,0.8,35
CARD002,2025-01-01T08:00:00Z,72,13,136,82,100,36.6,99,0.8,0.8,40
CARD002,2025-01-01T12:00:00Z,74,13,134,80,98,36.6,98,0.9,0.9,45
CARD003,2025-01-01T00:00:00Z,78,15,120,75,90,36.8,97,1.0,1.0,80
CARD003,2025-01-01T04:00:00Z,95,18,100,65,77,37.0,94,1.2,1.5,280
CARD003,2025-01-01T08:00:00Z,118,24,82,52,62,37.2,90,1.6,2.4,750
CARD003,2025-01-01T12:00:00Z,145,32,65,42,50,37.5,85,2.2,3.8,1800
CARD004,2025-01-01T00:00:00Z,70,13,135,82,100,36.5,98,0.8,0.8,40
CARD004,2025-01-01T04:00:00Z,72,14,132,80,97,36.6,98,0.9,0.9,50
CARD004,2025-01-01T08:00:00Z,76,15,128,78,95,36.7,97,0.9,1.0,65
CARD004,2025-01-01T12:00:00Z,80,16,124,76,92,36.8,96,1.0,1.1,85
CARD005,2025-01-01T00:00:00Z,82,16,110,68,82,37.0,96,1.1,1.2,120
CARD005,2025-01-01T04:00:00Z,100,20,90,55,67,37.2,92,1.4,1.8,400
CARD005,2025-01-01T08:00:00Z,125,28,72,44,53,37.4,87,1.9,2.8,1100
CARD005,2025-01-01T12:00:00Z,150,35,58,35,43,37.8,82,2.6,4.2,2200
CARD006,2025-01-01T00:00:00Z,66,11,145,88,107,36.4,99,0.7,0.7,25
CARD006,2025-01-01T04:00:00Z,68,12,142,86,105,36.5,99,0.7,0.7,28
CARD006,2025-01-01T08:00:00Z,70,12,140,85,103,36.5,99,0.8,0.8,32
CARD006,2025-01-01T12:00:00Z,72,13,138,84,102,36.6,99,0.8,0.8,38
CARD007,2025-01-01T00:00:00Z,88,18,105,65,78,37.1,94,1.2,1.4,200
CARD007,2025-01-01T04:00:00Z,108,24,85,52,63,37.3,90,1.6,2.2,600
CARD007,2025-01-01T08:00:00Z,132,32,68,42,51,37.6,85,2.2,3.5,1400
CARD007,2025-01-01T12:00:00Z,155,40,52,32,39,38.0,79,3.0,5.0,2800
CARD008,2025-01-01T00:00:00Z,75,14,125,78,94,36.7,97,0.9,0.9,70
CARD008,2025-01-01T04:00:00Z,80,15,120,75,90,36.8,96,1.0,1.0,90
CARD008,2025-01-01T08:00:00Z,88,17,112,70,84,37.0,95,1.1,1.2,130
CARD008,2025-01-01T12:00:00Z,95,19,105,65,78,37.2,93,1.2,1.4,180"""

# Expected: CARD003, CARD005, CARD007 high risk (cardiogenic shock)
# CARD001, CARD008 moderate risk (cardiac stress)
# CARD002, CARD004, CARD006 low risk (stable)

# ============================================================
# DATASET 3: RESPIRATORY FAILURE (8 patients)
# Focused on pulmonary deterioration
# ============================================================
DATASET_3_RESPIRATORY = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,fio2,pao2,paco2,ph,creatinine,lactate
RESP001,2025-01-01T00:00:00Z,78,16,125,36.8,97,21,85,38,7.40,0.9,1.0
RESP001,2025-01-01T04:00:00Z,88,22,118,37.2,93,35,70,42,7.36,1.0,1.3
RESP001,2025-01-01T08:00:00Z,102,30,108,37.8,88,60,55,48,7.30,1.2,1.8
RESP001,2025-01-01T12:00:00Z,118,38,95,38.2,82,100,45,58,7.22,1.5,2.5
RESP002,2025-01-01T00:00:00Z,70,13,130,36.5,99,21,95,36,7.42,0.8,0.8
RESP002,2025-01-01T04:00:00Z,72,14,128,36.6,98,21,92,37,7.41,0.8,0.8
RESP002,2025-01-01T08:00:00Z,74,14,126,36.6,98,21,90,38,7.40,0.9,0.9
RESP002,2025-01-01T12:00:00Z,76,15,124,36.7,97,21,88,38,7.40,0.9,0.9
RESP003,2025-01-01T00:00:00Z,82,18,118,37.2,95,28,78,40,7.38,1.0,1.1
RESP003,2025-01-01T04:00:00Z,95,26,108,37.8,90,50,62,46,7.32,1.2,1.6
RESP003,2025-01-01T08:00:00Z,115,36,95,38.5,83,80,48,54,7.24,1.6,2.4
RESP003,2025-01-01T12:00:00Z,138,45,80,39.0,75,100,38,65,7.15,2.2,3.8
RESP004,2025-01-01T00:00:00Z,68,12,135,36.4,99,21,98,35,7.43,0.7,0.7
RESP004,2025-01-01T04:00:00Z,70,12,133,36.5,99,21,96,36,7.42,0.8,0.7
RESP004,2025-01-01T08:00:00Z,72,13,132,36.5,99,21,94,36,7.42,0.8,0.8
RESP004,2025-01-01T12:00:00Z,74,13,130,36.6,98,21,92,37,7.41,0.8,0.8
RESP005,2025-01-01T00:00:00Z,85,20,115,37.4,94,32,75,42,7.36,1.1,1.2
RESP005,2025-01-01T04:00:00Z,100,28,105,38.0,88,55,58,50,7.28,1.4,1.9
RESP005,2025-01-01T08:00:00Z,120,38,92,38.6,80,85,42,60,7.18,1.9,3.0
RESP005,2025-01-01T12:00:00Z,142,48,78,39.2,72,100,32,72,7.08,2.5,4.5
RESP006,2025-01-01T00:00:00Z,72,14,128,36.6,98,21,90,37,7.41,0.8,0.9
RESP006,2025-01-01T04:00:00Z,76,16,124,36.8,96,25,82,40,7.38,0.9,1.0
RESP006,2025-01-01T08:00:00Z,82,18,118,37.2,94,32,75,43,7.35,1.0,1.2
RESP006,2025-01-01T12:00:00Z,90,22,110,37.6,92,40,68,46,7.32,1.1,1.5
RESP007,2025-01-01T00:00:00Z,90,24,110,37.8,92,45,65,45,7.34,1.2,1.4
RESP007,2025-01-01T04:00:00Z,110,34,98,38.4,85,70,50,55,7.25,1.6,2.2
RESP007,2025-01-01T08:00:00Z,132,44,84,39.0,76,95,38,68,7.14,2.2,3.5
RESP007,2025-01-01T12:00:00Z,155,52,68,39.6,68,100,28,78,7.02,3.0,5.2
RESP008,2025-01-01T00:00:00Z,75,15,124,36.7,97,21,88,38,7.40,0.9,0.9
RESP008,2025-01-01T04:00:00Z,80,17,120,36.9,95,28,80,41,7.37,1.0,1.1
RESP008,2025-01-01T08:00:00Z,88,20,114,37.2,93,38,72,44,7.34,1.1,1.3
RESP008,2025-01-01T12:00:00Z,96,24,108,37.6,90,48,64,48,7.30,1.2,1.6"""

# Expected: RESP003, RESP005, RESP007 high risk (ARDS/respiratory failure)
# RESP001, RESP006, RESP008 moderate risk (respiratory distress)
# RESP002, RESP004 low risk (stable)

# ============================================================
# DATASET 4: MIXED ACUITY (12 patients)
# Diverse presentations from stable to critical
# ============================================================
DATASET_4_MIXED = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets,gcs
MIX001,2025-01-01T00:00:00Z,68,12,135,36.5,99,0.7,0.7,6.0,290,15
MIX001,2025-01-01T06:00:00Z,70,13,133,36.5,99,0.8,0.8,6.2,285,15
MIX001,2025-01-01T12:00:00Z,72,13,131,36.6,99,0.8,0.8,6.5,280,15
MIX002,2025-01-01T00:00:00Z,140,40,60,39.8,78,4.5,8.0,28.0,30,8
MIX002,2025-01-01T06:00:00Z,145,42,55,40.0,75,5.0,9.0,32.0,25,6
MIX002,2025-01-01T12:00:00Z,150,45,50,40.2,72,5.5,10.0,35.0,20,4
MIX003,2025-01-01T00:00:00Z,75,14,128,36.7,98,0.9,0.9,7.0,265,15
MIX003,2025-01-01T06:00:00Z,78,15,125,36.8,97,0.9,1.0,7.5,258,15
MIX003,2025-01-01T12:00:00Z,82,16,122,37.0,96,1.0,1.1,8.0,250,15
MIX004,2025-01-01T00:00:00Z,120,32,75,39.0,84,3.0,5.0,22.0,70,10
MIX004,2025-01-01T06:00:00Z,128,35,70,39.3,81,3.5,6.0,25.0,55,9
MIX004,2025-01-01T12:00:00Z,135,38,65,39.6,78,4.0,7.0,28.0,40,7
MIX005,2025-01-01T00:00:00Z,72,13,132,36.6,98,0.8,0.8,6.5,275,15
MIX005,2025-01-01T06:00:00Z,74,14,130,36.7,98,0.8,0.8,6.8,270,15
MIX005,2025-01-01T12:00:00Z,76,14,128,36.7,98,0.9,0.9,7.0,265,15
MIX006,2025-01-01T00:00:00Z,95,22,105,37.8,93,1.5,2.0,12.0,180,14
MIX006,2025-01-01T06:00:00Z,105,26,95,38.2,90,1.8,2.5,15.0,150,13
MIX006,2025-01-01T12:00:00Z,115,30,85,38.6,86,2.2,3.2,18.0,120,12
MIX007,2025-01-01T00:00:00Z,70,12,130,36.5,99,0.8,0.8,6.0,280,15
MIX007,2025-01-01T06:00:00Z,72,13,128,36.6,99,0.8,0.8,6.2,275,15
MIX007,2025-01-01T12:00:00Z,74,13,126,36.6,98,0.9,0.9,6.5,270,15
MIX008,2025-01-01T00:00:00Z,88,18,110,37.5,95,1.2,1.4,10.0,200,14
MIX008,2025-01-01T06:00:00Z,98,22,100,38.0,92,1.5,1.8,13.0,165,13
MIX008,2025-01-01T12:00:00Z,110,26,90,38.5,88,1.9,2.4,16.0,130,12
MIX009,2025-01-01T00:00:00Z,66,11,140,36.4,99,0.7,0.7,5.5,300,15
MIX009,2025-01-01T06:00:00Z,68,12,138,36.5,99,0.7,0.7,5.8,295,15
MIX009,2025-01-01T12:00:00Z,70,12,136,36.5,99,0.7,0.8,6.0,290,15
MIX010,2025-01-01T00:00:00Z,130,35,70,39.2,82,3.5,5.5,24.0,60,9
MIX010,2025-01-01T06:00:00Z,138,38,65,39.5,79,4.0,6.5,27.0,45,8
MIX010,2025-01-01T12:00:00Z,145,42,58,39.8,75,4.5,7.5,30.0,32,6
MIX011,2025-01-01T00:00:00Z,80,15,120,37.0,96,1.0,1.1,8.0,240,15
MIX011,2025-01-01T06:00:00Z,85,17,115,37.3,95,1.1,1.3,9.0,225,15
MIX011,2025-01-01T12:00:00Z,92,19,108,37.6,93,1.3,1.5,10.5,205,14
MIX012,2025-01-01T00:00:00Z,78,14,125,36.8,97,0.9,0.9,7.5,255,15
MIX012,2025-01-01T06:00:00Z,82,15,122,37.0,96,1.0,1.0,8.0,245,15
MIX012,2025-01-01T12:00:00Z,86,17,118,37.2,95,1.1,1.1,8.5,235,15"""

# Expected: MIX002, MIX010 critical (multi-organ failure)
# MIX004 high risk (severe)
# MIX006, MIX008, MIX011 moderate risk
# MIX001, MIX003, MIX005, MIX007, MIX009, MIX012 low risk (stable)

# ============================================================
# DATASET 5: EDGE CASES (6 patients)
# Unusual patterns to test robustness
# ============================================================
DATASET_5_EDGE = """patient_id,timestamp,heart_rate,respiratory_rate,sbp,temperature,spo2,creatinine,lactate,wbc,platelets
EDGE001,2025-01-01T00:00:00Z,120,10,140,36.0,99,0.6,0.5,4.0,350
EDGE001,2025-01-01T06:00:00Z,118,11,138,36.1,99,0.6,0.5,4.2,345
EDGE001,2025-01-01T12:00:00Z,115,12,135,36.2,99,0.7,0.6,4.5,340
EDGE002,2025-01-01T00:00:00Z,45,8,160,35.5,100,0.5,0.4,3.5,400
EDGE002,2025-01-01T06:00:00Z,48,9,155,35.6,100,0.5,0.5,3.8,390
EDGE002,2025-01-01T12:00:00Z,50,10,150,35.8,99,0.6,0.5,4.0,380
EDGE003,2025-01-01T00:00:00Z,72,14,125,36.8,97,0.9,0.9,7.0,250
EDGE003,2025-01-01T06:00:00Z,72,14,125,36.8,97,0.9,0.9,7.0,250
EDGE003,2025-01-01T12:00:00Z,72,14,125,36.8,97,0.9,0.9,7.0,250
EDGE004,2025-01-01T00:00:00Z,90,18,110,37.5,94,1.3,1.5,10.0,200
EDGE004,2025-01-01T06:00:00Z,80,15,120,37.0,96,1.0,1.1,8.0,230
EDGE004,2025-01-01T12:00:00Z,72,13,130,36.6,98,0.8,0.8,6.5,260
EDGE005,2025-01-01T00:00:00Z,85,16,115,37.2,95,1.2,1.3,9.0,220
EDGE005,2025-01-01T06:00:00Z,110,25,90,38.5,88,2.5,3.5,18.0,100
EDGE005,2025-01-01T12:00:00Z,75,14,125,36.8,97,1.0,1.0,7.5,250
EDGE006,2025-01-01T00:00:00Z,95,20,100,38.0,92,2.0,2.5,15.0,150
EDGE006,2025-01-01T06:00:00Z,95,20,100,38.0,92,2.0,2.5,15.0,150
EDGE006,2025-01-01T12:00:00Z,95,20,100,38.0,92,2.0,2.5,15.0,150"""

# Expected:
# EDGE001: Tachycardia only, otherwise healthy - single domain, low-moderate
# EDGE002: Bradycardia + hypothermia + hypertension - unusual but not critical
# EDGE003: Perfectly stable (no change) - should be very low risk
# EDGE004: Improving patient (getting better) - should decrease risk
# EDGE005: Spike and recovery - transient deterioration
# EDGE006: Persistently abnormal but stable - moderate risk

DATASETS = {
    "sepsis_progression": {
        "data": DATASET_1_SEPSIS,
        "description": "10 patients with varying sepsis trajectories",
        "expected_high_risk": ["SEP003", "SEP005", "SEP007", "SEP009"],
        "expected_low_risk": ["SEP002", "SEP004", "SEP008"]
    },
    "cardiac_deterioration": {
        "data": DATASET_2_CARDIAC,
        "description": "8 patients with cardiac presentations",
        "expected_high_risk": ["CARD003", "CARD005", "CARD007"],
        "expected_low_risk": ["CARD002", "CARD004", "CARD006"]
    },
    "respiratory_failure": {
        "data": DATASET_3_RESPIRATORY,
        "description": "8 patients with respiratory presentations",
        "expected_high_risk": ["RESP003", "RESP005", "RESP007"],
        "expected_low_risk": ["RESP002", "RESP004"]
    },
    "mixed_acuity": {
        "data": DATASET_4_MIXED,
        "description": "12 patients with diverse acuity levels",
        "expected_high_risk": ["MIX002", "MIX004", "MIX010"],
        "expected_low_risk": ["MIX001", "MIX005", "MIX007", "MIX009"]
    },
    "edge_cases": {
        "data": DATASET_5_EDGE,
        "description": "6 patients with unusual patterns",
        "expected_high_risk": [],  # None expected
        "expected_low_risk": ["EDGE003"]  # Only the perfectly stable patient
    }
}


def test_dataset(name: str, dataset: Dict, mode: str = "balanced") -> Dict:
    """Test a dataset against the API and return results."""
    print(f"\n{'='*60}")
    print(f"TESTING: {name.upper()}")
    print(f"Description: {dataset['description']}")
    print(f"Mode: {mode}")
    print(f"{'='*60}")

    url = f"{API_BASE}/early_risk_discovery"

    try:
        # scoring_mode goes in the body, not query params
        response = requests.post(
            url,
            json={"csv": dataset["data"], "scoring_mode": mode},
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        if response.status_code != 200:
            print(f"ERROR: HTTP {response.status_code}")
            print(response.text[:500])
            return {"error": response.status_code, "message": response.text[:500]}

        result = response.json()

        # Extract hybrid scoring results
        cp = result.get("comparator_performance", {})
        hybrid = cp.get("hybrid_multisignal", {})

        print(f"\n--- HYBRID SCORING RESULTS ---")
        print(f"Risk Score: {hybrid.get('risk_score', 'N/A')}")
        print(f"Risk Level: {hybrid.get('risk_level', 'N/A')}")
        print(f"Domains Alerting: {hybrid.get('domains_alerting', 'N/A')}")
        print(f"Domain Breakdown: {hybrid.get('domain_alert_counts', {})}")
        print(f"Patients Alerting: {hybrid.get('patients_alerting', 'N/A')}/{hybrid.get('patients_analyzed', 'N/A')}")

        # Check high-risk patients
        high_risk = hybrid.get("high_risk_patients", [])
        if isinstance(high_risk, list):
            high_risk_ids = [p.get("patient_id") if isinstance(p, dict) else p for p in high_risk]
        else:
            high_risk_ids = []

        print(f"\nHigh Risk Patients Detected: {high_risk_ids}")
        print(f"Expected High Risk: {dataset['expected_high_risk']}")

        # Calculate accuracy
        expected_high = set(dataset['expected_high_risk'])
        detected_high = set(high_risk_ids)

        true_positives = expected_high & detected_high
        false_negatives = expected_high - detected_high
        false_positives = detected_high - expected_high

        print(f"\n--- ACCURACY ---")
        print(f"True Positives: {list(true_positives)}")
        print(f"Missed (FN): {list(false_negatives)}")
        print(f"Extra (FP): {list(false_positives)}")

        # Validation reference check
        vr = hybrid.get("validation_reference", {})
        print(f"\n--- VALIDATION REFERENCE ---")
        print(f"Note: {vr.get('note', 'N/A')}")
        print(f"Sensitivity: {vr.get('sensitivity', 'N/A')}")
        print(f"Specificity: {vr.get('specificity', 'N/A')}")
        print(f"PPV @ 5%: {vr.get('ppv_at_5_percent_prevalence', 'N/A')}")

        return {
            "dataset": name,
            "mode": mode,
            "risk_score": hybrid.get("risk_score"),
            "risk_level": hybrid.get("risk_level"),
            "domains_alerting": hybrid.get("domains_alerting"),
            "domain_counts": hybrid.get("domain_alert_counts"),
            "patients_analyzed": hybrid.get("patients_analyzed"),
            "patients_alerting": hybrid.get("patients_alerting"),
            "high_risk_detected": high_risk_ids,
            "expected_high_risk": dataset["expected_high_risk"],
            "true_positives": list(true_positives),
            "false_negatives": list(false_negatives),
            "false_positives": list(false_positives),
            "validation_reference": vr
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return {"error": str(e)}


def run_all_validations():
    """Run comprehensive validation across all datasets and modes."""
    print("="*60)
    print("HYPERCORE COMPREHENSIVE CROSS-DATASET VALIDATION")
    print("="*60)
    print(f"API: {API_BASE}")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Modes to test: high_confidence, balanced, screening")

    results = []

    # Test each dataset with each mode
    for mode in ["high_confidence", "balanced", "screening"]:
        print(f"\n\n{'#'*60}")
        print(f"# MODE: {mode.upper()}")
        print(f"{'#'*60}")

        for name, dataset in DATASETS.items():
            result = test_dataset(name, dataset, mode)
            results.append(result)
            time.sleep(1)  # Rate limiting

    # Summary
    print("\n\n" + "="*60)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("="*60)

    # Group by mode
    for mode in ["high_confidence", "balanced", "screening"]:
        mode_results = [r for r in results if r.get("mode") == mode and "error" not in r]

        print(f"\n--- {mode.upper()} MODE ---")

        total_tp = 0
        total_fn = 0
        total_fp = 0

        for r in mode_results:
            tp = len(r.get("true_positives", []))
            fn = len(r.get("false_negatives", []))
            fp = len(r.get("false_positives", []))
            total_tp += tp
            total_fn += fn
            total_fp += fp

            print(f"  {r['dataset']}: TP={tp}, FN={fn}, FP={fp}")

        if total_tp + total_fn > 0:
            sensitivity = total_tp / (total_tp + total_fn) * 100
            print(f"  TOTAL: TP={total_tp}, FN={total_fn}, FP={total_fp}")
            print(f"  Detection Rate: {sensitivity:.1f}%")

    return results


if __name__ == "__main__":
    results = run_all_validations()

    # Save results
    with open("validation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\nResults saved to validation_results.json")
