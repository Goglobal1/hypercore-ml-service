"""
Test Disease Detection with Aaron's Labs
Verify that AKI is NOT detected when kidney values are normal
"""

from app.core.discovery.disease_detection import analyze_patient_validated

# Aaron's labs - ALL NORMAL kidney values
aaron = {
    'patient_id': 'Aaron T Adams',
    'creatinine': 1.04,  # Normal (0.6-1.3)
    'bun': 11,           # Normal (7-20)
    'egfr': 89,          # Normal (>=60)
    'glucose': 115,      # Pre-diabetic (100-125)
    'sodium': 140,
    'potassium': 4.2,
    'ast': 28,
    'alt': 32,
    'alkaline_phosphatase': 148,  # Slightly elevated
    'albumin': 4.2,
    'hemoglobin': 14.5,
}

result = analyze_patient_validated(aaron)

print("=" * 60)
print("DISEASE DETECTION TEST - Aaron's Labs")
print("=" * 60)
print()
print(f"Patient: {result['patient_id']}")
print(f"Clinical State: {result['clinical_state']} - {result['state_label']}")
print(f"Risk Score: {result['risk_score']}")
print()
print("CONDITIONS DETECTED:")
if not result['conditions']:
    print("  (None)")
else:
    for c in result['conditions']:
        print(f"  - {c['disease']}: {c['confidence']*100:.0f}%")
        for e in c.get('evidence', []):
            print(f"      {e}")
print()
print("ABNORMAL VALUES:")
if not result['abnormal_values']:
    print("  (None)")
else:
    for a in result['abnormal_values']:
        print(f"  - {a['marker']}: {a['value']} {a['unit']} ({a['status']})")

print()
print("=" * 60)
print("VALIDATION:")
# Check that AKI was NOT detected
aki_detected = any('Kidney Injury' in c.get('disease', '') for c in result['conditions'])
ckd_detected = any('Kidney Disease' in c.get('disease', '') for c in result['conditions'])

if aki_detected:
    print("FAIL: AKI was incorrectly detected with normal kidney values!")
else:
    print("PASS: AKI correctly NOT detected (kidney values normal)")

if ckd_detected:
    print("FAIL: CKD was incorrectly detected with normal eGFR!")
else:
    print("PASS: CKD correctly NOT detected (eGFR 89 is normal)")

# Check that pre-diabetes WAS detected
prediabetes_detected = any('diabetes' in c.get('disease', '').lower() for c in result['conditions'])
if prediabetes_detected:
    print("PASS: Pre-diabetes correctly detected (glucose 115)")
else:
    print("NOTE: Pre-diabetes not detected")
print("=" * 60)
