import json
import sys

f = open('C:/Users/letsa/icu_result_4509.json')
d = json.load(f)

print('=== 4509 PATIENT PRODUCTION TEST ===')
print(f'Success: {d.get("success")}')
print(f'Patient Count: {d.get("patient_count")}')
print(f'Analysis Type: {d.get("analysis_type")}')
print(f'Endpoints: {len(d.get("endpoints_analyzed", []))}')

summary = d.get('summary', {})
print(f'Overall Risk: {summary.get("overall_risk")}')
print(f'Risk Distribution: {summary.get("risk_distribution")}')

# CSE Distribution
patients = d.get('patient_results', [])
states = {}
decisions = {}
for p in patients:
    s = p.get('state_label', '?')
    dc = p.get('decision', '?')
    states[s] = states.get(s, 0) + 1
    decisions[dc] = decisions.get(dc, 0) + 1

print()
print('Clinical State Distribution:')
for k, v in sorted(states.items(), key=lambda x: -x[1]):
    pct = round(v/len(patients)*100, 1)
    print(f'  {k}: {v} ({pct}%)')

print()
print('Utility Decision Distribution:')
for k, v in sorted(decisions.items(), key=lambda x: -x[1]):
    pct = round(v/len(patients)*100, 1)
    print(f'  {k}: {v} ({pct}%)')

# Verify all patients have CSE fields
has_cse = sum(1 for p in patients if p.get('clinical_state') is not None)
has_utility = sum(1 for p in patients if p.get('utility_score') is not None)
has_actions = sum(1 for p in patients if p.get('immediate_actions'))

print()
print('Field Coverage:')
print(f'  Patients with clinical_state: {has_cse}/{len(patients)}')
print(f'  Patients with utility_score: {has_utility}/{len(patients)}')
print(f'  Patients with immediate_actions: {has_actions}/{len(patients)}')
