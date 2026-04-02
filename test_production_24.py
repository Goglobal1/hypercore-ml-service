"""Test 24-endpoint system on Railway production."""
import requests
import json
import time

base_url = 'https://hypercore-ml-service-production.up.railway.app'
compare_url = f'{base_url}/compare'

# Check health first
print('Checking Railway health...')
try:
    health = requests.get(f'{base_url}/health', timeout=10)
    if health.status_code == 200:
        h = health.json()
        print(f"  Version: {h.get('version')}")
        print(f"  Status: {h.get('status')}")
except Exception as e:
    print(f"  Health check failed: {e}")
print()

# CSV format expected by production endpoint
csv_data = """patient_id,heart_rate,respiratory_rate,sbp,dbp,spo2,temperature,lactate,creatinine,glucose,wbc,outcome
P001,95,24,100,65,94,38.2,3.5,1.8,180,14.5,1
P002,72,16,120,80,98,36.8,1.0,1.0,95,7.5,0
P003,110,28,85,55,91,38.8,4.2,2.1,220,18.0,1
"""

payload = {
    'csv': csv_data,
    'mode': 'high_confidence'
}

url = compare_url

print('='*60)
print('Testing HyperCore 24-Endpoint System on Railway Production')
print('='*60)
print()
print(f'URL: {url}')
print()

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f'Status Code: {response.status_code}')

    if response.status_code == 200:
        result = response.json()
        print()
        print('=== HyperCore Response ===')
        print(f"Engine Version: {result.get('engine_version', 'unknown')}")

        # Check components from components_used (new structure)
        components_used = result.get('components_used', {})
        components = components_used.get('components', {})
        version = components.get('version', result.get('algorithm_version', 'unknown'))

        print(f"Engine Version: {version}")

        # Check for new 24-endpoint components
        has_24_endpoints = components.get('endpoints_24', False)
        has_cross_loop_v2 = components.get('cross_loop_v2', False)
        has_pathway_lib = components.get('pathway_library', False)
        has_handler_metrics = components.get('handler_metrics', False)

        print()
        print('New v3.0 Components:')
        print(f"  24 Endpoints: {'YES' if has_24_endpoints else 'NO (pending deploy)'}")
        print(f"  Cross-Loop V2: {'YES' if has_cross_loop_v2 else 'NO (pending deploy)'}")
        print(f"  Pathway Library: {'YES' if has_pathway_lib else 'NO (pending deploy)'}")
        print(f"  Handler Metrics: {'YES' if has_handler_metrics else 'NO (pending deploy)'}")
        print()

        # Get patient details (new response format)
        patients = result.get('patient_details', [])
        if patients:
            p = patients[0]
            print()
            print('Patient Analysis (P001):')
            print(f"  HyperCore Score: {p.get('hypercore_score', 'N/A')}")
            print(f"  Alert: {p.get('hypercore_alert', 'N/A')}")
            print(f"  Clinical State: {p.get('clinical_state', 'N/A')} ({p.get('state_name', '')})")
            print(f"  Domains: {p.get('domains_detected', [])}")

            cross_loop = p.get('cross_loop_analysis', {})
            if cross_loop:
                print()
                print('Cross-Loop Analysis:')
                print(f"  Endpoints Alerting: {cross_loop.get('endpoints_alerting', [])}")
                print(f"  N Alerting: {cross_loop.get('n_endpoints_alerting', 0)}")
                print(f"  Patterns: {cross_loop.get('cross_domain_patterns', [])}")
                print(f"  Convergence: {cross_loop.get('convergence_detected', False)}")
                print(f"  Convergence Score: {cross_loop.get('convergence_score', 0)}")
                print(f"  Multi-System: {cross_loop.get('multi_system_failure', False)}")
        else:
            hypercore = result.get('hypercore_result', {})
            print('HyperCore Analysis:')
            print(f"  Alert Fired: {hypercore.get('alert_fired')}")
            print(f"  Risk Score: {hypercore.get('risk_score')}")
            print(f"  Urgency: {hypercore.get('urgency')}")


        validation = result.get('clinical_validation', {})
        if validation:
            print()
            print('Clinical Validation (Handler Metrics):')
            print(f"  Sensitivity: {validation.get('sensitivity', 'N/A')}")
            print(f"  Specificity: {validation.get('specificity', 'N/A')}")
            print(f"  PPV: {validation.get('ppv', 'N/A')}")
            print(f"  PPV@5%: {validation.get('ppv_at_5_percent', 'N/A')}")
            print(f"  Lead Time: {validation.get('lead_time', 'N/A')}")

        # Test all three modes
        print()
        print('='*60)
        print('Testing All Three Modes')
        print('='*60)

        for mode in ['screening', 'balanced', 'high_confidence']:
            payload['mode'] = mode
            try:
                resp = requests.post(url, json=payload, timeout=30)
                if resp.status_code == 200:
                    r = resp.json()
                    patients = r.get('patient_details', [])
                    if patients:
                        p = patients[0]
                        cl = p.get('cross_loop_analysis', {})
                        score = p.get('hypercore_score', 0)
                        alert = p.get('hypercore_alert', False)
                        n_eps = cl.get('n_endpoints_alerting', 0)
                        print(f"\n{mode.upper()}:")
                        print(f"  Score: {score:.3f}, Alert: {alert}, Endpoints Alerting: {n_eps}")
            except Exception as e:
                print(f"\n{mode.upper()}: Error - {e}")

        print()
        print('='*60)
        print('PRODUCTION TEST COMPLETE')
        print('='*60)

    else:
        print(f'Error Response: {response.text[:500]}')

except requests.exceptions.RequestException as e:
    print(f'Request failed: {e}')
except Exception as e:
    print(f'Error: {e}')
