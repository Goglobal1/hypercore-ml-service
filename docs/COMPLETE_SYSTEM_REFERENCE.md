# HyperCore ML Service - Complete System Reference

**Version:** 5.1 (Production)
**Total Endpoints:** 152
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Architecture](#module-architecture)
3. [Endpoint Categories](#endpoint-categories)
4. [Data Flow](#data-flow)
5. [Base44 Integration Guide](#base44-integration-guide)
6. [Detailed Endpoint Reference](#detailed-endpoint-reference)

---

## System Overview

HyperCore ML Service is a unified medical AI API providing:

- **Clinical Decision Support** - Risk assessment, state tracking, alerting
- **Genomics Analysis** - Gene expression, variants, phenotype correlations
- **Pharmaceutical Intelligence** - Drug profiles, interactions, pharmacogenomics
- **Pathogen Detection** - Outbreak surveillance, AMR analysis
- **Multi-Omic Fusion** - Cross-source data integration
- **AI Agent Orchestration** - Multi-agent diagnostic reasoning

### Core Capabilities

| Capability | Status | Module |
|------------|--------|--------|
| Auto Risk Scoring | Active | `alert_system/risk_calculator.py` |
| Clinical State Engine | Active | `alert_system/engine.py` |
| Time-to-Harm Prediction | Active | `core/time_to_harm.py` |
| Real-time Alerts | Active | `alert_system/realtime.py` |
| Genomics Integration | Active | `core/genomics_integration.py` |
| Drug Response Prediction | Active | `core/drug_response_predictor.py` |
| Pathogen Detection | Active | `core/pathogen_detection.py` |
| Multi-Agent Orchestration | Active | `agents/` |

---

## Module Architecture

```
hypercore-ml-service/
├── main.py                      # FastAPI app, 70+ endpoints
├── app/
│   ├── agents/                  # AI Diagnostic Agents
│   │   ├── base_agent.py        # Agent base class, registry
│   │   ├── biomarker_agent.py   # Biomarker interpretation
│   │   ├── diagnostic_agent.py  # Differential diagnosis
│   │   ├── trial_rescue_agent.py# Trial matching, rescue therapy
│   │   └── surveillance_agent.py# Population surveillance
│   │
│   ├── core/                    # Core processing engines
│   │   ├── alert_system/        # Unified alert system (9 files)
│   │   │   ├── __init__.py      # Module exports
│   │   │   ├── models.py        # ClinicalState, AlertSeverity
│   │   │   ├── engine.py        # Clinical State Engine
│   │   │   ├── storage.py       # Event/state storage
│   │   │   ├── pipeline.py      # 11-step processing pipeline
│   │   │   ├── routing.py       # Alert routing rules
│   │   │   ├── realtime.py      # WebSocket/SSE hub
│   │   │   ├── config.py        # Biomarker thresholds
│   │   │   ├── risk_calculator.py # Auto risk scoring
│   │   │   └── router.py        # FastAPI router (26 endpoints)
│   │   │
│   │   ├── genomics_integration.py  # ClinVar, GEO integration
│   │   ├── drug_response_predictor.py # FAERS, AACT integration
│   │   ├── pathogen_detection.py    # WHO surveillance
│   │   ├── multiomic_fusion.py      # Multi-source fusion
│   │   ├── time_to_harm.py          # TTH prediction
│   │   └── clinical_state_engine.py # Legacy CSE
│   │
│   ├── routers/                 # API routers
│   │   ├── genomics_router.py   # 6 endpoints
│   │   ├── pharmaceutical_router.py # 11 endpoints
│   │   ├── pathogen_router.py   # 14 endpoints
│   │   ├── multiomic_router.py  # 11 endpoints
│   │   └── agents_router.py     # 17 endpoints
│   │
│   └── models/                  # Pydantic models
│       ├── genomics_models.py
│       ├── pharmaceutical_models.py
│       ├── pathogen_models.py
│       └── multiomic_models.py
```

---

## Endpoint Categories

### A. ALERT SYSTEM ENDPOINTS (/alerts/*) - 26 endpoints

Core patient monitoring and clinical alerting system.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/alerts/patient/intake` | **Primary intake** - processes patient data through 11-step pipeline |
| POST | `/alerts/evaluate` | Direct CSE evaluation |
| POST | `/alerts/acknowledge` | Acknowledge alert, stop escalation |
| GET | `/alerts/patient/{id}/state` | Get patient states across domains |
| GET | `/alerts/patient/{id}/state/{domain}` | Get state for specific domain |
| GET | `/alerts/episodes` | List open episodes |
| GET | `/alerts/episodes/{id}` | Get specific episode |
| GET | `/alerts/events` | Query audit log |
| GET | `/alerts/events/{id}` | Get specific event |
| GET | `/alerts/acknowledgments` | Get acknowledgment records |
| GET | `/alerts/routing/rules` | Get routing rules |
| GET | `/alerts/routing/escalations` | Get pending escalations |
| GET | `/alerts/config/domains` | Get all domain configs |
| GET | `/alerts/config/domains/{domain}` | Get domain config |
| GET | `/alerts/config/biomarkers` | Get all biomarker thresholds |
| GET | `/alerts/config/biomarkers/{domain}` | Get domain biomarkers |
| GET | `/alerts/stats` | Get system statistics |
| WS | `/alerts/ws` | WebSocket real-time updates |
| GET | `/alerts/sse` | Server-Sent Events stream |
| GET | `/alerts/health` | Alert system health check |

**Legacy Alert Endpoints (main.py):**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/alerts/evaluate` | Evaluate patient alert |
| POST | `/alerts/state` | Get/update patient state |
| POST | `/alerts/history` | Get alert history |
| GET | `/alerts/config` | Get ATC configuration |

---

### B. GENOMICS ENDPOINTS (/genomics/*) - 6 endpoints

Gene expression, variants, and clinical correlation.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/genomics/health` | Module health check |
| GET | `/genomics/expression/{gene}` | Get gene expression from GEO |
| GET | `/genomics/variants/{gene}` | Get ClinVar variants |
| POST | `/genomics/analyze` | Comprehensive genomics analysis |
| GET | `/genomics/series` | List available GEO series |
| GET | `/genomics/phenotype-icd10/{phenotype}` | Phenotype to ICD-10 mapping |

---

### C. PHARMACEUTICAL ENDPOINTS (/pharma/*) - 11 endpoints

Drug profiles, interactions, and pharmacogenomics.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/pharma/health` | Module health check |
| GET | `/pharma/drug/{name}` | Get comprehensive drug profile |
| GET | `/pharma/adverse-events/{drug}` | Get FAERS adverse events |
| GET | `/pharma/trials/{condition}` | Search trials by condition |
| GET | `/pharma/trials-by-drug/{drug}` | Search trials by drug |
| POST | `/pharma/predict-response` | Predict drug response (pharmacogenomics) |
| POST | `/pharma/interaction-check` | Check drug-drug interactions |
| GET | `/pharma/pharmacogenomics/{gene}` | Get gene's affected drugs |
| GET | `/pharma/genes` | List all pharmacogenes |
| GET | `/pharma/faers-quarters` | List available FAERS quarters |

---

### D. PATHOGEN ENDPOINTS (/pathogen/*) - 14 endpoints

Pathogen detection, surveillance, and AMR analysis.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/pathogen/health` | Module health check |
| GET | `/pathogen/info/{pathogen}` | Get pathogen details |
| GET | `/pathogen/disease/{disease}` | Get pathogens for disease |
| GET | `/pathogen/list` | List all pathogens |
| GET | `/pathogen/diseases` | List disease-pathogen mappings |
| POST | `/pathogen/outbreak-detection` | Detect outbreaks from surveillance |
| GET | `/pathogen/outbreak-quick` | Quick outbreak check |
| POST | `/pathogen/amr-analysis` | AMR pattern analysis |
| GET | `/pathogen/amr/{pathogen}` | Get AMR for pathogen |
| GET | `/pathogen/vaccination` | Get vaccination coverage |
| POST | `/pathogen/search` | Search WHO surveillance |
| GET | `/pathogen/surveillance` | Search surveillance (GET) |
| GET | `/pathogen/correlation/{pathogen}` | Clinical-pathogen correlation |

---

### E. MULTIOMIC ENDPOINTS (/multiomic/*) - 11 endpoints

Cross-source data fusion and queries.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/multiomic/health` | Module health check |
| GET | `/multiomic/sources` | List data sources |
| GET | `/multiomic/query/gene/{gene}` | Gene-centric query |
| GET | `/multiomic/query/disease/{disease}` | Disease-centric query |
| GET | `/multiomic/query/drug/{drug}` | Drug-centric query |
| POST | `/multiomic/query/unified` | Unified multi-omic query |
| POST | `/multiomic/analyze/fusion` | Multi-omic fusion analysis |
| GET | `/multiomic/layers` | List omic layers |
| GET | `/multiomic/pharmacogenomics/{drug}` | Drug pharmacogenomics |
| GET | `/multiomic/gene-disease/{gene}` | Gene-disease associations |

---

### F. AGENT ENDPOINTS (/agents/*) - 17 endpoints

AI diagnostic agent system.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/agents/health` | All agents health check |
| GET | `/agents/list` | List available agents |
| GET | `/agents/status/{type}` | Get agent status |
| POST | `/agents/biomarker/analyze` | Biomarker interpretation |
| POST | `/agents/diagnostic/analyze` | Differential diagnosis |
| POST | `/agents/trial-rescue/analyze` | Trial matching, rescue therapy |
| POST | `/agents/surveillance/analyze` | Population surveillance |
| POST | `/agents/orchestrate` | Multi-agent orchestration |
| GET | `/agents/findings` | Get shared findings |
| DELETE | `/agents/findings` | Clear session findings |
| GET | `/agents/diagnostic/diagnoses` | List diagnosis patterns |
| GET | `/agents/diagnostic/diagnoses/{dx}` | Get diagnosis info |
| GET | `/agents/biomarker/panels` | List biomarker panels |
| GET | `/agents/biomarker/panels/{cat}` | Get panel details |
| GET | `/agents/trial-rescue/categories` | List rescue categories |
| GET | `/agents/surveillance/regions` | List regional profiles |
| GET | `/agents/surveillance/regions/{r}` | Get regional profile |

---

### G. ANALYSIS & PREDICTION ENDPOINTS - 25 endpoints

Core ML analysis and prediction capabilities.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/analyze` | **Primary analysis** - HyperCore pipeline |
| POST | `/predict` | Mortality/outcome prediction |
| GET | `/predict/capabilities` | List prediction capabilities |
| POST | `/predict/time-to-harm` | Time-to-harm prediction |
| GET | `/predict/time-to-harm/domains` | TTH supported domains |
| POST | `/early_risk_discovery` | Early risk factor discovery |
| POST | `/multi_omic_fusion` | Multi-omic fusion analysis |
| POST | `/confounder_detection` | Detect confounders |
| POST | `/confounder_analysis` | Analyze confounders |
| POST | `/emerging_phenotype` | Detect emerging phenotypes |
| POST | `/responder_prediction` | Predict treatment responders |
| POST | `/trial_rescue` | Trial rescue recommendations |
| POST | `/outbreak_detection` | Outbreak detection |
| POST | `/predictive_modeling` | Build predictive models |
| POST | `/synthetic_cohort` | Generate synthetic cohort |
| POST | `/digital_twin_simulation` | Digital twin simulation |
| POST | `/population_risk` | Population risk assessment |
| POST | `/forecast_timeline` | Forecast patient timeline |
| POST | `/root_cause_sim` | Root cause simulation |
| POST | `/patient_report` | Generate patient report |
| POST | `/cross_loop` | Cross-loop analysis |
| POST | `/shap_explain` | SHAP explanations |
| POST | `/change_point_detect` | Change point detection |
| POST | `/lead_time_analysis` | Lead time analysis |

---

### H. SURVEILLANCE ENDPOINTS - 4 endpoints

Population-level disease surveillance.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/surveillance/unknown_diseases` | Unknown disease detection |
| POST | `/surveillance/outbreak_detection` | Outbreak detection |
| POST | `/surveillance/multisite_synthesis` | Multi-site synthesis |
| POST | `/surveillance/comprehensive` | Comprehensive surveillance |

---

### I. ORACLE/PROTEUS ENDPOINTS - 12 endpoints

Advanced AI system (internal).

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/oracle/execute` | Execute Oracle analysis |
| GET | `/oracle/agents` | List Oracle agents |
| GET | `/oracle/performance` | Oracle performance stats |
| GET | `/oracle/health` | Oracle health check |
| POST | `/oracle/activate_clone` | Activate Oracle clone |
| POST | `/proteus/generate_cohort` | Generate cohort |
| POST | `/proteus/validate_model` | Validate model |
| POST | `/proteus/ab_test` | A/B testing |
| POST | `/sentinel/assess` | Sentinel assessment |
| GET | `/sentinel/statistics` | Sentinel statistics |
| POST | `/honeypot/interact` | Honeypot interaction |
| GET | `/honeypot/telemetry` | Honeypot telemetry |

---

### J. GOVERNANCE ENDPOINTS - 12 endpoints

HIPAA compliance, audit, consent management.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/governance/policy/evaluate` | Evaluate governance policy |
| GET | `/governance/audit/{session_id}` | Get audit log |
| GET | `/governance/audit/{session_id}/verify` | Verify audit integrity |
| POST | `/governance/audit/append` | Append audit entry |
| POST | `/governance/evidence/build` | Build evidence package |
| POST | `/governance/blockchain/anchor` | Anchor to blockchain |
| POST | `/governance/blockchain/verify` | Verify blockchain anchor |
| POST | `/governance/consent/record` | Record patient consent |
| POST | `/governance/consent/withdraw` | Withdraw consent |
| GET | `/governance/consent/{patient_ref}` | Get consent status |
| GET | `/governance/status` | Governance system status |

---

### K. SECURITY ENDPOINTS - 4 endpoints

PHI protection and security scanning.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/security/phi-scan` | Scan for PHI exposure |
| POST | `/security/deidentify` | De-identify patient data |
| GET | `/security/audit-logs` | Get security audit logs |
| GET | `/security/status` | Security system status |

---

### L. ASTRA (Conversational AI) - 4 endpoints

Natural language interface.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/astra/query` | Submit natural language query |
| GET | `/astra/conversation/{session_id}` | Get conversation history |
| GET | `/astra/capabilities` | List ASTRA capabilities |
| GET | `/astra/status` | ASTRA system status |

---

### M. OTHER ENDPOINTS - 10 endpoints

Utility and system endpoints.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | System health check |
| POST | `/fluview_ingest` | FluView data ingestion |
| POST | `/create_digital_twin` | Create digital twin |
| POST | `/medication_interaction` | Medication interaction check |
| POST | `/lucian/respond` | Lucian response |
| GET | `/lucian/statistics` | Lucian statistics |
| GET | `/obsidian/validate` | Obsidian validation |
| GET | `/obsidian/summary` | Obsidian summary |
| POST | `/obsidian/add_block` | Add Obsidian block |
| POST | `/trinity/process` | Trinity processing |
| GET | `/trinity/posture` | Trinity posture |
| GET | `/trinity/integrity` | Trinity integrity |
| GET | `/encryption/status` | Encryption status |
| POST | `/encryption/hash` | Hash data |
| POST | `/encryption/split_secret` | Split secret |

---

## Data Flow

### Patient Intake Flow (Primary)

```
Client Request
     │
     ▼
POST /alerts/patient/intake
     │
     ├─► Step 1: Receive patient data
     │   {patient_id, risk_domain, lab_data, vitals_data}
     │
     ├─► Step 2: Calculate Risk Score (if not provided)
     │   risk_calculator.py → weighted biomarker analysis
     │
     ├─► Step 3: Clinical State Engine (CSE)
     │   engine.py → evaluate state transition (S0→S3)
     │
     ├─► Step 4: Agent Invocation (conditional)
     │   On state change or high severity → invoke agents
     │
     ├─► Step 5: Cross-Validation
     │   Validate across multiple data sources
     │
     ├─► Step 6: Time-to-Harm Prediction
     │   time_to_harm.py → predict deterioration timeline
     │
     ├─► Step 7: Confidence Scoring
     │   Calculate evidence confidence
     │
     ├─► Step 8: Alert Decision
     │   Determine if alert should fire
     │
     ├─► Step 9: Alert Routing (if fired)
     │   routing.py → determine recipients
     │
     ├─► Step 10: Audit Logging
     │   storage.py → persist event
     │
     └─► Step 11: Dashboard Notification
         realtime.py → WebSocket/SSE push
     │
     ▼
Response: {
  evaluation: {state, severity, risk_score, transition},
  alerts: [{recipients, severity}],
  tth: {predicted_hours, confidence},
  processing_time_ms
}
```

### Risk Score Calculation

```
Raw Biomarkers (lab_data + vitals_data)
     │
     ▼
Normalize Names → "WBC" → "wbc", "HR" → "heart_rate"
     │
     ▼
For each biomarker:
  ├─► Look up threshold (warning, critical)
  ├─► Determine direction (rising/falling)
  ├─► Calculate score (0.0 - 1.0)
  └─► Apply weight factor
     │
     ▼
Composite Score = Σ(score × weight) / Σ(weight)
     │
     ▼
Multi-Critical Boost (if applicable)
     │
     ▼
Final Risk Score (0.0 - 1.0)
```

### Clinical State Mapping

| Risk Score | State | Name | Action |
|------------|-------|------|--------|
| 0.00 - 0.25 | S0 | Stable | Routine monitoring |
| 0.25 - 0.50 | S1 | Watch | Enhanced monitoring |
| 0.50 - 0.75 | S2 | Escalating | Clinical evaluation |
| 0.75 - 1.00 | S3 | Critical | Immediate intervention |

---

## Base44 Integration Guide

### Overview

Base44 is the frontend dashboard that consumes HyperCore ML Service APIs. This section documents integration patterns.

### Primary Integration Points

#### 1. Patient Intake (Real-time Monitoring)

**Use Case:** Submit patient vitals/labs for continuous risk assessment

```javascript
// POST /alerts/patient/intake
const response = await fetch('/alerts/patient/intake', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    patient_id: 'P12345',
    risk_domain: 'sepsis',
    lab_data: {
      lactate: 4.5,
      wbc: 16.0,
      crp: 150
    },
    vitals_data: {
      heart_rate: 115,
      respiratory_rate: 24,
      temperature: 38.9
    },
    location: 'ICU-2A'
  })
});

const result = await response.json();
// result.evaluation.state → "S3"
// result.evaluation.severity → "critical"
// result.alerts → [{severity: "critical", recipients: [...]}]
```

#### 2. Real-time Updates (WebSocket)

**Use Case:** Receive push notifications for alerts

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/alerts/ws');

ws.onopen = () => {
  // Subscribe to specific patients/domains
  ws.send(JSON.stringify({
    type: 'subscribe',
    patient_ids: ['P12345', 'P12346'],
    risk_domains: ['sepsis', 'cardiac'],
    roles: ['attending']
  }));
};

ws.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  // Handle real-time alert
  if (alert.type === 'state_change') {
    updatePatientCard(alert.patient_id, alert.new_state);
  } else if (alert.type === 'alert_fired') {
    showNotification(alert);
  }
};
```

#### 3. Server-Sent Events (Alternative)

```javascript
// Connect to SSE
const eventSource = new EventSource(
  '/alerts/sse?patient_ids=P12345&risk_domains=sepsis'
);

eventSource.onmessage = (event) => {
  const alert = JSON.parse(event.data);
  handleAlert(alert);
};
```

#### 4. Patient State Queries

```javascript
// Get all states for a patient
const states = await fetch('/alerts/patient/P12345/state');

// Get specific domain state
const sepsisState = await fetch('/alerts/patient/P12345/state/sepsis');
```

#### 5. Acknowledge Alerts

```javascript
// Acknowledge alert
await fetch('/alerts/acknowledge', {
  method: 'POST',
  body: JSON.stringify({
    alert_id: 'EVT_abc123',
    acknowledged_by: 'Dr. Smith',
    action_taken: 'Initiated antibiotics',
    close_episode: false
  })
});
```

### Supported Risk Domains

| Domain | Description | Key Biomarkers |
|--------|-------------|----------------|
| sepsis | Infection/sepsis | lactate, WBC, CRP, procalcitonin |
| cardiac | Cardiovascular | troponin, BNP, heart_rate |
| kidney | Renal function | creatinine, BUN, potassium, GFR |
| respiratory | Pulmonary | SpO2, PaO2, respiratory_rate |
| hepatic | Liver function | ALT, AST, bilirubin, INR |
| neurological | Neurological | GCS, ICP, CPP |
| metabolic | Metabolic | glucose, pH, potassium, lactate |

### Response Schema Reference

#### Patient Intake Response

```json
{
  "patient_id": "P12345",
  "risk_domain": "sepsis",
  "pipeline_result": {
    "evaluation": {
      "patient_id": "P12345",
      "risk_domain": "sepsis",
      "risk_score": 0.85,
      "previous_state": "S1",
      "current_state": "S3",
      "state_changed": true,
      "transition_type": "escalation",
      "severity": "critical",
      "contributing_biomarkers": ["lactate", "wbc", "crp"],
      "timestamp": "2026-03-19T10:30:00Z"
    },
    "alerts": [
      {
        "severity": "critical",
        "recipients": ["attending", "charge_nurse"],
        "message": "Patient escalated to S3 Critical"
      }
    ],
    "tth_prediction": {
      "predicted_hours": 2.5,
      "confidence": 0.78,
      "harm_type": "organ_failure"
    },
    "processing_time_ms": 45
  }
}
```

#### State Query Response

```json
{
  "patient_id": "P12345",
  "risk_domain": "sepsis",
  "current_state": "S3",
  "state_name": "Critical",
  "risk_score": 0.85,
  "episode_id": "EP_xyz789",
  "last_transition": "2026-03-19T10:30:00Z",
  "time_in_state_minutes": 15
}
```

### Error Handling

| Status Code | Meaning | Action |
|-------------|---------|--------|
| 200 | Success | Process response |
| 400 | Bad request | Check request format |
| 404 | Not found | Patient/episode doesn't exist |
| 500 | Server error | Retry with backoff |

### Rate Limits

- Patient intake: 100 requests/minute per patient
- Queries: 1000 requests/minute
- WebSocket: 1 connection per dashboard session

---

## Detailed Endpoint Reference

For detailed API documentation of each endpoint, see:

- [API_REFERENCE.md](./API_REFERENCE.md) - Alert system endpoints
- [RISK_SCORING_GUIDE.md](./RISK_SCORING_GUIDE.md) - Risk calculation details

### Quick Reference: Most Common Endpoints

| Use Case | Endpoint |
|----------|----------|
| Submit patient data | `POST /alerts/patient/intake` |
| Get patient state | `GET /alerts/patient/{id}/state` |
| Real-time updates | `WS /alerts/ws` |
| Acknowledge alert | `POST /alerts/acknowledge` |
| Gene variants | `GET /genomics/variants/{gene}` |
| Drug interactions | `POST /pharma/interaction-check` |
| System health | `GET /health` |

---

## Appendix: Environment Configuration

### Required Environment Variables

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Data paths
GEO_DATA_PATH=F:/DATASETS/GENOMICS/GEO/
CLINVAR_PATH=F:/DATASETS/GENETICS/ClinVar/
FAERS_PATH=F:/DATASETS/PHARMACEUTICAL/FAERS/
AACT_PATH=F:/DATASETS/PHARMACEUTICAL/AACT/
WHO_GHO_PATH=F:/DATASETS/POPULATION/WHO-GHO/

# Alert system
ALERT_STORAGE_TYPE=memory  # or: postgres, redis
ALERT_ENABLE_REALTIME=true
```

### Starting the Server

```bash
cd hypercore-ml-service
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Risk calculator tests only
pytest tests/test_risk_calculator.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

---

**Document Version:** 1.0
**Last Updated:** 2026-03-19
**Maintained by:** HyperCore ML Team
