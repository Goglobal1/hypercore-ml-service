# Alert System Feature Matrix - Merge Analysis

## System A: hypercore-ml-service (Existing)
## System B: cse.py Requirements (Specified)

---

## State Model Features

| Feature | System A | System B | Merged |
|---------|----------|----------|--------|
| S0 Stable State | YES | YES | YES |
| S1 Watch State | YES | YES | YES |
| S2 Escalating State | YES | YES | YES |
| S3 Critical State | YES | YES | YES |
| Domain: sepsis | YES | NO | YES |
| Domain: cardiac | YES | NO | YES |
| Domain: kidney | YES | NO | YES |
| Domain: respiratory | YES | NO | YES |
| Domain: hepatic | YES | NO | YES |
| Domain: neurological | YES | NO | YES |
| Domain: oncology | YES | NO | YES |
| Domain: metabolic | YES | NO | YES |
| Domain: hematologic | YES | NO | YES |
| Domain: multi_system | YES | NO | YES |
| Domain: deterioration | NO | YES | YES |
| Domain: infection | NO | YES | YES |
| Domain: outbreak | NO | YES | YES |
| Domain: trial_confounder | NO | YES | YES |
| Domain: custom | NO | YES | YES |
| Per-site threshold overrides | NO | YES | YES |

## Biomarker Thresholds

| Domain | Biomarker | Warning | Critical | Direction | In Merged |
|--------|-----------|---------|----------|-----------|-----------|
| Sepsis | lactate | 2.0 | 4.0 | rising | YES |
| Sepsis | MAP | 70.0 | 65.0 | falling | YES |
| Sepsis | CRP | 50.0 | 100.0 | rising | YES |
| Sepsis | WBC | 10.0 | 12.0 | rising | YES |
| Sepsis | procalcitonin | 0.5 | 2.0 | rising | YES |
| Sepsis | temperature | 37.8 | 38.3 | rising | YES |
| Sepsis | heart_rate | 100.0 | 120.0 | rising | YES |
| Sepsis | respiratory_rate | 20.0 | 24.0 | rising | YES |
| Cardiac | troponin | 0.01 | 0.04 | rising | YES |
| Cardiac | troponin_i | 0.01 | 0.04 | rising | YES |
| Cardiac | troponin_t | 0.03 | 0.1 | rising | YES |
| Cardiac | BNP | 100.0 | 400.0 | rising | YES |
| Cardiac | NT-proBNP | 300.0 | 900.0 | rising | YES |
| Cardiac | heart_rate | 110.0 | 130.0 | rising | YES |
| Cardiac | systolic_bp | 100.0 | 90.0 | falling | YES |
| Cardiac | diastolic_bp | 65.0 | 60.0 | falling | YES |
| Cardiac | CK-MB | 5.0 | 25.0 | rising | YES |
| Kidney | creatinine | 1.3 | 2.0 | rising | YES |
| Kidney | BUN | 25.0 | 40.0 | rising | YES |
| Kidney | potassium | 5.0 | 5.5 | rising | YES |
| Kidney | GFR | 60.0 | 30.0 | falling | YES |
| Kidney | urine_output | 1.0 | 0.5 | falling | YES |
| Kidney | sodium | 130.0 | 125.0 | falling | YES |
| Respiratory | SpO2 | 94.0 | 90.0 | falling | YES |
| Respiratory | PaO2 | 80.0 | 60.0 | falling | YES |
| Respiratory | respiratory_rate | 24.0 | 30.0 | rising | YES |
| Respiratory | FiO2 | 0.4 | 0.6 | rising | YES |
| Respiratory | P/F ratio | 300.0 | 200.0 | falling | YES |
| Respiratory | PaCO2 | 45.0 | 50.0 | rising | YES |
| Hepatic | ALT | 200.0 | 1000.0 | rising | YES |
| Hepatic | AST | 200.0 | 1000.0 | rising | YES |
| Hepatic | bilirubin | 2.0 | 4.0 | rising | YES |
| Hepatic | INR | 1.5 | 2.0 | rising | YES |
| Hepatic | albumin | 3.0 | 2.5 | falling | YES |
| Hepatic | ammonia | 50.0 | 100.0 | rising | YES |
| Neurological | GCS | 12.0 | 8.0 | falling | YES |
| Neurological | ICP | 15.0 | 22.0 | rising | YES |
| Neurological | CPP | 70.0 | 60.0 | falling | YES |
| Hematologic | hemoglobin | 9.0 | 7.0 | falling | YES |
| Hematologic | platelets | 100.0 | 50.0 | falling | YES |
| Hematologic | INR | 1.5 | 2.5 | rising | YES |
| Hematologic | fibrinogen | 200.0 | 100.0 | falling | YES |

## Break Rules

| Rule | System A | System B | Merged |
|------|----------|----------|--------|
| Velocity spike (>threshold/hr) | YES | YES | YES |
| Novelty detection (new biomarker top-3) | YES | YES | YES |
| TTH shortening (>25% decrease) | NO | YES | YES |
| Dwell escalation (S2+ >4hr no ack) | NO | YES | YES |

## Cooldown Configuration (minutes)

| Domain | Default | Escalation | Critical | In Merged |
|--------|---------|------------|----------|-----------|
| Sepsis | 30 | 10 | 5 | YES |
| Cardiac | 45 | 15 | 5 | YES |
| Kidney | 60 | 20 | 10 | YES |
| Respiratory | 30 | 10 | 5 | YES |
| Hepatic | 60 | 20 | 10 | YES |
| Neurological | 20 | 10 | 3 | YES |
| Metabolic | 45 | 15 | 10 | YES |
| Hematologic | 60 | 20 | 10 | YES |
| Oncology | 120 | 60 | 30 | YES |
| Multi-System | 20 | 10 | 3 | YES |

## Confidence Scoring

| Feature | System A | System B | Merged |
|---------|----------|----------|--------|
| VERY_LOW (0.0-0.19) | YES | YES | YES |
| LOW (0.2-0.39) | YES | YES | YES |
| MODERATE (0.4-0.59) | YES | YES | YES |
| HIGH (0.6-0.79) | YES | YES | YES |
| VERY_HIGH (0.8-0.94) | YES | YES | YES |
| DEFINITIVE (0.95-0.99) | YES | YES | YES |
| Multi-source boost | YES | YES | YES |
| Cross-loop validation | YES | YES | YES |
| Contradiction penalty | YES | YES | YES |
| Gap penalty | YES | YES | YES |

## Episode Management

| Feature | System A | System B | Merged |
|---------|----------|----------|--------|
| Episode opens on state transition | YES | YES | YES |
| Episode closes on resolve to S0 | YES | YES | YES |
| Episode closes on higher-state transition | NO | YES | YES |
| Episode closes on max duration (24hr) | YES | YES | YES |
| Episode closes on manual ack | NO | YES | YES |
| Episode boundary at S0 >4hr | YES | YES | YES |
| Track episode_id | YES | YES | YES |
| Track episode_opened_at | YES | YES | YES |
| Track episode_closed_at | NO | YES | YES |

## Alert Types

| Type | Condition | System A | System B | Merged |
|------|-----------|----------|----------|--------|
| INTERRUPTIVE | S2→S3 | YES | YES | YES |
| INTERRUPTIVE | S0→S3 | YES | YES | YES |
| INTERRUPTIVE | S1→S3 | YES | YES | YES |
| INTERRUPTIVE | Dwell at S2+ | NO | YES | YES |
| INTERRUPTIVE | TTH shortening at S2+ | NO | YES | YES |
| NON_INTERRUPTIVE | S0→S1 | YES | YES | YES |
| NON_INTERRUPTIVE | S1→S2 | YES | YES | YES |
| NON_INTERRUPTIVE | S0→S2 | YES | YES | YES |
| NON_INTERRUPTIVE | Velocity spike | YES | YES | YES |
| NON_INTERRUPTIVE | Novelty | YES | YES | YES |
| NONE | Same state no break | YES | YES | YES |
| NONE | Downward not to S0 | NO | YES | YES |
| NONE | Cooldown suppression | YES | YES | YES |

## Suppression Reasons

| Reason | System A | System B | Merged |
|--------|----------|----------|--------|
| cooldown_active | YES | YES | YES |
| same_state_no_break | YES | YES | YES |
| downward_transition_not_resolve | NO | YES | YES |
| de-escalation | YES | YES | YES |

## Audit Logging Fields

| Field | System A | System B | Merged |
|-------|----------|----------|--------|
| event_id | YES | YES | YES |
| timestamp | YES | YES | YES |
| patient_id | YES | YES | YES |
| risk_domain | YES | YES | YES |
| scores | YES | YES | YES |
| state_previous | YES | YES | YES |
| state_current | YES | YES | YES |
| state_transition | YES | YES | YES |
| alert_fired | YES | YES | YES |
| alert_type | NO | YES | YES |
| alert_event | YES | YES | YES |
| suppression_reason | YES | YES | YES |
| break_rules | NO | YES | YES |
| thresholds_used | YES | YES | YES |
| confidence | YES | YES | YES |
| contributing_biomarkers | YES | YES | YES |
| velocity | YES | YES | YES |
| evaluation_duration_ms | NO | YES | YES |

## Agent Integration

| Feature | System A | System B | Merged |
|---------|----------|----------|--------|
| Biomarker Agent auto-query | YES | YES | YES |
| Diagnostic Agent correlation | YES | YES | YES |
| Trial Rescue Agent eligibility | YES | YES | YES |
| Surveillance Agent patterns | YES | YES | YES |
| Inter-agent communication | YES | YES | YES |
| Consensus protocol | YES | YES | YES |

## P0/P1 Features (NEW)

| Feature | System A | System B | Merged |
|---------|----------|----------|--------|
| Alert acknowledgment | NO | YES | YES |
| Alert routing rules | NO | YES | YES |
| Escalation timer | NO | YES | YES |
| WebSocket real-time push | NO | YES | YES |
| SSE alternative | NO | YES | YES |
| Persistence layer abstraction | NO | YES | YES |
| In-memory storage | YES | YES | YES |
| PostgreSQL storage | NO | YES | YES |
| Redis caching | NO | YES | YES |

## API Endpoints

| Endpoint | System A | System B | Merged |
|----------|----------|----------|--------|
| POST /alerts/evaluate | YES | YES | YES |
| POST /alerts/state | YES | YES | YES |
| POST /alerts/history | YES | YES | YES |
| GET /alerts/config | YES | YES | YES |
| POST /predict/time-to-harm | YES | YES | YES |
| GET /predict/time-to-harm/domains | YES | YES | YES |
| POST /alerts/{alert_id}/ack | NO | YES | YES |
| POST /patient/intake | NO | YES | YES |
| WS /alerts/stream | NO | YES | YES |
| GET /alerts/stream/sse | NO | YES | YES |

---

## Summary

| Category | System A Only | System B Only | Both | Total in Merged |
|----------|---------------|---------------|------|-----------------|
| State/Domain | 10 | 5 | 4 | 19 |
| Biomarkers | 40 | 0 | 0 | 40 |
| Break Rules | 2 | 2 | 0 | 4 |
| Cooldowns | 10 | 0 | 0 | 10 |
| Confidence | 0 | 0 | 10 | 10 |
| Episodes | 4 | 3 | 3 | 10 |
| Alert Types | 0 | 2 | 9 | 11 |
| Suppression | 0 | 1 | 3 | 4 |
| Audit Fields | 0 | 4 | 14 | 18 |
| Agent Integration | 0 | 0 | 6 | 6 |
| P0/P1 Features | 1 | 8 | 0 | 9 |
| Endpoints | 6 | 4 | 0 | 10 |

**TOTAL FEATURES IN MERGED SYSTEM: 151**
