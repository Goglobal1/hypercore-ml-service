# HyperCore Alert System - API Reference

**Version:** 2.0.0
**Base URL:** `http://localhost:8000`

---

## Table of Contents

1. [Health & Status](#health--status)
2. [Patient Operations](#patient-operations)
3. [Alert Operations](#alert-operations)
4. [Configuration](#configuration)
5. [Real-time Connections](#real-time-connections)
6. [Error Handling](#error-handling)

---

## Health & Status

### GET /alerts/health

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "storage_type": "in_memory",
  "timestamp": "2026-03-19T03:00:00.000000+00:00"
}
```

| Field | Type | Description |
|-------|------|-------------|
| status | string | "healthy" or "unhealthy" |
| storage_type | string | "in_memory" or "postgresql" |
| timestamp | string | ISO 8601 timestamp |

---

### GET /alerts/stats

Get system statistics.

**Response:**
```json
{
  "total_patients": 15,
  "total_events": 42,
  "total_episodes": 8,
  "active_episodes": 3,
  "events_by_severity": {
    "CRITICAL": 5,
    "URGENT": 12,
    "WARNING": 18,
    "INFO": 7
  },
  "events_by_domain": {
    "sepsis": 15,
    "cardiac": 10,
    "kidney": 8,
    "respiratory": 9
  }
}
```

---

## Patient Operations

### POST /alerts/patient/intake

**Primary endpoint for patient evaluation.** Processes patient data through the full 11-step alert pipeline.

**Request Body:**
```json
{
  "patient_id": "P12345",
  "risk_domain": "sepsis",
  "lab_data": {
    "lactate": 4.5,
    "WBC": 16.0,
    "CRP": 150.0,
    "procalcitonin": 2.5
  },
  "vitals": {
    "heart_rate": 115,
    "respiratory_rate": 26,
    "temperature": 39.2,
    "blood_pressure_systolic": 88
  },
  "clinical_context": {
    "location": "ICU",
    "admission_reason": "sepsis"
  },
  "risk_score": null
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| patient_id | string | Yes | Unique patient identifier |
| risk_domain | string | Yes | Clinical domain (sepsis, cardiac, kidney, etc.) |
| lab_data | object | No | Laboratory values |
| vitals | object | No | Vital signs |
| clinical_context | object | No | Additional clinical context |
| risk_score | float | No | Pre-calculated risk score (0.0-1.0). If not provided, auto-calculated from biomarkers |

**Response:**
```json
{
  "patient_id": "P12345",
  "risk_domain": "sepsis",
  "timestamp": "2026-03-19T03:00:00.000000+00:00",
  "success": true,
  "steps": [
    {
      "step_name": "data_received",
      "success": true,
      "duration_ms": 0,
      "data": {"has_labs": true, "has_vitals": true}
    },
    {
      "step_name": "risk_score_calculated",
      "success": true,
      "duration_ms": 0.1,
      "data": {"risk_score": 0.85}
    }
  ],
  "evaluation_result": {
    "patient_id": "P12345",
    "risk_domain": "sepsis",
    "risk_score": 0.85,
    "state_now": "S3",
    "state_name": "Critical",
    "state_previous": "S2",
    "state_transition": true,
    "alert_fired": true,
    "alert_type": "interruptive",
    "severity": "CRITICAL",
    "confidence": 0.89,
    "clinical_headline": "Rapid inflammatory marker rise detected",
    "clinical_rationale": "Patient risk state is critical (S3). Primary drivers: lactate, WBC, CRP.",
    "suggested_action": "Immediate bedside evaluation. Activate rapid response if appropriate.",
    "contributing_biomarkers": ["lactate", "WBC", "CRP", "procalcitonin"],
    "time_to_harm": {
      "hours": 1.3,
      "intervention_window": "IMMEDIATE",
      "confidence": 0.75
    },
    "episode": {
      "episode_id": "ep_abc123",
      "is_open": true,
      "highest_state": "S3",
      "alert_count": 1
    }
  },
  "routing_result": {
    "routed_to_roles": ["attending", "resident", "charge_nurse"],
    "notification_channels": ["pager", "dashboard", "sms"],
    "escalation_minutes": 5
  },
  "total_duration_ms": 5.2
}
```

---

### GET /alerts/patient/{patient_id}/state

Get current state for a patient across all domains.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| patient_id | string | Patient identifier |

**Response:**
```json
{
  "patient_id": "P12345",
  "states": {
    "sepsis": {
      "current_state": "S3",
      "risk_score": 0.85,
      "episode": {
        "episode_id": "ep_abc123",
        "is_open": true
      },
      "last_score_time": "2026-03-19T03:00:00.000000+00:00"
    }
  },
  "domain_count": 1
}
```

---

### GET /alerts/patient/{patient_id}/state/{domain}

Get current state for a specific domain.

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| patient_id | string | Patient identifier |
| domain | string | Risk domain (sepsis, cardiac, etc.) |

**Response:**
```json
{
  "patient_id": "P12345",
  "risk_domain": "sepsis",
  "current_state": "S3",
  "risk_score": 0.85,
  "episode": {
    "episode_id": "ep_abc123",
    "is_open": true,
    "highest_state": "S3"
  },
  "last_score_time": "2026-03-19T03:00:00.000000+00:00"
}
```

---

## Alert Operations

### GET /alerts/events

Query the audit log for alert events.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| patient_id | string | Filter by patient |
| risk_domain | string | Filter by domain |
| event_type | string | Filter by event type (ALERT_FIRED, ALERT_SUPPRESSED, etc.) |
| since | string | ISO 8601 timestamp - events after this time |
| until | string | ISO 8601 timestamp - events before this time |
| limit | int | Max events to return (default: 100) |

**Response:**
```json
{
  "events": [
    {
      "event_id": "evt_abc123",
      "event_type": "ALERT_FIRED",
      "timestamp": "2026-03-19T03:00:00.000000+00:00",
      "patient_id": "P12345",
      "risk_domain": "sepsis",
      "state_current": "S3",
      "risk_score": 0.85,
      "alert_type": "interruptive",
      "severity": "CRITICAL",
      "clinical_headline": "Rapid inflammatory marker rise detected",
      "routed_to": ["attending", "resident"]
    }
  ],
  "count": 1,
  "filters": {
    "patient_id": "P12345"
  }
}
```

---

### POST /alerts/acknowledge

Acknowledge an alert.

**Request Body:**
```json
{
  "alert_id": "evt_abc123",
  "acknowledged_by": "dr_smith",
  "notes": "Patient assessed, antibiotics started"
}
```

**Response:**
```json
{
  "success": true,
  "acknowledgment": {
    "ack_id": "ack_xyz789",
    "alert_id": "evt_abc123",
    "acknowledged_by": "dr_smith",
    "acknowledged_at": "2026-03-19T03:05:00.000000+00:00",
    "notes": "Patient assessed, antibiotics started"
  }
}
```

---

### GET /alerts/episodes

Get all episodes (optionally filtered).

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| patient_id | string | Filter by patient |
| risk_domain | string | Filter by domain |
| is_open | bool | Filter by open/closed status |

**Response:**
```json
{
  "episodes": [
    {
      "episode_id": "ep_abc123",
      "patient_id": "P12345",
      "risk_domain": "sepsis",
      "opened_at": "2026-03-19T02:00:00.000000+00:00",
      "opened_state": "S2",
      "highest_state": "S3",
      "is_open": true,
      "alert_count": 3,
      "duration_hours": 1.5,
      "acknowledged": false
    }
  ],
  "count": 1
}
```

---

## Configuration

### GET /alerts/config/domains

Get all domain configurations.

**Response:**
```json
{
  "domains": {
    "sepsis": {
      "s0_upper": 0.25,
      "s1_upper": 0.50,
      "s2_upper": 0.75,
      "default_cooldown_minutes": 30,
      "velocity_threshold": 0.12
    },
    "cardiac": {
      "s0_upper": 0.30,
      "s1_upper": 0.55,
      "s2_upper": 0.80,
      "default_cooldown_minutes": 45,
      "velocity_threshold": 0.15
    }
  }
}
```

---

### GET /alerts/config/domains/{domain}

Get configuration for a specific domain.

**Response:**
```json
{
  "domain": "sepsis",
  "config": {
    "s0_upper": 0.25,
    "s1_upper": 0.50,
    "s2_upper": 0.75,
    "default_cooldown_minutes": 30,
    "escalation_cooldown_minutes": 10,
    "critical_cooldown_minutes": 5,
    "velocity_threshold": 0.12,
    "velocity_window_hours": 1.0,
    "episode_break_hours": 4.0,
    "novelty_detection_enabled": true,
    "velocity_override_enabled": true,
    "tth_shortening_enabled": true,
    "dwell_escalation_enabled": true,
    "dwell_escalation_hours": 2.0
  }
}
```

---

### GET /alerts/config/biomarkers

Get all biomarker thresholds.

**Response:**
```json
{
  "biomarkers": {
    "sepsis": {
      "lactate": {
        "warning": 2.0,
        "critical": 4.0,
        "unit": "mmol/L",
        "direction": "rising",
        "weight": 1.0
      },
      "wbc": {
        "warning": 10.0,
        "critical": 12.0,
        "unit": "K/uL",
        "direction": "rising",
        "weight": 0.6
      }
    }
  }
}
```

---

## Real-time Connections

### WebSocket /alerts/ws

Real-time alert updates via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/alerts/ws');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    risk_domains: ['sepsis', 'cardiac'],
    roles: ['attending']
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Alert:', message);
};
```

**Message Types:**

| Type | Description |
|------|-------------|
| alert | New alert fired |
| evaluation | Patient evaluation completed |
| acknowledgment | Alert acknowledged |
| escalation | Alert escalated |
| heartbeat | Connection keepalive |

**Alert Message:**
```json
{
  "type": "alert",
  "timestamp": "2026-03-19T03:00:00.000000+00:00",
  "data": {
    "event_id": "evt_abc123",
    "patient_id": "P12345",
    "risk_domain": "sepsis",
    "severity": "CRITICAL",
    "clinical_headline": "Rapid inflammatory marker rise detected"
  }
}
```

---

### GET /alerts/sse

Server-Sent Events fallback for real-time updates.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| risk_domains | string | Comma-separated domains to subscribe to |
| roles | string | Comma-separated roles |

**Example:**
```
GET /alerts/sse?risk_domains=sepsis,cardiac&roles=attending
```

**Event Stream:**
```
event: alert
data: {"event_id":"evt_abc123","patient_id":"P12345","severity":"CRITICAL"}

event: heartbeat
data: {"timestamp":"2026-03-19T03:00:00.000000+00:00"}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Schema validation failed |
| 500 | Internal Server Error |

### Error Response Format

```json
{
  "detail": "Error message here",
  "errors": [
    {
      "loc": ["body", "patient_id"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Supported Risk Domains

| Domain | Description | Key Biomarkers |
|--------|-------------|----------------|
| sepsis | Sepsis/infection | lactate, WBC, CRP, procalcitonin |
| cardiac | Cardiac events | troponin, BNP, heart_rate |
| kidney | Kidney injury | creatinine, BUN, potassium, GFR |
| respiratory | Respiratory failure | SpO2, PaO2, respiratory_rate |
| hepatic | Liver dysfunction | ALT, AST, bilirubin, INR |
| neurological | Neurological events | GCS, ICP, CPP |
| metabolic | Metabolic disorders | glucose, pH, lactate, potassium |
| hematologic | Blood disorders | hemoglobin, platelets, INR |
| oncology | Cancer-related | CEA, CA125, PSA, AFP |
| multi_system | Multi-organ failure | Multiple domains |
| deterioration | General deterioration | Various |
| infection | General infection | WBC, CRP, temperature |
| outbreak | Population-level | Aggregated metrics |

---

## Clinical States

| State | Name | Risk Score Range | Description |
|-------|------|------------------|-------------|
| S0 | Stable | 0.00 - 0.25 | Patient stable, routine monitoring |
| S1 | Watch | 0.25 - 0.50 | Elevated risk, enhanced monitoring |
| S2 | Escalating | 0.50 - 0.75 | Significant risk, clinical evaluation |
| S3 | Critical | 0.75 - 1.00 | Critical risk, immediate intervention |

---

## Alert Types

| Type | Description | When Used |
|------|-------------|-----------|
| interruptive | Immediate attention needed | S3 Critical, rapid escalation |
| non_interruptive | Important but not urgent | S2 Escalating, stable critical |
| none | No alert | De-escalation, within cooldown |

---

## Severity Levels

| Severity | Description |
|----------|-------------|
| CRITICAL | Immediate life threat |
| URGENT | Requires prompt attention |
| WARNING | Monitor closely |
| INFO | Informational only |

---

## Rate Limits

Currently no rate limits are enforced, but for production:
- Recommended: 100 requests/minute per client
- WebSocket: 1 connection per client

---

**Last Updated:** 2026-03-19
