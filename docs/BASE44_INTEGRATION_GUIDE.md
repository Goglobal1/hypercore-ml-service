# Base44 Integration Guide - CSE Alert Dashboard

This guide explains how to integrate the Clinical State Engine (CSE) Alert Dashboard into your Base44 application.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [API Endpoints](#api-endpoints)
7. [Real-time Updates](#real-time-updates)
8. [Customization](#customization)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The CSE Alert Dashboard provides real-time clinical monitoring with:

- **15 Risk Domains**: sepsis, cardiac, kidney, respiratory, hepatic, neurological, metabolic, hematologic, oncology, multi_system, deterioration, infection, outbreak, trial_confounder, custom
- **4-State Clinical Model**: S0 (Stable), S1 (Watch), S2 (Escalating), S3 (Critical)
- **Real-time Updates**: WebSocket with SSE fallback
- **Alert Management**: Filtering, acknowledgment, routing
- **Patient Monitoring**: State tracking, biomarker trends, time-to-harm prediction

---

## Prerequisites

### Required Dependencies

```json
{
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "recharts": "^2.10.0",
    "lucide-react": "^0.300.0"
  }
}
```

### Install with npm

```bash
npm install react react-dom recharts lucide-react
```

### Install with yarn

```bash
yarn add react react-dom recharts lucide-react
```

---

## Installation

### Option 1: Copy Component Directly

1. Copy `CSEAlertDashboard.jsx` to your Base44 components directory:
   ```
   base44-app/
   └── src/
       └── components/
           └── CSEAlertDashboard.jsx
   ```

2. Import in your app:
   ```jsx
   import CSEAlertDashboard from './components/CSEAlertDashboard';
   ```

### Option 2: Use Production Build

1. Copy `CSEAlertDashboard.prod.jsx` which uses environment variables
2. Configure your environment (see Configuration section)

---

## Configuration

### Environment Variables

Create a `.env` file in your Base44 project root:

```env
# HyperCore ML Service URL
REACT_APP_HYPERCORE_API_URL=http://localhost:8000
REACT_APP_HYPERCORE_WS_URL=ws://localhost:8000/alerts/ws
REACT_APP_HYPERCORE_SSE_URL=http://localhost:8000/alerts/sse
```

### For Production

```env
REACT_APP_HYPERCORE_API_URL=https://api.yourcompany.com
REACT_APP_HYPERCORE_WS_URL=wss://api.yourcompany.com/alerts/ws
REACT_APP_HYPERCORE_SSE_URL=https://api.yourcompany.com/alerts/sse
```

### Configuration Object (Alternative)

If you prefer runtime configuration:

```jsx
// config.js
export const HYPERCORE_CONFIG = {
  apiBaseUrl: process.env.REACT_APP_HYPERCORE_API_URL || 'http://localhost:8000',
  wsUrl: process.env.REACT_APP_HYPERCORE_WS_URL || 'ws://localhost:8000/alerts/ws',
  sseUrl: process.env.REACT_APP_HYPERCORE_SSE_URL || 'http://localhost:8000/alerts/sse',
};
```

---

## Usage

### Basic Usage

```jsx
import React from 'react';
import CSEAlertDashboard from './components/CSEAlertDashboard';

function App() {
  return (
    <div className="App">
      <CSEAlertDashboard />
    </div>
  );
}

export default App;
```

### With Custom Styling

```jsx
import CSEAlertDashboard from './components/CSEAlertDashboard';
import './custom-dashboard-styles.css';

function ClinicalMonitor() {
  return (
    <div className="clinical-monitor-wrapper">
      <CSEAlertDashboard />
    </div>
  );
}
```

### Embedding in Existing Layout

```jsx
import CSEAlertDashboard from './components/CSEAlertDashboard';
import { Sidebar, Header } from './layout';

function Dashboard() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Header title="Clinical Monitoring" />
        <main className="flex-1 overflow-auto">
          <CSEAlertDashboard />
        </main>
      </div>
    </div>
  );
}
```

---

## API Endpoints

The dashboard connects to these HyperCore endpoints:

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/alerts/health` | GET | System health check |
| `/alerts/stats` | GET | System statistics |

### Patient Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/alerts/patient/intake` | POST | Full pipeline evaluation |
| `/alerts/patient/{id}/state` | GET | Patient state (all domains) |
| `/alerts/patient/{id}/state/{domain}` | GET | Patient state (specific domain) |

### Alert Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/alerts/events` | GET | Query audit log |
| `/alerts/acknowledge` | POST | Acknowledge an alert |
| `/alerts/episodes` | GET | Get episodes |

### Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/alerts/config/domains` | GET | All domain configurations |
| `/alerts/config/biomarkers` | GET | Biomarker thresholds |

### Real-time

| Endpoint | Protocol | Description |
|----------|----------|-------------|
| `/alerts/ws` | WebSocket | Real-time updates |
| `/alerts/sse` | SSE | Server-sent events (fallback) |

---

## Real-time Updates

### WebSocket Connection

The dashboard automatically establishes a WebSocket connection:

```javascript
// Automatic subscription on connect
ws.send(JSON.stringify({
  type: 'subscribe',
  risk_domains: ['sepsis', 'cardiac', ...], // All 15 domains
  roles: ['dashboard']
}));
```

### Message Types

```javascript
// Alert fired
{
  type: 'alert',
  data: {
    event_id: 'evt_xxx',
    patient_id: 'P12345',
    risk_domain: 'sepsis',
    severity: 'CRITICAL',
    // ... full alert data
  }
}

// State update
{
  type: 'evaluation',
  data: {
    patient_id: 'P12345',
    state_now: 'S2',
    risk_score: 0.65,
    // ... evaluation result
  }
}

// Acknowledgment
{
  type: 'acknowledgment',
  data: {
    alert_id: 'evt_xxx',
    acknowledged_by: 'Dr. Smith',
    // ... ack details
  }
}
```

### SSE Fallback

If WebSocket fails, the dashboard automatically falls back to Server-Sent Events:

```javascript
// SSE connection with filters
const eventSource = new EventSource(
  '/alerts/sse?risk_domains=sepsis,cardiac&roles=dashboard'
);
```

---

## Customization

### Custom Risk Domains

Edit the `RISK_DOMAINS` array in the component:

```javascript
const RISK_DOMAINS = [
  { id: 'sepsis', name: 'Sepsis', color: '#ef4444', icon: '🦠' },
  // Add your custom domains
  { id: 'custom_domain', name: 'Custom', color: '#6366f1', icon: '⚙️' },
];
```

### Custom State Colors

Modify the `CLINICAL_STATES` object:

```javascript
const CLINICAL_STATES = {
  S0: { name: 'Stable', color: '#22c55e', bgColor: '#dcfce7' },
  // Customize colors and names
};
```

### Custom Alert Actions

Override the `handleAcknowledge` function:

```javascript
const handleAcknowledge = async (alertId) => {
  // Custom logic before acknowledging
  await customPreAckHook(alertId);

  // Standard acknowledgment
  await acknowledgeAlert(alertId, getCurrentUser().id);

  // Custom logic after acknowledging
  await customPostAckHook(alertId);
};
```

---

## Troubleshooting

### Common Issues

#### 1. "Failed to connect to WebSocket"

**Cause**: Server not running or CORS issue

**Solution**:
```bash
# Ensure server is running
cd hypercore-ml-service
python -m uvicorn main:app --host 0.0.0.0 --port 8000

# Check CORS settings in main.py
```

#### 2. "CORS error in browser console"

**Cause**: Cross-origin request blocked

**Solution**: Add your Base44 domain to the CORS allowed origins in HyperCore:

```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-base44-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### 3. "Patient state shows 'null'"

**Cause**: Patient hasn't been evaluated yet

**Solution**: Submit a patient intake first:
```javascript
await submitPatientIntake({
  patient_id: 'P12345',
  risk_domain: 'sepsis',
  lab_data: { lactate: 2.5 }
});
```

#### 4. "Real-time updates not working"

**Cause**: WebSocket connection failed

**Solution**:
1. Check browser console for errors
2. Verify WebSocket URL is correct
3. Check if firewall/proxy blocks WebSocket
4. The dashboard will automatically fallback to SSE

#### 5. "Charts not rendering"

**Cause**: recharts dependency missing

**Solution**:
```bash
npm install recharts
```

### Debug Mode

Enable debug logging:

```javascript
// Add to top of component
const DEBUG = true;

// In useRealtimeConnection hook
if (DEBUG) console.log('[CSE Dashboard] Message:', message);
```

### Health Check

Test API connectivity:

```bash
# Health check
curl http://localhost:8000/alerts/health

# Expected response
{"status":"healthy","storage_type":"in_memory","timestamp":"..."}
```

---

## Support

- **Issues**: https://github.com/Goglobal1/hypercore-ml-service/issues
- **Documentation**: See `/docs` folder in the repository
- **API Reference**: `/docs` endpoint when server is running

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-03-19 | Full rewrite for merged alert system |
| 1.0.0 | 2026-03-01 | Initial release |
