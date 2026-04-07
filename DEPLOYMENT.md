# HyperCore ML Service - Deployment Info

## URLs

- **Production**: https://hypercore-ml-service-production.up.railway.app
- **GitHub**: https://github.com/Goglobal1/hypercore-ml-service

## Test Commands

```bash
# Health check
curl https://hypercore-ml-service-production.up.railway.app/health

# Discovery endpoint
curl -X POST https://hypercore-ml-service-production.up.railway.app/discover \
  -H "Content-Type: application/json" \
  -d '{"patients": [{"patient_id": "TEST001", "creatinine": 2.8, "BUN": 45, "potassium": 5.8, "glucose": 180}]}'

# Batch analysis
curl -X POST https://hypercore-ml-service-production.up.railway.app/discover/batch \
  -H "Content-Type: application/json" \
  -d @your_data.json
```

## Key Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/discover` | Single/batch patient discovery |
| `/discover/batch` | Per-patient batch analysis |
| `/discovery/patient` | Single patient analysis |
| `/discovery/hospital` | Hospital-wide aggregate |
| `/early_risk_discovery` | Early risk detection |
| `/alerts/evaluate` | CSE alert evaluation |

## Response Fields (Handler-Aligned)

The `/discover` endpoint returns:

- **CSE Fields**: `clinical_state`, `state_label`, `previous_state`, `state_changed`
- **Utility Fields**: `utility_score`, `decision`, `utility_components`
- **Actionable Fields**: `immediate_actions`, `conditions`, `convergence`
- **Discovery Fields**: `endpoint_results`, `identified_diseases`, `recommendations`
