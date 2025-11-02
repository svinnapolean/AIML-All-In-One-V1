# 🚀 Loan Default Prediction API - Deployment Guide

Complete guide for deploying the Loan Default Prediction API with Docker, validation pipeline, and client integration.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Development Deployment](#development-deployment)
4. [Production Deployment](#production-deployment)
5. [API Usage](#api-usage)
6. [Validation Pipeline](#validation-pipeline)
7. [Client Examples](#client-examples)
8. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

---

## 🚀 Quick Start

### 1. One-Command Deployment
```bash
# Clone and deploy in development mode
git clone <repository-url>
cd AIML-All-In-One-V1
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```

### 2. Verify Deployment
```bash
# Check API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

### 3. Test Prediction
```bash
# Run client examples
python examples/api_client_examples.py
```

---

## 📋 Prerequisites

### System Requirements
- **Docker**: Version 20.10+ 
- **Docker Compose**: Version 2.0+
- **Python**: 3.11+ (for local development)
- **Memory**: Minimum 4GB RAM
- **Storage**: 2GB free space

### Trained Models
- At least one trained model in `src/models/saved_models/`
- Train a model first: `python src/models/fast_deep_learning.py --save_model --model_name demo_model`

### Data Files
- Loan data in `loan_data/loan_data.csv` (optional for API operation)

---

## 🛠 Development Deployment

### Using Docker Compose
```bash
# Start development environment
docker-compose -f docker/docker-compose.dev.yml up -d

# View logs
docker-compose -f docker/docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.dev.yml down
```

### Direct Python Execution
```bash
# Install dependencies
pip install -r requirements-api.txt

# Set environment variables
export PYTHONPATH=./src
export MODEL_PATH=./src/models

# Start API server
cd src/api
uvicorn model_api:app --host 0.0.0.0 --port 8000 --reload
```

### Development Features
- 🔄 Hot reload for code changes
- 🐛 Debug mode enabled
- 📊 Detailed logging
- 🔧 Direct volume mounts

---

## 🏭 Production Deployment

### Full Production Stack
```bash
# Deploy with monitoring and validation
ENVIRONMENT=production ./scripts/deploy.sh --validate
```

### Production Services
- **API Server**: Main application (Port 8000)
- **Redis**: Caching and session management (Port 6379)
- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Monitoring dashboards (Port 3000)
- **Nginx**: Reverse proxy and load balancer (Port 80/443)
- **Validator**: Automated validation service

### Environment Configuration
```bash
# Set production environment variables
export API_PORT=8000
export REDIS_PASSWORD=your_secure_password
export GRAFANA_PASSWORD=your_secure_password
export LOG_LEVEL=INFO
export ENVIRONMENT=production
```

### SSL/TLS Configuration
```bash
# Place SSL certificates in docker/nginx/ssl/
docker/nginx/ssl/
├── cert.pem
└── key.pem
```

---

## 🌐 API Usage

### Core Endpoints

#### Health Check
```bash
GET /health
# Returns: API status, model status, uptime
curl http://localhost:8000/health
```

#### Model Management
```bash
# List available models
GET /models
curl http://localhost:8000/models

# Get current model info
GET /model-info
curl http://localhost:8000/model-info

# Load a specific model
POST /load-model/{model_name}
curl -X POST http://localhost:8000/load-model/my_model
```

#### Predictions
```bash
# Single prediction
POST /predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amt_credit": 450000,
    "amt_annuity": 25000,
    "amt_income_total": 150000,
    "code_gender": "M",
    "days_birth": -12000,
    "days_employed": -2000,
    "name_contract_type": "Cash loans",
    "name_income_type": "Working",
    "name_education_type": "Higher education",
    "name_family_status": "Married",
    "name_housing_type": "House / apartment",
    "region_population_relative": 0.02
  }'

# Batch predictions
POST /predict-batch
curl -X POST http://localhost:8000/predict-batch \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [
      { /* application 1 data */ },
      { /* application 2 data */ }
    ]
  }'
```

### Request/Response Examples

#### Single Prediction Request
```json
{
  "amt_credit": 450000.0,
  "amt_annuity": 25000.0,
  "amt_income_total": 150000.0,
  "amt_goods_price": 400000.0,
  "code_gender": "M",
  "days_birth": -12000,
  "days_employed": -2000,
  "name_contract_type": "Cash loans",
  "name_income_type": "Working",
  "name_education_type": "Higher education",
  "name_family_status": "Married",
  "name_housing_type": "House / apartment",
  "region_population_relative": 0.02,
  "ext_source_1": 0.7,
  "ext_source_2": 0.6,
  "ext_source_3": 0.8
}
```

#### Single Prediction Response
```json
{
  "application_id": "app_1699234567890",
  "prediction": 0.2847,
  "risk_level": "LOW",
  "model_used": "loan_model_v2",
  "prediction_timestamp": "2025-11-01T21:45:30.123456",
  "confidence": 0.8523
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `amt_credit` | float | ✅ | Credit amount of the loan |
| `amt_annuity` | float | ✅ | Loan annuity amount |
| `amt_income_total` | float | ✅ | Total income of the client |
| `code_gender` | string | ✅ | Gender: "M" or "F" |
| `days_birth` | int | ✅ | Days since birth (negative) |
| `name_contract_type` | string | ✅ | "Cash loans" or "Revolving loans" |
| `ext_source_1` | float | ❌ | External data source 1 (0-1) |
| `ext_source_2` | float | ❌ | External data source 2 (0-1) |
| `ext_source_3` | float | ❌ | External data source 3 (0-1) |

---

## ✅ Validation Pipeline

### Automated Validation
```bash
# Run comprehensive validation
python src/validation/model_validator.py --api-url http://localhost:8000

# Validation includes:
# - API health checks
# - Model endpoint validation
# - Single prediction testing
# - Batch prediction testing
# - Performance consistency testing
# - Model accuracy validation
```

### Validation Results
```bash
# View validation reports
ls validation_reports/
validation_report_20251101_214500.json

# Sample validation output:
{
  "validation_timestamp": "2025-11-01T21:45:00.123456",
  "summary": {
    "overall_status": "PASS",
    "total_tests": 6,
    "passed_tests": 6,
    "failed_tests": 0,
    "pass_rate_percent": 100.0
  },
  "tests": {
    "health_check": {"status": "pass", "response_time_ms": 45.2},
    "single_prediction": {"status": "pass", "response_time_ms": 123.4},
    "batch_prediction": {"status": "pass", "avg_prediction_time_ms": 89.1},
    "performance_consistency": {"status": "pass", "availability_percent": 100.0}
  }
}
```

### Continuous Validation
```bash
# Schedule validation (cron example)
# Run validation every 6 hours
0 */6 * * * cd /app && python src/validation/model_validator.py
```

---

## 💻 Client Examples

### Python Client
```python
from examples.api_client_examples import LoanPredictionClient

# Initialize client
client = LoanPredictionClient("http://localhost:8000")

# Check API health
health = client.check_api_health()
print(f"API Status: {health['status']}")

# Make a prediction
loan_data = {
    "amt_credit": 450000.0,
    "amt_annuity": 25000.0,
    "amt_income_total": 150000.0,
    "code_gender": "M",
    "days_birth": -12000,
    "days_employed": -2000,
    "name_contract_type": "Cash loans",
    "name_income_type": "Working",
    "name_education_type": "Higher education",
    "name_family_status": "Married",
    "name_housing_type": "House / apartment",
    "region_population_relative": 0.02
}

result = client.predict_single(loan_data)
print(f"Default Probability: {result['prediction']:.4f}")
print(f"Risk Level: {result['risk_level']}")
```

### JavaScript Client
```javascript
// Single prediction
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    amt_credit: 450000,
    amt_annuity: 25000,
    amt_income_total: 150000,
    code_gender: "M",
    days_birth: -12000,
    days_employed: -2000,
    name_contract_type: "Cash loans",
    name_income_type: "Working",
    name_education_type: "Higher education",
    name_family_status: "Married",
    name_housing_type: "House / apartment",
    region_population_relative: 0.02
  })
});

const result = await response.json();
console.log(`Default Probability: ${result.prediction}`);
console.log(`Risk Level: ${result.risk_level}`);
```

### cURL Examples
```bash
# Health check
curl -s http://localhost:8000/health | jq

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"amt_credit":450000,"amt_annuity":25000,"amt_income_total":150000,"code_gender":"M","days_birth":-12000,"days_employed":-2000,"name_contract_type":"Cash loans","name_income_type":"Working","name_education_type":"Higher education","name_family_status":"Married","name_housing_type":"House / apartment","region_population_relative":0.02}' \
  | jq
```

### CSV Processing
```python
import pandas as pd
from examples.api_client_examples import LoanPredictionClient

# Load CSV file
df = pd.read_csv('loan_applications.csv')

# Convert to API format
client = LoanPredictionClient()
applications = df.to_dict('records')

# Make batch prediction
result = client.predict_batch(applications)

# Add results to DataFrame
predictions = result['predictions']
df['default_probability'] = [p['prediction'] for p in predictions]
df['risk_level'] = [p['risk_level'] for p in predictions]

# Save results
df.to_csv('loan_predictions_output.csv', index=False)
```

---

## 📊 Monitoring & Troubleshooting

### Health Monitoring
```bash
# API health
curl http://localhost:8000/health

# Container health
docker ps
docker-compose -f docker/docker-compose.dev.yml ps

# View logs
docker-compose -f docker/docker-compose.dev.yml logs -f loan-api-dev
```

### Performance Metrics (Production)
- **Grafana Dashboards**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:9090
- **API Metrics**: http://localhost:8000/metrics

### Common Issues & Solutions

#### Issue: API not starting
```bash
# Check logs
docker-compose logs loan-api-dev

# Common causes:
# 1. Model files missing
# 2. Port already in use
# 3. Python dependencies issue
```

#### Issue: Predictions failing
```bash
# Check model status
curl http://localhost:8000/model-info

# Load a model if none loaded
curl -X POST http://localhost:8000/load-model/your_model_name
```

#### Issue: Slow response times
```bash
# Check performance
python src/validation/model_validator.py --performance-requests 100

# Optimize:
# 1. Increase Docker memory allocation
# 2. Use lighter model architecture
# 3. Enable model caching
```

### Logs Analysis
```bash
# View specific service logs
docker-compose logs -f loan-api-dev
docker-compose logs -f redis-dev

# Filter logs by level
docker-compose logs loan-api-dev | grep ERROR
docker-compose logs loan-api-dev | grep WARNING
```

---

## ⚙️ Advanced Configuration

### Environment Variables
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
PYTHONPATH=/app/src
MODEL_PATH=/app/models

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your_password

# Model Configuration
DEFAULT_MODEL=loan_model_v2
MODEL_CACHE_SIZE=3
PREDICTION_TIMEOUT=30

# Performance Configuration
MAX_BATCH_SIZE=1000
WORKER_PROCESSES=4
MAX_CONCURRENT_REQUESTS=100
```

### Custom Model Loading
```python
# src/api/model_api.py
@app.on_event("startup")
async def startup_event():
    # Load specific model on startup
    await load_model_internal("production_model_v1")
```

### Load Balancing
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  loan-api-1:
    # ... api configuration
  loan-api-2:
    # ... api configuration
  nginx:
    # ... load balancer configuration
```

### Security Configuration
```bash
# Enable HTTPS
export ENABLE_HTTPS=true
export SSL_CERT_PATH=/app/ssl/cert.pem
export SSL_KEY_PATH=/app/ssl/key.pem

# API key authentication
export REQUIRE_API_KEY=true
export API_KEY=your_secure_api_key
```

### Scaling Configuration
```yaml
# Production scaling
services:
  loan-api:
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

---

## 🎯 Summary

This deployment guide provides everything needed to deploy and operate the Loan Default Prediction API:

### ✅ **What You Get**
- 🚀 **Production-ready API** with FastAPI
- 🐳 **Docker containerization** for easy deployment
- ✅ **Comprehensive validation** pipeline
- 📊 **Monitoring and metrics** (production)
- 💻 **Client examples** in multiple languages
- 🔧 **Automated deployment** scripts

### 🚀 **Quick Commands**
```bash
# Deploy development
./scripts/deploy.sh

# Deploy production with validation
ENVIRONMENT=production ./scripts/deploy.sh --validate

# Run client examples
python examples/api_client_examples.py

# Run validation
python src/validation/model_validator.py
```

### 📞 **API Access**
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

The API is now ready to serve loan default predictions in production! 🎉