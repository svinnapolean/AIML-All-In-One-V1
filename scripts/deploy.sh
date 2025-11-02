#!/bin/bash
# Deployment Script for Loan Default Prediction API
# This script deploys the API locally using Docker

set -e

echo "🚀 Starting Loan Default Prediction API Deployment"
echo "=================================================="

# Configuration
API_PORT=${API_PORT:-8000}
ENVIRONMENT=${ENVIRONMENT:-development}
COMPOSE_FILE="docker/docker-compose.dev.yml"

if [ "$ENVIRONMENT" = "production" ]; then
    COMPOSE_FILE="docker/docker-compose.prod.yml"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
print_status "Checking prerequisites..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker first."
    exit 1
fi

print_success "Prerequisites check passed"

# Check if models exist
print_status "Checking for trained models..."
if [ -d "src/models/saved_models" ] && [ "$(ls -A src/models/saved_models)" ]; then
    model_count=$(find src/models/saved_models -name "*.h5" 2>/dev/null | wc -l)
    print_success "Found $model_count trained models"
else
    print_warning "No trained models found in src/models/saved_models"
    print_warning "The API will start but predictions may fail until models are trained"
    
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Training a model first..."
        cd src/models
        python fast_deep_learning.py --save_model --model_name demo_model
        cd ../..
        print_success "Demo model trained successfully"
    fi
fi

# Check if loan data exists
print_status "Checking for loan data..."
if [ ! -d "loan_data" ] || [ ! -f "loan_data/loan_data.csv" ]; then
    print_warning "Loan data not found at loan_data/loan_data.csv"
    print_warning "Some API features may not work properly"
    
    # Create dummy loan data directory
    mkdir -p loan_data
    echo "Creating placeholder loan_data directory"
fi

# Stop any existing containers
print_status "Stopping existing containers..."
docker-compose -f $COMPOSE_FILE down --remove-orphans 2>/dev/null || true

# Build and start services
print_status "Building Docker images..."
docker-compose -f $COMPOSE_FILE build

print_status "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

# Wait for API to be ready
print_status "Waiting for API to be ready..."
MAX_ATTEMPTS=30
ATTEMPT=1

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
    if curl -s http://localhost:$API_PORT/health > /dev/null 2>&1; then
        break
    fi
    
    echo -n "."
    sleep 2
    ATTEMPT=$((ATTEMPT + 1))
done

echo ""

if [ $ATTEMPT -gt $MAX_ATTEMPTS ]; then
    print_error "API failed to start within 60 seconds"
    print_error "Check logs with: docker-compose -f $COMPOSE_FILE logs"
    exit 1
fi

# Verify deployment
print_status "Verifying deployment..."

# Check API health
HEALTH_RESPONSE=$(curl -s http://localhost:$API_PORT/health)
if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    print_success "API health check passed"
else
    print_error "API health check failed"
    echo "Response: $HEALTH_RESPONSE"
fi

# Check if model is loaded
if echo "$HEALTH_RESPONSE" | grep -q '"model_loaded":true'; then
    MODEL_NAME=$(echo "$HEALTH_RESPONSE" | grep -o '"model_name":"[^"]*"' | cut -d'"' -f4)
    print_success "Model '$MODEL_NAME' is loaded and ready"
else
    print_warning "No model is currently loaded"
fi

# Run validation if requested
if [ "$1" = "--validate" ]; then
    print_status "Running deployment validation..."
    
    if [ -f "src/validation/model_validator.py" ]; then
        cd src/validation
        python model_validator.py --api-url "http://localhost:$API_PORT"
        cd ../..
    else
        print_warning "Validation script not found, skipping validation"
    fi
fi

# Display deployment information
echo ""
echo "🎉 Deployment Completed Successfully!"
echo "======================================"
echo ""
echo "🌐 API URLs:"
echo "   Main API: http://localhost:$API_PORT"
echo "   Health Check: http://localhost:$API_PORT/health"
echo "   API Documentation: http://localhost:$API_PORT/docs"
echo "   Interactive Docs: http://localhost:$API_PORT/redoc"
echo ""

if [ "$ENVIRONMENT" = "production" ]; then
    echo "📊 Monitoring URLs:"
    echo "   Grafana Dashboard: http://localhost:3000 (admin/admin)"
    echo "   Prometheus Metrics: http://localhost:9090"
    echo ""
fi

echo "🔧 Useful Commands:"
echo "   View logs: docker-compose -f $COMPOSE_FILE logs -f"
echo "   Stop services: docker-compose -f $COMPOSE_FILE down"
echo "   Restart services: docker-compose -f $COMPOSE_FILE restart"
echo "   Run validation: python examples/api_client_examples.py"
echo ""

echo "📝 Example API Calls:"
echo "   # Check health"
echo "   curl http://localhost:$API_PORT/health"
echo ""
echo "   # List models"
echo "   curl http://localhost:$API_PORT/models"
echo ""
echo "   # Make prediction"
echo "   curl -X POST http://localhost:$API_PORT/predict \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"amt_credit\": 450000, \"amt_annuity\": 25000, \"amt_income_total\": 150000, \"code_gender\": \"M\", \"days_birth\": -12000, \"days_employed\": -2000, \"name_contract_type\": \"Cash loans\", \"name_income_type\": \"Working\", \"name_education_type\": \"Higher education\", \"name_family_status\": \"Married\", \"name_housing_type\": \"House / apartment\", \"region_population_relative\": 0.02}'"
echo ""

print_success "API is ready to accept requests!"

# Optionally run client examples
read -p "Do you want to run client examples now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Running client examples..."
    if [ -f "examples/api_client_examples.py" ]; then
        cd examples
        python api_client_examples.py
        cd ..
    else
        print_warning "Client examples not found"
    fi
fi

echo ""
print_success "Deployment script completed!"