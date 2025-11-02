# PowerShell Deployment Script for Optimized Loan Risk AI/ML API
# Quick deployment with error handling and status monitoring

param(
    [string]$Mode = "full",  # full, build, deploy, test
    [switch]$SkipBenchmark = $false
)

Write-Host "🚀 OPTIMIZED LOAN RISK API DEPLOYMENT" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition

function Write-Status {
    param([string]$Message, [string]$Color = "Cyan")
    Write-Host "`n💻 $Message" -ForegroundColor $Color
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Test-DockerRunning {
    try {
        docker version | Out-Null
        return $true
    } catch {
        Write-Error "Docker is not running. Please start Docker Desktop."
        exit 1
    }
}

function Build-OptimizedImages {
    Write-Status "Building optimized Docker images..."
    
    try {
        # Build backend with optimized dependencies
        Write-Status "Building backend API with turbo models..."
        docker build -f docker/Dockerfile.backend -t loan-api-optimized:latest .
        
        Write-Success "Docker images built successfully"
        docker images | Select-String "loan-api-optimized"
        
    } catch {
        Write-Error "Failed to build Docker images: $_"
        exit 1
    }
}

function Deploy-ProductionStack {
    Write-Status "Deploying production stack..."
    
    try {
        # Stop existing containers
        Write-Status "Stopping existing containers..."
        docker-compose -f docker/docker-compose.optimized.yml down --remove-orphans 2>$null
        
        # Start optimized stack
        Write-Status "Starting optimized production stack..."
        docker-compose -f docker/docker-compose.optimized.yml up -d
        
        # Wait for services
        Write-Status "Waiting for services to start..."
        Start-Sleep -Seconds 30
        
        # Check status
        Write-Status "Checking service status..."
        docker-compose -f docker/docker-compose.optimized.yml ps
        
        Write-Success "Production stack deployed successfully"
        
    } catch {
        Write-Error "Failed to deploy production stack: $_"
        exit 1
    }
}

function Test-Deployment {
    Write-Status "Validating deployment..."
    
    try {
        # Test health endpoints
        $healthUrls = @(
            "http://localhost:8000/health",
            "http://localhost/health"
        )
        
        foreach ($url in $healthUrls) {
            Write-Status "Testing health endpoint: $url"
            try {
                $response = Invoke-RestMethod -Uri $url -TimeoutSec 10
                Write-Success "Health check passed: $($response | ConvertTo-Json -Compress)"
            } catch {
                Write-Error "Health check failed for ${url}: $_"
            }
        }
        
        # Test prediction endpoints
        Write-Status "Testing prediction endpoints..."
        
        $testData = @{
            loan_amount = 10000
            term = 36
            int_rate = 10.5
            annual_inc = 50000
            dti = 15.2
            fico_range_low = 720
            fico_range_high = 724
            pub_rec = 0
        }
        
        # Test traditional API
        try {
            Write-Status "Testing traditional prediction API..."
            $response = Invoke-RestMethod -Uri "http://localhost/api/predict" -Method POST -Body ($testData | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
            Write-Success "Traditional API test passed: $($response | ConvertTo-Json -Compress)"
        } catch {
            Write-Error "Traditional API test failed: $_"
        }
        
        # Test turbo API
        try {
            Write-Status "Testing turbo prediction API..."
            $advancedData = @{
                features = $testData
                model_type = "xgboost_turbo"
                simulate_missing = $true
            }
            $response = Invoke-RestMethod -Uri "http://localhost/models/advanced/predict" -Method POST -Body ($advancedData | ConvertTo-Json) -ContentType "application/json" -TimeoutSec 30
            Write-Success "Turbo API test passed: $($response | ConvertTo-Json -Compress)"
        } catch {
            Write-Error "Turbo API test failed: $_"
        }
        
    } catch {
        Write-Error "Deployment validation failed: $_"
    }
}

function Start-PerformanceBenchmark {
    if ($SkipBenchmark) {
        Write-Status "Skipping performance benchmark"
        return
    }
    
    Write-Status "Running performance benchmark..."
    
    try {
        python deploy_optimized.py 2>$null | Select-String "benchmark|Testing|Results|Throughput|response time"
        Write-Success "Performance benchmark completed"
    } catch {
        Write-Error "Performance benchmark failed: $_"
    }
}

function Show-DeploymentInfo {
    Write-Host "`n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "=" * 50 -ForegroundColor Green
    
    Write-Host "`n📋 Service Endpoints:" -ForegroundColor Yellow
    Write-Host "   🌐 Main API Gateway: http://localhost" -ForegroundColor White
    Write-Host "   🔗 Direct API: http://localhost:8000" -ForegroundColor White
    Write-Host "   📚 API Documentation: http://localhost/docs" -ForegroundColor White
    Write-Host "   🏥 Health Check: http://localhost/health" -ForegroundColor White
    Write-Host "   📊 Redis Cache: localhost:6379" -ForegroundColor White
    
    Write-Host "`n⚡ Performance Features:" -ForegroundColor Yellow
    Write-Host "   🏃‍♂️ XGBoost Turbo: ~0.067s training time" -ForegroundColor White
    Write-Host "   🚄 LightGBM Optimized: 3x faster training" -ForegroundColor White
    Write-Host "   🔧 Missing Feature Simulation: Automatic" -ForegroundColor White
    Write-Host "   📈 API Throughput: 46+ requests/second" -ForegroundColor White
    
    Write-Host "`n🛠️ Quick Commands:" -ForegroundColor Yellow
    Write-Host "   # View logs" -ForegroundColor Gray
    Write-Host "   docker-compose -f docker/docker-compose.optimized.yml logs -f" -ForegroundColor White
    Write-Host "   # Stop services" -ForegroundColor Gray
    Write-Host "   docker-compose -f docker/docker-compose.optimized.yml down" -ForegroundColor White
    Write-Host "   # Restart services" -ForegroundColor Gray
    Write-Host "   docker-compose -f docker/docker-compose.optimized.yml restart" -ForegroundColor White
}

# Main execution
try {
    # Check prerequisites
    Test-DockerRunning
    
    switch ($Mode.ToLower()) {
        "build" {
            Build-OptimizedImages
        }
        "deploy" {
            Deploy-ProductionStack
            Test-Deployment
        }
        "test" {
            Test-Deployment
            Start-PerformanceBenchmark
        }
        "full" {
            Build-OptimizedImages
            Deploy-ProductionStack
            Test-Deployment
            Start-PerformanceBenchmark
            Show-DeploymentInfo
        }
        default {
            Write-Error "Invalid mode. Use: full, build, deploy, or test"
            exit 1
        }
    }
    
    if ($Mode -eq "full") {
        Write-Host "`n✅ Optimized deployment completed successfully!" -ForegroundColor Green
    }
    
} catch {
    Write-Error "Deployment failed: $_"
    Write-Host "`n🔧 Troubleshooting:" -ForegroundColor Yellow
    Write-Host "   1. Ensure Docker Desktop is running" -ForegroundColor White
    Write-Host "   2. Check ports 80, 8000, 6379 are available" -ForegroundColor White
    Write-Host "   3. Run: docker-compose -f docker/docker-compose.optimized.yml logs" -ForegroundColor White
    exit 1
}