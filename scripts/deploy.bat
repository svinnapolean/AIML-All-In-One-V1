@echo off
REM Windows Deployment Script for Loan Default Prediction API
REM This script deploys the API locally using Docker on Windows

echo 🚀 Starting Loan Default Prediction API Deployment on Windows
echo ==============================================================

REM Configuration
set API_PORT=8000
if "%ENVIRONMENT%"=="" set ENVIRONMENT=development
set COMPOSE_FILE=docker/docker-compose.dev.yml

if "%ENVIRONMENT%"=="production" (
    set COMPOSE_FILE=docker/docker-compose.prod.yml
)

echo 📋 Environment Configuration:
echo    - Environment: %ENVIRONMENT%
echo    - API Port: %API_PORT%
echo    - Compose File: %COMPOSE_FILE%

REM Check prerequisites
echo 📋 Checking prerequisites...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not in PATH. Please install Docker Desktop for Windows.
    echo Download from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Desktop which includes Compose.
    pause
    exit /b 1
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    echo Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    echo Waiting for Docker to start...
    timeout /t 30 /nobreak
    
    REM Check again
    docker info >nul 2>&1
    if errorlevel 1 (
        echo ❌ Docker failed to start. Please start Docker Desktop manually and try again.
        pause
        exit /b 1
    )
)

echo ✅ Prerequisites check passed

REM Check for trained models
echo 📊 Checking for trained models...
if exist "src\models\saved_models" (
    dir /b "src\models\saved_models\*.h5" >nul 2>&1
    if not errorlevel 1 (
        for /f %%i in ('dir /b "src\models\saved_models\*.h5" 2^>nul ^| find /c /v ""') do set model_count=%%i
        echo ✅ Found !model_count! trained models
    ) else (
        echo ⚠️ No trained models found in src\models\saved_models
        echo ⚠️ The API will start but predictions may fail until models are trained
        
        set /p continue="Do you want to train a demo model first? (y/N): "
        if /i "!continue!"=="y" (
            echo 🚀 Training a demo model...
            cd src\models
            python fast_deep_learning.py --save_model --model_name demo_model
            cd ..\..
            echo ✅ Demo model trained successfully
        )
    )
) else (
    echo ⚠️ Models directory not found. Creating...
    mkdir "src\models\saved_models"
)

REM Check for loan data
echo 📊 Checking for loan data...
if not exist "loan_data\loan_data.csv" (
    echo ⚠️ Loan data not found at loan_data\loan_data.csv
    echo ⚠️ Some API features may not work properly
    
    REM Create dummy loan data directory
    if not exist "loan_data" mkdir loan_data
    echo Creating placeholder loan_data directory
)

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose -f %COMPOSE_FILE% down --remove-orphans 2>nul

REM Build and start services
echo 🔨 Building Docker images...
docker-compose -f %COMPOSE_FILE% build

if errorlevel 1 (
    echo ❌ Docker build failed. Check the error messages above.
    pause
    exit /b 1
)

echo 🚀 Starting services...
docker-compose -f %COMPOSE_FILE% up -d

if errorlevel 1 (
    echo ❌ Failed to start services. Check the error messages above.
    pause
    exit /b 1
)

REM Wait for API to be ready
echo ⏳ Waiting for API to be ready...
set MAX_ATTEMPTS=30
set ATTEMPT=1

:wait_loop
if %ATTEMPT% GTR %MAX_ATTEMPTS% goto timeout

curl -s http://localhost:%API_PORT%/health >nul 2>&1
if not errorlevel 1 goto api_ready

echo|set /p="."
timeout /t 2 /nobreak >nul
set /a ATTEMPT+=1
goto wait_loop

:timeout
echo.
echo ❌ API failed to start within 60 seconds
echo 📋 Check logs with: docker-compose -f %COMPOSE_FILE% logs
pause
exit /b 1

:api_ready
echo.
echo ✅ API is ready!

REM Verify deployment
echo 🔍 Verifying deployment...

REM Check API health
curl -s http://localhost:%API_PORT%/health > temp_health.json
findstr "healthy" temp_health.json >nul
if not errorlevel 1 (
    echo ✅ API health check passed
) else (
    echo ❌ API health check failed
    type temp_health.json
)

REM Check if model is loaded
findstr "model_loaded.*true" temp_health.json >nul
if not errorlevel 1 (
    for /f "tokens=2 delims=:" %%a in ('findstr "model_name" temp_health.json') do (
        set model_name=%%a
        set model_name=!model_name:"=!
        set model_name=!model_name:,=!
        echo ✅ Model '!model_name!' is loaded and ready
    )
) else (
    echo ⚠️ No model is currently loaded
)

del temp_health.json 2>nul

echo.
echo 🎉 Deployment Completed Successfully!
echo ======================================
echo.
echo 🌐 API URLs:
echo    Main API: http://localhost:%API_PORT%
echo    Health Check: http://localhost:%API_PORT%/health
echo    API Documentation: http://localhost:%API_PORT%/docs
echo    Interactive Docs: http://localhost:%API_PORT%/redoc
echo.

if "%ENVIRONMENT%"=="production" (
    echo 📊 Monitoring URLs:
    echo    Grafana Dashboard: http://localhost:3000 ^(admin/admin^)
    echo    Prometheus Metrics: http://localhost:9090
    echo.
)

echo 🔧 Useful Commands:
echo    View logs: docker-compose -f %COMPOSE_FILE% logs -f
echo    Stop services: docker-compose -f %COMPOSE_FILE% down
echo    Restart services: docker-compose -f %COMPOSE_FILE% restart
echo    Run validation: python examples\api_client_examples.py
echo.

echo 📝 Example API Calls:
echo    # Check health
echo    curl http://localhost:%API_PORT%/health
echo.
echo    # List models
echo    curl http://localhost:%API_PORT%/models
echo.
echo    # Make prediction ^(PowerShell^)
echo    $body = @{
echo        amt_credit = 450000
echo        amt_annuity = 25000
echo        amt_income_total = 150000
echo        code_gender = "M"
echo        days_birth = -12000
echo        days_employed = -2000
echo        name_contract_type = "Cash loans"
echo        name_income_type = "Working"
echo        name_education_type = "Higher education"
echo        name_family_status = "Married"
echo        name_housing_type = "House / apartment"
echo        region_population_relative = 0.02
echo    } ^| ConvertTo-Json
echo.
echo    Invoke-RestMethod -Uri "http://localhost:%API_PORT%/predict" -Method Post -ContentType "application/json" -Body $body
echo.

echo ✅ API is ready to accept requests!

REM Optionally run client examples
set /p run_examples="Do you want to run client examples now? (y/N): "
if /i "!run_examples!"=="y" (
    echo 🚀 Running client examples...
    if exist "examples\api_client_examples.py" (
        cd examples
        python api_client_examples.py
        cd ..
    ) else (
        echo ⚠️ Client examples not found
    )
)

echo.
echo ✅ Deployment script completed!
echo 🌐 Open http://localhost:%API_PORT%/docs to explore the API!
pause