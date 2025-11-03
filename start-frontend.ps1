#!/usr/bin/env powershell
# PowerShell script to start React frontend development server

Write-Host "🚀 Starting React Frontend Development Server" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Green

# Check if we're in the right directory
$currentDir = Get-Location
Write-Host "📂 Current directory: $currentDir" -ForegroundColor Cyan

# Navigate to frontend directory
$frontendDir = Join-Path $currentDir "frontend"
if (Test-Path $frontendDir) {
    Write-Host "📁 Navigating to frontend directory..." -ForegroundColor Cyan
    Set-Location $frontendDir
    
    # Verify package.json exists
    if (Test-Path "package.json") {
        Write-Host "✅ Found package.json" -ForegroundColor Green
        
        # Check if node_modules exists
        if (-not (Test-Path "node_modules")) {
            Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
            npm install
        }
        
        Write-Host "🌐 Starting development server..." -ForegroundColor Green
        Write-Host "🔗 Your React app will open at: http://localhost:3000" -ForegroundColor Yellow
        Write-Host "🔗 Your API is running at: http://localhost:8000" -ForegroundColor Yellow
        Write-Host "" -ForegroundColor White
        Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
        Write-Host "" -ForegroundColor White
        
        # Start the React development server
        npm start
        
    } else {
        Write-Host "❌ package.json not found in frontend directory" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ Frontend directory not found!" -ForegroundColor Red
    Write-Host "Please run this script from the project root directory" -ForegroundColor Yellow
    exit 1
}