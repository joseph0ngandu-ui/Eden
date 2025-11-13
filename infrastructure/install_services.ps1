# Eden Trading System - Service Installation Script
# Installs both the API and Trading Bot as Windows services

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Eden Trading System - Service Installer" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check if running as Administrator
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host "❌ ERROR: This script must be run as Administrator" -ForegroundColor Red
    Write-Host "   Right-click PowerShell and select 'Run as Administrator'" -ForegroundColor Yellow
    exit 1
}

# Get Python path
$pythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonPath) {
    Write-Host "❌ ERROR: Python not found in PATH" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python found: $pythonPath" -ForegroundColor Green

# Install pywin32 if not already installed
Write-Host ""
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$pywin32Check = python -c "import win32serviceutil" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing pywin32..." -ForegroundColor Yellow
    pip install --quiet pywin32
}
Write-Host "✓ pywin32 installed" -ForegroundColor Green

# Service installation function
function Install-EdenService {
    param(
        [string]$ServiceScript,
        [string]$ServiceName,
        [string]$DisplayName
    )
    
    Write-Host ""
    Write-Host "Installing $DisplayName..." -ForegroundColor Cyan
    
    # Stop and remove existing service if it exists
    $existingService = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($existingService) {
        Write-Host "  Stopping existing service..." -ForegroundColor Yellow
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
        
        Write-Host "  Removing existing service..." -ForegroundColor Yellow
        python $ServiceScript remove
        Start-Sleep -Seconds 2
    }
    
    # Install service
    Write-Host "  Installing service..." -ForegroundColor Yellow
    python $ServiceScript install
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $DisplayName installed successfully" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Failed to install $DisplayName" -ForegroundColor Red
        return $false
    }
}

# Service start function
function Start-EdenService {
    param(
        [string]$ServiceScript,
        [string]$ServiceName,
        [string]$DisplayName
    )
    
    Write-Host "  Starting $DisplayName..." -ForegroundColor Yellow
    python $ServiceScript start
    
    Start-Sleep -Seconds 3
    
    $service = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    if ($service -and $service.Status -eq 'Running') {
        Write-Host "✓ $DisplayName is running" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Failed to start $DisplayName" -ForegroundColor Red
        return $false
    }
}

# Install Eden Trading API
$apiInstalled = Install-EdenService `
    -ServiceScript "C:\Users\Administrator\Eden\backend\api_service.py" `
    -ServiceName "EdenTradingAPI" `
    -DisplayName "Eden Trading API"

# Install Eden Trading Bot
$botInstalled = Install-EdenService `
    -ServiceScript "C:\Users\Administrator\Eden\bot_service.py" `
    -ServiceName "EdenTradingBot" `
    -DisplayName "Eden Trading Bot"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Starting Services" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan

# Start services
$apiStarted = $false
$botStarted = $false

if ($apiInstalled) {
    $apiStarted = Start-EdenService `
        -ServiceScript "C:\Users\Administrator\Eden\backend\api_service.py" `
        -ServiceName "EdenTradingAPI" `
        -DisplayName "Eden Trading API"
}

if ($botInstalled) {
    $botStarted = Start-EdenService `
        -ServiceScript "C:\Users\Administrator\Eden\bot_service.py" `
        -ServiceName "EdenTradingBot" `
        -DisplayName "Eden Trading Bot"
}

# Summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Installation Summary" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

if ($apiStarted) {
    Write-Host "✓ Eden Trading API:  RUNNING on https://13.50.226.20:443" -ForegroundColor Green
} else {
    Write-Host "❌ Eden Trading API:  FAILED" -ForegroundColor Red
}

if ($botStarted) {
    Write-Host "✓ Eden Trading Bot:  RUNNING in LIVE MODE" -ForegroundColor Green
} else {
    Write-Host "❌ Eden Trading Bot:  FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Service Management Commands:" -ForegroundColor Yellow
Write-Host "  View status:   Get-Service EdenTradingAPI, EdenTradingBot" -ForegroundColor Gray
Write-Host "  Stop API:      Stop-Service EdenTradingAPI" -ForegroundColor Gray
Write-Host "  Stop Bot:      Stop-Service EdenTradingBot" -ForegroundColor Gray
Write-Host "  Start API:     Start-Service EdenTradingAPI" -ForegroundColor Gray
Write-Host "  Start Bot:     Start-Service EdenTradingBot" -ForegroundColor Gray
Write-Host "  View logs:     Get-EventLog -LogName Application -Source Eden* -Newest 50" -ForegroundColor Gray
Write-Host ""
Write-Host "Both services will automatically start on system boot." -ForegroundColor Cyan
Write-Host "They will continue running even when you disconnect RDP." -ForegroundColor Cyan
Write-Host ""
