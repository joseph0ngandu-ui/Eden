# Eden Trading System - Background Process Starter
# Starts both API and Trading Bot as detached background processes

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Starting Eden Trading System" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$EdenRoot = "C:\Users\Administrator\Eden"

# Stop existing processes if any
Write-Host "Stopping existing processes..." -ForegroundColor Yellow
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*Eden*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start API as background job
Write-Host "Starting Eden Trading API (HTTPS on port 8443)..." -ForegroundColor Cyan
$apiJob = Start-Job -ScriptBlock {
    Set-Location "C:\Users\Administrator\Eden\backend"
    $env:PYTHONUNBUFFERED = "1"
    python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-certfile "C:/Users/Administrator/Eden/backend/ssl/cert.pem" --ssl-keyfile "C:/Users/Administrator/Eden/backend/ssl/key.pem" --log-config null 2>&1
} -Name "EdenAPI"

Start-Sleep -Seconds 3

# Start Bot as background job
Write-Host "Starting Eden Trading Bot (LIVE MODE)..." -ForegroundColor Cyan
$botJob = Start-Job -ScriptBlock {
    Set-Location "C:\Users\Administrator\Eden"
    $env:PYTHONUNBUFFERED = "1"
    $env:EDEN_SHADOW = "0"
    python bot_runner.py 2>&1
} -Name "EdenBot"

Start-Sleep -Seconds 2

# Check status
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Status" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$api = Get-Job -Name "EdenAPI" -ErrorAction SilentlyContinue
$bot = Get-Job -Name "EdenBot" -ErrorAction SilentlyContinue

if ($api -and $api.State -eq "Running") {
    Write-Host "✓ Eden Trading API:  RUNNING" -ForegroundColor Green
    Write-Host "  URL: https://13.50.226.20:8443" -ForegroundColor Gray
} else {
    Write-Host "❌ Eden Trading API:  FAILED" -ForegroundColor Red
}

if ($bot -and $bot.State -eq "Running") {
    Write-Host "✓ Eden Trading Bot:  RUNNING in LIVE MODE" -ForegroundColor Green
} else {
    Write-Host "❌ Eden Trading Bot:  FAILED" -ForegroundColor Red
}

Write-Host ""
Write-Host "Management Commands:" -ForegroundColor Yellow
Write-Host "  View jobs:     Get-Job" -ForegroundColor Gray
Write-Host "  View API logs: Receive-Job -Name EdenAPI -Keep" -ForegroundColor Gray
Write-Host "  View Bot logs: Receive-Job -Name EdenBot -Keep" -ForegroundColor Gray
Write-Host "  Stop API:      Stop-Job -Name EdenAPI; Remove-Job -Name EdenAPI" -ForegroundColor Gray
Write-Host "  Stop Bot:      Stop-Job -Name EdenBot; Remove-Job -Name EdenBot" -ForegroundColor Gray
Write-Host ""
Write-Host "NOTE: These jobs run in the current PowerShell session." -ForegroundColor Yellow
Write-Host "To make them persist across RDP disconnections, use Task Scheduler:" -ForegroundColor Yellow
Write-Host "  .\create_scheduled_tasks.ps1" -ForegroundColor Cyan
Write-Host ""
