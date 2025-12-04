# Restart Bot Script
# Stops old bot processes and starts fresh with fixes

Write-Host "=== Eden Bot Restart Script ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Stop old Python processes
Write-Host "🛑 Stopping old bot processes..." -ForegroundColor Yellow
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    $pythonProcesses | ForEach-Object {
        Write-Host "  Killing PID: $($_.Id) (Started: $($_.StartTime))" -ForegroundColor Gray
        Stop-Process -Id $_.Id -Force
    }
    Start-Sleep -Seconds 2
    Write-Host "✅ Old processes stopped" -ForegroundColor Green
} else {
    Write-Host "ℹ️  No existing Python processes found" -ForegroundColor Gray
}

Write-Host ""

# Step 2: Verify symbols are valid
Write-Host "🔍 Verifying trading symbols..." -ForegroundColor Yellow
python verify_symbols.py
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Symbol verification passed" -ForegroundColor Green
} else {
    Write-Host "❌ Symbol verification failed - check MT5 connection" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 3: Start watchdog
Write-Host "🚀 Starting Eden Bot Watchdog..." -ForegroundColor Yellow
Write-Host "   The bot will now:" -ForegroundColor Gray
Write-Host "   • Reset daily DD at midnight" -ForegroundColor Gray
Write-Host "   • Trade 7 forex pairs + gold (with 'm' suffix)" -ForegroundColor Gray
Write-Host "   • Use ML-optimized position sizing" -ForegroundColor Gray
Write-Host ""
Write-Host "   Press Ctrl+C to stop the bot" -ForegroundColor Yellow
Write-Host ""

# Start in foreground so you can see logs
python watchdog.py
