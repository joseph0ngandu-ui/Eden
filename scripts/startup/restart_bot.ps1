# Restart Bot Script - Plain Text

Write-Host "=== Eden Bot Restart Script ==="
Write-Host ""

# Step 1: Stop old Python processes
Write-Host "Stopping old bot processes..."
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    $pythonProcesses | ForEach-Object {
        Write-Host "  Killing PID: $($_.Id)"
        Stop-Process -Id $_.Id -Force
    }
    Start-Sleep -Seconds 2
    Write-Host "Old processes stopped"
}
else {
    Write-Host "No existing Python processes found"
}

Write-Host ""

# Step 2: Start Watchdog
Write-Host "Starting Eden Bot Watchdog..."

# Start in foreground with unbuffered output
$env:PYTHONUNBUFFERED = "1"
python watchdog.py
