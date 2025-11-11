# Check Backend API Status

Write-Host "Checking Eden Backend API..." -ForegroundColor Cyan
Write-Host ""

# Check if backend process is running
$backendProcess = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*main:app*"
}

if ($backendProcess) {
    Write-Host "[PROCESS] Backend running (PID: $($backendProcess.Id))" -ForegroundColor Green
} else {
    Write-Host "[PROCESS] Backend NOT running" -ForegroundColor Red
}

# Check if API responds
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/docs" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "[API] Backend responding: HTTP $($response.StatusCode)" -ForegroundColor Green
        Write-Host "[URL] http://localhost:8000/docs" -ForegroundColor Cyan
        Write-Host "[URL] http://localhost:8000/redoc" -ForegroundColor Cyan
    }
} catch {
    Write-Host "[API] Backend NOT responding" -ForegroundColor Red
}

Write-Host ""
Write-Host "Default Login:" -ForegroundColor Yellow
Write-Host "  Email: admin@eden.com" -ForegroundColor White
Write-Host "  Password: admin123" -ForegroundColor White
