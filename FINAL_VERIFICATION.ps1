# Eden Final Verification Before Deployment

Write-Host "================================" -ForegroundColor Cyan
Write-Host "EDEN FINAL VERIFICATION" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check Python
Write-Host "[1/8] Python Environment" -ForegroundColor Yellow
$pythonVersion = python --version
Write-Host "  $pythonVersion" -ForegroundColor Green

# 2. Check MT5
Write-Host "[2/8] MT5 Connection" -ForegroundColor Yellow
$mt5Test = python -c "import MetaTrader5 as mt5; mt5.initialize(); info = mt5.account_info(); mt5.shutdown(); print('OK' if info else 'FAIL')"
if ($mt5Test -eq "OK") {
    Write-Host "  MT5 Connected" -ForegroundColor Green
} else {
    Write-Host "  MT5 NOT Connected" -ForegroundColor Red
}

# 3. Check SSL Certificates
Write-Host "[3/8] SSL Certificates" -ForegroundColor Yellow
if (Test-Path "C:\Users\Administrator\Eden\backend\ssl\cert.pem") {
    Write-Host "  Cert: Found" -ForegroundColor Green
} else {
    Write-Host "  Cert: Missing" -ForegroundColor Red
}
if (Test-Path "C:\Users\Administrator\Eden\backend\ssl\key.pem") {
    Write-Host "  Key: Found" -ForegroundColor Green
} else {
    Write-Host "  Key: Missing" -ForegroundColor Red
}

# 4. Check Backend HTTPS
Write-Host "[4/8] Backend HTTPS" -ForegroundColor Yellow
$backendTest = python -c "import requests; requests.packages.urllib3.disable_warnings(); r = requests.get('https://localhost:8443/docs', verify=False); print('OK' if r.status_code == 200 else 'FAIL')"
if ($backendTest -eq "OK") {
    Write-Host "  HTTPS API: Running" -ForegroundColor Green
} else {
    Write-Host "  HTTPS API: NOT Running" -ForegroundColor Red
}

# 5. Check Network IP
Write-Host "[5/8] Network Configuration" -ForegroundColor Yellow
$ipConfig = Get-NetIPAddress | Where-Object {$_.AddressFamily -eq "IPv4" -and $_.IPAddress -notlike "127.*"} | Select-Object -First 1
Write-Host "  Local IP: $($ipConfig.IPAddress)" -ForegroundColor Green

# Try to get public IP
try {
    $publicIP = (Invoke-WebRequest -Uri "https://api.ipify.org" -UseBasicParsing -TimeoutSec 3).Content
    Write-Host "  Public IP: $publicIP" -ForegroundColor Green
} catch {
    Write-Host "  Public IP: Unable to determine" -ForegroundColor Yellow
}

# 6. Check Git Status
Write-Host "[6/8] Git Repository" -ForegroundColor Yellow
Set-Location "C:\Users\Administrator\Eden"
$gitStatus = git status --porcelain | Measure-Object | Select-Object -ExpandProperty Count
Write-Host "  Modified files: $gitStatus" -ForegroundColor $(if ($gitStatus -eq 0) {"Green"} else {"Yellow"})

# 7. Check Files
Write-Host "[7/8] Essential Files" -ForegroundColor Yellow
$files = @(
    "deployment_manager.py",
    "autonomous_optimizer.py",
    "test_mt5_connection.py",
    "backend\main.py",
    "backend\ssl\cert.pem",
    "backend\ssl\key.pem"
)
foreach ($file in $files) {
    $exists = Test-Path $file
    $status = if ($exists) {"✓"} else {"✗"}
    $color = if ($exists) {"Green"} else {"Red"}
    Write-Host "  $status $file" -ForegroundColor $color
}

# 8. Summary
Write-Host ""
Write-Host "[8/8] Mobile App Connection" -ForegroundColor Yellow
Write-Host "  API URL: https://$($ipConfig.IPAddress):8443" -ForegroundColor Cyan
Write-Host "  Email: admin@eden.com" -ForegroundColor White
Write-Host "  Password: admin123" -ForegroundColor White
Write-Host ""
Write-Host "  Note: Accept certificate warning on first connection" -ForegroundColor Yellow

Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "VERIFICATION COMPLETE" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
