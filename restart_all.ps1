Write-Host "================================================================================"
Write-Host "Restarting Eden Backend and Bot"
Write-Host "================================================================================"

# Step 1: Stop existing processes
Write-Host "Step 1: Stopping any running processes..."
taskkill /F /FI "WINDOWTITLE eq *uvicorn*" 2>$null
taskkill /F /FI "IMAGENAME eq python.exe" /FI "MEMUSAGE gt 50000" 2>$null
Start-Sleep -Seconds 2

# Step 2: Start Backend
Write-Host "Step 2: Starting backend..."
$backendArgs = '/k "cd /d c:\Users\Administrator\Desktop\Eden\backend && python -m uvicorn main:app --host 0.0.0.0 --port 8443 --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem"'
Start-Process -FilePath "cmd.exe" -ArgumentList $backendArgs -WindowStyle Minimized

# Step 3: Start Bot
Write-Host "Step 3: Starting bot..."
$botArgs = '/k "cd /d c:\Users\Administrator\Desktop\Eden && python watchdog.py"'
Start-Process -FilePath "cmd.exe" -ArgumentList $botArgs -WindowStyle Minimized

# Step 4: Wait
Write-Host "Step 4: Waiting for services to start (15s)..."
Start-Sleep -Seconds 15

# Step 5: Verify
Write-Host "Step 5: Running verification..."
Set-Location "c:\Users\Administrator\Desktop\Eden"
python verify_api_contract.py

Write-Host "================================================================================"
Write-Host "Restart Sequence Complete"
Write-Host "================================================================================"
