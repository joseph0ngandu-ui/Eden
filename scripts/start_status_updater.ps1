# Eden Status Uploader launcher
# Starts the eden-status-uploader.py as a persistent background job

$EdenRoot = "C:\Users\Administrator\Eden"
$logDir = "$EdenRoot\logs"
If (!(Test-Path $logDir)) { New-Item -Path $logDir -ItemType Directory -Force | Out-Null }

Write-Host "Starting Eden Status Uploader..." -ForegroundColor Cyan

# Stop any existing instances of the uploader
$existing = Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*eden-status-uploader*" }
If ($existing) {
    Write-Host "Stopping existing eden-status-uploader processes..." -ForegroundColor Yellow
    $existing | Stop-Process -Force
    Start-Sleep -Seconds 2
}

# Run eden-status-uploader as a background PowerShell Job
$job = Start-Job -Name "EdenStatusUploader" -ScriptBlock {
    # Ensure python from venv is used, where boto3 + psutil + pandas + mt5 are installed
    $venvPython = "C:\Users\Administrator\Eden\venv\Scripts\python.exe"
    if (!(Test-Path $venvPython)) {
        Write-Host "Eden venv python not found at $venvPython – falling back to system python" -ForegroundColor Yellow
        $venvPython = "python"
    }
    & $venvPython "C:\Users\Administrator\Eden\scripts\eden-status-uploader.py"
}

# Auto-kill and restart helper: watcher PowerShell job
Start-Job -Name "EdenStatusUploaderWatchdog" -ScriptBlock {
    while ($true) {
        try {
            $updater = Get-Job -Name "EdenStatusUploader" -ErrorAction SilentlyContinue
            if (-not $updater -or $updater.State -eq "Failed" -or $updater.State -eq "Stopped") {
                Write-Host "Restarting Eden Status Uploader..." -ForegroundColor Yellow
                Remove-Job -Name "EdenStatusUploader" -ErrorAction SilentlyContinue
                $newJob = Start-Job -Name "EdenStatusUploader" -ScriptBlock {
                    $venvPython = "C:\Users\Administrator\Eden\venv\Scripts\python.exe"
                    if (!(Test-Path $venvPython)) { $venvPython = "python" }
                    & $venvPython "C:\Users\Administrator\Eden\scripts\eden-status-uploader.py"
                }
            }
        } catch {
            Write-Host "Watchdog error, ignoring: $_" -ForegroundColor Gray
        }
        Start-Sleep -Seconds 60  # check every minute
    }
}

Write-Host "Eden Status Uploader started in the background." -ForegroundColor Green
Write-Host ""
Write-Host "Commands:"
Write-Host "  View live log: Get-Job -Name EdenStatusUploader | Receive-Job -Keep" -ForegroundColor Gray
Write-Host "  Stop uploader: Stop-Job -Name EdenStatusUploader,Stop-Job -Name EdenStatusUploaderWatchdog" -ForegroundColor Gray
Write-Host ""
Write-Host "It will now push live balances and bot status to DynamoDB every 15 seconds for userId='demo-user'." -ForegroundColor Cyan