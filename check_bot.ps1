# Check if the bot is running
Get-Process | Where-Object { $_.ProcessName -eq "python" } | Select-Object Id, StartTime | Sort-Object -Property StartTime -Descending | Select-Object -First 5

# Check for watchdog logs
if (Test-Path "C:\Users\Administrator\Eden\watchdog.log") {
    Write-Host "`n-------- Watchdog Log (last 10 lines) --------" -ForegroundColor Green
    Get-Content "C:\Users\Administrator\Eden\watchdog.log" -Tail 10
}

# Check for trading logs
$logDir = "C:\Users\Administrator\Eden\logs"
if (Test-Path $logDir) {
    $latestLog = Get-ChildItem -Path $logDir -Filter "*.log" | Sort-Object -Property LastWriteTime -Descending | Select-Object -First 1
    if ($latestLog) {
        Write-Host "`n-------- Trading Log (last 5 lines) --------" -ForegroundColor Green
        Get-Content $latestLog.FullName -Tail 5
    }
}

# Print summary
Write-Host "`n-------- Summary --------" -ForegroundColor Yellow
Write-Host "Bot processes running: $(Get-Process -Name python -ErrorAction SilentlyContinue | Measure-Object).Count"
Write-Host "Scheduled task status:"
schtasks /query /tn "TradingBot" /fo LIST | Find-Str "Status"