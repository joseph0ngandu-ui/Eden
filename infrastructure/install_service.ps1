# Eden Windows Service Installer
# Uses Task Scheduler for reliable background execution

$ServiceName = "EdenTradingBot"
$EdenPath = "C:\Users\Administrator\Eden"
$PythonExe = "C:\Program Files\Python312\python.exe"
$ScriptPath = "$EdenPath\deployment_manager.py"

Write-Host "Installing Eden as Windows Service..." -ForegroundColor Cyan
Write-Host ""

# Method 1: Task Scheduler (Recommended)
Write-Host "Creating Scheduled Task..." -ForegroundColor Yellow

# Remove existing task if it exists
Unregister-ScheduledTask -TaskName $ServiceName -Confirm:$false -ErrorAction SilentlyContinue

# Create action
$action = New-ScheduledTaskAction -Execute $PythonExe -Argument $ScriptPath -WorkingDirectory $EdenPath

# Create trigger (at startup)
$trigger = New-ScheduledTaskTrigger -AtStartup

# Create settings
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 999

# Create principal (run with highest privileges)
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Register the task
Register-ScheduledTask -TaskName $ServiceName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description "Eden Trading Bot - Autonomous Trading System"

Write-Host ""
Write-Host "SUCCESS: Eden service installed!" -ForegroundColor Green
Write-Host ""
Write-Host "Service Name: $ServiceName" -ForegroundColor Cyan
Write-Host "Method: Task Scheduler" -ForegroundColor Cyan
Write-Host ""
Write-Host "Management Commands:" -ForegroundColor Yellow
Write-Host "  Start:   Start-ScheduledTask -TaskName '$ServiceName'" -ForegroundColor White
Write-Host "  Stop:    Stop-ScheduledTask -TaskName '$ServiceName'" -ForegroundColor White
Write-Host "  Status:  Get-ScheduledTask -TaskName '$ServiceName'" -ForegroundColor White
Write-Host "  Remove:  Unregister-ScheduledTask -TaskName '$ServiceName' -Confirm:`$false" -ForegroundColor White
Write-Host ""
Write-Host "Logs: $EdenPath\logs\deployment_manager.log" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting service now..." -ForegroundColor Yellow
Start-ScheduledTask -TaskName $ServiceName

Start-Sleep -Seconds 5

$task = Get-ScheduledTask -TaskName $ServiceName
Write-Host ""
Write-Host "Service Status: $($task.State)" -ForegroundColor Green
Write-Host ""
Write-Host "Eden is now running autonomously!" -ForegroundColor Green
Write-Host "It will auto-start on system reboot." -ForegroundColor Green
