# Eden Trading System - Scheduled Task Creator
# Creates Windows scheduled tasks that run on system boot and persist across RDP disconnections

#Requires -RunAsAdministrator

Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Eden Trading System - Scheduled Task Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Remove existing tasks if they exist
Write-Host "Removing existing tasks (if any)..." -ForegroundColor Yellow
Unregister-ScheduledTask -TaskName "EdenTradingAPI" -Confirm:$false -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName "EdenTradingBot" -Confirm:$false -ErrorAction SilentlyContinue

# Create API task
Write-Host "Creating Eden Trading API task..." -ForegroundColor Cyan
$apiAction = New-ScheduledTaskAction `
    -Execute "C:\Users\Administrator\Eden\start_api.bat" `
    -WorkingDirectory "C:\Users\Administrator\Eden"

$apiTrigger = New-ScheduledTaskTrigger -AtStartup

$apiSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$apiPrincipal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName "EdenTradingAPI" `
    -Description "Eden Trading Bot HTTPS API on port 8443" `
    -Action $apiAction `
    -Trigger $apiTrigger `
    -Settings $apiSettings `
    -Principal $apiPrincipal | Out-Null

Write-Host "✓ Eden Trading API task created" -ForegroundColor Green

# Create Bot task
Write-Host "Creating Eden Trading Bot task..." -ForegroundColor Cyan
$botAction = New-ScheduledTaskAction `
    -Execute "C:\Users\Administrator\Eden\start_bot.bat" `
    -WorkingDirectory "C:\Users\Administrator\Eden"

$botTrigger = New-ScheduledTaskTrigger -AtStartup

$botSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$botPrincipal = New-ScheduledTaskPrincipal `
    -UserId "SYSTEM" `
    -LogonType ServiceAccount `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName "EdenTradingBot" `
    -Description "Eden Trading Bot in LIVE mode" `
    -Action $botAction `
    -Trigger $botTrigger `
    -Settings $botSettings `
    -Principal $botPrincipal | Out-Null

Write-Host "✓ Eden Trading Bot task created" -ForegroundColor Green

# Start the tasks
Write-Host ""
Write-Host "Starting tasks..." -ForegroundColor Cyan
Start-ScheduledTask -TaskName "EdenTradingAPI"
Start-Sleep -Seconds 3
Start-ScheduledTask -TaskName "EdenTradingBot"
Start-Sleep -Seconds 2

# Check status
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Status" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

$apiTask = Get-ScheduledTask -TaskName "EdenTradingAPI"
$botTask = Get-ScheduledTask -TaskName "EdenTradingBot"

if ($apiTask.State -eq "Running") {
    Write-Host "✓ Eden Trading API:  RUNNING" -ForegroundColor Green
    Write-Host "  URL: https://13.50.226.20:8443" -ForegroundColor Gray
} else {
    Write-Host "❌ Eden Trading API:  $($apiTask.State)" -ForegroundColor Yellow
}

if ($botTask.State -eq "Running") {
    Write-Host "✓ Eden Trading Bot:  RUNNING in LIVE MODE" -ForegroundColor Green
} else {
    Write-Host "❌ Eden Trading Bot:  $($botTask.State)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Management Commands:" -ForegroundColor Yellow
Write-Host "  View tasks:    Get-ScheduledTask -TaskName Eden*" -ForegroundColor Gray
Write-Host "  Stop API:      Stop-ScheduledTask -TaskName EdenTradingAPI" -ForegroundColor Gray
Write-Host "  Stop Bot:      Stop-ScheduledTask -TaskName EdenTradingBot" -ForegroundColor Gray
Write-Host "  Start API:     Start-ScheduledTask -TaskName EdenTradingAPI" -ForegroundColor Gray
Write-Host "  Start Bot:     Start-ScheduledTask -TaskName EdenTradingBot" -ForegroundColor Gray
Write-Host ""
Write-Host "✓ Both tasks will automatically start on system boot" -ForegroundColor Green
Write-Host "✓ They will continue running when you disconnect RDP" -ForegroundColor Green
Write-Host ""
