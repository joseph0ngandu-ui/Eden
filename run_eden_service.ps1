# Eden Service Runner
# Simplified version for Windows Service compatibility

$EdenPath = "C:\Users\Administrator\Eden"
$LogFile = "$EdenPath\logs\service_runner.log"

function Write-ServiceLog {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Add-Content -Path $LogFile -Value $logMessage
}

Write-ServiceLog "Eden Service Starting..."

try {
    Set-Location $EdenPath
    
    # Start deployment manager
    Write-ServiceLog "Launching deployment manager..."
    python "$EdenPath\deployment_manager.py"
    
} catch {
    Write-ServiceLog "ERROR: $_"
    Start-Sleep -Seconds 30
}

Write-ServiceLog "Eden Service Stopped"
