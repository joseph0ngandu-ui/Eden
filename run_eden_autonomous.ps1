# Eden Autonomous Management Script
# Keeps Eden running with automatic updates, health checks, and git sync

$ErrorActionPreference = "Continue"
$EdenPath = "C:\Users\Administrator\Eden"
$LogFile = "$EdenPath\logs\autonomous_runner.log"

function Write-Log {
    param($Message, $Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor $Color
    Add-Content -Path $LogFile -Value $logMessage
}

function Test-BackendHealth {
    try {
        $response = Invoke-WebRequest -Uri 'http://localhost:8000/docs' -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Test-MT5Connection {
    try {
        $result = python -c "import MetaTrader5 as mt5; mt5.initialize(); info = mt5.account_info(); mt5.shutdown(); print('OK' if info else 'FAIL')"
        return $result -eq "OK"
    } catch {
        return $false
    }
}

function Start-EdenComponents {
    Write-Log "Starting Eden deployment manager..." "Cyan"
    
    # Start deployment manager in background
    $process = Start-Process -FilePath "python" `
        -ArgumentList "$EdenPath\deployment_manager.py" `
        -WorkingDirectory $EdenPath `
        -PassThru `
        -WindowStyle Hidden
    
    Start-Sleep -Seconds 10
    
    if ($process -and !$process.HasExited) {
        Write-Log "✅ Eden deployment manager started (PID: $($process.Id))" "Green"
        return $process
    } else {
        Write-Log "❌ Failed to start Eden deployment manager" "Red"
        return $null
    }
}

function Stop-EdenComponents {
    Write-Log "Stopping Eden components..." "Yellow"
    
    # Stop Python processes related to Eden
    Get-Process -Name "python" -ErrorAction SilentlyContinue | Where-Object {
        $_.Path -like "*Python*" -and 
        ($_.CommandLine -like "*deployment_manager*" -or 
         $_.CommandLine -like "*autonomous_optimizer*" -or
         $_.CommandLine -like "*uvicorn*")
    } | Stop-Process -Force -ErrorAction SilentlyContinue
    
    Start-Sleep -Seconds 3
    Write-Log "✅ Eden components stopped" "Green"
}

function Sync-Repository {
    Write-Log "Syncing repository..." "Cyan"
    Set-Location $EdenPath
    
    try {
        # Fetch latest changes
        git fetch origin main 2>&1 | Out-Null
        
        # Check if there are updates
        $localHash = git rev-parse HEAD
        $remoteHash = git rev-parse origin/main
        
        if ($localHash -ne $remoteHash) {
            Write-Log "📥 Updates detected, pulling changes..." "Yellow"
            git reset --hard origin/main 2>&1 | Out-Null
            Write-Log "✅ Repository updated" "Green"
            return $true
        } else {
            Write-Log "✅ Repository up to date" "Green"
            return $false
        }
    } catch {
        Write-Log "⚠️ Git sync failed: $_" "Yellow"
        return $false
    }
}

function Push-LocalChanges {
    Set-Location $EdenPath
    
    try {
        # Check if there are local changes
        $status = git status --porcelain
        
        if ($status) {
            Write-Log "📤 Pushing local changes..." "Cyan"
            git add .
            git commit -m "auto: sync Eden updates $(Get-Date -Format 'yyyy-MM-dd HH:mm')" 2>&1 | Out-Null
            git push origin main 2>&1 | Out-Null
            Write-Log "✅ Local changes pushed" "Green"
        }
    } catch {
        Write-Log "⚠️ Git push failed: $_" "Yellow"
    }
}

function Update-Dependencies {
    Write-Log "Checking dependencies..." "Cyan"
    Set-Location $EdenPath
    
    try {
        # Update Python packages quietly
        python -m pip install --quiet --upgrade pip 2>&1 | Out-Null
        
        if (Test-Path "$EdenPath\requirements.txt") {
            python -m pip install --quiet -r requirements.txt 2>&1 | Out-Null
            Write-Log "✅ Python dependencies updated" "Green"
        }
    } catch {
        Write-Log "⚠️ Dependency update failed: $_" "Yellow"
    }
}

# Main autonomous loop
Write-Log "================================" "Cyan"
Write-Log "Eden Autonomous Manager Started" "Cyan"
Write-Log "================================" "Cyan"

$edenProcess = $null
$cycleCount = 0

while ($true) {
    try {
        $cycleCount++
        Write-Log "" "White"
        Write-Log "--- Cycle #$cycleCount ---" "Cyan"
        
        # Step 1: Sync repository
        $repoUpdated = Sync-Repository
        
        # Step 2: Update dependencies if repo was updated
        if ($repoUpdated) {
            Update-Dependencies
        }
        
        # Step 3: Check MT5 connection
        Write-Log "Checking MT5 connection..." "Cyan"
        if (Test-MT5Connection) {
            Write-Log "✅ MT5 connection healthy" "Green"
        } else {
            Write-Log "❌ MT5 connection failed" "Red"
        }
        
        # Step 4: Check backend health
        Write-Log "Checking backend health..." "Cyan"
        if (Test-BackendHealth) {
            Write-Log "✅ Backend API healthy" "Green"
        } else {
            Write-Log "⚠️ Backend API not responding" "Yellow"
        }
        
        # Step 5: Check if Eden process is running
        if ($edenProcess -and !$edenProcess.HasExited) {
            Write-Log "✅ Eden deployment manager running (PID: $($edenProcess.Id))" "Green"
        } else {
            Write-Log "⚠️ Eden deployment manager not running, restarting..." "Yellow"
            Stop-EdenComponents
            $edenProcess = Start-EdenComponents
        }
        
        # Step 6: Push local changes (logs, performance data)
        Push-LocalChanges
        
        # Step 7: Generate status report
        Write-Log "Generating status report..." "Cyan"
        python "$EdenPath\generate_status_report.py" | Out-Null
        
        Write-Log "🔁 Cycle complete. Waiting 10 minutes..." "Green"
        Start-Sleep -Seconds 600  # 10 minutes
        
    } catch {
        Write-Log "❌ Error in autonomous cycle: $_" "Red"
        Write-Log "Retrying in 1 minute..." "Yellow"
        Start-Sleep -Seconds 60
    }
}
