#!/usr/bin/env pwsh
<#
.SYNOPSIS
Launch MT5 and run the backtest with real data
.DESCRIPTION
This script:
1. Checks if MT5 terminal is running
2. If not, launches it
3. Waits for connection
4. Runs the export and backtest
#>

# Configuration
$MT5_PATH = "C:\Program Files\MetaTrader 5\terminal64.exe"
$SCRIPT_PATH = "C:\Users\Sal\Documents\Eden\export_mt5_data.py"
$WAIT_TIME = 30  # seconds to wait for MT5 to connect

function Check-MT5-Running {
    $mt5_process = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
    return $null -ne $mt5_process
}

function Start-MT5 {
    Write-Host "Starting MT5..." -ForegroundColor Cyan
    
    if (-not (Test-Path $MT5_PATH)) {
        Write-Host "MT5 not found at $MT5_PATH" -ForegroundColor Red
        Write-Host "Please install MT5 or update the path." -ForegroundColor Yellow
        return $false
    }
    
    & $MT5_PATH
    
    # Wait for MT5 to fully start
    Write-Host "Waiting for MT5 to initialize..." -ForegroundColor Yellow
    Start-Sleep -Seconds $WAIT_TIME
    
    $connected = $false
    for ($i = 0; $i -lt 10; $i++) {
        if (Check-MT5-Running) {
            $connected = $true
            break
        }
        Start-Sleep -Seconds 2
    }
    
    if ($connected) {
        Write-Host "✓ MT5 is running" -ForegroundColor Green
        return $true
    } else {
        Write-Host "✗ Failed to start MT5" -ForegroundColor Red
        return $false
    }
}

function Run-Backtest {
    Write-Host "`nStarting backtest script..." -ForegroundColor Cyan
    
    # Run the Python export and backtest
    python $SCRIPT_PATH
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ Backtest completed successfully" -ForegroundColor Green
    } else {
        Write-Host "`n✗ Backtest failed with exit code $LASTEXITCODE" -ForegroundColor Red
    }
}

# Main execution
Clear-Host
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║      MT5 Real Data Backtest - Risk Ladder Strategy       ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Check if MT5 is already running
if (Check-MT5-Running) {
    Write-Host "✓ MT5 is already running" -ForegroundColor Green
} else {
    Write-Host "⚠ MT5 is not running" -ForegroundColor Yellow
    Write-Host ""
    
    $launch_mt5 = Read-Host "Launch MT5 now? (y/n)"
    if ($launch_mt5 -eq "y" -or $launch_mt5 -eq "Y") {
        if (-not (Start-MT5)) {
            exit 1
        }
    } else {
        Write-Host "MT5 is required to run this backtest." -ForegroundColor Yellow
        exit 1
    }
}

Write-Host ""
Write-Host "Prerequisites:" -ForegroundColor Yellow
Write-Host "  ✓ MT5 is running" -ForegroundColor Green
Write-Host "  ✓ Account is logged in" 
Write-Host "  ✓ Required symbols are available: VIX75, VIX100, VIX50, VIX25, etc."
Write-Host ""

$proceed = Read-Host "Ready to export data and run backtest? (y/n)"
if ($proceed -eq "y" -or $proceed -eq "Y") {
    Run-Backtest
} else {
    Write-Host "Cancelled." -ForegroundColor Yellow
    exit 0
}
