# Eden VIX 100 Trading Bot Startup Script
# This script checks dependencies and starts the bot

param(
    [switch]$DemoMode = $true,
    [switch]$CheckOnly = $false
)

# Colors for output
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Red = [System.ConsoleColor]::Red
$Cyan = [System.ConsoleColor]::Cyan

function Write-ColoredOutput {
    param($Message, $Color = [System.ConsoleColor]::White)
    Write-Host $Message -ForegroundColor $Color
}

function Test-PythonPackage {
    param($PackageName)
    try {
        $result = python -c "import $PackageName; print('OK')" 2>$null
        return $result -eq "OK"
    } catch {
        return $false
    }
}

Write-ColoredOutput "============================================" $Cyan
Write-ColoredOutput "    EDEN VIX 100 TRADING BOT STARTUP       " $Cyan
Write-ColoredOutput "============================================" $Cyan
Write-ColoredOutput ""

# Check Python installation
Write-ColoredOutput "Checking Python..." $Yellow
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-ColoredOutput "✅ $pythonVersion found" $Green
    } else {
        Write-ColoredOutput "❌ Python not found" $Red
        Write-ColoredOutput "Please install Python 3.11+ from python.org" $Yellow
        exit 1
    }
} catch {
    Write-ColoredOutput "❌ Python not accessible" $Red
    exit 1
}

# Check required packages
Write-ColoredOutput "`nChecking required packages..." $Yellow

$requiredPackages = @("pandas", "numpy", "MetaTrader5")
$missingPackages = @()

foreach ($package in $requiredPackages) {
    if (Test-PythonPackage $package) {
        Write-ColoredOutput "✅ $package installed" $Green
    } else {
        Write-ColoredOutput "❌ $package missing" $Red
        $missingPackages += $package
    }
}

# Install missing packages if found
if ($missingPackages.Count -gt 0) {
    Write-ColoredOutput "`n📦 Installing missing packages..." $Yellow
    foreach ($package in $missingPackages) {
        Write-ColoredOutput "Installing $package..." $Yellow
        try {
            python -m pip install $package
            Write-ColoredOutput "✅ $package installed successfully" $Green
        } catch {
            Write-ColoredOutput "❌ Failed to install $package" $Red
        }
    }
}

# Check if MetaTrader 5 is running
Write-ColoredOutput "`nChecking MetaTrader 5..." $Yellow
$mt5Process = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
if ($mt5Process) {
    Write-ColoredOutput "✅ MetaTrader 5 is running" $Green
} else {
    Write-ColoredOutput "⚠️  MetaTrader 5 not detected" $Yellow
    Write-ColoredOutput "Please start MT5 and log in to your account before running the bot" $Yellow
}

# Check if config file exists
Write-ColoredOutput "`nChecking configuration..." $Yellow
if (Test-Path "config.yaml") {
    Write-ColoredOutput "✅ Configuration file found" $Green
} else {
    Write-ColoredOutput "⚠️  No config.yaml found, bot will use default settings" $Yellow
}

# Check if this is just a dependency check
if ($CheckOnly) {
    Write-ColoredOutput "`n🔍 Dependency check complete!" $Green
    exit 0
}

# Start the bot
Write-ColoredOutput "`n🚀 Starting Eden VIX 100 Trading Bot..." $Green
Write-ColoredOutput "Mode: $(if ($DemoMode) { 'DEMO' } else { 'LIVE' })" $(if ($DemoMode) { $Yellow } else { $Red })

if (-not $DemoMode) {
    Write-ColoredOutput "`n⚠️  WARNING: You are about to start LIVE TRADING!" $Red
    Write-ColoredOutput "This will use real money. Are you sure? (y/N): " -NoNewline $Red
    $confirmation = Read-Host
    if ($confirmation -ne "y" -and $confirmation -ne "Y") {
        Write-ColoredOutput "Bot startup cancelled." $Yellow
        exit 0
    }
}

Write-ColoredOutput "`nPress Ctrl+C to stop the bot at any time." $Yellow
Write-ColoredOutput "============================================`n" $Cyan

try {
    # Set demo mode in environment if specified
    if ($DemoMode) {
        $env:EDEN_DEMO_MODE = "true"
    }
    
    # Start the bot
    python eden_vix100_bot.py
} catch {
    Write-ColoredOutput "`n❌ Error starting bot: $_" $Red
} finally {
    Write-ColoredOutput "`n🛑 Bot stopped." $Yellow
}