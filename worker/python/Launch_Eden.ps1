#!/usr/bin/env pwsh
#
# Eden Launcher
# Professional deployment and development tool
#

param(
    [switch]$Build,
    [switch]$Install,
    [switch]$Run
)

$ErrorActionPreference = "Stop"

Write-Host "Eden" -ForegroundColor White
Write-Host "The Origin of Order" -ForegroundColor Gray
Write-Host ""

# Functions
function Test-AdminRights {
    return ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")
}

function Install-Dependencies {
    Write-Host "Installing dependencies..." -ForegroundColor White
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "Python found: $pythonVersion" -ForegroundColor Gray
    }
    catch {
        Write-Host "Python not found. Please install Python 3.11+ from python.org" -ForegroundColor Red
        exit 1
    }
    
    # Install packages
    Write-Host "Installing Python packages..." -ForegroundColor Gray
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    
    Write-Host "Dependencies installed successfully" -ForegroundColor White
}

function Start-EdenDevelopment {
    Write-Host "Starting Eden in development mode..." -ForegroundColor White
    
    # Check if virtual environment exists
    if (Test-Path ".venv") {
        Write-Host "Activating virtual environment..." -ForegroundColor Gray
        & ".\.venv\Scripts\Activate.ps1"
    }
    
    # Install dependencies if needed
    Install-Dependencies
    
    # Launch Eden
    Write-Host "Launching Eden Dashboard..." -ForegroundColor White
    python Eden.py
}

function Build-EdenDeployment {
    Write-Host "Building Eden deployment..." -ForegroundColor White
    
    # Install dependencies
    Install-Dependencies
    
    # Run deployment builder
    python build_deployment.py
}

function Install-Eden {
    Write-Host "Running Eden installer..." -ForegroundColor White
    
    # Check if installer exists
    if (Test-Path "installer.py") {
        python installer.py
    }
    else {
        Write-Host "Installer not found. Building and running development version..." -ForegroundColor Gray
        Start-EdenDevelopment
    }
}

# Main execution
try {
    if ($Build) {
        Build-EdenDeployment
    }
    elseif ($Install) {
        Install-Eden
    }
    elseif ($Run) {
        Start-EdenDevelopment
    }
    else {
        # Interactive menu
        Write-Host "What would you like to do?" -ForegroundColor White
        Write-Host ""
        Write-Host "1. Run Eden (Development)" -ForegroundColor Gray
        Write-Host "2. Install Eden (User-friendly installer)" -ForegroundColor Gray
        Write-Host "3. Build Eden (Create deployment package)" -ForegroundColor Gray
        Write-Host "4. Exit" -ForegroundColor Gray
        Write-Host ""
        
        $choice = Read-Host "Enter your choice (1-4)"
        
        switch ($choice) {
            "1" { Start-EdenDevelopment }
            "2" { Install-Eden }
            "3" { Build-EdenDeployment }
            "4" { 
                Write-Host "Goodbye" -ForegroundColor Gray
                exit 0 
            }
            default { 
                Write-Host "Invalid choice. Please run the script again." -ForegroundColor Red
                exit 1
            }
        }
    }
    
    Write-Host ""
    Write-Host "Operation completed successfully" -ForegroundColor White
    Write-Host "Eden - The Origin of Order" -ForegroundColor Gray
}
catch {
    Write-Host ""
    Write-Host "An error occurred: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try running as administrator or check the error above." -ForegroundColor Gray
    exit 1
}

# Pause to let user see the output
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
