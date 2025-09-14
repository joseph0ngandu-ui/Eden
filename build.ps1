# Eden Trading System - Build Script
# This script builds the hybrid C++/Python trading system

param(
    [Parameter()]
    [string]$BuildType = "Release",
    
    [Parameter()]
    [switch]$Clean,
    
    [Parameter()]
    [switch]$Install,
    
    [Parameter()]
    [switch]$Portable
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
$Green = [System.ConsoleColor]::Green
$Yellow = [System.ConsoleColor]::Yellow
$Red = [System.ConsoleColor]::Red
$Cyan = [System.ConsoleColor]::Cyan

function Write-ColoredOutput {
    param($Message, $Color = [System.ConsoleColor]::White)
    Write-Host $Message -ForegroundColor $Color
}

function Test-Dependency {
    param($Command, $Name, $InstallHint = "")
    try {
        $null = Get-Command $Command -ErrorAction Stop
        Write-ColoredOutput "✓ $Name found" $Green
        return $true
    } catch {
        Write-ColoredOutput "✗ $Name not found" $Red
        if ($InstallHint) {
            Write-ColoredOutput "  Install hint: $InstallHint" $Yellow
        }
        return $false
    }
}

# Header
Write-ColoredOutput @"
 ______     _             
|  ____|   | |            
| |__   __| | ___ _ __    
|  __| / _` |/ _ \ '_ \   
| |___| (_| |  __/ | | |  
|______\__,_|\___|_| |_|  Trading System

Hybrid C++/Python Professional Trading Platform
"@ $Cyan

Write-ColoredOutput "Build Configuration: $BuildType" $Yellow

# Check dependencies
Write-ColoredOutput "`n=== Checking Dependencies ===" $Cyan

$allDepsOk = $true

$allDepsOk = (Test-Dependency "cmake" "CMake" "winget install Kitware.CMake") -and $allDepsOk
$allDepsOk = (Test-Dependency "python" "Python" "winget install Python.Python.3.11") -and $allDepsOk

# Check for Qt6
try {
    $qtPath = $env:QT_ROOT
    if (-not $qtPath) {
        $qtPath = "${env:ProgramFiles}\Qt"
        if (Test-Path $qtPath) {
            $qtVersions = Get-ChildItem $qtPath -Directory | Where-Object { $_.Name -match "^6\." } | Sort-Object Name -Descending
            if ($qtVersions) {
                $qtPath = Join-Path $qtVersions[0].FullName "msvc2022_64"
                if (Test-Path $qtPath) {
                    $env:Qt6_DIR = $qtPath
                    Write-ColoredOutput "✓ Qt6 found at $qtPath" $Green
                } else {
                    throw "Qt6 MSVC build not found"
                }
            } else {
                throw "No Qt6 versions found"
            }
        } else {
            throw "Qt installation not found"
        }
    } else {
        Write-ColoredOutput "✓ Qt6 found at $qtPath" $Green
    }
} catch {
    Write-ColoredOutput "✗ Qt6 not found - install from https://www.qt.io/download" $Red
    $allDepsOk = $false
}

if (-not $allDepsOk) {
    Write-ColoredOutput "`nPlease install missing dependencies and try again." $Red
    exit 1
}

# Set up environment
$ProjectRoot = $PSScriptRoot
$BuildDir = Join-Path $ProjectRoot "build"
$InstallDir = Join-Path $ProjectRoot "install"

Write-ColoredOutput "`n=== Setting Up Build Environment ===" $Cyan

# Clean if requested
if ($Clean -and (Test-Path $BuildDir)) {
    Write-ColoredOutput "Cleaning build directory..." $Yellow
    Remove-Item $BuildDir -Recurse -Force
}

# Create build directory
if (-not (Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir | Out-Null
}

# Install Python dependencies
Write-ColoredOutput "`nInstalling Python dependencies..." $Yellow
Set-Location (Join-Path $ProjectRoot "worker" "python")
python -m pip install -r requirements.txt
python -m pip install pyzmq onnxruntime

# Configure CMake
Set-Location $BuildDir
Write-ColoredOutput "`n=== Configuring CMake ===" $Cyan

$cmakeArgs = @(
    "-DCMAKE_BUILD_TYPE=$BuildType"
    "-DCMAKE_INSTALL_PREFIX=$InstallDir"
)

if ($env:Qt6_DIR) {
    $cmakeArgs += "-DQt6_DIR=$env:Qt6_DIR"
}

try {
    cmake .. @cmakeArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configuration failed" }
    Write-ColoredOutput "✓ CMake configuration successful" $Green
} catch {
    Write-ColoredOutput "✗ CMake configuration failed" $Red
    exit 1
}

# Build
Write-ColoredOutput "`n=== Building Eden ===" $Cyan
try {
    cmake --build . --config $BuildType --parallel
    if ($LASTEXITCODE -ne 0) { throw "Build failed" }
    Write-ColoredOutput "✓ Build successful" $Green
} catch {
    Write-ColoredOutput "✗ Build failed" $Red
    exit 1
}

# Install if requested
if ($Install) {
    Write-ColoredOutput "`n=== Installing Eden ===" $Cyan
    try {
        cmake --install . --config $BuildType
        if ($LASTEXITCODE -ne 0) { throw "Install failed" }
        Write-ColoredOutput "✓ Installation successful" $Green
        Write-ColoredOutput "Eden installed to: $InstallDir" $Yellow
    } catch {
        Write-ColoredOutput "✗ Installation failed" $Red
        exit 1
    }
}

# Create portable package if requested
if ($Portable) {
    Write-ColoredOutput "`n=== Creating Portable Package ===" $Cyan
    $PortableDir = Join-Path $ProjectRoot "Eden-Portable"
    
    if (Test-Path $PortableDir) {
        Remove-Item $PortableDir -Recurse -Force
    }
    
    # Copy application files
    Copy-Item $InstallDir $PortableDir -Recurse
    
    # Create portable launcher
    $launcherScript = @"
@echo off
cd /d "%~dp0"
set PATH=%~dp0;%PATH%
start Eden.exe %*
"@
    $launcherScript | Out-File (Join-Path $PortableDir "Eden-Portable.bat") -Encoding ascii
    
    Write-ColoredOutput "✓ Portable package created: $PortableDir" $Green
}

# Summary
Write-ColoredOutput "`n=== Build Summary ===" $Cyan
Write-ColoredOutput "Build Type: $BuildType" $Yellow
Write-ColoredOutput "Build Directory: $BuildDir" $Yellow
if ($Install) {
    Write-ColoredOutput "Install Directory: $InstallDir" $Yellow
}
if ($Portable) {
    Write-ColoredOutput "Portable Package: $PortableDir" $Yellow
}

Write-ColoredOutput "`n=== Next Steps ===" $Cyan
if ($Install) {
    Write-ColoredOutput "1. Run: $InstallDir\Eden.exe" $Green
} else {
    Write-ColoredOutput "1. Run: $BuildDir\Eden.exe" $Green
}
Write-ColoredOutput "2. Or build installer: setup\installers\eden_installer.iss" $Green
Write-ColoredOutput "3. Check documentation: docs\" $Green

Write-ColoredOutput "`n🎉 Eden build completed successfully!" $Green

Set-Location $ProjectRoot