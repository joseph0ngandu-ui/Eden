@echo off
title Eden Quick Installer
color 0B

echo.
echo =============================================
echo    EDEN - Professional Trading Platform
echo =============================================
echo.
echo Welcome to Eden Setup!
echo.
echo This will install Eden with full Windows integration:
echo   * Desktop shortcut
echo   * Start Menu entries  
echo   * Add/Remove Programs support
echo   * File associations
echo.
echo Press any key to start installation...
pause >nul

echo.
echo Starting Eden Setup...
echo.

if exist "Eden Setup.exe" (
    echo Running Eden Setup.exe with administrator privileges...
    powershell -Command "Start-Process 'Eden Setup.exe' -Verb RunAs"
    echo.
    echo Installation started! Please follow the setup wizard.
    echo.
    pause
) else (
    echo ERROR: Eden Setup.exe not found in this directory!
    echo Please make sure all setup files are present.
    echo.
    pause
)

echo.
echo Thank you for installing Eden!
echo Happy Trading! 📈
echo.
pause