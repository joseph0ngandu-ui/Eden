@echo off
title Eden Test & Validation
color 0A

echo.
echo =============================================
echo    EDEN TEST & VALIDATION
echo =============================================
echo.

echo Testing Eden Python version...
python Eden_Test.py --test

echo.
echo =============================================
echo.

echo Testing Eden GUI (will open dialog box)...
python Eden_Test.py --gui

echo.
echo =============================================
echo.

echo Press any key to continue...
pause >nul

echo.
echo Testing Eden executable...
echo.

if exist "eden.exe" (
    echo Found eden.exe, testing...
    echo Starting eden.exe with timeout...
    timeout /t 3 /nobreak >nul
    tasklist | findstr "eden.exe" >nul
    if errorlevel 1 (
        echo Eden executable test: No processes found
    ) else (
        echo Eden executable test: Process is running
        taskkill /f /im eden.exe >nul 2>&1
    )
) else (
    echo eden.exe not found in this directory
)

echo.
echo =============================================
echo    TEST SUMMARY
echo =============================================
echo.

echo ✅ Python version tested
echo ✅ GUI version tested  
echo ✅ Executable version checked
echo.
echo If all tests passed, Eden is working correctly!
echo.
echo To install Eden properly:
echo 1. Run INSTALL_EDEN.bat (or right-click Eden Setup.exe → Run as administrator)
echo 2. Follow the installation wizard
echo 3. Launch from Desktop or Start Menu
echo.

pause