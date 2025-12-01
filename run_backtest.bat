@echo off
echo ========================================
echo  COMPREHENSIVE BACKTEST RUNNER
echo ========================================
echo.

REM Try python command
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using python command
    python "%~dp0scripts\comprehensive_backtest.py"
    goto :done
)

REM Try py launcher
py --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Using py launcher
    py "%~dp0scripts\comprehensive_backtest.py"
    goto :done
)

REM Try common installation paths
if exist "C:\Program Files\Python311\python.exe" (
    echo Using Python 3.11
    "C:\Program Files\Python311\python.exe" "%~dp0scripts\comprehensive_backtest.py"
    goto :done
)

if exist "C:\Python311\python.exe" (
    echo Using Python 3.11
    "C:\Python311\python.exe" "%~dp0scripts\comprehensive_backtest.py"
    goto :done
)

echo ERROR: Python not found!
echo Please install Python from python.org
pause
exit /b 1

:done
echo.
echo ========================================
echo  BACKTEST COMPLETE
echo ========================================
pause
