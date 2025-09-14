@echo off
title Eden - The Origin of Order

echo.
echo Eden
echo The Origin of Order
echo.

rem Check if Python launcher script exists
if exist "launch.py" (
    echo Starting Eden Launcher...
    python launch.py %*
) else if exist "Eden.py" (
    echo Launching Eden directly...
    python Eden.py
) else if exist "installer.py" (
    echo Running Eden installer...
    python installer.py
) else (
    echo Eden files not found.
    echo Please make sure you're in the correct directory.
    echo.
    pause
    exit /b 1
)

echo.
echo Thank you for using Eden.
echo The Origin of Order
echo.
pause
