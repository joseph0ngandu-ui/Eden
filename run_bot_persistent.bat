@echo off
cd /d "C:\Users\Administrator\Eden"
set PYTHONPATH=C:\Users\Administrator\Eden\trading;C:\Users\Administrator\Eden
venv\Scripts\activate
echo Starting Trading Bot... %date% %time%
start /B /MIN venv\Scripts\python.exe infrastructure\bot_runner.py
echo Bot launched in background. Check logs for status.
timeout /t 3 > nul