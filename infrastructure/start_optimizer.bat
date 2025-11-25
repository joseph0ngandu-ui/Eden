@echo off
:: Start Autonomous Optimizer
echo Starting Autonomous Optimizer...
cd /d %~dp0
cd ..
set PYTHONPATH=%PYTHONPATH%;%CD%
python infrastructure/autonomous_optimizer.py
pause
