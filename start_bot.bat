@echo off
cd /d C:\Users\Administrator\Eden
set PYTHONUNBUFFERED=1
set PATH=C:\Program Files\MetaTrader 5 Terminal;%PATH%
C:\Users\Administrator\Eden\venv\Scripts\python.exe watchdog.py
