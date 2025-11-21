#!/usr/bin/env python3
"""
Bot watchdog that ensures the trading bot keeps running.
If bot crashes, it will restart automatically.
"""

import os
import subprocess
import sys
import time
import signal
from datetime import datetime

# Get the absolute path of the directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHONPATH = f"{os.path.join(BASE_DIR, 'trading')};{BASE_DIR}"
BOT_SCRIPT = os.path.join(BASE_DIR, "infrastructure", "bot_runner.py")
LOG_FILE = os.path.join(BASE_DIR, "watchdog.log")

def log(message):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[{timestamp}] {message}")

def start_bot():
    """Start the trading bot process."""
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH
    
    log("Starting trading bot...")
    
    try:
        # Determine python executable
        if os.path.exists(os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")):
            python_exe = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
        else:
            python_exe = sys.executable

        # Start the bot process
        process = subprocess.Popen(
            [
                python_exe,
                BOT_SCRIPT
            ],
            cwd=BASE_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        log(f"Trading bot started with PID: {process.pid}")
        
        # Monitor the process
        for line in iter(process.stdout.readline, ""):
            if line.strip():
                log(f"[BOT] {line.strip()}")
        
        # Process has terminated
        return_code = process.wait()
        log(f"Trading bot terminated with code: {return_code}")
        return False
        
    except Exception as e:
        log(f"Error starting bot: {e}")
        return False

def main():
    """Main watchdog loop."""
    log("Watchdog started")
    
    # Set up signal handlers
    def handle_sigterm(signum, frame):
        log(f"Received signal {signum}, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    # Main loop
    while True:
        try:
            # Start bot and wait for it to finish
            start_bot()
            
            # If we get here, bot stopped, wait a bit before restarting
            log("Bot stopped, waiting 30 seconds before restart...")
            time.sleep(30)
            
        except Exception as e:
            log(f"Watchdog error: {e}")
            time.sleep(60)  # Wait longer on unexpected errors
            
if __name__ == "__main__":
    main()