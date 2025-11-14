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

PYTHONPATH = "C:\\Users\\Administrator\\Eden\\trading;C:\\Users\\Administrator\\Eden"
BOT_SCRIPT = "C:\\Users\\Administrator\\Eden\\infrastructure\\bot_runner.py"
LOG_FILE = "C:\\Users\\Administrator\\Eden\\watchdog.log"

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
        # Start the bot process
        process = subprocess.Popen(
            [
                "C:\\Users\\Administrator\\Eden\\venv\\Scripts\\python.exe",
                BOT_SCRIPT
            ],
            cwd="C:\\Users\\Administrator\\Eden",
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