#!/usr/bin/env python3
"""
Bot Health Monitor - Ensures 24/7 Uptime
"""

import subprocess
import time
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Setup logging
log_file = Path(__file__).parent / 'logs' / 'monitor.log'
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

BOT_SCRIPT = "trading/trading_bot.py"
PYTHON_EXE = r"C:\Program Files\Cloudbase Solutions\Cloudbase-Init\Python\python.exe"

def is_bot_running():
    """Check if bot process is running."""
    # Simple check using tasklist (Windows)
    try:
        # This is a rough check, ideally we check for the specific python script
        # But for now, we'll rely on the subprocess handle
        return False 
    except:
        return False

def run_bot():
    """Run the bot and monitor it."""
    while True:
        logging.info(f"Starting Bot: {BOT_SCRIPT}")
        
        try:
            # Start the bot
            process = subprocess.Popen(
                [PYTHON_EXE, BOT_SCRIPT],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
            
            logging.info(f"Bot started with PID: {process.pid}")
            
            # Wait for it to finish (crash or stop)
            process.wait()
            
            logging.warning(f"Bot stopped! Exit code: {process.returncode}. Restarting in 5 seconds...")
            time.sleep(5)
            
        except Exception as e:
            logging.error(f"Monitor error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    print(f"Health Monitor Active. Logging to {log_file}")
    print("Press Ctrl+C to stop monitor (and bot).")
    run_bot()
