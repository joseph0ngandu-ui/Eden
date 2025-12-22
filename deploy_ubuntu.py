#!/usr/bin/env python3
"""
Eden Trading Bot - Ubuntu Server Deployment Script
Deploys Eden bot to Ubuntu server with real MT5 connection
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Server configuration
SERVER_IP = "84.8.142.27"
SSH_KEY = "/Users/josephngandu/Downloads/ssh-key-2025-12-22.key"
SERVER_USER = "ubuntu"
REMOTE_DIR = "/home/ubuntu/Eden"

def run_ssh_command(command, capture_output=True):
    """Execute command on remote server via SSH"""
    ssh_cmd = [
        "ssh", "-i", SSH_KEY, f"{SERVER_USER}@{SERVER_IP}", command
    ]
    
    if capture_output:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    else:
        return subprocess.run(ssh_cmd).returncode == 0

def upload_files():
    """Upload Eden bot files to server"""
    print("📤 Uploading Eden bot files to server...")
    
    # Files to upload
    files_to_upload = [
        "trading/",
        "config/",
        "scripts/",
        ".env.eden",
        "requirements.txt",
        "watchdog.py"
    ]
    
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            scp_cmd = [
                "scp", "-i", SSH_KEY, "-r", file_path, 
                f"{SERVER_USER}@{SERVER_IP}:{REMOTE_DIR}/"
            ]
            success = subprocess.run(scp_cmd).returncode == 0
            if success:
                print(f"✅ Uploaded {file_path}")
            else:
                print(f"❌ Failed to upload {file_path}")
                return False
    
    return True

def setup_python_environment():
    """Set up Python environment on server"""
    print("🐍 Setting up Python environment...")
    
    commands = [
        "cd Eden && python3 -m pip install --upgrade pip",
        "cd Eden && pip3 install -r requirements.txt",
        "cd Eden && pip3 install MetaTrader5"
    ]
    
    for cmd in commands:
        success, stdout, stderr = run_ssh_command(cmd)
        if success:
            print(f"✅ {cmd.split('&&')[-1].strip()}")
        else:
            print(f"❌ Failed: {cmd}")
            print(f"Error: {stderr}")
            return False
    
    return True

def create_mt5_config():
    """Create MT5 configuration for server"""
    print("⚙️ Creating MT5 configuration...")
    
    mt5_config = '''#!/usr/bin/env python3
"""
MT5 Connection Module for Ubuntu Server
Uses Wine-installed MT5 with proper environment setup
"""

import MetaTrader5 as mt5
import os
import time
import logging

# Set up Wine environment for MT5
os.environ['DISPLAY'] = ':99'
os.environ['WINEPREFIX'] = os.path.expanduser('~/.wine-mt5')
os.environ['WINEARCH'] = 'win64'

def initialize_mt5():
    """Initialize MT5 connection with Wine environment"""
    try:
        # Initialize MT5
        if not mt5.initialize():
            logging.error(f"MT5 initialization failed: {mt5.last_error()}")
            return False
        
        # Check connection
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            return False
        
        logging.info(f"MT5 connected - Account: {account_info.login}, Server: {account_info.server}")
        return True
        
    except Exception as e:
        logging.error(f"MT5 initialization error: {e}")
        return False

def test_connection():
    """Test MT5 connection"""
    if initialize_mt5():
        print("✅ MT5 connection successful")
        
        # Get account info
        account_info = mt5.account_info()
        if account_info:
            print(f"Account: {account_info.login}")
            print(f"Server: {account_info.server}")
            print(f"Balance: ${account_info.balance:.2f}")
            print(f"Equity: ${account_info.equity:.2f}")
        
        # Test symbol data
        symbols = ["USTECm", "US500m", "EURUSDm", "USDJPYm"]
        for symbol in symbols:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                print(f"{symbol}: {tick.bid:.5f}")
            else:
                print(f"❌ {symbol}: No data")
        
        mt5.shutdown()
        return True
    else:
        print("❌ MT5 connection failed")
        return False

if __name__ == "__main__":
    test_connection()
'''
    
    # Upload MT5 config
    with open("/tmp/mt5_config.py", "w") as f:
        f.write(mt5_config)
    
    scp_cmd = [
        "scp", "-i", SSH_KEY, "/tmp/mt5_config.py",
        f"{SERVER_USER}@{SERVER_IP}:{REMOTE_DIR}/mt5_config.py"
    ]
    
    success = subprocess.run(scp_cmd).returncode == 0
    if success:
        print("✅ MT5 configuration uploaded")
        os.remove("/tmp/mt5_config.py")
        return True
    else:
        print("❌ Failed to upload MT5 configuration")
        return False

def create_startup_script():
    """Create startup script for Eden bot"""
    print("🚀 Creating startup script...")
    
    startup_script = '''#!/bin/bash
# Eden Trading Bot Startup Script for Ubuntu Server

export DISPLAY=:99
export WINEPREFIX=~/.wine-mt5
export WINEARCH=win64

cd /home/ubuntu/Eden

echo "🤖 Starting Eden Trading Bot..."
echo "📊 Environment: LIVE TRADING"
echo "🖥️  Server: Ubuntu with Wine MT5"
echo "⏰ Started: $(date)"

# Test MT5 connection first
echo "🔌 Testing MT5 connection..."
python3 mt5_config.py

if [ $? -eq 0 ]; then
    echo "✅ MT5 connection verified"
    echo "🚀 Starting trading bot..."
    
    # Start the bot with logging
    python3 -u trading/trading_bot.py 2>&1 | tee -a logs/eden_$(date +%Y%m%d).log
else
    echo "❌ MT5 connection failed - check Wine/MT5 setup"
    exit 1
fi
'''
    
    # Upload startup script
    with open("/tmp/start_eden.sh", "w") as f:
        f.write(startup_script)
    
    scp_cmd = [
        "scp", "-i", SSH_KEY, "/tmp/start_eden.sh",
        f"{SERVER_USER}@{SERVER_IP}:{REMOTE_DIR}/start_eden.sh"
    ]
    
    success = subprocess.run(scp_cmd).returncode == 0
    if success:
        # Make executable
        run_ssh_command(f"chmod +x {REMOTE_DIR}/start_eden.sh")
        print("✅ Startup script created")
        os.remove("/tmp/start_eden.sh")
        return True
    else:
        print("❌ Failed to upload startup script")
        return False

def test_mt5_connection():
    """Test MT5 connection on server"""
    print("🔌 Testing MT5 connection on server...")
    
    success, stdout, stderr = run_ssh_command(f"cd {REMOTE_DIR} && python3 mt5_config.py")
    
    if success:
        print("✅ MT5 connection test successful")
        print(stdout)
        return True
    else:
        print("❌ MT5 connection test failed")
        print(f"Error: {stderr}")
        return False

def deploy():
    """Main deployment function"""
    print("🚀 Eden Trading Bot - Ubuntu Deployment")
    print("=" * 50)
    
    # Create logs directory
    run_ssh_command(f"mkdir -p {REMOTE_DIR}/logs")
    
    # Step 1: Upload files
    if not upload_files():
        print("❌ File upload failed")
        return False
    
    # Step 2: Setup Python environment
    if not setup_python_environment():
        print("❌ Python environment setup failed")
        return False
    
    # Step 3: Create MT5 configuration
    if not create_mt5_config():
        print("❌ MT5 configuration failed")
        return False
    
    # Step 4: Create startup script
    if not create_startup_script():
        print("❌ Startup script creation failed")
        return False
    
    # Step 5: Test MT5 connection
    if not test_mt5_connection():
        print("❌ MT5 connection test failed")
        return False
    
    print("\n🎉 Deployment completed successfully!")
    print("\nTo start Eden bot:")
    print(f"ssh -i {SSH_KEY} {SERVER_USER}@{SERVER_IP}")
    print("cd Eden && ./start_eden.sh")
    
    return True

if __name__ == "__main__":
    deploy()
