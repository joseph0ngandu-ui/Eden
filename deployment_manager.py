#!/usr/bin/env python3
"""
Eden Deployment & Recovery Manager

Handles full deployment, error recovery, and operational monitoring.
Ensures all components run smoothly with automatic error handling.
"""

import sys
import os
import subprocess
import time
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
import psutil

# Configure logging
log_dir = Path("C:/Users/Administrator/Eden/logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "deployment_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ComponentStatus:
    """Track status of Eden components."""
    def __init__(self):
        self.mt5_terminal = False
        self.mt5_connection = False
        self.backend_api = False
        self.optimizer = False
        self.last_check = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "mt5_terminal": self.mt5_terminal,
            "mt5_connection": self.mt5_connection,
            "backend_api": self.backend_api,
            "optimizer": self.optimizer,
            "last_check": self.last_check.isoformat()
        }


class DeploymentManager:
    """Manages Eden deployment, monitoring, and recovery."""
    
    def __init__(self):
        self.eden_path = Path("C:/Users/Administrator/Eden")
        self.mt5_path = Path("C:/Program Files/MetaTrader 5 Terminal")
        self.status = ComponentStatus()
        self.backend_process: Optional[subprocess.Popen] = None
        self.optimizer_process: Optional[subprocess.Popen] = None
        self.retry_count = {}
        self.max_retries = 3
    
    def check_mt5_terminal(self) -> bool:
        """Check if MT5 terminal is running."""
        try:
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] == 'terminal64.exe':
                    self.status.mt5_terminal = True
                    return True
            self.status.mt5_terminal = False
            return False
        except Exception as e:
            logger.error(f"Error checking MT5 terminal: {e}")
            return False
    
    def start_mt5_terminal(self) -> bool:
        """Start MT5 terminal if not running."""
        if self.check_mt5_terminal():
            logger.info("✓ MT5 terminal already running")
            return True
        
        try:
            mt5_exe = self.mt5_path / "terminal64.exe"
            if not mt5_exe.exists():
                logger.error(f"MT5 executable not found at {mt5_exe}")
                return False
            
            logger.info("Starting MT5 terminal...")
            subprocess.Popen([str(mt5_exe)], shell=True)
            
            # Wait for terminal to start
            for i in range(30):
                time.sleep(1)
                if self.check_mt5_terminal():
                    logger.info("✓ MT5 terminal started")
                    return True
            
            logger.error("MT5 terminal failed to start within 30 seconds")
            return False
        
        except Exception as e:
            logger.error(f"Error starting MT5 terminal: {e}")
            return False
    
    def check_mt5_connection(self) -> bool:
        """Check if MT5 Python connection works."""
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                self.status.mt5_connection = False
                return False
            
            account = mt5.account_info()
            mt5.shutdown()
            
            if account is None:
                self.status.mt5_connection = False
                return False
            
            self.status.mt5_connection = True
            return True
        
        except Exception as e:
            logger.error(f"Error checking MT5 connection: {e}")
            self.status.mt5_connection = False
            return False
    
    def check_backend_api(self) -> bool:
        """Check if backend API is running."""
        try:
            import requests
            response = requests.get("http://localhost:8000/docs", timeout=5)
            self.status.backend_api = response.status_code == 200
            return self.status.backend_api
        except:
            self.status.backend_api = False
            return False
    
    def start_backend_api(self) -> bool:
        """Start the backend API server."""
        if self.check_backend_api():
            logger.info("✓ Backend API already running")
            return True
        
        try:
            logger.info("Starting backend API...")
            backend_dir = self.eden_path / "backend"
            
            # Start uvicorn server
            self.backend_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "main:app", 
                 "--host", "0.0.0.0", "--port", "8000"],
                cwd=str(backend_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for API to be ready
            for i in range(30):
                time.sleep(1)
                if self.check_backend_api():
                    logger.info("✓ Backend API started at http://localhost:8000")
                    return True
            
            logger.warning("Backend API may not be fully ready")
            return False
        
        except Exception as e:
            logger.error(f"Error starting backend API: {e}")
            return False
    
    def start_optimizer(self) -> bool:
        """Start the autonomous optimizer."""
        try:
            logger.info("Starting autonomous optimizer...")
            optimizer_script = self.eden_path / "autonomous_optimizer.py"
            
            self.optimizer_process = subprocess.Popen(
                [sys.executable, str(optimizer_script)],
                cwd=str(self.eden_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            time.sleep(2)
            if self.optimizer_process.poll() is None:
                logger.info("✓ Autonomous optimizer started")
                self.status.optimizer = True
                return True
            else:
                logger.error("Optimizer process terminated unexpectedly")
                return False
        
        except Exception as e:
            logger.error(f"Error starting optimizer: {e}")
            return False
    
    def deploy_all_components(self) -> bool:
        """Deploy all Eden components."""
        logger.info("=" * 80)
        logger.info("Starting Eden deployment...")
        logger.info("=" * 80)
        
        # 1. Check/Start MT5 Terminal
        logger.info("\n[1/4] Checking MT5 Terminal...")
        if not self.start_mt5_terminal():
            logger.error("✗ Failed to start MT5 terminal")
            return False
        
        # 2. Verify MT5 Connection
        logger.info("\n[2/4] Verifying MT5 Connection...")
        if not self.check_mt5_connection():
            logger.error("✗ MT5 connection failed")
            return False
        logger.info("✓ MT5 connection verified")
        
        # 3. Start Backend API
        logger.info("\n[3/4] Starting Backend API...")
        if not self.start_backend_api():
            logger.warning("⚠ Backend API may have issues")
        
        # 4. Start Optimizer
        logger.info("\n[4/4] Starting Autonomous Optimizer...")
        if not self.start_optimizer():
            logger.warning("⚠ Optimizer may have issues")
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ Eden deployment complete!")
        logger.info("=" * 80)
        
        return True
    
    def monitor_and_recover(self, check_interval: int = 60):
        """Continuously monitor components and recover from failures."""
        logger.info(f"Starting continuous monitoring (interval: {check_interval}s)...")
        
        while True:
            try:
                time.sleep(check_interval)
                
                logger.info("\n--- Health Check ---")
                
                # Check MT5 terminal
                if not self.check_mt5_terminal():
                    logger.warning("⚠ MT5 terminal not running, attempting restart...")
                    self.start_mt5_terminal()
                
                # Check MT5 connection
                if not self.check_mt5_connection():
                    logger.warning("⚠ MT5 connection lost, attempting recovery...")
                    time.sleep(5)
                    self.check_mt5_connection()
                
                # Check backend API
                if not self.check_backend_api():
                    logger.warning("⚠ Backend API down, attempting restart...")
                    self.start_backend_api()
                
                # Check optimizer
                if self.optimizer_process and self.optimizer_process.poll() is not None:
                    logger.warning("⚠ Optimizer stopped, attempting restart...")
                    self.start_optimizer()
                
                # Log status
                logger.info(f"Status: MT5={self.status.mt5_terminal} | "
                           f"Connection={self.status.mt5_connection} | "
                           f"API={self.status.backend_api} | "
                           f"Optimizer={self.status.optimizer}")
                
                # Save status
                self.save_status()
            
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(10)
    
    def save_status(self):
        """Save current status to file."""
        status_file = log_dir / "deployment_status.json"
        with open(status_file, 'w') as f:
            json.dump(self.status.to_dict(), f, indent=2)
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report."""
        import MetaTrader5 as mt5
        
        report = []
        report.append("=" * 80)
        report.append("EDEN DEPLOYMENT STATUS REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Python Environment
        report.append("Python Environment:")
        report.append(f"  Version: {sys.version.split()[0]}")
        report.append(f"  Path: {sys.executable}")
        report.append("")
        
        # MT5 Status
        report.append("MT5 Status:")
        report.append(f"  Terminal Running: {'✓' if self.status.mt5_terminal else '✗'}")
        report.append(f"  Python Connection: {'✓' if self.status.mt5_connection else '✗'}")
        
        if mt5.initialize():
            info = mt5.terminal_info()
            account = mt5.account_info()
            if info:
                report.append(f"  Terminal: {info.name}")
                report.append(f"  Company: {info.company}")
            if account:
                report.append(f"  Account: {account.login}")
                report.append(f"  Server: {account.server}")
                report.append(f"  Balance: ${account.balance:.2f}")
            mt5.shutdown()
        report.append("")
        
        # Backend API
        report.append("Backend API:")
        report.append(f"  Status: {'✓ Running' if self.status.backend_api else '✗ Stopped'}")
        report.append(f"  URL: http://localhost:8000")
        report.append("")
        
        # Optimizer
        report.append("Autonomous Optimizer:")
        report.append(f"  Status: {'✓ Running' if self.status.optimizer else '✗ Stopped'}")
        report.append("")
        
        # Performance Snapshot
        snapshot_file = log_dir / "performance_snapshot.json"
        if snapshot_file.exists():
            with open(snapshot_file) as f:
                snapshot = json.load(f)
            report.append("Active Strategy:")
            report.append(f"  {snapshot.get('active_strategy', 'Unknown')}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def shutdown(self):
        """Shutdown all components."""
        logger.info("Shutting down Eden components...")
        
        if self.backend_process:
            self.backend_process.terminate()
            logger.info("✓ Backend API stopped")
        
        if self.optimizer_process:
            self.optimizer_process.terminate()
            logger.info("✓ Optimizer stopped")
        
        logger.info("✓ Shutdown complete")


def main():
    """Main entry point."""
    manager = DeploymentManager()
    
    try:
        # Deploy all components
        if manager.deploy_all_components():
            # Generate and print status report
            print("\n" + manager.generate_status_report())
            
            # Start monitoring
            logger.info("\nStarting continuous monitoring...")
            logger.info("Press Ctrl+C to stop\n")
            manager.monitor_and_recover(check_interval=60)
        else:
            logger.error("Deployment failed")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    finally:
        manager.shutdown()


if __name__ == "__main__":
    main()
