"""
Eden Trading Bot - Windows Service Wrapper

Runs the trading bot as a Windows service in LIVE mode that persists across RDP disconnections.
"""

import sys
import os
import time
import json
import servicemanager
import socket
import win32event
import win32service
import win32serviceutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.chdir(str(PROJECT_ROOT))

# Set LIVE mode environment variable
os.environ['EDEN_SHADOW'] = '0'  # 0 = LIVE, 1 = PAPER

class EdenBotService(win32serviceutil.ServiceFramework):
    _svc_name_ = "EdenTradingBot"
    _svc_display_name_ = "Eden Trading Bot (Live Mode)"
    _svc_description_ = "Eden autonomous trading bot for Volatility 75 Index"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.stop_event = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_alive = True

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.stop_event)
        self.is_alive = False

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, '')
        )
        self.main()

    def main(self):
        """Run the trading bot in LIVE mode with automatic restart on failure."""
        from trading_bot import TradingBot
        from dotenv import load_dotenv
        load_dotenv()
        
        # Ensure strategies.json has at least one LIVE active strategy
        self.ensure_live_strategy()
        
        servicemanager.LogInfoMsg("Starting Eden Trading Bot in LIVE mode")
        
        while self.is_alive:
            try:
                # Force LIVE mode
                os.environ['EDEN_SHADOW'] = '0'
                
                # Start bot
                config_path = str(PROJECT_ROOT / 'config.yaml')
                bot = TradingBot(symbols=None, config_path=config_path)
                
                servicemanager.LogInfoMsg(f"Bot initialized. Shadow mode: {bot.shadow_mode}")
                
                # Run bot with 5-minute check interval
                bot.start(check_interval=300)
                
            except Exception as e:
                servicemanager.LogErrorMsg(f"Bot error: {e}")
                time.sleep(30)  # Wait before restart
            
            finally:
                if self.is_alive:
                    servicemanager.LogInfoMsg("Bot stopped, restarting in 10 seconds...")
                    time.sleep(10)
    
    def ensure_live_strategy(self):
        """Ensure at least one active LIVE strategy exists in strategies.json"""
        strategies_file = PROJECT_ROOT / 'data' / 'strategies.json'
        
        try:
            if strategies_file.exists():
                with open(strategies_file, 'r') as f:
                    strategies = json.load(f)
                
                # Activate first validated strategy in LIVE mode if none are active
                has_active_live = any(
                    s.get('is_active') and s.get('mode') == 'LIVE' 
                    for s in strategies.values()
                )
                
                if not has_active_live:
                    for sid, strat in strategies.items():
                        if strat.get('is_validated', False):
                            strat['is_active'] = True
                            strat['mode'] = 'LIVE'
                            servicemanager.LogInfoMsg(f"Activated strategy {sid} in LIVE mode")
                            break
                    
                    with open(strategies_file, 'w') as f:
                        json.dump(strategies, f, indent=2)
        
        except Exception as e:
            servicemanager.LogErrorMsg(f"Strategy file error: {e}")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(EdenBotService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(EdenBotService)
