"""
Eden Trading API - Windows Service Wrapper

Runs the FastAPI backend as a Windows service that persists across RDP disconnections.
"""

import sys
import os
import time
import servicemanager
import socket
import win32event
import win32service
import win32serviceutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT / 'backend'))

class EdenAPIService(win32serviceutil.ServiceFramework):
    _svc_name_ = "EdenTradingAPI"
    _svc_display_name_ = "Eden Trading Bot API"
    _svc_description_ = "FastAPI backend for Eden iOS trading application"

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
        """Run the FastAPI server."""
        try:
            # Import uvicorn and the app here so the service context has all dependencies
            import uvicorn
            from dotenv import load_dotenv
            load_dotenv()
            
            # Import the FastAPI app
            from main import app
            from app.settings import settings
            
            servicemanager.LogInfoMsg(f"Starting Eden API on {settings.host}:{settings.port}")
            
            # Configure uvicorn
            config = uvicorn.Config(
                app,
                host=settings.host,
                port=settings.port,
                ssl_certfile=settings.ssl_certfile if settings.ssl_certfile else None,
                ssl_keyfile=settings.ssl_keyfile if settings.ssl_keyfile else None,
                proxy_headers=True,
                forwarded_allow_ips="*",
                log_config=None,  # Disable uvicorn's logging config for Windows service
                access_log=False
            )
            server = uvicorn.Server(config)
            
            # Run server
            server.run()
            
        except Exception as e:
            servicemanager.LogErrorMsg(f"Eden API Service error: {e}")
            raise

if __name__ == '__main__':
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(EdenAPIService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(EdenAPIService)
