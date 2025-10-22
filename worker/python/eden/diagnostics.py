"""
Eden Advanced Logging and Diagnostics System
Professional logging, crash reporting, and system diagnostics
"""

import logging
import logging.handlers
import os
import sys
import traceback
import datetime
import json
import platform
import psutil
import socket
from pathlib import Path
from typing import Dict, Any, List


class EdenLogger:
    """Professional logging system with multiple handlers and formatters"""

    def __init__(self, app_name: str = "Eden", log_level: str = "INFO"):
        self.app_name = app_name
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)

        # Create logs directory
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Initialize loggers
        self.main_logger = None
        self.error_logger = None
        self.performance_logger = None
        self.user_action_logger = None

        self._setup_loggers()

    def _setup_loggers(self):
        """Setup multiple specialized loggers"""

        # Main application logger
        self.main_logger = logging.getLogger(f"{self.app_name}.main")
        self.main_logger.setLevel(self.log_level)
        self._configure_main_logger()

        # Error/crash logger
        self.error_logger = logging.getLogger(f"{self.app_name}.error")
        self.error_logger.setLevel(logging.ERROR)
        self._configure_error_logger()

        # Performance monitoring logger
        self.performance_logger = logging.getLogger(f"{self.app_name}.performance")
        self.performance_logger.setLevel(logging.INFO)
        self._configure_performance_logger()

        # User action logger
        self.user_action_logger = logging.getLogger(f"{self.app_name}.user_actions")
        self.user_action_logger.setLevel(logging.INFO)
        self._configure_user_action_logger()

    def _configure_main_logger(self):
        """Configure main application logger"""
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name.lower()}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )

        # Console handler for development
        console_handler = logging.StreamHandler()

        # Detailed formatter
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Simple console formatter
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(console_formatter)

        # Only add console handler in development
        self.main_logger.addHandler(file_handler)
        if __debug__:  # Only in development/debug mode
            self.main_logger.addHandler(console_handler)

    def _configure_error_logger(self):
        """Configure error/crash logger"""
        # Error file handler - no rotation, keep all errors
        error_handler = logging.FileHandler(
            self.log_dir / f"{self.app_name.lower()}_errors.log"
        )

        # Critical error formatter with full context
        error_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s\n"
            "Function: %(funcName)s:%(lineno)d\n"
            "Message: %(message)s\n"
            "%(exc_info)s\n"
            "--- End Error ---\n",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)

    def _configure_performance_logger(self):
        """Configure performance monitoring logger"""
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name.lower()}_performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )

        # JSON formatter for structured data
        perf_formatter = JsonFormatter()
        perf_handler.setFormatter(perf_formatter)

        self.performance_logger.addHandler(perf_handler)

    def _configure_user_action_logger(self):
        """Configure user action logger"""
        action_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.app_name.lower()}_user_actions.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )

        # Structured formatter for user actions
        action_formatter = logging.Formatter(
            "%(asctime)s - USER_ACTION - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        action_handler.setFormatter(action_formatter)
        self.user_action_logger.addHandler(action_handler)

    def log_info(self, message: str, **kwargs):
        """Log info message with optional context"""
        if kwargs:
            message = f"{message} - Context: {json.dumps(kwargs)}"
        self.main_logger.info(message)

    def log_warning(self, message: str, **kwargs):
        """Log warning message"""
        if kwargs:
            message = f"{message} - Context: {json.dumps(kwargs)}"
        self.main_logger.warning(message)

    def log_error(self, message: str, exception: Exception = None, **kwargs):
        """Log error with full context"""
        if kwargs:
            message = f"{message} - Context: {json.dumps(kwargs)}"

        if exception:
            self.error_logger.error(message, exc_info=exception)
        else:
            self.error_logger.error(message)

        # Also log to main logger
        self.main_logger.error(message)

    def log_user_action(self, action: str, details: Dict = None):
        """Log user action for analytics"""
        action_data = {
            "action": action,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": details or {},
        }
        self.user_action_logger.info(json.dumps(action_data))

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "", **context
    ):
        """Log performance metric"""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.datetime.now().isoformat(),
            "context": context,
        }
        self.performance_logger.info(json.dumps(metric_data))


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class SystemDiagnostics:
    """System diagnostics and health monitoring"""

    def __init__(self):
        self.logger = EdenLogger().main_logger

    def collect_system_info(self) -> Dict[str, Any]:
        """Collect comprehensive system information"""
        try:
            system_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "architecture": platform.architecture(),
                },
                "python": {
                    "version": sys.version,
                    "executable": sys.executable,
                    "path": sys.path[:5],  # First 5 paths only
                },
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent_used": psutil.virtual_memory().percent,
                },
                "disk": {
                    "total": (
                        psutil.disk_usage("/").total
                        if os.name != "nt"
                        else psutil.disk_usage("C:\\").total
                    ),
                    "free": (
                        psutil.disk_usage("/").free
                        if os.name != "nt"
                        else psutil.disk_usage("C:\\").free
                    ),
                    "percent_used": (
                        psutil.disk_usage("/").percent
                        if os.name != "nt"
                        else psutil.disk_usage("C:\\").percent
                    ),
                },
                "cpu": {
                    "count": psutil.cpu_count(),
                    "percent_used": psutil.cpu_percent(interval=1),
                },
                "network": self._get_network_info(),
            }

            return system_info

        except Exception as e:
            self.logger.error(f"Failed to collect system info: {e}")
            return {"error": str(e)}

    def _get_network_info(self) -> Dict[str, Any]:
        """Get network connectivity information"""
        try:
            # Test internet connectivity
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            internet_connected = True
        except OSError:
            internet_connected = False

        return {
            "internet_connected": internet_connected,
            "hostname": socket.gethostname(),
        }

    def collect_application_info(self) -> Dict[str, Any]:
        """Collect application-specific diagnostic information"""
        try:
            app_info = {
                "timestamp": datetime.datetime.now().isoformat(),
                "version": self._get_app_version(),
                "installation_path": os.path.abspath("."),
                "config_files": self._check_config_files(),
                "dependencies": self._check_critical_dependencies(),
                "log_files": self._get_log_file_info(),
            }

            return app_info

        except Exception as e:
            self.logger.error(f"Failed to collect application info: {e}")
            return {"error": str(e)}

    def _get_app_version(self) -> str:
        """Get application version"""
        version_file = Path("eden/version.txt")
        if version_file.exists():
            return version_file.read_text().strip()
        return "Unknown"

    def _check_config_files(self) -> List[Dict[str, Any]]:
        """Check status of configuration files"""
        config_files = [
            "Eden.py",
            "run_ui.py",
            "splash_screen.py",
            "eden/version.txt",
        ]

        file_status = []
        for file_path in config_files:
            path = Path(file_path)
            status = {
                "file": file_path,
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "modified": path.stat().st_mtime if path.exists() else 0,
            }
            file_status.append(status)

        return file_status

    def _check_critical_dependencies(self) -> Dict[str, bool]:
        """Check if critical dependencies are available"""
        critical_modules = [
            "PyQt5",
            "requests",
            "pandas",
            "numpy",
        ]

        dependency_status = {}
        for module in critical_modules:
            try:
                __import__(module)
                dependency_status[module] = True
            except ImportError:
                dependency_status[module] = False

        return dependency_status

    def _get_log_file_info(self) -> List[Dict[str, Any]]:
        """Get information about log files"""
        log_dir = Path("logs")
        if not log_dir.exists():
            return []

        log_files = []
        for log_file in log_dir.glob("*.log"):
            file_info = {
                "name": log_file.name,
                "size": log_file.stat().st_size,
                "modified": log_file.stat().st_mtime,
                "lines": self._count_lines(log_file),
            }
            log_files.append(file_info)

        return log_files

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file efficiently"""
        try:
            with open(file_path, "rb") as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    def generate_diagnostic_report(self) -> str:
        """Generate comprehensive diagnostic report"""
        report_data = {
            "report_generated": datetime.datetime.now().isoformat(),
            "system_info": self.collect_system_info(),
            "application_info": self.collect_application_info(),
        }

        # Save detailed JSON report
        report_file = f"diagnostic_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Generate human-readable summary
        summary = self._generate_summary_report(report_data)
        summary_file = f"diagnostic_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, "w") as f:
            f.write(summary)

        self.logger.info(f"Diagnostic report generated: {report_file}, {summary_file}")
        return summary_file

    def _generate_summary_report(self, report_data: Dict) -> str:
        """Generate human-readable diagnostic summary"""
        system = report_data["system_info"]
        app = report_data["application_info"]

        summary = f"""
Eden Diagnostic Report
Generated: {report_data['report_generated']}

SYSTEM INFORMATION
==================
Platform: {system.get('platform', {}).get('system', 'Unknown')} {system.get('platform', {}).get('release', '')}
Architecture: {system.get('platform', {}).get('architecture', ['Unknown'])[0]}
Python: {system.get('python', {}).get('version', 'Unknown').split()[0]}

RESOURCE USAGE
==============
Memory: {system.get('memory', {}).get('percent_used', 0):.1f}% used
Disk: {system.get('disk', {}).get('percent_used', 0):.1f}% used
CPU: {system.get('cpu', {}).get('percent_used', 0):.1f}% used
Internet: {'Connected' if system.get('network', {}).get('internet_connected', False) else 'Disconnected'}

APPLICATION STATUS
==================
Version: {app.get('version', 'Unknown')}
Installation Path: {app.get('installation_path', 'Unknown')}

DEPENDENCIES
============
"""

        dependencies = app.get("dependencies", {})
        for dep, status in dependencies.items():
            summary += f"{dep}: {'✅ Available' if status else '❌ Missing'}\n"

        summary += "\nCONFIGURATION FILES\n===================\n"

        config_files = app.get("config_files", [])
        for file_info in config_files:
            status = "✅ OK" if file_info["exists"] else "❌ Missing"
            size_kb = file_info["size"] / 1024 if file_info["size"] > 0 else 0
            summary += f"{file_info['file']}: {status} ({size_kb:.1f}KB)\n"

        return summary


class CrashReporter:
    """Automated crash reporting system"""

    def __init__(self, app_name: str = "Eden"):
        self.app_name = app_name
        self.logger = EdenLogger().error_logger
        self.diagnostics = SystemDiagnostics()

    def install_crash_handler(self):
        """Install global exception handler"""

        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Allow keyboard interrupts to work normally
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            # Log the crash
            self.logger.error(
                "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
            )

            # Generate crash report
            self.generate_crash_report(exc_type, exc_value, exc_traceback)

        # Install the handler
        sys.excepthook = handle_exception

    def generate_crash_report(self, exc_type, exc_value, exc_traceback):
        """Generate detailed crash report"""
        crash_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "exception_type": exc_type.__name__,
            "exception_message": str(exc_value),
            "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback),
            "system_info": self.diagnostics.collect_system_info(),
            "application_info": self.diagnostics.collect_application_info(),
        }

        # Save crash report
        crash_file = (
            f"crash_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(crash_file, "w") as f:
            json.dump(crash_data, f, indent=2, default=str)

        print(f"💥 Eden has crashed! Crash report saved to: {crash_file}")
        print("Please send this file to support for assistance.")

        return crash_file


# Global logger instance
_global_logger = None


def get_logger() -> EdenLogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = EdenLogger()
    return _global_logger


def setup_logging(log_level: str = "INFO"):
    """Setup global logging system"""
    global _global_logger
    _global_logger = EdenLogger(log_level=log_level)

    # Install crash handler
    crash_reporter = CrashReporter()
    crash_reporter.install_crash_handler()

    return _global_logger


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""

    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = datetime.datetime.now()

        try:
            result = func(*args, **kwargs)

            # Log successful execution
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.log_performance_metric(
                f"{func.__module__}.{func.__name__}",
                duration,
                "seconds",
                args_count=len(args),
                kwargs_count=len(kwargs),
            )

            return result

        except Exception as e:
            # Log error with performance context
            duration = (datetime.datetime.now() - start_time).total_seconds()
            logger.log_error(
                f"Function {func.__name__} failed after {duration:.2f}s",
                exception=e,
                function=func.__name__,
                module=func.__module__,
            )
            raise

    return wrapper


if __name__ == "__main__":
    # Test the logging system
    logger = setup_logging("INFO")

    logger.log_info("Eden logging system initialized")
    logger.log_user_action("system_test", {"test": True})
    logger.log_performance_metric("startup_time", 2.5, "seconds")

    # Generate diagnostic report
    diagnostics = SystemDiagnostics()
    report_file = diagnostics.generate_diagnostic_report()
    print(f"Diagnostic report generated: {report_file}")

    # Test crash reporting (uncomment to test)
    # raise Exception("Test crash for reporting")
