#!/usr/bin/env python3
"""
Eden - Professional Trading System
Unified application entry point with Apple-style design, automatic dependency bootstrap,
full-screen UI, and top-down multi-timeframe analysis.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def _bootstrap_dependencies() -> bool:
    """Check and install missing dependencies with a minimal fallback UI/progress.
    Returns True if environment is ready (or was fixed), False otherwise.
    """
    try:
        # Try importing critical GUI and core deps first
        import PyQt5  # type: ignore
        import numpy  # type: ignore
        import pandas  # type: ignore
        import requests  # type: ignore
        return True
    except Exception:
        # Attempt to self-heal using bootstrapper
        try:
            from eden import bootstrap  # type: ignore
        except Exception:
            # If bootstrap not importable, try running the installer script directly
            bootstrap_script = project_root / "dependency_installer.py"
            if bootstrap_script.exists():
                print("Missing dependencies detected. Launching installer...")
                import subprocess
                result = subprocess.run([sys.executable, str(bootstrap_script)], shell=False)
                return result.returncode == 0
            else:
                print("Missing dependencies and no installer available. Please run: pip install -r requirements.txt")
                return False
        # Use bootstrap module
        try:
            ready = bootstrap.check_and_install(show_ui=True)
            return bool(ready)
        except Exception as e:
            print(f"Bootstrap failed: {e}")
            return False


def main():
    """Main entry point for Eden application."""
    try:
        # Configure logging first
        try:
            from eden.logging_conf import configure_logging
            configure_logging("INFO")
        except Exception:
            pass

        # If running GUI, bootstrap dependencies first
        if len(sys.argv) <= 1:
            if not _bootstrap_dependencies():
                # Last-chance fallback to CLI
                print("Continuing in CLI mode due to unresolved GUI dependencies...")
                from eden.cli import main as cli_main
                cli_main()
                return

        # CLI vs GUI mode
        if len(sys.argv) > 1:
            # CLI mode
            from eden.cli import main as cli_main
            cli_main()
        else:
            # GUI mode - show splash screen then main UI
            try:
                # Prefer the modern PySide6 UI; it manages its own splash/app lifecycle
                from run_ui import run_main_ui
                run_main_ui(None)

            except ImportError as e:
                # Fallback to basic CLI if GUI dependencies missing
                print(f"GUI dependencies not available: {e}")
                print("Running in CLI mode...")
                from eden.cli import main as cli_main
                cli_main()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
