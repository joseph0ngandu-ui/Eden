#!/usr/bin/env python3
"""
Eden Launcher
Professional deployment and development tool
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

# Terminal colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    END = '\033[0m'
    BOLD = '\033[1m'

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_banner():
    """Print the Eden Bot banner"""
    print(f"{Colors.WHITE}Eden{Colors.END}")
    print(f"{Colors.GRAY}The Origin of Order{Colors.END}")
    print()

def check_python():
    """Check if Python is available and version is correct"""
    try:
        result = subprocess.run([sys.executable, '--version'], 
                              capture_output=True, text=True)
        version = result.stdout.strip()
        print(f"{Colors.GRAY}Python found: {version}{Colors.END}")
        
        # Check if Python version is 3.8+
        version_parts = version.split()[1].split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        if major < 3 or (major == 3 and minor < 8):
            print(f"{Colors.RED}Python 3.8+ required, found {version}{Colors.END}")
            return False
        return True
    except Exception as e:
        print(f"{Colors.RED}Python not found or error checking version: {e}{Colors.END}")
        return False

def install_dependencies():
    """Install required Python packages"""
    print(f"{Colors.WHITE}Installing dependencies...{Colors.END}")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print(f"{Colors.GRAY}requirements.txt not found, installing core packages...{Colors.END}")
        packages = ["PySide6", "pyqtgraph", "MetaTrader5", "numpy", "pandas", "requests"]
        for package in packages:
            print(f"Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package])
    else:
        print(f"{Colors.GRAY}Installing packages from requirements.txt...{Colors.END}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    print(f"{Colors.WHITE}Dependencies installed successfully{Colors.END}")

def activate_venv():
    """Check for and activate virtual environment if it exists"""
    venv_paths = [".venv", "venv", "env"]
    
    for venv_path in venv_paths:
        if Path(venv_path).exists():
            print(f"{Colors.GRAY}Virtual environment found at {venv_path}{Colors.END}")
            
            if platform.system() == "Windows":
                activate_script = Path(venv_path) / "Scripts" / "activate.bat"
                if activate_script.exists():
                    print(f"{Colors.GRAY}Note: Using virtual environment at {venv_path}{Colors.END}")
                    # On Windows, we can't easily activate in the same process
                    # The Python executable should be used from the venv
                    python_exe = Path(venv_path) / "Scripts" / "python.exe"
                    if python_exe.exists():
                        return str(python_exe)
            else:
                activate_script = Path(venv_path) / "bin" / "activate"
                if activate_script.exists():
                    print(f"{Colors.GRAY}Note: Using virtual environment at {venv_path}{Colors.END}")
                    python_exe = Path(venv_path) / "bin" / "python"
                    if python_exe.exists():
                        return str(python_exe)
    
    return sys.executable

def run_eden_development():
    """Run Eden Bot in development mode"""
    print(f"{Colors.WHITE}Starting Eden in development mode...{Colors.END}")
    
    if not check_python():
        return False
    
    python_exe = activate_venv()
    
    try:
        install_dependencies()
        
        # Check if main Eden file exists
        if Path("Eden.py").exists():
            print(f"{Colors.WHITE}Launching Eden Dashboard...{Colors.END}")
            subprocess.run([python_exe, "Eden.py"])
        else:
            print(f"{Colors.RED}Main UI file (run_ui.py) not found{Colors.END}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n{Colors.GRAY}Application interrupted by user{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}Error running Eden: {e}{Colors.END}")
        return False
    
    return True

def build_deployment():
    """Build Eden Bot deployment package"""
    print(f"{Colors.WHITE}Building Eden deployment...{Colors.END}")
    
    if not check_python():
        return False
    
    try:
        install_dependencies()
        
        if Path("build_deployment.py").exists():
            print(f"{Colors.GRAY}Running deployment builder...{Colors.END}")
            subprocess.run([sys.executable, "build_deployment.py"])
        else:
            print(f"{Colors.RED}Deployment builder (build_deployment.py) not found{Colors.END}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}Error building deployment: {e}{Colors.END}")
        return False
    
    return True

def install_eden():
    """Run the Eden Bot installer"""
    print(f"{Colors.WHITE}Running Eden installer...{Colors.END}")
    
    if Path("installer.py").exists():
        try:
            subprocess.run([sys.executable, "installer.py"])
        except Exception as e:
            print(f"{Colors.RED}Error running installer: {e}{Colors.END}")
            return False
    else:
        print(f"{Colors.GRAY}Installer not found. Running development version...{Colors.END}")
        return run_eden_development()
    
    return True

def show_menu():
    """Show the interactive menu"""
    while True:
        clear_screen()
        print_banner()
        
        print(f"{Colors.WHITE}What would you like to do?{Colors.END}")
        print()
        print(f"{Colors.GRAY}1. Run Eden (Development){Colors.END}")
        print(f"{Colors.GRAY}2. Install Eden (User-friendly installer){Colors.END}")
        print(f"{Colors.GRAY}3. Build Eden (Create deployment package){Colors.END}")
        print(f"{Colors.GRAY}4. Install Dependencies Only{Colors.END}")
        print(f"{Colors.GRAY}5. Exit{Colors.END}")
        print()
        
        try:
            choice = input(f"{Colors.WHITE}Enter your choice (1-5): {Colors.END}").strip()
            
            if choice == "1":
                success = run_eden_development()
            elif choice == "2":
                success = install_eden()
            elif choice == "3":
                success = build_deployment()
            elif choice == "4":
                if check_python():
                    install_dependencies()
                    success = True
                else:
                    success = False
            elif choice == "5":
                print(f"{Colors.GRAY}Goodbye{Colors.END}")
                sys.exit(0)
            else:
                print(f"{Colors.RED}Invalid choice. Please try again.{Colors.END}")
                input(f"{Colors.GRAY}Press Enter to continue...{Colors.END}")
                continue
            
            print()
            if success:
                print(f"{Colors.WHITE}Operation completed successfully{Colors.END}")
            else:
                print(f"{Colors.RED}Operation failed. Check the errors above.{Colors.END}")
            
            print(f"{Colors.GRAY}Eden - The Origin of Order{Colors.END}")
            print()
            input(f"{Colors.GRAY}Press Enter to return to menu...{Colors.END}")
            
        except KeyboardInterrupt:
            print(f"\n{Colors.GRAY}Goodbye{Colors.END}")
            sys.exit(0)
        except Exception as e:
            print(f"{Colors.RED}An error occurred: {e}{Colors.END}")
            input(f"{Colors.GRAY}Press Enter to continue...{Colors.END}")

def main():
    """Main entry point"""
    try:
        # Check command line arguments
        if len(sys.argv) > 1:
            arg = sys.argv[1].lower()
            if arg in ["run", "dev", "development"]:
                clear_screen()
                print_banner()
                run_eden_development()
            elif arg in ["build", "deploy", "deployment"]:
                clear_screen()
                print_banner()
                build_deployment()
            elif arg in ["install", "installer"]:
                clear_screen()
                print_banner()
                install_eden()
            elif arg in ["deps", "dependencies"]:
                clear_screen()
                print_banner()
                if check_python():
                    install_dependencies()
            else:
                print(f"{Colors.RED}Unknown argument: {arg}{Colors.END}")
                print(f"{Colors.GRAY}Valid arguments: run, build, install, deps{Colors.END}")
                sys.exit(1)
        else:
            # Interactive menu
            show_menu()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.GRAY}Goodbye{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"{Colors.RED}Fatal error: {e}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()