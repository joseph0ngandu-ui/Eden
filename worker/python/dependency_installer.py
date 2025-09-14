"""
Eden Dependency Installer
Automatically installs all required dependencies with progress tracking
"""

import subprocess
import sys
import os
import time
from pathlib import Path
import importlib
import pkg_resources
from typing import List, Tuple, Optional

class DependencyInstaller:
    """Professional dependency installer with progress tracking"""
    
    def __init__(self):
        self.required_packages = [
            # Core GUI and visualization
            ('PyQt5', 'PyQt5>=5.15.0'),
            ('pyqtgraph', 'pyqtgraph>=0.13.0'),
            
            # Data analysis and machine learning  
            ('numpy', 'numpy>=1.21.0'),
            ('pandas', 'pandas>=1.5.0'),
            ('scikit-learn', 'scikit-learn>=1.0.0'),
            ('scipy', 'scipy>=1.7.0'),
            
            # Plotting and visualization
            ('matplotlib', 'matplotlib>=3.5.0'),
            ('seaborn', 'seaborn>=0.11.0'),
            ('plotly', 'plotly>=5.0.0'),
            
            # Trading and financial data
            ('yfinance', 'yfinance>=0.2.0'),
            ('requests', 'requests>=2.28.0'),
            ('websockets', 'websockets>=10.0'),
            
            # Technical analysis
            ('TA-Lib', 'TA-Lib>=0.4.0'),
            ('talib-binary', 'talib-binary>=0.4.19'),  # Fallback for TA-Lib
            
            # Utilities
            ('pydantic', 'pydantic>=2.0.0'),
            ('python-dotenv', 'python-dotenv>=0.19.0'),
            ('PyYAML', 'PyYAML>=6.0'),
            ('psutil', 'psutil>=5.9.0'),
            ('packaging', 'packaging>=21.0'),
            
            # Crypto and security
            ('cryptography', 'cryptography>=38.0.0'),
            
            # Build tools
            ('pyinstaller', 'pyinstaller>=5.0.0'),
        ]
        
        self.optional_packages = [
            # MetaTrader 5 integration
            ('MetaTrader5', 'MetaTrader5>=5.0.37'),
            
            # Alternative plotting
            ('bokeh', 'bokeh>=2.4.0'),
            
            # Database support
            ('sqlalchemy', 'SQLAlchemy>=1.4.0'),
            ('sqlite3', 'sqlite3'),  # Built-in, but check
            
            # Advanced ML
            ('xgboost', 'xgboost>=1.6.0'),
            ('lightgbm', 'lightgbm>=3.3.0'),
            
            # Time series analysis
            ('statsmodels', 'statsmodels>=0.13.0'),
            
            # Jupyter support
            ('jupyter', 'jupyter>=1.0.0'),
            ('ipykernel', 'ipykernel>=6.0.0'),
        ]
        
        self.installed_packages = set()
        self.failed_packages = []
        
    def check_package_installed(self, package_name: str) -> bool:
        """Check if a package is already installed"""
        try:
            if package_name == 'sqlite3':
                import sqlite3
                return True
            elif package_name == 'TA-Lib':
                import talib
                return True
            elif package_name == 'talib-binary':
                # Skip if TA-Lib is already installed
                try:
                    import talib
                    return True
                except ImportError:
                    pass
                # Check talib-binary
                try:
                    import talib
                    return True
                except ImportError:
                    return False
            else:
                importlib.import_module(package_name.lower().replace('-', '_'))
                return True
        except ImportError:
            return False
    
    def install_package(self, package_spec: str, package_name: str = None) -> bool:
        """Install a single package"""
        try:
            print(f"📦 Installing {package_spec}...")
            
            # Special handling for problematic packages
            if 'TA-Lib' in package_spec:
                # Try TA-Lib first, fallback to talib-binary
                try:
                    result = subprocess.run([
                        sys.executable, '-m', 'pip', 'install', 'TA-Lib',
                        '--prefer-binary', '--no-cache-dir'
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode != 0:
                        print(f"   ⚠️ TA-Lib failed, trying talib-binary...")
                        result = subprocess.run([
                            sys.executable, '-m', 'pip', 'install', 'talib-binary',
                            '--no-cache-dir'
                        ], capture_output=True, text=True, timeout=300)
                except subprocess.TimeoutExpired:
                    print(f"   ⚠️ Installation timeout for {package_spec}")
                    return False
            else:
                # Standard installation
                result = subprocess.run([
                    sys.executable, '-m', 'pip', 'install', package_spec,
                    '--upgrade', '--no-cache-dir'
                ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"   ✅ Successfully installed {package_name or package_spec}")
                return True
            else:
                print(f"   ❌ Failed to install {package_name or package_spec}")
                print(f"   Error: {result.stderr.strip()[:200]}...")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"   ⚠️ Installation timeout for {package_spec}")
            return False
        except Exception as e:
            print(f"   ❌ Exception installing {package_spec}: {e}")
            return False
    
    def install_all_dependencies(self, include_optional: bool = False) -> Tuple[int, int, int]:
        """Install all required dependencies with progress tracking"""
        packages_to_install = self.required_packages.copy()
        if include_optional:
            packages_to_install.extend(self.optional_packages)
        
        total_packages = len(packages_to_install)
        installed = 0
        skipped = 0
        failed = 0
        
        print("🚀 Eden Dependency Installer")
        print("=" * 50)
        print(f"📊 Total packages to check: {total_packages}")
        print()
        
        for i, (package_name, package_spec) in enumerate(packages_to_install, 1):
            print(f"[{i}/{total_packages}] Checking {package_name}...")
            
            # Skip talib-binary if TA-Lib is already available
            if package_name == 'talib-binary':
                try:
                    import talib
                    print(f"   ⏭️ Skipping talib-binary (TA-Lib already available)")
                    skipped += 1
                    continue
                except ImportError:
                    pass
            
            if self.check_package_installed(package_name):
                print(f"   ✅ Already installed: {package_name}")
                self.installed_packages.add(package_name)
                skipped += 1
            else:
                print(f"   📦 Installing {package_name}...")
                if self.install_package(package_spec, package_name):
                    self.installed_packages.add(package_name)
                    installed += 1
                else:
                    self.failed_packages.append((package_name, package_spec))
                    failed += 1
            
            # Progress indicator
            progress = (i / total_packages) * 100
            print(f"   📈 Progress: {progress:.1f}%")
            print()
            time.sleep(0.1)  # Brief pause for readability
        
        return installed, skipped, failed
    
    def create_requirements_file(self):
        """Create requirements.txt file"""
        requirements_path = Path("requirements.txt")
        
        with open(requirements_path, 'w') as f:
            f.write("# Eden Trading Bot Dependencies\n")
            f.write(f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("# Core GUI and visualization\n")
            f.write("PyQt5>=5.15.0\n")
            f.write("pyqtgraph>=0.13.0\n\n")
            
            f.write("# Data analysis and machine learning\n")
            f.write("numpy>=1.21.0\n")
            f.write("pandas>=1.5.0\n")
            f.write("scikit-learn>=1.0.0\n")
            f.write("scipy>=1.7.0\n\n")
            
            f.write("# Plotting and visualization\n")
            f.write("matplotlib>=3.5.0\n")
            f.write("seaborn>=0.11.0\n")
            f.write("plotly>=5.0.0\n\n")
            
            f.write("# Trading and financial data\n")
            f.write("yfinance>=0.2.0\n")
            f.write("requests>=2.28.0\n")
            f.write("websockets>=10.0\n\n")
            
            f.write("# Technical analysis\n")
            f.write("# TA-Lib>=0.4.0  # May require manual installation\n")
            f.write("talib-binary>=0.4.19  # Fallback for TA-Lib\n\n")
            
            f.write("# Utilities\n")
            f.write("pydantic>=2.0.0\n")
            f.write("python-dotenv>=0.19.0\n")
            f.write("PyYAML>=6.0\n")
            f.write("psutil>=5.9.0\n")
            f.write("packaging>=21.0\n\n")
            
            f.write("# Security\n")
            f.write("cryptography>=38.0.0\n\n")
            
            f.write("# Build tools\n")
            f.write("pyinstaller>=5.0.0\n\n")
            
            f.write("# Optional packages (install separately if needed)\n")
            f.write("# MetaTrader5>=5.0.37\n")
            f.write("# bokeh>=2.4.0\n")
            f.write("# SQLAlchemy>=1.4.0\n")
            f.write("# xgboost>=1.6.0\n")
            f.write("# lightgbm>=3.3.0\n")
            f.write("# statsmodels>=0.13.0\n")
        
        print(f"✅ Created requirements.txt file")
        return requirements_path
    
    def print_summary(self, installed: int, skipped: int, failed: int):
        """Print installation summary"""
        total = installed + skipped + failed
        
        print("\n🎯 INSTALLATION SUMMARY")
        print("=" * 50)
        print(f"📊 Total packages processed: {total}")
        print(f"✅ Already installed: {skipped}")
        print(f"📦 Newly installed: {installed}")
        print(f"❌ Failed installations: {failed}")
        
        if self.failed_packages:
            print(f"\n⚠️ Failed packages:")
            for package_name, package_spec in self.failed_packages:
                print(f"   - {package_name} ({package_spec})")
            print("\nℹ️ You can try installing failed packages manually:")
            for package_name, package_spec in self.failed_packages:
                print(f"   pip install {package_spec}")
        
        success_rate = ((skipped + installed) / total) * 100 if total > 0 else 0
        print(f"\n📈 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 Excellent! Eden should work perfectly.")
        elif success_rate >= 80:
            print("✅ Good! Eden should work with minor limitations.")
        elif success_rate >= 70:
            print("⚠️ Fair. Some features may not work properly.")
        else:
            print("❌ Poor. Many features may not work. Please resolve failed installations.")
    
    def verify_critical_packages(self) -> bool:
        """Verify that critical packages are available"""
        critical_packages = ['numpy', 'pandas', 'PyQt5', 'requests', 'yfinance']
        
        print("\n🔍 Verifying critical packages...")
        all_available = True
        
        for package in critical_packages:
            try:
                if package == 'PyQt5':
                    from PyQt5.QtWidgets import QApplication
                    print(f"   ✅ {package} - OK")
                elif package == 'numpy':
                    import numpy as np
                    print(f"   ✅ {package} - OK")
                elif package == 'pandas':
                    import pandas as pd
                    print(f"   ✅ {package} - OK")
                elif package == 'requests':
                    import requests
                    print(f"   ✅ {package} - OK")
                elif package == 'yfinance':
                    import yfinance as yf
                    print(f"   ✅ {package} - OK")
                else:
                    importlib.import_module(package)
                    print(f"   ✅ {package} - OK")
            except ImportError as e:
                print(f"   ❌ {package} - FAILED: {e}")
                all_available = False
        
        if all_available:
            print("🎉 All critical packages verified successfully!")
        else:
            print("⚠️ Some critical packages are missing. Eden may not work properly.")
        
        return all_available
    
    def run_full_installation(self, include_optional: bool = False) -> bool:
        """Run complete installation process"""
        print("🚀 Starting Eden dependency installation...\n")
        
        # Update pip first
        print("📦 Updating pip...")
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ], capture_output=True, text=True, timeout=60)
            print("   ✅ pip updated successfully\n")
        except Exception as e:
            print(f"   ⚠️ pip update failed: {e}\n")
        
        # Install dependencies
        installed, skipped, failed = self.install_all_dependencies(include_optional)
        
        # Print summary
        self.print_summary(installed, skipped, failed)
        
        # Create requirements file
        self.create_requirements_file()
        
        # Verify critical packages
        verification_success = self.verify_critical_packages()
        
        return verification_success and failed < 3  # Allow up to 2 failed non-critical packages


def main():
    """Main installation function"""
    installer = DependencyInstaller()
    
    print("🌟 Welcome to Eden Dependency Installer!")
    print("This will install all required packages for Eden to work properly.\n")
    
    # Ask about optional packages
    try:
        choice = input("Install optional packages (MetaTrader5, ML libraries, etc.)? (y/N): ").strip().lower()
        include_optional = choice in ('y', 'yes', '1', 'true')
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
        return 1
    
    print()
    
    # Run installation
    success = installer.run_full_installation(include_optional)
    
    if success:
        print("\n🎉 Installation completed successfully!")
        print("🚀 You can now run Eden with: python Eden.py")
        return 0
    else:
        print("\n⚠️ Installation completed with issues.")
        print("💡 You may need to manually install some packages.")
        print("📋 Check the summary above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())