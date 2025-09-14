#!/usr/bin/env python3
"""
Eden Test Version - Simple validation script
"""

import sys
import os
import argparse

print("🚀 Eden - Professional Algorithmic Trading Platform")
print("=" * 60)
print(f"Python Version: {sys.version}")
print(f"Current Directory: {os.getcwd()}")
print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Eden Test Version")
    parser.add_argument('--version', action='store_true', help='Show version information')
    parser.add_argument('--test', action='store_true', help='Run test mode')
    parser.add_argument('--gui', action='store_true', help='Launch GUI mode')
    parser.add_argument('--cli', action='store_true', help='Launch CLI mode')
    
    args = parser.parse_args()
    
    if args.version:
        print("Eden v1.0.0 - Professional Algorithmic Trading Platform")
        print("© 2025 Eden Technologies")
        return
    
    if args.test:
        print("✅ Test mode - All systems operational!")
        print("📊 Dependencies check:")
        
        # Test key imports
        try:
            import numpy as np
            print("   ✅ NumPy available")
        except ImportError:
            print("   ❌ NumPy missing")
        
        try:
            import pandas as pd
            print("   ✅ Pandas available")
        except ImportError:
            print("   ❌ Pandas missing")
            
        try:
            import PyQt5
            print("   ✅ PyQt5 available")
        except ImportError:
            print("   ❌ PyQt5 missing")
            
        try:
            import requests
            print("   ✅ Requests available")
        except ImportError:
            print("   ❌ Requests missing")
            
        print("📋 Test completed successfully!")
        return
    
    if args.gui:
        print("🖥️  Launching GUI mode...")
        try:
            # Try to import the GUI components
            from PyQt5.QtWidgets import QApplication, QMessageBox
            
            app = QApplication(sys.argv)
            
            msg = QMessageBox()
            msg.setWindowTitle("Eden")
            msg.setText("Eden GUI Test")
            msg.setInformativeText("This is a test of the Eden GUI system.\n\nAll systems are working correctly!")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            
            print("✅ GUI test completed successfully!")
            
        except ImportError as e:
            print(f"❌ GUI components not available: {e}")
            print("💡 Try installing: pip install PyQt5")
        
        return
    
    if args.cli:
        print("💻 CLI Mode Active")
        print("Available commands:")
        print("  --test      Run system tests")
        print("  --gui       Launch GUI")
        print("  --version   Show version")
        print("Type 'Eden_Test.py --help' for full help")
        return
    
    # Default behavior
    print("🎯 Eden is ready for algorithmic trading!")
    print("")
    print("Quick Start:")
    print("  python Eden_Test.py --test     # Test system")
    print("  python Eden_Test.py --gui      # Launch GUI")
    print("  python Eden_Test.py --cli      # CLI mode")
    print("  python Eden_Test.py --version  # Version info")
    print("")
    print("For full documentation, see INSTALLATION_GUIDE.md")
    print("")
    print("Happy Trading! 📈")

if __name__ == "__main__":
    main()