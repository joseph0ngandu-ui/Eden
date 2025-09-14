#!/usr/bin/env python3
"""
Eden Setup Builder
Creates a professional Windows installer named "Eden Setup.exe"
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import zipfile
import json

def create_setup_exe():
    """Create the Eden Setup.exe using PyInstaller and Inno Setup or auto-extracting archive."""
    project_root = Path(__file__).parent
    
    print("Building Eden Setup.exe...")
    print("=" * 50)
    
    # Step 1: Create standalone executable using PyInstaller
    print("Step 1: Building standalone application...")
    
    # PyInstaller spec for the installer
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['installer.py'],
    pathex=[r'{project_root}'],
    binaries=[],
    datas=[
        ('theme_manager.py', '.'),
        ('ui_main.py', '.'),
        ('Eden.py', '.'),
        ('eden', 'eden'),
        ('requirements.txt', '.'),
        ('QUICK_START.md', '.'),
        ('eden/config/symbol_map.yaml', 'eden/config'),
    ],
    hiddenimports=[
'PyQt5',
        'winreg',
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'numpy',
        'pytest',
        'jupyterlab',
        'PySide6',
        'shiboken6'
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Eden Setup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    cofile=None,
    icon=None,
    version=r'version_info.txt'
)
"""
    
    # Write PyInstaller spec file
    spec_file = project_root / "eden_setup.spec"
    spec_file.write_text(spec_content, encoding='utf-8')
    
    # Create version info file for Windows executable
    version_info = """
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(2, 0, 0, 0),
    prodvers=(2, 0, 0, 0),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
        StringTable(
          u'040904B0',
          [
            StringStruct(u'CompanyName', u'Eden Trading Systems'),
            StringStruct(u'FileDescription', u'Eden - Professional Trading System Installer'),
            StringStruct(u'FileVersion', u'2.0.0.0'),
            StringStruct(u'InternalName', u'Eden Setup'),
            StringStruct(u'LegalCopyright', u'Copyright (c) 2025 Eden Trading Systems'),
            StringStruct(u'OriginalFilename', u'Eden Setup.exe'),
            StringStruct(u'ProductName', u'Eden - The Origin of Order'),
            StringStruct(u'ProductVersion', u'2.0.0.0')
          ]
        )
      ]
    ),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"""
    
    version_file = project_root / "version_info.txt"
    version_file.write_text(version_info, encoding='utf-8')
    
    try:
        # Run PyInstaller
        cmd = [
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            str(spec_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode != 0:
            print(f"PyInstaller failed: {result.stderr}")
            return False
        
        print("✅ Standalone executable created successfully")
        
        # Step 2: Check if the executable was created
        exe_path = project_root / "dist" / "Eden Setup.exe"
        if not exe_path.exists():
            print(f"❌ Expected executable not found at: {exe_path}")
            return False
        
        print(f"✅ Eden Setup.exe created at: {exe_path}")
        print(f"📦 File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error building setup: {e}")
        return False
    
    finally:
        # Cleanup spec files
        if spec_file.exists():
            spec_file.unlink()
        if version_file.exists():
            version_file.unlink()

def create_portable_package():
    """Create a portable ZIP package as an alternative."""
    project_root = Path(__file__).parent
    
    print("\nCreating portable package as backup...")
    
    # Create portable directory structure
    portable_dir = project_root / "build" / "Eden_Portable"
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    portable_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy essential files
    essential_files = [
        "Eden.py",
        "ui_main.py", 
        "installer.py",
        "launch.py",
        "launch.bat",
        "Launch_Eden.ps1",
        "splash_screen.py",
        "theme_manager.py",
        "requirements.txt",
        "QUICK_START.md",
        "README.md",
        "eden\\config\\symbol_map.yaml",
    ]
    
    for file in essential_files:
        src = project_root / file
        if src.exists():
            dest = portable_dir / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    
    # Copy eden package
    eden_src = project_root / "eden"
    eden_dst = portable_dir / "eden"
    if eden_src.exists():
        if eden_dst.exists():
            shutil.rmtree(eden_dst)
        shutil.copytree(eden_src, eden_dst, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    
    # Create portable launcher batch file
    launcher_content = """@echo off
title Eden - The Origin of Order
echo Eden - The Origin of Order
echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Installing dependencies...
python -m pip install -r requirements.txt

echo Launching Eden...
python Eden.py

pause
"""
    
    launcher_file = portable_dir / "Start_Eden.bat" 
    launcher_file.write_text(launcher_content, encoding='utf-8')
    
    # Create ZIP package
    zip_path = project_root / "dist" / "Eden_Portable.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(portable_dir):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']
            
            for file in files:
                if not file.endswith('.pyc'):
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(portable_dir)
                    zipf.write(file_path, arc_path)
    
    print(f"✅ Portable package created: {zip_path}")
    print(f"📦 File size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Cleanup temporary directory
    shutil.rmtree(portable_dir)
    
    return True

def main():
    """Main build process."""
    print("Eden Setup Builder")
    print("Creating professional Windows installer...")
    print()
    
    # Ensure dist directory exists
    project_root = Path(__file__).parent
    dist_dir = project_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    
    success = False
    
    # Try to create setup executable
    if create_setup_exe():
        success = True
        print("\n🎉 Eden Setup.exe created successfully!")
        print("\nTo install Eden:")
        print("1. Double-click 'Eden Setup.exe'")
        print("2. Follow the installation wizard")
        print("3. Eden will be installed automatically with all dependencies")
    else:
        print("\n⚠️ Failed to create setup executable")
    
    # Always create portable package as backup
    if create_portable_package():
        print("\n📦 Portable package also created as backup")
        print("\nTo use portable version:")
        print("1. Extract 'Eden_Portable.zip'")
        print("2. Double-click 'Start_Eden.bat'")
    
    if success:
        # Show final distribution structure
        print(f"\n📁 Distribution files in: {dist_dir}")
        for file in dist_dir.glob("*"):
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                print(f"   {file.name} ({size_mb:.1f} MB)")
        
        print(f"\n✨ Setup complete! Your professional Eden installer is ready.")
        return True
    else:
        print(f"\n❌ Setup creation failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)