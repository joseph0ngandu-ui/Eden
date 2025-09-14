# 🚀 Eden Installation Guide

Welcome to Eden - Professional Algorithmic Trading Platform

## 📦 Available Installation Options

This folder contains multiple installation options to suit different user needs:

---

## 🎯 **RECOMMENDED: Easy Installation**

### **Eden Setup.exe** (Professional Installer)
- **Size**: ~48.8 MB
- **Best for**: Most users who want a complete installation experience
- **Features**:
  - ✅ Full Windows integration
  - ✅ Desktop shortcut creation
  - ✅ Start Menu entries
  - ✅ Add/Remove Programs support
  - ✅ Automatic uninstaller
  - ✅ File associations for .eden files

**Installation Steps:**
1. **Right-click** `Eden Setup.exe` and select **"Run as administrator"**
2. Follow the installation wizard
3. Choose installation directory (default: `C:\Program Files\Eden`)
4. Select components to install
5. Click **"Install"** and wait for completion
6. Launch Eden from Desktop shortcut or Start Menu

---

## ⚡ **Alternative: Standalone Installation**

### **eden.exe** (Portable Executable)
- **Size**: ~110.1 MB  
- **Best for**: Users who prefer portable applications
- **Features**:
  - ✅ No installation required
  - ✅ Run from any folder
  - ✅ Full functionality included
  - ✅ Easy to move or backup

**Usage Steps:**
1. Copy `eden.exe` to your preferred folder
2. Double-click to run (no installation needed)
3. First run may take a few moments to initialize

---

## 📱 **Portable Package**

### **Eden_Portable.zip**
- **Size**: ~0.1 MB (download package)
- **Best for**: Distribution or network installations
- **Contents**: Compressed portable version

**Usage Steps:**
1. Extract `Eden_Portable.zip` to desired location
2. Run the extracted executable
3. Suitable for USB drives or shared folders

---

## 🔧 **Advanced: Custom Installation**

For developers or advanced users who want to customize the installation:

### **NSIS Installer Source** (Windows)
- **File**: `eden_installer.nsi`
- **Requirements**: NSIS installed
- **Build**: `makensis eden_installer.nsi`

### **WiX MSI Source** (Enterprise)
- **File**: `eden_installer.wxs`
- **Requirements**: WiX Toolset
- **Build**: 
  1. `candle eden_installer.wxs`
  2. `light eden_installer.wixobj -out Eden.msi`

---

## 🛡️ System Requirements

### **Minimum Requirements:**
- Windows 10 (64-bit) or later
- 4 GB RAM
- 500 MB free disk space
- Internet connection (for updates and data)

### **Recommended:**
- Windows 11 (64-bit)
- 8 GB RAM or more
- 1 GB free disk space
- High-speed internet connection
- SSD storage for better performance

---

## 🔒 Security Information

### **Digital Signatures**
- All executables are built from verified source code
- Automated security scanning performed during build
- No malware or unwanted software included

### **Windows SmartScreen**
If Windows shows "Windows protected your PC":
1. Click **"More info"**
2. Click **"Run anyway"**
3. This is normal for new software without expensive certificates

### **Antivirus Software**
Some antivirus programs may flag new executables:
- This is a false positive (common with PyInstaller builds)
- Add Eden to your antivirus exceptions if needed
- All source code is available for inspection

---

## 🚀 Quick Start

### **After Installation:**

1. **Launch Eden** using your preferred method
2. **First-time setup** wizard will appear
3. **Choose mode**:
   - **GUI Mode**: Full graphical interface (recommended)
   - **CLI Mode**: Command-line interface (advanced users)
4. **Configure settings** in the Settings panel
5. **Start trading!**

### **Key Features:**
- 📊 Advanced charting and analysis
- 🤖 Algorithmic trading strategies
- 📈 Real-time market data
- 🔔 Professional update system
- 🛡️ Enterprise-grade security
- 📋 Comprehensive logging and diagnostics

---

## ❓ Troubleshooting

### **Installation Issues**

**Problem**: "Access denied" during installation
**Solution**: Right-click installer and select "Run as administrator"

**Problem**: Installation fails or gets stuck
**Solution**: 
1. Disable antivirus temporarily
2. Close other applications
3. Restart computer and try again

**Problem**: "Missing DLL" errors
**Solution**: Install Microsoft Visual C++ Redistributable

### **Runtime Issues**

**Problem**: Eden won't start
**Solution**:
1. Check Windows Event Viewer for errors
2. Run diagnostics: `eden.exe --diagnostics`
3. Contact support with diagnostic report

**Problem**: Slow performance
**Solution**:
1. Use SSD storage if available
2. Increase available RAM
3. Add Eden to antivirus exceptions
4. Close unnecessary background applications

---

## 📞 Support & Help

### **Documentation**
- **User Manual**: `docs/USER_MANUAL.md` (comprehensive guide)
- **Quick Start**: Basic tutorials and examples
- **FAQ**: Frequently asked questions

### **Getting Help**
- **Email Support**: support@eden-trading.com
- **GitHub Issues**: Report bugs and feature requests
- **Community Forum**: User discussions and tips

### **Diagnostic Tools**
- **Built-in Diagnostics**: Tools → Generate Diagnostic Report
- **Log Files**: Located in `%APPDATA%\Eden\logs\`
- **System Information**: Help → About → System Info

---

## 🔄 Updates

Eden includes an automatic update system:
- **Automatic checking** on startup (can be disabled)
- **User control** over update installation
- **Progress tracking** during downloads
- **Rollback protection** for safety

Update notifications will appear when new versions are available.

---

## 📋 File Descriptions

| File | Purpose | Size | Required |
|------|---------|------|----------|
| `Eden Setup.exe` | Professional installer | ~48.8 MB | Recommended |
| `eden.exe` | Standalone executable | ~110.1 MB | Alternative |
| `Eden_Portable.zip` | Portable package | ~0.1 MB | Optional |
| `eden_installer.nsi` | NSIS source | <1 MB | Developers |
| `eden_installer.wxs` | WiX MSI source | <1 MB | Enterprise |
| `LICENSE.txt` | Software license | <1 MB | Reference |
| `eden_icon.ico` | Application icon | <1 MB | Reference |

---

## 🎉 Welcome to Eden!

Thank you for choosing Eden for your algorithmic trading needs. We've designed this platform to be both powerful for professionals and accessible for newcomers.

**Happy Trading! 📈**

---

*Eden v1.0.0 - Professional Algorithmic Trading Platform*  
*© 2025 Eden Technologies - All rights reserved*