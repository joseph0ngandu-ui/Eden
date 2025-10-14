# 🚀 Eden Trading System - Complete Setup Guide

## 📋 Overview

This guide will help you set up the **Eden Trading System** - a professional hybrid C++/Python algorithmic trading platform with advanced ML capabilities and GPU acceleration.

## 🎯 What You Have Successfully Completed

✅ **Git Installation & Configuration**  
✅ **Repository Cloning** - Eden bot downloaded from GitHub  
✅ **Auto-Push Setup** - Changes will be automatically committed and pushed  
✅ **Architecture Analysis** - Complete understanding of the system components  

## 🔧 Required Dependencies (Manual Installation Needed)

Since the automated installers require user interaction, please install these manually:

### 1. **Python 3.11** (Critical)
```powershell
# Download and install from: https://www.python.org/downloads/
# Or use Windows Store: search for "Python 3.11"
# After installation, verify with: python --version
```

### 2. **CMake 3.22+** (For C++ Build)
```powershell
# Download from: https://cmake.org/download/
# Or use: winget install Kitware.CMake
# Verify with: cmake --version
```

### 3. **Qt6** (For GUI)
```powershell
# Download Qt6 from: https://www.qt.io/download
# Install Qt6.6.0+ with these components:
# - Qt Quick
# - Qt Charts  
# - Qt Network
# - Qt SQL
# - MSVC 2022 64-bit compiler
```

### 4. **Python Packages**
Once Python is installed, run in the Eden directory:
```bash
cd worker/python
pip install -r requirements.txt
pip install pyzmq onnxruntime MetaTrader5
```

## 🎮 Usage Instructions

### **Automatic Git Sync**
To push your changes automatically:
```powershell
./auto_push_setup.ps1 -CommitMessage "Your custom message here"
```

### **Building Eden** (After installing dependencies)
```powershell
# Run the build script
./build.ps1 -BuildType Release -Install

# Or manual build:
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### **Running Eden**
```bash
# Python-only mode (strategies and backtesting)
cd worker/python
python Eden.py

# Full GUI mode (after C++ build)
./build/Eden.exe
```

## 🏗️ System Architecture

```
Eden Trading System
├── 🖥️  C++ Qt6 Frontend (Professional UI)
├── 🐍 Python Backend (Trading Logic)
├── ⚡ GPU Acceleration (CUDA/DirectML) 
├── 📊 MetaTrader 5 Integration
├── 🤖 ML Pipeline (LightGBM/Neural Networks)
└── 🔄 ZeroMQ IPC Communication
```

## 📈 Trading Strategies Available

1. **ICT Strategy** - Inner Circle Trading methodology
   - Fair Value Gaps (FVG) detection
   - Liquidity sweep analysis
   - Order block identification

2. **Mean Reversion** - Statistical trading
   - Bollinger Bands
   - Z-score analysis
   - Oversold/overbought conditions

3. **Momentum** - Trend following
   - Moving average crossovers
   - Momentum indicators

4. **ML-Generated** - AI-powered trading
   - LightGBM models
   - Feature engineering pipeline
   - Reinforcement learning

## 🎨 Features Highlights

- **Apple-class UI Design** with Eden Dark theme
- **Real-time backtesting** with streaming results
- **GPU-accelerated** computations
- **Multi-timeframe analysis**
- **Monte Carlo optimization**
- **Live trading** via MetaTrader 5
- **Advanced risk management**

## 🔒 MetaTrader 5 Integration

Eden supports live trading through MT5 with:
- **Auto-installation** prompts for MT5
- **Symbol mapping** (EURUSDm format support)
- **Real-time data feeds**  
- **Position management**
- **Risk controls**

## 🚨 Important Notes

1. **Dependencies** - All tools need manual installation due to UAC restrictions
2. **MT5 Required** - For live trading functionality  
3. **GPU Optional** - Will fallback to CPU if GPU acceleration unavailable
4. **Broker Account** - Needed for live trading (demo accounts supported)

## 🆘 Troubleshooting

**Python Not Found:**
- Install Python 3.11 from python.org
- Add to PATH during installation
- Restart PowerShell after installation

**CMake Errors:**
- Install Visual Studio Build Tools 2022
- Ensure CMake is in PATH

**Qt6 Issues:**
- Set QT_ROOT environment variable
- Install MSVC 2022 components

**Git Authentication:**
- Run: `gh auth login` (if GitHub CLI installed)
- Or setup SSH keys for GitHub

## 🎯 Next Steps

1. **Install dependencies** listed above
2. **Run build script**: `./build.ps1`
3. **Test Python components**: `cd worker/python && python Eden.py`
4. **Configure MetaTrader 5** for live trading
5. **Start trading** with paper money first!

---

**🌟 Eden Trading System** - Professional algorithmic trading at your fingertips!