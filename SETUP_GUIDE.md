# 🚀 Eden VIX 100 Trading Bot - Setup Guide

## 📋 Overview

This guide will help you set up the **Eden VIX 100 Trading Bot** - a specialized Python bot for automated VIX 100 trading via MetaTrader 5.

## 🎯 What You Have Successfully Completed

✅ **Git Installation & Configuration**  
✅ **Repository Cloning** - Eden VIX 100 bot downloaded from GitHub  
✅ **Auto-Push Setup** - Changes will be automatically committed and pushed  
✅ **Bot Conversion** - Converted from hybrid system to pure Python VIX 100 bot

## 🔧 Required Dependencies (Manual Installation Needed)

Since the automated installers require user interaction, please install these manually:

### 1. **Python 3.11+** (Critical)
```powershell
# Download and install from: https://www.python.org/downloads/
# Or use Windows Store: search for "Python 3.11"
# After installation, verify with: python --version
```

### 2. **MetaTrader 5** (Required for trading)
```powershell
# Download from: https://www.metatrader5.com/en/download
# Install and set up with your broker account
# Ensure it's running and logged in before starting the bot
```

### 3. **Python Packages**
Once Python is installed, run in the Eden directory:
```bash
pip install -r requirements.txt
```

This will install:
- MetaTrader5 Python library
- pandas and numpy for data analysis
- TA-Lib for technical indicators
- Other required packages

## 🎮 Usage Instructions

### **Running the VIX 100 Trading Bot**
```bash
# Run the main trading bot
python eden_vix100_bot.py
```

### **Configuration**
Edit `config.yaml` to customize:
- Trading parameters (lot size, risk management)
- Strategy settings (RSI periods, thresholds)
- MT5 connection settings
- Logging and monitoring options

### **Automatic Git Sync**
To push your changes automatically:
```powershell
./auto_push_setup.ps1 -CommitMessage "Your custom message here"
```

## 🏧 Bot Architecture

```
Eden VIX 100 Trading Bot
├── 🐍 Python Core (Main Bot Logic)
├── 📊 MetaTrader 5 Integration
├── 📈 Technical Analysis (RSI, Moving Averages)
├── 🛡️ Risk Management (Stop Loss, Position Limits)
├── 📁 Configuration Management (YAML)
└── 📝 Logging & Monitoring
```

## 📈 VIX 100 Trading Strategy

The bot uses a **RSI + Moving Average** strategy specifically optimized for VIX 100:

**🔴 Buy Signals:**
- RSI below 30 (oversold)
- Price below 20-period moving average
- Risk management: 50-pip stop loss, 100-pip take profit

**🔴 Sell Signals:**
- RSI above 70 (overbought)
- Price above 20-period moving average
- Risk management: 50-pip stop loss, 100-pip take profit

**🛡️ Safety Features:**
- Maximum 3 concurrent positions
- Maximum 5 consecutive losses before pause
- Daily loss limits
- Emergency stop functionality

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