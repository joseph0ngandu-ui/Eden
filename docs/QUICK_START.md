# Eden - Quick Start Guide

**The Origin of Order**

Welcome to Eden! This guide will help you get up and running quickly.

## Getting Started

### Option 1: Super Simple (Windows)
1. Double-click `launch.bat`
2. Follow the interactive menu
3. Choose option 1 to run Eden

### Option 2: Cross-Platform
1. Open terminal/command prompt in the Eden Bot folder
2. Run: `python launch.py`
3. Choose your option from the menu

### Option 3: Command Line
```bash
# Run directly
python launch.py run

# Install for all users
python launch.py install

# Build deployment package
python launch.py build

# Install dependencies only
python launch.py deps
```

### Option 4: PowerShell (Windows)
```powershell
# Interactive menu
.\Launch_Eden.ps1

# Direct commands
.\Launch_Eden.ps1 -Run
.\Launch_Eden.ps1 -Install
.\Launch_Eden.ps1 -Build
```

## Prerequisites

- **Python 3.8+** (required)
- **Windows 10/11** (recommended for MetaTrader 5 integration)
- **4GB RAM** minimum, 8GB recommended
- **Internet connection** for downloading dependencies

## First-Time Setup

1. **Install Python**: Download from [python.org](https://python.org)
2. **Run Launcher**: Use any of the methods above
3. **Choose Install**: Select option 2 for user-friendly installation
4. **Follow Prompts**: The installer will handle everything else

## What Gets Installed

- **Core Application**: Main Eden interface
- **Dependencies**: All required Python packages
- **MetaTrader 5**: Optional trading platform integration
- **Desktop Shortcuts**: Easy access from desktop and Start Menu
- **Uninstaller**: Clean removal when needed

## Features Overview

### Trading Dashboard
- Real-time market data visualization
- Portfolio tracking and analysis
- Risk management tools

### MetaTrader 5 Integration
- Auto-detection and setup
- Live trading connectivity
- Account monitoring

### Data Management
- Trade history analysis
- Performance reporting
- Export capabilities

### User Interface
- Professional themed design
- Motivational splash screen
- Intuitive navigation

## Troubleshooting

### Common Issues

**"Python not found"**
- Install Python from python.org
- Make sure to check "Add to PATH" during installation

**"Permission denied"**
- Run as Administrator (right-click → "Run as administrator")
- Or use the regular installer instead

**"Dependencies failed to install"**
- Check your internet connection
- Try running: `python -m pip install --upgrade pip`

**"MetaTrader 5 not detected"**
- Install MT5 from metaquotes.net
- The app will guide you through the process

### Getting Help

1. **Check Logs**: Look in the installation folder for log files
2. **Restart**: Try restarting the application
3. **Reinstall**: Use the uninstaller and reinstall
4. **Support**: Contact support with error details

## Updates

The app will check for updates automatically. You can also:
- Use the deployment builder to create updated packages
- Reinstall using the latest installer
- Pull updates from the source repository

## Tips for Best Experience

1. **Close Other Trading Apps**: Avoid conflicts with MT5
2. **Stable Internet**: Ensure reliable connection for live data
3. **Regular Backups**: Export your settings periodically
4. **Monitor Resources**: Keep an eye on CPU and memory usage

## Advanced Usage

### Development Mode
```bash
python launch.py dev
```

### Custom Configuration
- Edit `config.json` for advanced settings
- Modify themes in the `assets/` folder
- Add custom indicators via the plugin system

### Building from Source
```bash
git clone [repository-url]
cd eden_bot
python launch.py build
```

---

## Ready to Trade?

1. Launch Eden
2. Connect to MetaTrader 5
3. Configure your trading parameters
4. Start with paper trading to learn
5. Go live when you're confident

**Remember: Trading involves risk. Never trade more than you can afford to lose.**

---

**Eden - The Origin of Order**

*"In every chaos lies the seed of perfect order."*
