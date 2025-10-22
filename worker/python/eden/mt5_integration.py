"""
Eden Bot - MetaTrader 5 Integration
Robust MT5 connectivity with auto-detection, installation prompts, and seamless trading
"""

import subprocess
import threading
import winreg
import requests
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    from PySide6.QtWidgets import (
        QDialog,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QProgressBar,
        QMessageBox,
        QGroupBox,
    )
    from PySide6.QtCore import QThread, Signal, QTimer
    from PySide6.QtGui import QFont

    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class MT5ConnectionStatus(Enum):
    """MetaTrader 5 connection status."""

    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting"
    CONNECTED = "Connected"
    ERROR = "Error"
    NOT_INSTALLED = "Not Installed"


@dataclass
class MT5AccountInfo:
    """MetaTrader 5 account information."""

    login: int
    server: str
    name: str
    company: str
    currency: str
    balance: float
    equity: float
    margin: float
    free_margin: float
    profit: float
    leverage: int
    connected: bool


@dataclass
class MT5Position:
    """MetaTrader 5 position information."""

    ticket: int
    symbol: str
    type: int
    volume: float
    price_open: float
    price_current: float
    profit: float
    swap: float
    commission: float
    time: datetime
    comment: str


class MT5Installer(QDialog if PYSIDE6_AVAILABLE else object):
    """MetaTrader 5 installer dialog."""

    def __init__(self, parent=None):
        if not PYSIDE6_AVAILABLE:
            super().__init__()
            return

        super().__init__(parent)
        self.setWindowTitle("MetaTrader 5 Installation")
        self.setFixedSize(500, 400)
        self.setModal(True)

        # Apply Eden theme
        self.setStyleSheet(
            f"""
            QDialog {{
                background-color: #0B1220;
                color: #FFFFFF;
            }}
            QLabel {{
                color: #FFFFFF;
                font-size: 12px;
            }}
            QPushButton {{
                background-color: #1ABC9C;
                color: #FFFFFF;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #16A085;
            }}
            QPushButton:disabled {{
                background-color: #34495E;
                color: #7F8C8D;
            }}
            QProgressBar {{
                background-color: #2C3E50;
                border: 1px solid #34495E;
                border-radius: 3px;
                text-align: center;
                color: white;
            }}
            QProgressBar::chunk {{
                background-color: #1ABC9C;
                border-radius: 3px;
            }}
        """
        )

        self.setup_ui()
        self.installation_complete = False

    def setup_ui(self):
        """Setup the installer UI."""
        layout = QVBoxLayout()

        # Header
        header = QLabel("MetaTrader 5 Installation")
        header.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header.setStyleSheet("color: #1ABC9C; margin-bottom: 10px;")
        layout.addWidget(header)

        # Description
        description = QLabel(
            """
        Eden Bot requires MetaTrader 5 to execute live trades.
        
        MetaTrader 5 is a professional trading platform that provides:
        • Access to global financial markets
        • Advanced charting and analysis tools
        • Automated trading capabilities
        • Secure connection to brokers
        
        Click 'Download & Install' to get MetaTrader 5 from the official website.
        """
        )
        description.setWordWrap(True)
        description.setStyleSheet("margin: 10px 0px; line-height: 1.4;")
        layout.addWidget(description)

        # Progress section
        self.progress_group = QGroupBox("Installation Progress")
        progress_layout = QVBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready to install")
        self.status_label.setStyleSheet("color: #BDC3C7;")
        progress_layout.addWidget(self.status_label)

        self.progress_group.setLayout(progress_layout)
        layout.addWidget(self.progress_group)

        # Buttons
        button_layout = QHBoxLayout()

        self.download_btn = QPushButton("🔽 Download & Install MT5")
        self.download_btn.clicked.connect(self.download_mt5)
        button_layout.addWidget(self.download_btn)

        self.manual_btn = QPushButton("🌐 Open MT5 Website")
        self.manual_btn.clicked.connect(self.open_mt5_website)
        button_layout.addWidget(self.manual_btn)

        self.skip_btn = QPushButton("Skip")
        self.skip_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.skip_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def download_mt5(self):
        """Download and install MetaTrader 5."""
        self.download_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setText("Downloading MetaTrader 5...")

        # Start download in separate thread
        self.download_thread = MT5DownloadThread()
        self.download_thread.progress_updated.connect(self.update_progress)
        self.download_thread.download_complete.connect(self.on_download_complete)
        self.download_thread.error_occurred.connect(self.on_download_error)
        self.download_thread.start()

    def update_progress(self, percentage: int, status: str):
        """Update download progress."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(status)

    def on_download_complete(self, installer_path: str):
        """Handle download completion."""
        self.status_label.setText("Download complete! Launching installer...")

        try:
            # Launch the installer
            subprocess.Popen(installer_path, shell=True)
            self.status_label.setText(
                "MetaTrader 5 installer launched. Please complete the installation."
            )

            # Replace buttons
            self.download_btn.setText("✅ Installation Launched")
            self.download_btn.setEnabled(False)

            self.manual_btn.setText("Done")
            self.manual_btn.clicked.disconnect()
            self.manual_btn.clicked.connect(self.accept)

            self.installation_complete = True

        except Exception as e:
            self.on_download_error(f"Failed to launch installer: {e}")

    def on_download_error(self, error: str):
        """Handle download error."""
        self.status_label.setText(f"Error: {error}")
        self.download_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        QMessageBox.warning(
            self,
            "Download Error",
            f"Failed to download MetaTrader 5:\n{error}\n\nPlease try downloading manually.",
        )

    def open_mt5_website(self):
        """Open MetaTrader 5 official website."""
        import webbrowser

        webbrowser.open("https://www.metatrader5.com/en/download")
        self.accept()


class MT5DownloadThread(QThread if PYSIDE6_AVAILABLE else threading.Thread):
    """Thread for downloading MetaTrader 5."""

    if PYSIDE6_AVAILABLE:
        progress_updated = Signal(int, str)
        download_complete = Signal(str)
        error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self.mt5_download_url = "https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"

    def run(self):
        """Download MetaTrader 5 installer."""
        try:
            # Create temp directory
            temp_dir = Path(tempfile.gettempdir()) / "eden_mt5"
            temp_dir.mkdir(exist_ok=True)

            installer_path = temp_dir / "mt5setup.exe"

            if PYSIDE6_AVAILABLE:
                self.progress_updated.emit(10, "Starting download...")

            # Download with progress
            response = requests.get(self.mt5_download_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(installer_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percentage = int((downloaded / total_size) * 90) + 10
                            if PYSIDE6_AVAILABLE:
                                self.progress_updated.emit(
                                    percentage,
                                    f"Downloading... {downloaded // 1024}KB / {total_size // 1024}KB",
                                )

            if PYSIDE6_AVAILABLE:
                self.progress_updated.emit(100, "Download complete!")
                self.download_complete.emit(str(installer_path))

        except Exception as e:
            if PYSIDE6_AVAILABLE:
                self.error_occurred.emit(str(e))
            else:
                print(f"Download error: {e}")


# ---- Lightweight OHLCV helpers (no UI deps) ----


def _mt5_timeframe(tf: str):
    if not MT5_AVAILABLE:
        return None
    tfu = tf.upper()
    mapping = {
        "1M": mt5.TIMEFRAME_M1,
        "5M": mt5.TIMEFRAME_M5,
        "15M": mt5.TIMEFRAME_M15,
        "30M": mt5.TIMEFRAME_M30,
        "1H": mt5.TIMEFRAME_H1,
        "4H": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "1D": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    return mapping.get(tfu)


def mt5_fetch_ohlcv(symbol: str, timeframe: str, start: str, end: str):
    """Fetch OHLCV from MT5 between start and end (ISO date strings). Returns pandas DataFrame or None."""
    if not MT5_AVAILABLE:
        return None
    import pandas as pd
    from datetime import datetime

    # Initialize MT5 terminal
    try:
        if not mt5.initialize():
            return None
    except Exception:
        return None

    # Ensure symbol is selected
    try:
        mt5.symbol_select(symbol, True)
    except Exception:
        pass

    tf_const = _mt5_timeframe(timeframe)
    if tf_const is None:
        return None

    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except Exception:
        # Fallback: last 2 years
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=730)

    try:
        rates = mt5.copy_rates_range(symbol, tf_const, start_dt, end_dt)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        if df.empty:
            return None
        # Convert time to datetime (UTC)
        if "time" in df.columns:
            df["datetime"] = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df.drop(columns=["time"])
            df = df.set_index("datetime").sort_index()
        # Normalize columns
        rename = {c: c for c in df.columns}
        for src, dst in [
            ("open", "open"),
            ("high", "high"),
            ("low", "low"),
            ("close", "close"),
            ("tick_volume", "volume"),
            ("real_volume", "volume"),
        ]:
            if src in df.columns:
                rename[src] = dst
        df = df.rename(columns=rename)
        # Ensure required columns
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                df[c] = 0
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return None


class MT5Manager:
    """Comprehensive MetaTrader 5 integration manager."""

    def __init__(self):
        self.status = MT5ConnectionStatus.DISCONNECTED
        self.account_info: Optional[MT5AccountInfo] = None
        self.positions: List[MT5Position] = []
        self.connection_callbacks = []
        self.error_callbacks = []
        self.trading_enabled = False

        # Connection monitoring
        self.monitor_timer = None
        self.last_heartbeat = None

    def add_connection_callback(self, callback):
        """Add callback for connection status changes."""
        self.connection_callbacks.append(callback)

    def add_error_callback(self, callback):
        """Add callback for error events."""
        self.error_callbacks.append(callback)

    def notify_connection_change(self):
        """Notify listeners of connection status change."""
        for callback in self.connection_callbacks:
            try:
                callback(self.status, self.account_info)
            except Exception as e:
                print(f"Error in connection callback: {e}")

    def notify_error(self, error_message: str):
        """Notify listeners of error."""
        for callback in self.error_callbacks:
            try:
                callback(error_message)
            except Exception as e:
                print(f"Error in error callback: {e}")

    def check_mt5_installed(self) -> bool:
        """Check if MetaTrader 5 is installed."""
        try:
            # Check registry for MT5 installation
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall",
            )

            i = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, i)
                    subkey = winreg.OpenKey(key, subkey_name)

                    try:
                        display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                        if "MetaTrader 5" in display_name:
                            winreg.CloseKey(subkey)
                            winreg.CloseKey(key)
                            return True
                    except FileNotFoundError:
                        pass

                    winreg.CloseKey(subkey)
                    i += 1

                except OSError:
                    break

            winreg.CloseKey(key)

            # Also check common installation paths
            common_paths = [
                Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal",
                Path("C:") / "Program Files" / "MetaTrader 5",
                Path("C:") / "Program Files (x86)" / "MetaTrader 5",
            ]

            for path in common_paths:
                if path.exists() and (path / "terminal64.exe").exists():
                    return True

            return False

        except Exception:
            return False

    def prompt_mt5_installation(self, parent=None) -> bool:
        """Prompt user to install MetaTrader 5."""
        if not PYSIDE6_AVAILABLE:
            print(
                "MetaTrader 5 not found. Please install it manually from https://www.metatrader5.com/"
            )
            return False

        installer = MT5Installer(parent)
        result = installer.exec()

        return result == QDialog.Accepted and installer.installation_complete

    def connect(
        self, login: int = None, password: str = None, server: str = None
    ) -> bool:
        """Connect to MetaTrader 5."""
        try:
            if not MT5_AVAILABLE:
                # Try to import again in case it was just installed
                try:
                    import MetaTrader5 as mt5

                    globals()["MT5_AVAILABLE"] = True
                    globals()["mt5"] = mt5
                except ImportError:
                    self.status = MT5ConnectionStatus.NOT_INSTALLED
                    self.notify_connection_change()
                    return False

            self.status = MT5ConnectionStatus.CONNECTING
            self.notify_connection_change()

            # Initialize MT5
            if not mt5.initialize():
                error = f"MT5 initialization failed: {mt5.last_error()}"
                self.notify_error(error)
                self.status = MT5ConnectionStatus.ERROR
                self.notify_connection_change()
                return False

            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password, server):
                    error = f"MT5 login failed: {mt5.last_error()}"
                    self.notify_error(error)
                    self.status = MT5ConnectionStatus.ERROR
                    self.notify_connection_change()
                    return False

            # Get account info
            account = mt5.account_info()
            if account is None:
                error = f"Failed to get account info: {mt5.last_error()}"
                self.notify_error(error)
                self.status = MT5ConnectionStatus.ERROR
                self.notify_connection_change()
                return False

            # Store account info
            self.account_info = MT5AccountInfo(
                login=account.login,
                server=account.server,
                name=account.name,
                company=account.company,
                currency=account.currency,
                balance=account.balance,
                equity=account.equity,
                margin=account.margin,
                free_margin=account.margin_free,
                profit=account.profit,
                leverage=account.leverage,
                connected=True,
            )

            self.status = MT5ConnectionStatus.CONNECTED
            self.last_heartbeat = datetime.now()

            # Start connection monitoring
            self.start_monitoring()

            self.notify_connection_change()
            return True

        except Exception as e:
            error = f"MT5 connection error: {str(e)}"
            self.notify_error(error)
            self.status = MT5ConnectionStatus.ERROR
            self.notify_connection_change()
            return False

    def disconnect(self):
        """Disconnect from MetaTrader 5."""
        try:
            if MT5_AVAILABLE:
                mt5.shutdown()

            self.status = MT5ConnectionStatus.DISCONNECTED
            self.account_info = None
            self.positions.clear()
            self.trading_enabled = False

            # Stop monitoring
            self.stop_monitoring()

            self.notify_connection_change()

        except Exception as e:
            self.notify_error(f"Error during disconnect: {str(e)}")

    def start_monitoring(self):
        """Start connection monitoring."""
        if PYSIDE6_AVAILABLE:
            self.monitor_timer = QTimer()
            self.monitor_timer.timeout.connect(self.check_connection)
            self.monitor_timer.start(5000)  # Check every 5 seconds

    def stop_monitoring(self):
        """Stop connection monitoring."""
        if self.monitor_timer:
            self.monitor_timer.stop()
            self.monitor_timer = None

    def check_connection(self):
        """Check if MT5 connection is still active."""
        try:
            if not MT5_AVAILABLE or self.status != MT5ConnectionStatus.CONNECTED:
                return

            # Try to get account info as heartbeat
            account = mt5.account_info()
            if account is None:
                # Connection lost
                self.status = MT5ConnectionStatus.ERROR
                self.notify_error("MT5 connection lost")
                self.notify_connection_change()
                return

            # Update account info
            self.account_info.balance = account.balance
            self.account_info.equity = account.equity
            self.account_info.margin = account.margin
            self.account_info.free_margin = account.margin_free
            self.account_info.profit = account.profit

            self.last_heartbeat = datetime.now()

            # Update positions
            self.update_positions()

        except Exception as e:
            self.notify_error(f"Connection check failed: {str(e)}")

    def update_positions(self):
        """Update current positions."""
        try:
            if not MT5_AVAILABLE or self.status != MT5ConnectionStatus.CONNECTED:
                return

            positions = mt5.positions_get()
            if positions is None:
                return

            self.positions.clear()
            for pos in positions:
                position = MT5Position(
                    ticket=pos.ticket,
                    symbol=pos.symbol,
                    type=pos.type,
                    volume=pos.volume,
                    price_open=pos.price_open,
                    price_current=pos.price_current,
                    profit=pos.profit,
                    swap=pos.swap,
                    commission=pos.commission,
                    time=datetime.fromtimestamp(pos.time),
                    comment=pos.comment,
                )
                self.positions.append(position)

        except Exception as e:
            self.notify_error(f"Failed to update positions: {str(e)}")

    def enable_trading(self):
        """Enable trading capabilities."""
        if self.status == MT5ConnectionStatus.CONNECTED:
            self.trading_enabled = True
            return True
        return False

    def disable_trading(self):
        """Disable trading capabilities."""
        self.trading_enabled = False

    def place_order(
        self,
        symbol: str,
        order_type: int,
        volume: float,
        price: float = None,
        sl: float = None,
        tp: float = None,
        comment: str = "Eden Bot",
    ) -> bool:
        """Place a trading order."""
        try:
            if not self.trading_enabled or self.status != MT5ConnectionStatus.CONNECTED:
                self.notify_error("Trading not enabled or not connected")
                return False

            if not MT5_AVAILABLE:
                self.notify_error("MT5 not available")
                return False

            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "deviation": 10,
                "comment": comment,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            if price is not None:
                request["price"] = price

            if sl is not None:
                request["sl"] = sl

            if tp is not None:
                request["tp"] = tp

            # Send order
            result = mt5.order_send(request)

            if result is None:
                error = f"Order failed: {mt5.last_error()}"
                self.notify_error(error)
                return False

            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error = f"Order failed with retcode: {result.retcode}"
                self.notify_error(error)
                return False

            return True

        except Exception as e:
            self.notify_error(f"Order execution error: {str(e)}")
            return False

    def close_position(self, ticket: int) -> bool:
        """Close a position."""
        try:
            if not self.trading_enabled or self.status != MT5ConnectionStatus.CONNECTED:
                return False

            position = None
            for pos in self.positions:
                if pos.ticket == ticket:
                    position = pos
                    break

            if position is None:
                self.notify_error("Position not found")
                return False

            # Determine opposite order type
            if position.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
            else:
                order_type = mt5.ORDER_TYPE_BUY

            # Close position
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": position.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": ticket,
                "deviation": 10,
                "comment": "Eden Bot - Close",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = f"Close position failed: {mt5.last_error() if result is None else result.retcode}"
                self.notify_error(error)
                return False

            return True

        except Exception as e:
            self.notify_error(f"Close position error: {str(e)}")
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get symbol information."""
        try:
            if not MT5_AVAILABLE or self.status != MT5ConnectionStatus.CONNECTED:
                return None

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None

            return {
                "name": symbol_info.name,
                "bid": symbol_info.bid,
                "ask": symbol_info.ask,
                "spread": symbol_info.spread,
                "digits": symbol_info.digits,
                "point": symbol_info.point,
                "trade_contract_size": symbol_info.trade_contract_size,
                "volume_min": symbol_info.volume_min,
                "volume_max": symbol_info.volume_max,
                "volume_step": symbol_info.volume_step,
            }

        except Exception as e:
            self.notify_error(f"Failed to get symbol info: {str(e)}")
            return None


# Global MT5 manager instance
mt5_manager = MT5Manager()


def get_mt5_manager() -> MT5Manager:
    """Get the global MT5 manager instance."""
    return mt5_manager
