#!/usr/bin/env python3
"""
Eden - Professional Installer
Apple-inspired design with clean interface and seamless setup
"""

import sys
import shutil
import winreg
import subprocess
from pathlib import Path
from datetime import datetime
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, messagebox

# Import Apple-style theme detection
try:
    from theme_manager import WindowsThemeDetector

    THEME_DETECTION_AVAILABLE = True
except ImportError:
    THEME_DETECTION_AVAILABLE = False


class EdenInstaller:
    """Professional Eden Bot installer for Windows."""

    def __init__(self):
        self.app_name = "Eden"
        self.app_description = "Professional Algorithmic Trading System"
        self.app_version = "2.0.0"
        self.publisher = "Eden Trading Systems"

        # Paths
        self.temp_dir = Path(tempfile.gettempdir()) / "eden_installer"
        self.install_dir = Path.home() / "AppData" / "Local" / "Eden"
        self.desktop_path = Path.home() / "Desktop"
        self.start_menu_path = (
            Path.home()
            / "AppData"
            / "Roaming"
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
        )

        # Dependencies
        self.python_required = "3.11"
        self.required_packages = [
            "PySide6==6.6.0",
            "pyqtgraph==0.13.3",
            "pandas==2.2.2",
            "numpy==1.26.4",
            "scipy==1.11.4",
            "scikit-learn==1.4.2",
            "lightgbm==4.3.0",
            "yfinance==0.2.38",
            "requests==2.32.3",
            "matplotlib==3.8.4",
            "optuna==3.6.1",
            "tqdm==4.66.4",
            "pydantic==2.8.2",
            "python-dotenv==1.0.1",
            "PyYAML==6.0.2",
            "ccxt==4.3.77",
            "ta==0.11.0",
            "MetaTrader5==5.0.45",
        ]

        # UI
        self.root = None
        self.progress_var = None
        self.status_var = None
        self.log_text = None

        # Installation state
        self.installation_cancelled = False
        self.installation_complete = False

    def create_ui(self):
        """Create the installer UI."""
        self.root = tk.Tk()
        self.root.title(f"{self.app_name} Installer")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        # Detect system theme
        is_dark_mode = False
        if THEME_DETECTION_AVAILABLE:
            theme_detector = WindowsThemeDetector()
            is_dark_mode = theme_detector.get_system_theme() == "dark"

        # Set theme-appropriate colors
        if is_dark_mode:
            bg_color = "#1C1C1E"
            fg_color = "#FFFFFF"
            secondary_fg = "#EBEBF5"
            tertiary_bg = "#2C2C2E"
            primary_color = "#0A84FF"
            primary_hover = "#0A70E8"
            primary_pressed = "#085DD1"
            trough_color = "#2C2C2E"
        else:
            bg_color = "#FFFFFF"
            fg_color = "#1D1D1F"
            secondary_fg = "#86868B"
            tertiary_bg = "#F2F2F7"
            primary_color = "#007AFF"
            primary_hover = "#0056CC"
            primary_pressed = "#004499"
            trough_color = "#F2F2F7"

        self.root.configure(bg=bg_color)

        # Configure Apple-style theme
        style = ttk.Style()
        style.theme_use("clam")

        # Configure Apple-inspired colors and typography with theme support
        style.configure(
            "TLabel",
            background=bg_color,
            foreground=fg_color,
            font=("-apple-system", "SF Pro Text", 13),
        )
        style.configure(
            "Title.TLabel",
            background=bg_color,
            foreground=fg_color,
            font=("-apple-system", "SF Pro Display", 28, "normal"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=bg_color,
            foreground=secondary_fg,
            font=("-apple-system", "SF Pro Text", 16),
        )
        style.configure(
            "TButton",
            background=primary_color,
            foreground="#FFFFFF",
            font=("-apple-system", "SF Pro Text", 13, "normal"),
            relief="flat",
            borderwidth=0,
        )
        style.map(
            "TButton",
            background=[("active", primary_hover), ("pressed", primary_pressed)],
        )
        style.configure(
            "TProgressbar",
            background=primary_color,
            troughcolor=trough_color,
            borderwidth=0,
            lightcolor=primary_color,
            darkcolor=primary_color,
        )

        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 20))

        title_label = ttk.Label(header_frame, text=self.app_name, style="Title.TLabel")
        title_label.pack(pady=(0, 8))

        subtitle_label = ttk.Label(
            header_frame, text="The Origin of Order", style="Subtitle.TLabel"
        )
        subtitle_label.pack(pady=(0, 16))

        description_label = ttk.Label(
            header_frame, text=self.app_description, style="TLabel"
        )
        description_label.pack()

        # Installation info
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill="x", pady=(0, 20))

        ttk.Label(info_frame, text=f"Version: {self.app_version}", style="TLabel").pack(
            anchor="w"
        )
        ttk.Label(info_frame, text=f"Publisher: {self.publisher}", style="TLabel").pack(
            anchor="w"
        )
        ttk.Label(
            info_frame, text=f"Install Location: {self.install_dir}", style="TLabel"
        ).pack(anchor="w")

        # Progress section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill="x", pady=(0, 20))

        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="Ready to install")

        ttk.Label(progress_frame, text="Installation Progress:", style="TLabel").pack(
            anchor="w"
        )

        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill="x", pady=(5, 5))

        self.status_label = ttk.Label(
            progress_frame, textvariable=self.status_var, style="TLabel"
        )
        self.status_label.pack(anchor="w")

        # Log section
        log_frame = ttk.Frame(main_frame)
        log_frame.pack(fill="both", expand=True, pady=(0, 20))

        ttk.Label(log_frame, text="Installation Log:", style="TLabel").pack(anchor="w")

        log_scroll = tk.Scrollbar(log_frame)
        log_scroll.pack(side="right", fill="y")

        # Apply theme to log text area
        log_bg = tertiary_bg if is_dark_mode else "#F9F9F9"
        log_fg = fg_color

        self.log_text = tk.Text(
            log_frame,
            height=8,
            bg=log_bg,
            fg=log_fg,
            font=("-apple-system", "SF Mono", 12),
            yscrollcommand=log_scroll.set,
            borderwidth=1,
            relief="solid",
            highlightthickness=0,
        )
        self.log_text.pack(fill="both", expand=True)
        log_scroll.config(command=self.log_text.yview)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x")

        self.install_btn = ttk.Button(
            button_frame, text="Install", command=self.start_installation
        )
        self.install_btn.pack(side="left", padx=(0, 12))

        self.cancel_btn = ttk.Button(
            button_frame, text="Cancel", command=self.cancel_installation
        )
        self.cancel_btn.pack(side="right")

        # Center the window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.cancel_installation)

    def log(self, message: str, level: str = "INFO"):
        """Add message to installation log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}\n"

        if self.log_text:
            self.log_text.insert(tk.END, log_message)
            self.log_text.see(tk.END)
            self.root.update_idletasks()

        print(log_message.strip())

    def update_progress(self, percentage: float, status: str = ""):
        """Update installation progress."""
        if self.progress_var:
            self.progress_var.set(percentage)

        if status and self.status_var:
            self.status_var.set(status)

        if self.root:
            self.root.update_idletasks()

    def check_python(self) -> bool:
        """Check if Python is available and compatible."""
        try:
            result = subprocess.run(
                [sys.executable, "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                version_str = result.stdout.strip()
                self.log(f"Found Python: {version_str}")
                return True
            else:
                self.log("Python not found", "ERROR")
                return False
        except Exception as e:
            self.log(f"Error checking Python: {e}", "ERROR")
            return False

    def create_directories(self):
        """Create installation directories."""
        try:
            self.install_dir.mkdir(parents=True, exist_ok=True)
            self.log(f"Created installation directory: {self.install_dir}")

            # Create subdirectories
            (self.install_dir / "assets").mkdir(exist_ok=True)
            (self.install_dir / "logs").mkdir(exist_ok=True)
            (self.install_dir / "data").mkdir(exist_ok=True)
            (self.install_dir / "config").mkdir(exist_ok=True)

        except Exception as e:
            self.log(f"Error creating directories: {e}", "ERROR")
            raise

    def install_python_packages(self):
        """Install required Python packages."""
        try:
            self.log("Installing Python packages...")

            # Update pip first
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
                check=True,
                capture_output=True,
            )

            total_packages = len(self.required_packages)

            for i, package in enumerate(self.required_packages):
                if self.installation_cancelled:
                    return False

                self.log(f"Installing {package}...")
                self.update_progress(
                    30 + (i / total_packages) * 40,
                    f"Installing {package.split('==')[0]}...",
                )

                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )

                    if result.returncode != 0:
                        self.log(
                            f"Warning: Failed to install {package}: {result.stderr}",
                            "WARNING",
                        )
                    else:
                        self.log(f"Installed {package}")

                except subprocess.TimeoutExpired:
                    self.log(f"Timeout installing {package}", "WARNING")
                except Exception as e:
                    self.log(f"Error installing {package}: {e}", "WARNING")

            return True

        except Exception as e:
            self.log(f"Error installing packages: {e}", "ERROR")
            return False

    def copy_application_files(self):
        """Copy application files to installation directory."""
        try:
            self.log("Copying application files...")

            # List of files to copy
            files_to_copy = [
                "Eden.py",
                "ui_main.py",
                "splash_screen.py",
                "theme_manager.py",
                "config.yml",
                "requirements.txt",
                "QUICK_START.md",
                "eden\\config\\symbol_map.yaml",
            ]

            # Copy Eden package
            source_eden = Path("eden")
            dest_eden = self.install_dir / "eden"

            if source_eden.exists():
                if dest_eden.exists():
                    shutil.rmtree(dest_eden)
                shutil.copytree(source_eden, dest_eden)
                self.log("Copied Eden package")

            # Copy individual files
            for file_path in files_to_copy:
                source_file = Path(file_path)
                if source_file.exists():
                    dest_file = self.install_dir / file_path
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, dest_file)

            self.log("Application files copied successfully")
            return True

        except Exception as e:
            self.log(f"Error copying files: {e}", "ERROR")
            return False

    def create_launcher_script(self):
        """Create the main launcher script."""
        try:
            # Copy the main Eden.py launcher from source
            source_eden = Path("Eden.py")
            launcher_path = self.install_dir / "Eden.py"

            if source_eden.exists():
                shutil.copy2(source_eden, launcher_path)
                self.log("Copied main Eden.py launcher")
            else:
                # Create fallback launcher if source doesn't exist
                launcher_content = f'''#!/usr/bin/env python3
"""
Eden - Professional Trading System
Unified application entry point
"""

import sys
import os
from pathlib import Path

# Add application directory to Python path
app_dir = Path(__file__).parent
sys.path.insert(0, str(app_dir))

# Change to application directory
os.chdir(app_dir)

def main():
    """Main entry point for Eden application."""
    try:
        # Configure logging first
        from eden.logging_conf import configure_logging
        configure_logging("INFO")
        
        # Check if we should run CLI or GUI
        if len(sys.argv) > 1:
            # CLI mode
            from eden.cli import main as cli_main
            cli_main()
        else:
            # GUI mode
            try:
                from ui_main import run_main_ui
                run_main_ui()
            except ImportError:
                print("GUI dependencies not available, running CLI mode")
                from eden.cli import main as cli_main
                cli_main()
                
    except Exception as e:
        print(f"Error starting Eden: {{e}}")
        input("Press Enter to exit...")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
                launcher_path.write_text(launcher_content, encoding="utf-8")
                self.log("Created fallback Eden.py launcher")

            self.log("Created launcher script")
            return True

        except Exception as e:
            self.log(f"Error creating launcher: {e}", "ERROR")
            return False

    def create_desktop_shortcut(self):
        """Create desktop shortcut."""
        try:
            import win32com.client

            shell = win32com.client.Dispatch("WScript.Shell")
            shortcut_path = self.desktop_path / "Eden.lnk"
            shortcut = shell.CreateShortCut(str(shortcut_path))
            shortcut.Targetpath = sys.executable
            shortcut.Arguments = f'"{self.install_dir / "Eden.py"}"'
            shortcut.WorkingDirectory = str(self.install_dir)
            shortcut.IconLocation = sys.executable
            shortcut.Description = self.app_description
            shortcut.save()

            self.log("Created desktop shortcut")
            return True

        except ImportError:
            # Fallback method without win32com
            try:
                # Create a batch file as shortcut
                batch_content = f"""@echo off
cd /d "{self.install_dir}"
"{sys.executable}" "Eden.py"
pause
"""
                batch_path = self.desktop_path / "Eden.bat"
                batch_path.write_text(batch_content, encoding="utf-8")
                self.log("Created desktop batch file")
                return True
            except Exception as e:
                self.log(f"Error creating desktop shortcut: {e}", "WARNING")
                return False
        except Exception as e:
            self.log(f"Error creating desktop shortcut: {e}", "WARNING")
            return False

    def create_start_menu_shortcut(self):
        """Create Start Menu shortcut."""
        try:
            # Create Eden folder in Start Menu
            eden_start_menu = self.start_menu_path / "Eden"
            eden_start_menu.mkdir(exist_ok=True)

            # Create batch file for Start Menu
            batch_content = f"""@echo off
cd /d "{self.install_dir}"
"{sys.executable}" "Eden.py"
"""
            batch_path = eden_start_menu / "Eden.bat"
            batch_path.write_text(batch_content, encoding="utf-8")

            self.log("Created Start Menu shortcut")
            return True

        except Exception as e:
            self.log(f"Error creating Start Menu shortcut: {e}", "WARNING")
            return False

    def register_uninstaller(self):
        """Register the application in Windows Add/Remove Programs."""
        try:
            # Create uninstaller script
            uninstaller_content = f'''#!/usr/bin/env python3
"""
Eden Uninstaller
"""

import os
import shutil
import winreg
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

def uninstall():
    try:
        # Remove installation directory
        install_dir = Path(r"{self.install_dir}")
        if install_dir.exists():
            shutil.rmtree(install_dir)
        
        # Remove desktop shortcut
        desktop_shortcut = Path.home() / "Desktop" / "Eden.lnk"
        if desktop_shortcut.exists():
            desktop_shortcut.unlink()
        
        desktop_batch = Path.home() / "Desktop" / "Eden.bat"
        if desktop_batch.exists():
            desktop_batch.unlink()
        
        # Remove Start Menu shortcuts
        start_menu_eden = Path.home() / "AppData" / "Roaming" / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Eden"
        if start_menu_eden.exists():
            shutil.rmtree(start_menu_eden)
        
        # Remove registry entry
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                               r"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall", 
                               0, winreg.KEY_ALL_ACCESS)
            winreg.DeleteKey(key, "Eden")
            winreg.CloseKey(key)
        except:
            pass
        
        messagebox.showinfo("Uninstall Complete", "Eden has been successfully removed.")
        
    except Exception as e:
        messagebox.showerror("Uninstall Error", f"Error during uninstall: {{e}}")

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    
    if messagebox.askyesno("Uninstall Eden", "Are you sure you want to remove Eden?"):
        uninstall()
'''

            uninstaller_path = self.install_dir / "uninstall.py"
            uninstaller_path.write_text(uninstaller_content, encoding="utf-8")

            # Register in Windows Registry
            try:
                key = winreg.CreateKey(
                    winreg.HKEY_CURRENT_USER,
                    r"Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\Eden",
                )
                winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, self.app_name)
                winreg.SetValueEx(
                    key, "DisplayVersion", 0, winreg.REG_SZ, self.app_version
                )
                winreg.SetValueEx(key, "Publisher", 0, winreg.REG_SZ, self.publisher)
                winreg.SetValueEx(
                    key, "InstallLocation", 0, winreg.REG_SZ, str(self.install_dir)
                )
                winreg.SetValueEx(
                    key,
                    "UninstallString",
                    0,
                    winreg.REG_SZ,
                    f'"{sys.executable}" "{uninstaller_path}"',
                )
                winreg.SetValueEx(key, "DisplayIcon", 0, winreg.REG_SZ, sys.executable)
                winreg.CloseKey(key)

                self.log("Registered uninstaller")
            except Exception as e:
                self.log(f"Warning: Could not register uninstaller: {e}", "WARNING")

            return True

        except Exception as e:
            self.log(f"Error creating uninstaller: {e}", "WARNING")
            return False

    def installation_thread(self):
        """Main installation thread."""
        try:
            self.update_progress(0, "Starting installation...")

            # Check Python
            self.update_progress(5, "Checking Python installation...")
            if not self.check_python():
                raise Exception("Python not found or incompatible")

            # Create directories
            self.update_progress(10, "Creating directories...")
            self.create_directories()

            # Install packages
            self.update_progress(20, "Installing Python packages...")
            if not self.install_python_packages():
                raise Exception("Failed to install required packages")

            if self.installation_cancelled:
                return

            # Copy application files
            self.update_progress(70, "Copying application files...")
            if not self.copy_application_files():
                raise Exception("Failed to copy application files")

            # Create launcher
            self.update_progress(80, "Creating launcher...")
            if not self.create_launcher_script():
                raise Exception("Failed to create launcher")

            # Create shortcuts
            self.update_progress(90, "Creating shortcuts...")
            self.create_desktop_shortcut()
            self.create_start_menu_shortcut()

            # Register uninstaller
            self.update_progress(95, "Registering application...")
            self.register_uninstaller()

            # Complete
            self.update_progress(100, "Installation complete")
            self.log("Eden installation completed successfully")
            self.installation_complete = True

            # Update UI
            self.install_btn.config(text="Launch Eden", command=self.launch_eden)
            self.cancel_btn.config(text="Close")

        except Exception as e:
            self.log(f"Installation failed: {e}", "ERROR")
            self.update_progress(0, f"Installation failed: {e}")

            # Show error dialog
            if self.root:
                messagebox.showerror(
                    "Installation Error",
                    f"Installation failed: {e}\n\nPlease check the log for details.",
                )

    def start_installation(self):
        """Start the installation process."""
        if self.installation_complete:
            self.launch_eden()
            return

        self.install_btn.config(state="disabled")
        self.installation_cancelled = False

        # Start installation in separate thread
        thread = threading.Thread(target=self.installation_thread, daemon=True)
        thread.start()

    def cancel_installation(self):
        """Cancel the installation."""
        if self.installation_complete:
            self.root.quit()
            return

        if messagebox.askyesno(
            "Cancel Installation", "Are you sure you want to cancel the installation?"
        ):
            self.installation_cancelled = True
            self.root.quit()

    def launch_eden(self):
        """Launch Eden after installation."""
        try:
            launcher_path = self.install_dir / "Eden.py"
            subprocess.Popen(
                [sys.executable, str(launcher_path)], cwd=str(self.install_dir)
            )
            self.root.quit()
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch Eden: {e}")

    def run(self):
        """Run the installer."""
        self.create_ui()
        self.root.mainloop()


def main():
    """Main entry point."""
    installer = EdenInstaller()
    installer.run()


if __name__ == "__main__":
    main()
