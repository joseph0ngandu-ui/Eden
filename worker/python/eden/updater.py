"""
Eden Automatic Update System
Professional update management with GitHub integration
"""

import requests
import os
import subprocess
import tempfile
from pathlib import Path
from packaging import version
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class EdenUpdater:
    """Professional automatic update system for Eden"""

    def __init__(self, current_version: str = "1.0.0"):
        self.current_version = current_version
        self.github_repo = "user/eden"  # Update with actual repo
        self.github_api_url = (
            f"https://api.github.com/repos/{self.github_repo}/releases"
        )
        self.update_check_url = f"{self.github_api_url}/latest"

    def check_for_updates(self) -> Optional[Dict[str, Any]]:
        """Check GitHub for latest release"""
        try:
            response = requests.get(self.update_check_url, timeout=10)
            response.raise_for_status()

            release_data = response.json()
            latest_version = release_data["tag_name"].lstrip("v")

            if version.parse(latest_version) > version.parse(self.current_version):
                return {
                    "version": latest_version,
                    "download_url": self._get_installer_download_url(release_data),
                    "changelog": release_data.get("body", ""),
                    "release_date": release_data.get("published_at", ""),
                    "release_name": release_data.get("name", f"Eden v{latest_version}"),
                }

            return None

        except Exception as e:
            logger.error(f"Update check failed: {e}")
            return None

    def _get_installer_download_url(
        self, release_data: Dict[str, Any]
    ) -> Optional[str]:
        """Extract Windows installer download URL from release assets"""
        for asset in release_data.get("assets", []):
            if asset["name"].endswith("Setup.exe") or "setup" in asset["name"].lower():
                return asset["browser_download_url"]
        return None

    def download_update(
        self, download_url: str, progress_callback=None
    ) -> Optional[str]:
        """Download update installer with progress tracking"""
        try:
            response = requests.get(download_url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            # Create temporary file for download
            temp_dir = tempfile.gettempdir()
            installer_path = os.path.join(temp_dir, "Eden_Update.exe")

            downloaded_size = 0
            with open(installer_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)

                        if progress_callback and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            progress_callback(progress)

            return installer_path

        except Exception as e:
            logger.error(f"Update download failed: {e}")
            return None

    def install_update(self, installer_path: str, silent: bool = True) -> bool:
        """Install downloaded update"""
        try:
            if not os.path.exists(installer_path):
                logger.error(f"Installer not found: {installer_path}")
                return False

            # Prepare installer arguments
            cmd = [installer_path]
            if silent:
                cmd.extend(["/S", "/silent"])  # Silent install flags

            # Launch installer
            process = subprocess.Popen(cmd, shell=True)

            # Don't wait for installer to complete - it will replace this process
            logger.info("Update installer launched successfully")
            return True

        except Exception as e:
            logger.error(f"Update installation failed: {e}")
            return False

    def get_current_version(self) -> str:
        """Get current application version"""
        return self.current_version

    def set_current_version(self, version_str: str):
        """Update current version"""
        self.current_version = version_str


class UpdateNotificationUI:
    """Professional update notification interface"""

    def __init__(self, parent=None):
        self.parent = parent

    def show_update_available(self, update_info: Dict[str, Any]) -> bool:
        """Show update available dialog and return user choice"""
        try:
            from PyQt5.QtWidgets import QMessageBox, QApplication

            if not QApplication.instance():
                return False

            msg = QMessageBox(self.parent)
            msg.setWindowTitle("Eden Update Available")
            msg.setIcon(QMessageBox.Information)

            msg.setText(f"Eden {update_info['version']} is now available!")
            msg.setInformativeText(
                f"Current version: {EdenUpdater().get_current_version()}\n"
                f"New version: {update_info['version']}\n\n"
                "Would you like to download and install the update now?"
            )

            if update_info.get("changelog"):
                msg.setDetailedText(f"What's New:\n{update_info['changelog']}")

            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Later | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.Yes)

            # Apply Apple-style design
            msg.setStyleSheet(
                """
                QMessageBox {
                    background-color: #f8f8f8;
                    color: #333333;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    font-size: 13px;
                }
                QPushButton {
                    background-color: #007AFF;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 500;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #0051D0;
                }
                QPushButton:pressed {
                    background-color: #003C9B;
                }
            """
            )

            result = msg.exec_()
            return result == QMessageBox.Yes

        except ImportError:
            # Fallback to console prompt
            print(f"\n🔄 Eden Update Available!")
            print(f"Current: {EdenUpdater().get_current_version()}")
            print(f"Latest: {update_info['version']}")
            print(
                f"\nChangelog:\n{update_info.get('changelog', 'No changelog available')}"
            )

            choice = input("\nDownload and install update? (y/n): ").strip().lower()
            return choice in ("y", "yes")

    def show_download_progress(self, progress: float):
        """Show download progress"""
        try:
            from PyQt5.QtWidgets import QProgressDialog, QApplication
            from PyQt5.QtCore import Qt

            if hasattr(self, "progress_dialog"):
                self.progress_dialog.setValue(int(progress))
                QApplication.processEvents()
            else:
                self.progress_dialog = QProgressDialog(
                    "Downloading Eden update...", "Cancel", 0, 100, self.parent
                )
                self.progress_dialog.setWindowTitle("Eden Update")
                self.progress_dialog.setWindowModality(Qt.WindowModal)
                self.progress_dialog.show()

        except ImportError:
            # Console fallback
            print(f"\rDownloading update: {progress:.1f}%", end="", flush=True)


# Version management
def get_app_version() -> str:
    """Get current application version from various sources"""

    # Try to get from version file
    version_file = Path(__file__).parent / "version.txt"
    if version_file.exists():
        return version_file.read_text().strip()

    # Try to get from git tags
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
        if result.returncode == 0:
            return result.stdout.strip().lstrip("v")
    except:
        pass

    # Default version
    return "1.0.0"


# Convenience function for easy integration
def check_and_handle_updates(parent_widget=None, auto_check: bool = True) -> bool:
    """
    Complete update check and handling workflow
    Returns True if update was initiated
    """
    if not auto_check:
        return False

    updater = EdenUpdater(get_app_version())
    update_info = updater.check_for_updates()

    if not update_info:
        logger.info("No updates available")
        return False

    ui = UpdateNotificationUI(parent_widget)
    if not ui.show_update_available(update_info):
        return False

    # Download update
    def progress_callback(progress):
        ui.show_download_progress(progress)

    installer_path = updater.download_update(
        update_info["download_url"], progress_callback
    )

    if installer_path:
        # Install update (this will typically exit the current process)
        return updater.install_update(installer_path)

    return False


if __name__ == "__main__":
    # Test the update system
    logging.basicConfig(level=logging.INFO)

    updater = EdenUpdater()
    print(f"Current version: {updater.get_current_version()}")

    update_info = updater.check_for_updates()
    if update_info:
        print(f"Update available: {update_info['version']}")
        print(f"Download URL: {update_info['download_url']}")
    else:
        print("No updates available")
