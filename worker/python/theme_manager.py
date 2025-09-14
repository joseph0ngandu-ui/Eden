#!/usr/bin/env python3
"""
Eden - Apple-Style Theme Manager
Automatic light/dark mode detection based on Windows system theme preferences
"""

import sys
import os
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

try:
    import winreg
except ImportError:
    winreg = None

try:
    from PySide6.QtCore import QObject, Signal, QTimer
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication
    PYSIDE6_AVAILABLE = True
except ImportError:
    PYSIDE6_AVAILABLE = False


class AppleTheme:
    """Apple-inspired theme definitions for light and dark modes."""
    
    # Light Mode Colors (iOS/macOS light theme)
    LIGHT = {
        # Primary colors
        'primary': '#007AFF',           # Apple blue
        'secondary': '#5856D6',         # Purple
        'success': '#34C759',           # Green
        'warning': '#FF9500',           # Orange
        'danger': '#FF3B30',            # Red
        'info': '#5AC8FA',              # Light blue
        
        # Background colors
        'bg_primary': '#FFFFFF',        # Pure white
        'bg_secondary': '#F2F2F7',      # Light gray
        'bg_tertiary': '#FFFFFF',       # Card backgrounds
        'bg_grouped': '#F2F2F7',        # Grouped backgrounds
        'bg_sidebar': '#F9F9F9',        # Sidebar
        
        # Text colors
        'text_primary': '#000000',      # Black
        'text_secondary': '#3C3C43',    # Dark gray
        'text_tertiary': '#3C3C4399',   # Gray (60% opacity)
        'text_quaternary': '#3C3C4360', # Light gray (38% opacity)
        'text_placeholder': '#3C3C434D', # Placeholder (30% opacity)
        
        # Border and separator colors
        'separator': '#C6C6C8',         # Light separator
        'separator_opaque': '#C6C6C8',  # Opaque separator
        'border': '#E5E5E7',            # Very light border
        
        # System colors
        'fill_primary': '#78788033',     # 20% gray
        'fill_secondary': '#78788028',   # 16% gray
        'fill_tertiary': '#7676801F',    # 12% gray
        'fill_quaternary': '#74748014',  # 8% gray
        
        # Trading specific
        'profit': '#34C759',            # Green
        'loss': '#FF3B30',              # Red
        'neutral': '#8E8E93',           # Gray
    }
    
    # Dark Mode Colors (iOS/macOS dark theme)
    DARK = {
        # Primary colors
        'primary': '#0A84FF',           # Lighter blue for dark mode
        'secondary': '#5E5CE6',         # Lighter purple
        'success': '#32D74B',           # Brighter green
        'warning': '#FF9F0A',           # Brighter orange
        'danger': '#FF453A',            # Brighter red
        'info': '#64D2FF',              # Brighter light blue
        
        # Background colors
        'bg_primary': '#000000',        # Pure black
        'bg_secondary': '#1C1C1E',      # Dark gray
        'bg_tertiary': '#2C2C2E',       # Card backgrounds
        'bg_grouped': '#1C1C1E',        # Grouped backgrounds
        'bg_sidebar': '#1A1A1A',        # Sidebar
        
        # Text colors
        'text_primary': '#FFFFFF',      # White
        'text_secondary': '#EBEBF5',    # Light gray
        'text_tertiary': '#EBEBF599',   # Light gray (60% opacity)
        'text_quaternary': '#EBEBF54D', # Light gray (30% opacity)
        'text_placeholder': '#EBEBF54D', # Placeholder
        
        # Border and separator colors
        'separator': '#38383A',         # Dark separator
        'separator_opaque': '#38383A',  # Opaque separator
        'border': '#2C2C2E',            # Dark border
        
        # System colors
        'fill_primary': '#FFFFFF1F',     # 12% white
        'fill_secondary': '#FFFFFF19',   # 10% white
        'fill_tertiary': '#FFFFFF14',    # 8% white
        'fill_quaternary': '#FFFFFF0F',  # 6% white
        
        # Trading specific
        'profit': '#32D74B',            # Bright green
        'loss': '#FF453A',              # Bright red
        'neutral': '#98989D',           # Light gray
    }


class WindowsThemeDetector:
    """Windows system theme detection."""
    
    @staticmethod
    def get_system_theme() -> str:
        """Detect Windows system theme preference."""
        if not winreg:
            return 'light'  # Default fallback
        
        try:
            # Check Windows 10/11 dark mode setting
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                # AppsUseLightTheme: 0 = dark mode, 1 = light mode
                apps_use_light_theme, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
                
                if apps_use_light_theme == 0:
                    return 'dark'
                else:
                    return 'light'
                    
        except (FileNotFoundError, OSError, WindowsError):
            # Fallback: check system theme
            try:
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                    system_uses_light_theme, _ = winreg.QueryValueEx(key, "SystemUsesLightTheme")
                    return 'light' if system_uses_light_theme == 1 else 'dark'
            except:
                pass
        
        return 'light'  # Default fallback
    
    @staticmethod
    def get_accent_color() -> Optional[str]:
        """Get Windows accent color."""
        if not winreg:
            return None
            
        try:
            key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Accent"
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                accent_color, _ = winreg.QueryValueEx(key, "AccentColorMenu")
                
                # Convert DWORD to hex color
                r = (accent_color >> 0) & 0xFF
                g = (accent_color >> 8) & 0xFF
                b = (accent_color >> 16) & 0xFF
                
                return f"#{r:02x}{g:02x}{b:02x}"
                
        except (FileNotFoundError, OSError, WindowsError):
            pass
            
        return None


class ThemeManager(QObject if PYSIDE6_AVAILABLE else object):
    """Apple-style theme manager with automatic system theme detection."""
    
    # Signals for theme changes (only if PySide6 available)
    if PYSIDE6_AVAILABLE:
        theme_changed = Signal(str)  # 'light' or 'dark'
        colors_updated = Signal(dict)  # Updated color dictionary
    
    def __init__(self):
        if PYSIDE6_AVAILABLE:
            super().__init__()
        
        self._current_theme = None
        self._current_colors = {}
        self._theme_detector = WindowsThemeDetector()
        
        # Monitor for theme changes
        if PYSIDE6_AVAILABLE:
            self._monitor_timer = QTimer()
            self._monitor_timer.timeout.connect(self._check_theme_change)
            self._monitor_timer.start(1000)  # Check every second
        
        # Initialize theme
        self.update_theme()
    
    def get_current_theme(self) -> str:
        """Get the current theme ('light' or 'dark')."""
        return self._current_theme or 'light'
    
    def get_colors(self) -> Dict[str, str]:
        """Get current theme colors."""
        return self._current_colors.copy()
    
    def get_color(self, color_name: str, fallback: str = '#000000') -> str:
        """Get a specific color by name."""
        return self._current_colors.get(color_name, fallback)
    
    def update_theme(self) -> bool:
        """Update theme based on system settings. Returns True if theme changed."""
        system_theme = self._theme_detector.get_system_theme()
        
        if system_theme != self._current_theme:
            self._current_theme = system_theme
            
            # Update colors based on theme
            if system_theme == 'dark':
                self._current_colors = AppleTheme.DARK.copy()
            else:
                self._current_colors = AppleTheme.LIGHT.copy()
            
            # Try to use system accent color for primary
            accent_color = self._theme_detector.get_accent_color()
            if accent_color:
                self._current_colors['primary'] = accent_color
            
            # Emit signals if PySide6 available
            if PYSIDE6_AVAILABLE:
                self.theme_changed.emit(system_theme)
                self.colors_updated.emit(self._current_colors)
            
            return True
        
        return False
    
    def _check_theme_change(self):
        """Internal method to check for theme changes."""
        self.update_theme()
    
    def get_qt_palette(self) -> Optional['QPalette']:
        """Generate Qt palette for current theme."""
        if not PYSIDE6_AVAILABLE:
            return None
        
        palette = QPalette()
        colors = self._current_colors
        
        # Window colors
        palette.setColor(QPalette.Window, QColor(colors['bg_primary']))
        palette.setColor(QPalette.WindowText, QColor(colors['text_primary']))
        
        # Base colors
        palette.setColor(QPalette.Base, QColor(colors['bg_tertiary']))
        palette.setColor(QPalette.AlternateBase, QColor(colors['bg_secondary']))
        
        # Text colors
        palette.setColor(QPalette.Text, QColor(colors['text_primary']))
        palette.setColor(QPalette.BrightText, QColor(colors['text_primary']))
        palette.setColor(QPalette.PlaceholderText, QColor(colors['text_placeholder']))
        
        # Button colors
        palette.setColor(QPalette.Button, QColor(colors['primary']))
        palette.setColor(QPalette.ButtonText, QColor('#FFFFFF' if self._current_theme == 'light' else '#FFFFFF'))
        
        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(colors['primary']))
        palette.setColor(QPalette.HighlightedText, QColor('#FFFFFF'))
        
        return palette
    
    def get_stylesheet_variables(self) -> Dict[str, str]:
        """Get CSS variable definitions for stylesheets."""
        colors = self._current_colors
        variables = {}
        
        for key, value in colors.items():
            css_key = key.replace('_', '-')
            variables[f'--{css_key}'] = value
        
        return variables
    
    def apply_to_application(self, app: Optional['QApplication'] = None):
        """Apply theme to QApplication."""
        if not PYSIDE6_AVAILABLE:
            return
        
        if app is None:
            app = QApplication.instance()
        
        if app is not None:
            palette = self.get_qt_palette()
            if palette:
                app.setPalette(palette)


# Global theme manager instance
_theme_manager = None

def get_theme_manager() -> ThemeManager:
    """Get global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager

def get_current_theme() -> str:
    """Get current system theme."""
    return get_theme_manager().get_current_theme()

def get_theme_colors() -> Dict[str, str]:
    """Get current theme colors."""
    return get_theme_manager().get_colors()

def get_theme_color(color_name: str, fallback: str = '#000000') -> str:
    """Get a specific theme color."""
    return get_theme_manager().get_color(color_name, fallback)


if __name__ == "__main__":
    # Test theme detection
    theme_manager = ThemeManager()
    print(f"System theme: {theme_manager.get_current_theme()}")
    print(f"Colors: {theme_manager.get_colors()}")
    
    if PYSIDE6_AVAILABLE:
        app = QApplication(sys.argv)
        theme_manager.apply_to_application(app)
        print("Applied theme to QApplication")