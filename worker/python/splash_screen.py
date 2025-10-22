#!/usr/bin/env python3
"""
Eden Bot - Splash Screen
Professional loading screen with inspirational quotes
"""

import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel, 
                               QProgressBar, QGraphicsDropShadowEffect)
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor

# Import Apple-style theme manager
try:
    from theme_manager import get_theme_colors, get_theme_color
    THEME_MANAGER_AVAILABLE = True
except ImportError:
    THEME_MANAGER_AVAILABLE = False

class SplashScreen(QWidget):
    """Apple-style professional splash screen with motivational quotes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_quote_index = 0
        self.current_loading_index = 0
        self.init_ui()
        self.setup_animations()
        self.start_loading_sequence()
        
    def init_ui(self):
        """Initialize the user interface with Apple design principles"""
        self.setFixedSize(500, 350)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Get theme colors
        if THEME_MANAGER_AVAILABLE:
            colors = get_theme_colors()
            bg_color = colors.get('bg_primary', '#FFFFFF')
            bg_secondary = colors.get('bg_secondary', '#F2F2F7')
            border_color = colors.get('border', '#E5E5E7')
        else:
            bg_color = '#FFFFFF'
            bg_secondary = '#F2F2F7'
            border_color = '#E5E5E7'
        
        # Center window on screen
        self.center_on_screen()
        
        # Main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main container widget with dynamic theming
        container = QWidget()
        container.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {bg_color}, stop:1 {bg_secondary});
                border-radius: 20px;
                border: 1px solid {border_color};
            }}
        """)
        
        # Add subtle shadow to container
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 25))
        shadow.setOffset(0, 5)
        container.setGraphicsEffect(shadow)
        
        # Container layout
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 40, 30, 30)
        container_layout.setSpacing(20)
        
        # Main title with dynamic theming
        title = QLabel("Eden")
        title.setAlignment(Qt.AlignCenter)
        
        if THEME_MANAGER_AVAILABLE:
            text_color = get_theme_color('text_primary', '#1d1d1f')
        else:
            text_color = '#1d1d1f'
            
        title.setStyleSheet(f"""
            QLabel {{
                font-size: 42px;
                font-weight: 300;
                color: {text_color};
                margin: 20px 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Display', sans-serif;
                letter-spacing: -0.8px;
            }}
        """)
        
        # Subtitle with dynamic theming
        subtitle = QLabel("The Origin of Order")
        subtitle.setAlignment(Qt.AlignCenter)
        
        if THEME_MANAGER_AVAILABLE:
            text_secondary = get_theme_color('text_secondary', '#86868b')
        else:
            text_secondary = '#86868b'
            
        subtitle.setStyleSheet(f"""
            QLabel {{
                font-size: 18px;
                font-weight: 400;
                color: {text_secondary};
                margin-bottom: 30px;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
                letter-spacing: 0.1px;
            }}
        """)
        
        # Progress bar with Apple-style design and dynamic theming
        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        
        if THEME_MANAGER_AVAILABLE:
            primary_color = get_theme_color('primary', '#007aff')
            bg_secondary_color = get_theme_color('bg_secondary', '#e5e5e7')
        else:
            primary_color = '#007aff'
            bg_secondary_color = '#e5e5e7'
            
        self.progress.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: {bg_secondary_color};
                height: 6px;
                margin: 20px 40px;
            }}
            QProgressBar::chunk {{
                background: {primary_color};
                border-radius: 3px;
            }}
        """)
        
        # Loading text with dynamic theming
        self.loading_label = QLabel("Initializing...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        
        self.loading_label.setStyleSheet(f"""
            QLabel {{
                font-size: 14px;
                color: {text_secondary};
                margin: 5px 0 20px 0;
                font-weight: 400;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
            }}
        """)
        
        # Quote display with Apple-style design and dynamic theming
        self.quote_label = QLabel()
        self.quote_label.setAlignment(Qt.AlignCenter)
        self.quote_label.setWordWrap(True)
        self.quote_label.setMinimumHeight(60)
        
        if THEME_MANAGER_AVAILABLE:
            quote_bg = get_theme_color('bg_tertiary', '#FFFFFF')
            quote_border = get_theme_color('border', '#E5E5E7')
        else:
            quote_bg = 'rgba(255, 255, 255, 0.6)'
            quote_border = 'rgba(0, 0, 0, 0.04)'
            
        self.quote_label.setStyleSheet(f"""
            QLabel {{
                font-size: 15px;
                font-weight: 400;
                color: {text_color};
                background-color: {quote_bg};
                border: 1px solid {quote_border};
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
                line-height: 1.4;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'SF Pro Text', sans-serif;
            }}
        """)
        
        # Add widgets to layout
        container_layout.addWidget(title)
        container_layout.addWidget(subtitle)
        container_layout.addWidget(self.progress)
        container_layout.addWidget(self.loading_label)
        container_layout.addWidget(self.quote_label)
        container_layout.addStretch()
        
        # Add container to main layout
        layout.addWidget(container)
        self.setLayout(layout)
        
        # Set initial quote
        self.update_quote()
        
    def center_on_screen(self):
        """Center the splash screen on the screen"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        x = (screen.width() - size.width()) // 2
        y = (screen.height() - size.height()) // 2
        self.move(x, y)
        
    def setup_animations(self):
        """Setup smooth animations for progress and content updates"""
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(300)
        self.fade_animation.setEasingCurve(QEasingCurve.OutCubic)
        
        # Start with fade in
        self.setWindowOpacity(0)
        self.fade_animation.setStartValue(0)
        self.fade_animation.setEndValue(1)
        self.fade_animation.start()
        
    def start_loading_sequence(self):
        """Start the loading sequence with progress updates"""
        self.loading_steps = [
            "Starting up...",
            "Loading components...",
            "Initializing dashboard...",
            "Setting up tools...",
            "Connecting services...",
            "Almost ready...",
            "Ready to trade"
        ]
        
        # Timer for progress updates
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(800)  # Update every 800ms for smooth progress
        
        # Timer for quote updates
        self.quote_timer = QTimer()
        self.quote_timer.timeout.connect(self.update_quote)
        self.quote_timer.start(3500)  # Change quote every 3.5 seconds
        
    def update_progress(self):
        """Update progress bar and loading text"""
        if self.current_loading_index < len(self.loading_steps):
            # Update loading text
            self.loading_label.setText(self.loading_steps[self.current_loading_index])
            
            # Update progress bar
            progress_value = int((self.current_loading_index + 1) * (100 / len(self.loading_steps)))
            self.progress.setValue(progress_value)
            
            self.current_loading_index += 1
        else:
            # Loading complete
            self.progress_timer.stop()
            self.quote_timer.stop()
            QTimer.singleShot(1000, self.fade_out_and_close)
    
    def update_quote(self):
        """Update the motivational quote"""
        quotes = [
            "Success is not final, failure is not fatal: it is the courage to continue that counts.",
            "The way to get started is to quit talking and begin doing.",
            "Don't be afraid to give up the good to go for the great.",
            "Innovation distinguishes between a leader and a follower.",
            "The future belongs to those who believe in the beauty of their dreams.",
            "It is during our darkest moments that we must focus to see the light.",
            "Success is not how high you have climbed, but how you make a positive difference.",
            "Believe you can and you're halfway there.",
            "The only impossible journey is the one you never begin.",
            "In the midst of winter, I found there was, within me, an invincible summer.",
            "Your limitation—it's only your imagination.",
            "Great things never come from comfort zones.",
            "Dream it. Wish it. Do it.",
            "Success doesn't just find you. You have to go out and get it.",
            "The harder you work for something, the greater you'll feel when you achieve it.",
            "Don't stop when you're tired. Stop when you're done.",
            "Wake up with determination. Go to bed with satisfaction.",
            "Do something today that your future self will thank you for.",
            "Little things make big days.",
            "It's going to be hard, but hard does not mean impossible.",
            "Don't wait for opportunity. Create it.",
            "Sometimes we're tested not to show our weaknesses, but to discover our strengths.",
            "The key to success is to focus on goals, not obstacles.",
            "Dream bigger. Do bigger.",
            "Don't be pushed around by the fears in your mind. Be led by the dreams in your heart.",
            "Work hard in silence, let your success be the noise.",
            "The expert in anything was once a beginner.",
            "You are never too old to set another goal or to dream a new dream.",
            "If you want to achieve greatness, stop asking for permission.",
            "A year from now you may wish you had started today.",
            "You don't have to be great to get started, but you have to get started to be great.",
            "The only person you are destined to become is the person you decide to be."
        ]
        
        if self.current_quote_index >= len(quotes):
            self.current_quote_index = 0
            
        quote = quotes[self.current_quote_index]
        self.quote_label.setText(f'"{quote}"')
        self.current_quote_index += 1
        
    def fade_out_and_close(self):
        """Fade out and close the splash screen"""
        self.fade_animation.setStartValue(1)
        self.fade_animation.setEndValue(0)
        self.fade_animation.finished.connect(self.close)
        self.fade_animation.start()

def show_splash_screen():
    """Show the splash screen"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    splash = SplashScreen()
    splash.show()
    
    return splash, app

if __name__ == "__main__":
    splash, app = show_splash_screen()
    sys.exit(app.exec())