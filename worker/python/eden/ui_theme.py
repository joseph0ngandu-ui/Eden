import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any

# Modern dark theme optimized for trading applications
DEFAULT_THEME = {
    "window_size": "1200x800",
    "min_window_size": "800x600",
    "padding": 16,
    "font_family": "Segoe UI",
    "header_font_size": 24,
    "subheader_font_size": 16,
    "body_font_size": 11,
    "small_font_size": 9,
    
    # Color scheme - Professional dark theme
    "bg": "#0D1117",           # Primary background
    "fg": "#F0F6FC",           # Primary text
    "panel_bg": "#161B22",     # Panel/card background
    "card_bg": "#21262D",      # Card background
    "border": "#30363D",       # Border color
    "hover_bg": "#262C36",     # Hover states
    
    # Accent colors
    "accent_primary": "#238636", # Success/profit green
    "accent_secondary": "#1F6FEB", # Info blue
    "accent_warning": "#D29922",   # Warning orange
    "accent_danger": "#DA3633",    # Danger/loss red
    "accent_purple": "#8957E5",    # Purple accent
    
    # Trading specific colors
    "profit_color": "#26A641",
    "loss_color": "#F85149",
    "neutral_color": "#7D8590",
    
    # Input elements
    "input_bg": "#0D1117",
    "input_border": "#30363D",
    "input_focus": "#1F6FEB",
    "input_error": "#F85149",
    
    # Special elements
    "scrollbar": "#484F58",
    "scrollbar_hover": "#6E7681",
    "selection": "#1F6FEB40",
    "inactive": "#484F58",
}

THEME_PATHS = [
    Path("UI design.json"),             # top-level file the user can edit
    Path("eden/ui_theme.json"),        # package default override
]


def load_theme() -> dict:
    # Load first found theme file, otherwise default
    for p in THEME_PATHS:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return DEFAULT_THEME.copy()


def apply_theme(root: tk.Tk, theme: Dict[str, Any], **kwargs):
    """Apply comprehensive theming to the application."""
    try:
        style = ttk.Style(root)
        
        # Use a clean base theme
        try:
            available_themes = style.theme_names()
            if 'clam' in available_themes:
                style.theme_use('clam')
            elif 'vista' in available_themes:
                style.theme_use('vista')
        except Exception:
            pass
        
        # Extract colors
        bg = theme.get("bg", "#0D1117")
        fg = theme.get("fg", "#F0F6FC")
        panel_bg = theme.get("panel_bg", "#161B22")
        card_bg = theme.get("card_bg", "#21262D")
        border = theme.get("border", "#30363D")
        hover_bg = theme.get("hover_bg", "#262C36")
        accent_primary = theme.get("accent_primary", "#238636")
        accent_secondary = theme.get("accent_secondary", "#1F6FEB")
        accent_danger = theme.get("accent_danger", "#DA3633")
        input_bg = theme.get("input_bg", "#0D1117")
        input_border = theme.get("input_border", "#30363D")
        input_focus = theme.get("input_focus", "#1F6FEB")
        
        # Configure root window
        root.configure(bg=bg)
        
        # Configure basic elements
        style.configure('TFrame', background=bg, borderwidth=0)
        style.configure('TLabel', background=bg, foreground=fg, 
                       font=(theme.get("font_family", "Segoe UI"), theme.get("body_font_size", 11)))
        
        # Configure panels and cards
        style.configure('Panel.TFrame', background=panel_bg, relief='flat', borderwidth=1)
        style.configure('Card.TFrame', background=card_bg, relief='flat', borderwidth=1)
        style.configure('Panel.TLabel', background=panel_bg, foreground=fg)
        style.configure('Card.TLabel', background=card_bg, foreground=fg)
        
        # Configure headers
        style.configure('Header.TLabel', background=bg, foreground=fg, 
                       font=(theme.get("font_family", "Segoe UI"), theme.get("header_font_size", 24), 'bold'))
        style.configure('Subheader.TLabel', background=bg, foreground=fg,
                       font=(theme.get("font_family", "Segoe UI"), theme.get("subheader_font_size", 16), 'bold'))
        
        # Configure buttons
        style.configure('TButton', 
                       background=accent_primary,
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       font=(theme.get("font_family", "Segoe UI"), theme.get("body_font_size", 11), 'bold'))
        style.map('TButton',
                 background=[('active', hover_bg), ('pressed', accent_primary)],
                 foreground=[('active', 'white')])
        
        # Configure secondary buttons
        style.configure('Secondary.TButton',
                       background=accent_secondary,
                       foreground='white')
        style.map('Secondary.TButton',
                 background=[('active', hover_bg)])
        
        # Configure danger buttons
        style.configure('Danger.TButton',
                       background=accent_danger,
                       foreground='white')
        
        # Configure input elements
        style.configure('TEntry',
                       fieldbackground=input_bg,
                       background=input_bg,
                       foreground=fg,
                       bordercolor=input_border,
                       insertcolor=fg,
                       borderwidth=1,
                       relief='solid')
        style.map('TEntry',
                 bordercolor=[('focus', input_focus)],
                 background=[('focus', input_bg)])
        
        style.configure('TCombobox',
                       fieldbackground=input_bg,
                       background=panel_bg,
                       foreground=fg,
                       bordercolor=input_border,
                       arrowcolor=fg,
                       borderwidth=1,
                       relief='solid')
        style.map('TCombobox',
                 bordercolor=[('focus', input_focus)],
                 fieldbackground=[('focus', input_bg)])
        
        # Configure radio buttons and checkboxes
        style.configure('TRadiobutton',
                       background=bg,
                       foreground=fg,
                       focuscolor='none',
                       borderwidth=0)
        style.map('TRadiobutton',
                 background=[('active', hover_bg)])
        
        style.configure('TCheckbutton',
                       background=bg,
                       foreground=fg,
                       focuscolor='none',
                       borderwidth=0)
        style.map('TCheckbutton',
                 background=[('active', hover_bg)])
        
        # Configure notebook (tabs)
        style.configure('TNotebook', background=bg, borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=panel_bg,
                       foreground=fg,
                       padding=[20, 10],
                       borderwidth=0)
        style.map('TNotebook.Tab',
                 background=[('selected', accent_secondary), ('active', hover_bg)],
                 foreground=[('selected', 'white')])
        
        # Configure separators
        style.configure('TSeparator', background=border)
        
        # Configure progressbar
        style.configure('TProgressbar',
                       background=accent_primary,
                       troughcolor=panel_bg,
                       borderwidth=0,
                       lightcolor=accent_primary,
                       darkcolor=accent_primary)
        
        # Configure scrollbars
        style.configure('TScrollbar',
                       background=panel_bg,
                       troughcolor=bg,
                       borderwidth=0,
                       arrowcolor=fg)
        
        # Apply to text widgets if provided
        for widget_name, widget in kwargs.items():
            if isinstance(widget, tk.Text):
                widget.configure(
                    bg=card_bg,
                    fg=fg,
                    insertbackground=fg,
                    selectbackground=input_focus + '40',
                    selectforeground=fg,
                    borderwidth=1,
                    relief='solid',
                    font=(theme.get("font_family", "Segoe UI"), theme.get("body_font_size", 11))
                )
        
    except Exception as e:
        print(f"Theme application error: {e}")
        pass
