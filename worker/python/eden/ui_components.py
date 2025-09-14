"""
Modern UI components for Eden Trading System.
Provides reusable widgets, charts, and interface elements.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any, List, Optional, Callable
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
from datetime import datetime
import threading


class StatusCard(ttk.Frame):
    """A modern status card widget."""
    
    def __init__(self, parent, title: str, value: str = "", 
                 color: str = "neutral", **kwargs):
        super().__init__(parent, style='Card.TFrame', **kwargs)
        
        self.title_label = ttk.Label(self, text=title, 
                                   style='Card.TLabel', 
                                   font=('Segoe UI', 10, 'bold'))
        self.title_label.pack(anchor='w', padx=12, pady=(8, 2))
        
        self.value_label = ttk.Label(self, text=value,
                                   style='Card.TLabel',
                                   font=('Segoe UI', 16, 'bold'))
        self.value_label.pack(anchor='w', padx=12, pady=(0, 8))
        
        self.update_color(color)
    
    def update_value(self, value: str, color: str = None):
        """Update the card value and optionally color."""
        self.value_label.config(text=value)
        if color:
            self.update_color(color)
    
    def update_color(self, color: str):
        """Update card color based on status."""
        color_map = {
            'profit': '#26A641',
            'loss': '#F85149', 
            'warning': '#D29922',
            'info': '#1F6FEB',
            'neutral': '#7D8590'
        }
        fg_color = color_map.get(color, '#7D8590')
        self.value_label.config(foreground=fg_color)


class ProgressCard(ttk.Frame):
    """A card with progress indicator."""
    
    def __init__(self, parent, title: str, **kwargs):
        super().__init__(parent, style='Card.TFrame', **kwargs)
        
        # Title
        self.title_label = ttk.Label(self, text=title,
                                   style='Card.TLabel',
                                   font=('Segoe UI', 10, 'bold'))
        self.title_label.pack(anchor='w', padx=12, pady=(8, 4))
        
        # Progress bar
        self.progress = ttk.Progressbar(self, mode='determinate', length=200)
        self.progress.pack(fill='x', padx=12, pady=(0, 4))
        
        # Status text
        self.status_label = ttk.Label(self, text="Ready",
                                    style='Card.TLabel',
                                    font=('Segoe UI', 9))
        self.status_label.pack(anchor='w', padx=12, pady=(0, 8))
    
    def update_progress(self, value: float, status: str = ""):
        """Update progress value (0-100) and status text."""
        self.progress['value'] = value
        if status:
            self.status_label.config(text=status)


class TradingChart(ttk.Frame):
    """Interactive trading chart widget using matplotlib."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        self.figure = Figure(figsize=(12, 6), dpi=100, 
                           facecolor='#21262D', edgecolor='none')
        self.axis = self.figure.add_subplot(111)
        self.axis.set_facecolor('#21262D')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Toolbar for zooming/panning
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        
        self.data = None
        
    def plot_ohlcv(self, data: pd.DataFrame, symbol: str = ""):
        """Plot OHLCV candlestick chart."""
        if data is None or data.empty:
            return
            
        self.axis.clear()
        self.data = data
        
        # Simple line chart for now (can be enhanced with candlesticks)
        if 'close' in data.columns:
            self.axis.plot(data.index, data['close'], 
                         color='#1F6FEB', linewidth=1.5, label='Close Price')
        
        self.axis.set_title(f'{symbol} Price Chart' if symbol else 'Price Chart',
                          color='#F0F6FC', fontsize=14, pad=20)
        self.axis.set_xlabel('Date', color='#F0F6FC')
        self.axis.set_ylabel('Price', color='#F0F6FC')
        self.axis.grid(True, alpha=0.3, color='#30363D')
        self.axis.tick_params(colors='#F0F6FC')
        
        if symbol:
            self.axis.legend(facecolor='#161B22', edgecolor='#30363D', 
                           labelcolor='#F0F6FC')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def plot_performance(self, returns: pd.Series, benchmark: pd.Series = None):
        """Plot performance comparison chart."""
        self.axis.clear()
        
        # Plot cumulative returns
        cumulative = (1 + returns).cumprod()
        self.axis.plot(cumulative.index, cumulative.values,
                      color='#26A641', linewidth=2, label='Strategy')
        
        if benchmark is not None:
            cum_benchmark = (1 + benchmark).cumprod()
            self.axis.plot(cum_benchmark.index, cum_benchmark.values,
                          color='#7D8590', linewidth=1.5, label='Benchmark')
        
        self.axis.set_title('Performance Comparison', 
                          color='#F0F6FC', fontsize=14, pad=20)
        self.axis.set_xlabel('Date', color='#F0F6FC')
        self.axis.set_ylabel('Cumulative Return', color='#F0F6FC')
        self.axis.grid(True, alpha=0.3, color='#30363D')
        self.axis.tick_params(colors='#F0F6FC')
        self.axis.legend(facecolor='#161B22', edgecolor='#30363D',
                        labelcolor='#F0F6FC')
        
        self.figure.tight_layout()
        self.canvas.draw()


class LogViewer(ttk.Frame):
    """Enhanced log viewer with filtering and search."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(fill='x', padx=8, pady=(8, 4))
        
        ttk.Label(toolbar, text="Filter:").pack(side='left', padx=(0, 4))
        
        self.filter_var = tk.StringVar(value="ALL")
        filter_combo = ttk.Combobox(toolbar, textvariable=self.filter_var,
                                  values=["ALL", "INFO", "WARNING", "ERROR"],
                                  state="readonly", width=10)
        filter_combo.pack(side='left', padx=(0, 8))
        filter_combo.bind('<<ComboboxSelected>>', self.on_filter_change)
        
        ttk.Button(toolbar, text="Clear", 
                  command=self.clear_logs).pack(side='right')
        
        # Text widget with scrollbar
        text_frame = ttk.Frame(self)
        text_frame.pack(fill='both', expand=True, padx=8, pady=(0, 8))
        
        self.text_widget = tk.Text(text_frame, wrap='word', height=10,
                                 font=('Consolas', 10))
        scrollbar = ttk.Scrollbar(text_frame, orient='vertical',
                                command=self.text_widget.yview)
        self.text_widget.configure(yscrollcommand=scrollbar.set)
        
        self.text_widget.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # Configure text tags for different log levels
        self.text_widget.tag_configure('INFO', foreground='#F0F6FC')
        self.text_widget.tag_configure('WARNING', foreground='#D29922')
        self.text_widget.tag_configure('ERROR', foreground='#F85149')
        self.text_widget.tag_configure('SUCCESS', foreground='#26A641')
        
        self.all_logs = []
    
    def add_log(self, message: str, level: str = "INFO"):
        """Add a log message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'full_text': f"[{timestamp}] {level}: {message}"
        }
        self.all_logs.append(log_entry)
        
        # Apply current filter
        self.refresh_display()
    
    def clear_logs(self):
        """Clear all log messages."""
        self.all_logs.clear()
        self.text_widget.delete(1.0, tk.END)
    
    def on_filter_change(self, event=None):
        """Handle filter change."""
        self.refresh_display()
    
    def refresh_display(self):
        """Refresh the display based on current filter."""
        self.text_widget.delete(1.0, tk.END)
        filter_level = self.filter_var.get()
        
        for log_entry in self.all_logs:
            if filter_level == "ALL" or log_entry['level'] == filter_level:
                self.text_widget.insert(tk.END, log_entry['full_text'] + "\n",
                                      log_entry['level'])
        
        self.text_widget.see(tk.END)


class ConfigForm(ttk.Frame):
    """Dynamic configuration form generator."""
    
    def __init__(self, parent, config_schema: Dict[str, Any], **kwargs):
        super().__init__(parent, style='Panel.TFrame', **kwargs)
        self.config_schema = config_schema
        self.form_vars = {}
        self.create_form()
    
    def create_form(self):
        """Create form fields based on schema."""
        row = 0
        
        for key, config in self.config_schema.items():
            field_type = config.get('type', 'string')
            label_text = config.get('label', key.replace('_', ' ').title())
            default_value = config.get('default', '')
            options = config.get('options', [])
            
            # Label
            ttk.Label(self, text=f"{label_text}:",
                     style='Panel.TLabel').grid(row=row, column=0, 
                                               sticky='w', padx=8, pady=4)
            
            # Input widget based on type
            if field_type == 'boolean':
                var = tk.BooleanVar(value=bool(default_value))
                widget = ttk.Checkbutton(self, variable=var)
            elif field_type == 'choice' and options:
                var = tk.StringVar(value=str(default_value))
                widget = ttk.Combobox(self, textvariable=var, 
                                    values=options, state='readonly')
            elif field_type == 'number':
                var = tk.DoubleVar(value=float(default_value or 0))
                widget = ttk.Entry(self, textvariable=var)
            else:  # string
                var = tk.StringVar(value=str(default_value))
                widget = ttk.Entry(self, textvariable=var)
            
            widget.grid(row=row, column=1, sticky='ew', padx=8, pady=4)
            self.grid_columnconfigure(1, weight=1)
            
            self.form_vars[key] = var
            row += 1
    
    def get_values(self) -> Dict[str, Any]:
        """Get current form values."""
        return {key: var.get() for key, var in self.form_vars.items()}
    
    def set_values(self, values: Dict[str, Any]):
        """Set form values."""
        for key, value in values.items():
            if key in self.form_vars:
                self.form_vars[key].set(value)


class AsyncTaskManager:
    """Manage background tasks with progress updates."""
    
    def __init__(self):
        self.tasks = {}
        self.callbacks = {}
    
    def start_task(self, task_id: str, target: Callable, 
                  progress_callback: Callable = None, 
                  completion_callback: Callable = None):
        """Start a background task."""
        
        def task_wrapper():
            try:
                if progress_callback:
                    self.callbacks[task_id] = progress_callback
                
                result = target()
                
                if completion_callback:
                    completion_callback(result, None)
            except Exception as e:
                if completion_callback:
                    completion_callback(None, e)
            finally:
                if task_id in self.tasks:
                    del self.tasks[task_id]
                if task_id in self.callbacks:
                    del self.callbacks[task_id]
        
        thread = threading.Thread(target=task_wrapper, daemon=True)
        self.tasks[task_id] = thread
        thread.start()
    
    def update_progress(self, task_id: str, progress: float, status: str = ""):
        """Update task progress."""
        if task_id in self.callbacks:
            self.callbacks[task_id](progress, status)
    
    def is_running(self, task_id: str) -> bool:
        """Check if a task is running."""
        return task_id in self.tasks and self.tasks[task_id].is_alive()