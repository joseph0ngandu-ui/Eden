"""
Eden bootstrapper: checks and installs critical dependencies.
Provides optional minimal UI progress using Tkinter.
"""

import sys
import subprocess
import threading
import time
from typing import List

try:
    import tkinter as tk
    from tkinter import ttk
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

CRITICAL = [
    ("PyQt5", "PyQt5>=5.15.0"),
    ("pyqtgraph", "pyqtgraph>=0.13.0"),
    ("numpy", "numpy>=1.21.0"),
    ("pandas", "pandas>=1.5.0"),
    ("requests", "requests>=2.28.0"),
    ("yfinance", "yfinance>=0.2.0"),
]


def _is_installed(name: str) -> bool:
    try:
        if name == "PyQt5":
            import PyQt5  # noqa: F401
        else:
            __import__(name)
        return True
    except Exception:
        return False


def _pip_install(spec: str) -> bool:
    try:
        r = subprocess.run([sys.executable, "-m", "pip", "install", spec, "--upgrade", "--no-cache-dir"],
                           capture_output=True, text=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


def check_and_install(show_ui: bool = False) -> bool:
    missing = [(n, s) for n, s in CRITICAL if not _is_installed(n)]
    if not missing:
        return True

    if show_ui and TK_AVAILABLE:
        return _install_with_ui(missing)
    else:
        # Console fallback
        for name, spec in missing:
            print(f"Installing {spec} ...")
            ok = _pip_install(spec)
            if not ok:
                print(f"Failed to install {spec}")
                return False
        return True


def _install_with_ui(missing: List[tuple]) -> bool:
    # Minimal Tk progress window
    done = {"result": False}

    def worker():
        total = len(missing)
        for i, (_, spec) in enumerate(missing, 1):
            status.set(f"Installing {spec} ({i}/{total}) ...")
            bar['value'] = int((i-1) / total * 100)
            root.update_idletasks()
            ok = _pip_install(spec)
            if not ok:
                status.set(f"Failed to install {spec}")
                root.update_idletasks()
                done["result"] = False
                root.after(600, root.destroy)
                return
        bar['value'] = 100
        status.set("All dependencies installed. Launching Eden...")
        root.update_idletasks()
        done["result"] = True
        root.after(600, root.destroy)

    root = tk.Tk()
    root.title("Eden — Installing Dependencies")
    root.geometry("480x140")
    root.resizable(False, False)
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill='both', expand=True)
    lbl = ttk.Label(frm, text="Preparing your environment...")
    lbl.pack(anchor='w')
    bar = ttk.Progressbar(frm, length=440, mode='determinate', maximum=100)
    bar.pack(pady=8)
    status = tk.StringVar(value="Checking packages ...")
    st = ttk.Label(frm, textvariable=status)
    st.pack(anchor='w')

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    try:
        root.mainloop()
    except Exception:
        pass
    return bool(done["result"])
