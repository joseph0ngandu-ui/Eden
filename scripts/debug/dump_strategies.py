from pathlib import Path
import os

data_dir = Path(os.getcwd()) / 'data'
strategies_file = data_dir / 'strategies.json'

if strategies_file.exists():
    print(strategies_file.read_text())
