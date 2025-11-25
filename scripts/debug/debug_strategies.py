import json
from pathlib import Path
import os

data_dir = Path(os.getcwd()) / 'data'
strategies_file = data_dir / 'strategies.json'

print(f"Checking file: {strategies_file}")
if strategies_file.exists():
    print("File exists.")
    try:
        content = strategies_file.read_text()
        print(f"Content length: {len(content)}")
        print(f"Content preview: {content[:100]}")
        data = json.loads(content)
        print(f"Parsed JSON keys: {list(data.keys())}")
    except Exception as e:
        print(f"Error reading/parsing: {e}")
else:
    print("File does not exist.")
