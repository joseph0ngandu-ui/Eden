import json
from pathlib import Path
import os

data_dir = Path(os.getcwd()) / 'data'
strategies_file = data_dir / 'strategies.json'

data = {
  "volatility_burst": {
    "id": "volatility_burst",
    "name": "Volatility Burst v1.3",
    "type": "momentum",
    "is_active": True,
    "mode": "LIVE",
    "parameters": {}
  },
  "ict_silver_bullet": {
    "id": "ict_silver_bullet",
    "name": "ICT Silver Bullet",
    "type": "ict",
    "is_active": True,
    "mode": "LIVE",
    "parameters": {}
  },
  "ict_unicorn": {
    "id": "ict_unicorn",
    "name": "ICT Unicorn",
    "type": "ict",
    "is_active": True,
    "mode": "LIVE",
    "parameters": {}
  },
  "ict_venom": {
    "id": "ict_venom",
    "name": "ICT Venom",
    "type": "ict",
    "is_active": True,
    "mode": "LIVE",
    "parameters": {}
  }
}

with open(strategies_file, 'w') as f:
    json.dump(data, f, indent=2)

print("Fixed strategies.json")
