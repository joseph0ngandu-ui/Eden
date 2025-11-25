import urllib.request
import json
import time

url = "http://localhost:8000/strategies"

print(f"Querying {url}...")
try:
    with urllib.request.urlopen(url) as response:
        if response.status == 200:
            data = json.loads(response.read().decode())
            print("Response Status: 200 OK")
            print(f"Response Data: {json.dumps(data, indent=2)}")
            if len(data) > 0:
                print("SUCCESS: Strategies found.")
            else:
                print("WARNING: Empty strategy list returned.")
        else:
            print(f"Error: Status {response.status}")
except Exception as e:
    print(f"Error querying API: {e}")
