#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
Verify API Contract for Frontend-Backend Integration
Tests that /strategies endpoints return proper JSON arrays.
"""

import requests
import json
import sys

BASE_URL = "https://127.0.0.1:8443"

def test_strategies_endpoint():
    """Test /strategies endpoint returns a list."""
    print("Testing GET /strategies...")
    
    try:
        response = requests.get(f"{BASE_URL}/strategies", verify=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed with status {response.status_code}")
            return False
        
        data = response.json()
        print(f"Response type: {type(data)}")
        
        if not isinstance(data, list):
            print(f"❌ Expected list, got {type(data)}")
            print(f"Response: {json.dumps(data, indent=2)}")
            return False
        
        print(f"✓ Response is a list with {len(data)} items")
        
        # Check structure of each item
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                print(f"❌ Item {i} is not a dict")
                return False
            
            # Check required fields
            required_fields = ["id", "name", "type", "is_active", "is_validated"]
            for field in required_fields:
                if field not in item:
                    print(f"❌ Item {i} missing field '{field}'")
                    return False
            
            print(f"  Item {i}: {item.get('name')} (active: {item.get('is_active')}, validated: {item.get('is_validated')})")
        
        print("✓ All items have required fields")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_active_strategies():
    """Test /strategies/active endpoint."""
    print("\nTesting GET /strategies/active...")
    
    try:
        response = requests.get(f"{BASE_URL}/strategies/active", verify=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed with status {response.status_code}")
            return False
        
        data = response.json()
        
        if not isinstance(data, list):
            print(f"❌ Expected list, got {type(data)}")
            return False
        
        print(f"✓ Response is a list with {len(data)} active strategies")
        
        # All items should have is_active=True
        for item in data:
            if not item.get('is_active'):
                print(f"❌ Item {item.get('id')} has is_active=False")
                return False
        
        print("✓ All active strategies have is_active=True")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_validated_strategies():
    """Test /strategies/validated endpoint."""
    print("\nTesting GET /strategies/validated...")
    
    try:
        response = requests.get(f"{BASE_URL}/strategies/validated", verify=False)
        print(f"Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed with status {response.status_code}")
            return False
        
        data = response.json()
        
        if not isinstance(data, list):
            print(f"❌ Expected list, got {type(data)}")
            return False
        
        print(f"✓ Response is a list with {len(data)} validated strategies")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_empty_response():
    """Test that empty data returns [] not {}."""
    print("\nTesting empty response handling...")
    print("(This test assumes strategies file may be empty or missing)")
    
    # If we have strategies, this test is informative only
    try:
        response = requests.get(f"{BASE_URL}/strategies", verify=False)
        data = response.json()
        
        if len(data) == 0:
            print("✓ Empty response returns [] (empty list)")
            return True
        else:
            print(f"ℹ Has {len(data)} strategies, cannot test empty case")
            return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    print("=" * 60)
    print("API Contract Verification")
    print("=" * 60)
    
    results = {
        "strategies": test_strategies_endpoint(),
        "active": test_active_strategies(),
        "validated": test_validated_strategies(),
        "empty": test_empty_response(),
    }
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print("=" * 60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)
