#!/usr/bin/env python3
"""
Comprehensive API endpoint tester for Eden Trading Bot
Tests all 40 endpoints and generates a detailed report
"""

import requests
import json
import urllib3
from datetime import datetime

# Disable SSL warnings for local testing
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE_URL = "http://localhost:8000"
TOKEN = None  # Will be set after login

def log(message, level="INFO"):
    """Print formatted log message"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {message}")

def test_endpoint(method, endpoint, data=None, auth=False, description=""):
    """Test a single endpoint and return result"""
    url = f"{BASE_URL}{endpoint}"
    headers = {}
    
    if auth and TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, verify=False, timeout=10)
        elif method == "POST":
            headers["Content-Type"] = "application/json"
            response = requests.post(url, json=data, headers=headers, verify=False, timeout=10)
        elif method == "PUT":
            headers["Content-Type"] = "application/json"
            response = requests.put(url, json=data, headers=headers, verify=False, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers, verify=False, timeout=10)
        elif method == "PATCH":
            headers["Content-Type"] = "application/json"
            response = requests.patch(url, json=data, headers=headers, verify=False, timeout=10)
        else:
            return {"status": "ERROR", "message": f"Unsupported method: {method}"}
        
        result = {
            "method": method,
            "endpoint": endpoint,
            "description": description,
            "status_code": response.status_code,
            "status": "✅ PASS" if 200 <= response.status_code < 300 else "❌ FAIL",
            "response_time": f"{response.elapsed.total_seconds():.3f}s"
        }
        
        try:
            result["response"] = response.json()
        except:
            result["response"] = response.text[:200]
        
        return result
    
    except Exception as e:
        return {
            "method": method,
            "endpoint": endpoint,
            "description": description,
            "status": "❌ ERROR",
            "error": str(e)
        }

def main():
    global TOKEN
    results = []
    
    log("=" * 80)
    log("Eden Trading Bot - Comprehensive Endpoint Test")
    log("=" * 80)
    
    # ===== HEALTH CHECK =====
    log("\n🏥 Testing Health & System Endpoints...")
    results.append(test_endpoint("GET", "/health", description="Health check"))
    results.append(test_endpoint("GET", "/info", description="API info"))
    
    # ===== AUTHENTICATION =====
    log("\n🔐 Testing Authentication Endpoints...")
    
    # Try login first (in case user exists)
    login_data = {
        "email": "tester@eden.com",
        "password": "test12345"
    }
    login_result = test_endpoint("POST", "/auth/login-local", login_data, 
                                 description="Login local user")
    
    if login_result["status"] == "✅ PASS":
        results.append(login_result)
        TOKEN = login_result["response"].get("access_token")
        log("✓ Login successful, skipping registration")
    else:
        # Register new user
        register_data = {
            "email": "tester@eden.com",
            "password": "test12345",
            "full_name": "API Tester"
        }
        reg_result = test_endpoint("POST", "/auth/register-local", register_data, 
                                     description="Register local user")
        results.append(reg_result)
        
        # Login after registration
        login_result = test_endpoint("POST", "/auth/login-local", login_data, 
                                     description="Login local user")
        results.append(login_result)
        
        if login_result["status"] == "✅ PASS":
            TOKEN = login_result["response"].get("access_token")

    if TOKEN:
        log(f"✓ JWT Token obtained: {TOKEN[:20]}...")
    else:
        log("❌ Failed to obtain token, subsequent tests will fail", "ERROR")
    
    # ===== BOT CONTROL =====
    log("\n🤖 Testing Bot Control Endpoints...")
    results.append(test_endpoint("GET", "/bot/status", description="Get bot status (public)"))
    results.append(test_endpoint("POST", "/bot/start", auth=True, description="Start bot (requires auth)"))
    results.append(test_endpoint("POST", "/bot/pause", auth=True, description="Pause bot (requires auth)"))
    results.append(test_endpoint("POST", "/bot/stop", auth=True, description="Stop bot (requires auth)"))
    
    # ===== TRADING =====
    log("\n📊 Testing Trading Endpoints...")
    results.append(test_endpoint("GET", "/trades/open", description="Get open positions (public)"))
    results.append(test_endpoint("GET", "/trades/history?limit=10", auth=True, description="Get trade history"))
    results.append(test_endpoint("GET", "/trades/recent?days=7", auth=True, description="Get recent trades"))
    results.append(test_endpoint("GET", "/trades/logs?limit=10", description="Get trade logs"))
    results.append(test_endpoint("POST", "/trades/close?symbol=TEST", auth=True, description="Close position (test)"))
    
    # Test order
    test_order = {
        "symbol": "Volatility 75 Index",
        "side": "BUY",
        "volume": 0.01
    }
    results.append(test_endpoint("POST", "/orders/test", test_order, description="Place test order"))
    
    # ===== PERFORMANCE =====
    log("\n📈 Testing Performance Endpoints...")
    results.append(test_endpoint("GET", "/performance/stats", description="Get performance stats (public)"))
    results.append(test_endpoint("GET", "/performance/equity-curve", auth=True, description="Get equity curve"))
    results.append(test_endpoint("GET", "/performance/daily-summary", auth=True, description="Get daily summary"))
    
    # ===== STRATEGY CONFIGURATION =====
    log("\n🎯 Testing Strategy Configuration Endpoints...")
    results.append(test_endpoint("GET", "/strategy/config", description="Get strategy config (public)"))
    results.append(test_endpoint("GET", "/strategy/symbols", auth=True, description="Get trading symbols"))
    
    # Test symbol update
    symbols_update = {"symbols": ["EURUSD", "GBPUSD"]}
    results.append(test_endpoint("POST", "/symbols/update", symbols_update, auth=True, description="Update symbols"))
    
    # ===== STRATEGY MANAGEMENT =====
    log("\n🧬 Testing Strategy Management Endpoints...")
    results.append(test_endpoint("GET", "/strategies", description="List all strategies"))
    results.append(test_endpoint("GET", "/strategies/validated", description="List validated strategies"))
    results.append(test_endpoint("GET", "/strategies/active", description="List active strategies"))
    
    # Upload a test strategy
    test_strategy = {
        "id": "test_strategy_001",
        "name": "Test ICT Strategy",
        "type": "ICT_ML",
        "parameters": {"confidence_threshold": 0.7, "rr": 3.0}
    }
    results.append(test_endpoint("POST", "/strategies", test_strategy, description="Upload new strategy"))
    
    # ===== MT5 ACCOUNT MANAGEMENT =====
    log("\n💼 Testing MT5 Account Endpoints...")
    results.append(test_endpoint("GET", "/account/mt5", auth=True, description="List MT5 accounts"))
    results.append(test_endpoint("GET", "/account/mt5/primary", auth=True, description="Get primary MT5 account"))
    
    # Create test MT5 account
    mt5_account = {
        "account_number": "99999999",
        "account_name": "Test Account",
        "broker": "Deriv",
        "server": "Deriv-Demo",
        "password": "testpass",
        "is_primary": False
    }
    create_result = test_endpoint("POST", "/account/mt5", mt5_account, auth=True, 
                                 description="Create MT5 account")
    results.append(create_result)
    
    if create_result["status"] == "✅ PASS":
        acc_id = create_result["response"].get("id")
        if acc_id:
            # Test set primary
            results.append(test_endpoint("PUT", f"/account/mt5/{acc_id}/primary", auth=True, description="Set primary account"))
            # Test delete
            results.append(test_endpoint("DELETE", f"/account/mt5/{acc_id}", auth=True, description="Delete account"))

    # Test paper reset
    results.append(test_endpoint("POST", "/account/paper/reset", auth=True, description="Reset paper account"))
    
    # ===== SYSTEM =====
    log("\n⚙️ Testing System Endpoints...")
    results.append(test_endpoint("GET", "/system/status", auth=True, description="Get system status"))
    
    # ===== DEVICE REGISTRATION =====
    log("\n🔔 Testing Device Registration...")
    device_data = {"token": "test_device_token_12345"}
    results.append(test_endpoint("POST", "/device/register", device_data, auth=True, 
                                 description="Register device for notifications"))
    
    # ===== GENERATE REPORT =====
    log("\n" + "=" * 80)
    log("📋 TEST SUMMARY")
    log("=" * 80)
    
    passed = sum(1 for r in results if r["status"] == "✅ PASS")
    failed = sum(1 for r in results if r["status"] == "❌ FAIL")
    errors = sum(1 for r in results if r["status"] == "❌ ERROR")
    total = len(results)
    
    log(f"\nTotal Endpoints Tested: {total}")
    log(f"✅ Passed: {passed}")
    log(f"❌ Failed: {failed}")
    log(f"❌ Errors: {errors}")
    log(f"Success Rate: {(passed/total*100):.1f}%\n")
    
    # Detailed results
    for r in results:
        status_icon = r["status"]
        log(f"{status_icon} {r['method']:6s} {r['endpoint']:40s} {r.get('description', '')}")
        if r["status"] != "✅ PASS":
            if "error" in r:
                log(f"       Error: {r['error']}", "ERROR")
            elif "status_code" in r:
                log(f"       HTTP {r['status_code']}: {r.get('response', '')}", "WARN")
    
    # Save detailed JSON report
    report_file = "../test_results.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log(f"\n📄 Detailed report saved to: {report_file}")
    
    log("\n" + "=" * 80)
    log("✓ Testing Complete")
    log("=" * 80)

if __name__ == "__main__":
    main()
