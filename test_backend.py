#!/usr/bin/env python3
"""
Test script to verify backend can start without errors.
"""
import sys
import os

# Change to backend directory
os.chdir(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    print("Testing backend imports...")
    
    from app.models import BotStatus, Trade
    print("✓ Models imported")
    
    from app.auth import create_access_token
    print("✓ Auth imported")
    
    from app.database import init_db
    print("✓ Database imported")
    
    # Initialize database
    print("✓ Initializing database...")
    init_db()
    print("✓ Database initialized")
    
    # Import main app
    from main import app
    print("✓ FastAPI app imported")
    
    print("\n✓ Backend is ready to start!")
    print("To start the backend server, run:")
    print("  cd C:\\Users\\Administrator\\Eden\\backend")
    print("  python -m uvicorn main:app --host 0.0.0.0 --port 8000")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
