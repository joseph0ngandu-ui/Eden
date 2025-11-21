import asyncio
import sys
import os
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'backend'))

from app.notifications import NotificationService
from app.routers.strategies import upload_strategy, _load_json, STRATEGIES_FILE

async def test_notification_service():
    print("\nTesting NotificationService...")
    service = NotificationService()
    
    # Test token loading
    print(f"Loaded {len(service.device_tokens)} tokens")
    
    # Test registration
    test_token = "test_device_token_123"
    service.register_device(test_token)
    assert test_token in service.device_tokens
    print("✓ Device registration works")
    
    # Clean up
    if test_token in service.device_tokens:
        service.device_tokens.remove(test_token)
        service._save_tokens()

async def test_strategy_upload():
    print("\nTesting Strategy Upload...")
    
    test_strategy = {
        "id": "test_strat_001",
        "name": "Test Strategy",
        "parameters": {"risk": 0.5}
    }
    
    try:
        result = await upload_strategy(test_strategy)
        print(f"Upload result: {result}")
        
        # Verify it was saved
        saved = _load_json(STRATEGIES_FILE)
        assert "test_strat_001" in saved
        assert saved["test_strat_001"]["source"] == "mac_app_upload"
        print("✓ Strategy upload and save works")
        
        # Clean up
        del saved["test_strat_001"]
        from app.routers.strategies import _save_json
        _save_json(STRATEGIES_FILE, saved)
        
    except Exception as e:
        print(f"❌ Strategy upload failed: {e}")
        raise

async def main():
    try:
        await test_notification_service()
        await test_strategy_upload()
        print("\n✅ All verification tests passed!")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
