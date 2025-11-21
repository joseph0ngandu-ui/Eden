"""
Notification Service for Eden Bot
Handles sending push notifications to iOS devices via APNs.
"""

import time
import jwt
import httpx
import logging
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class NotificationService:
    def __init__(self, key_id: str = None, team_id: str = None, bundle_id: str = "com.eden.ios", p8_path: str = None):
        self.key_id = key_id
        self.team_id = team_id
        self.bundle_id = bundle_id
        self.p8_path = p8_path
        self.token = None
        self.token_expiry = 0
        self.device_tokens = set()
        self.tokens_file = Path("data/device_tokens.json")
        
        # Load saved tokens
        self._load_tokens()

    def _load_tokens(self):
        """Load registered device tokens from disk."""
        if self.tokens_file.exists():
            try:
                with open(self.tokens_file, 'r') as f:
                    data = json.load(f)
                    self.device_tokens = set(data.get('tokens', []))
                logger.info(f"Loaded {len(self.device_tokens)} device tokens")
            except Exception as e:
                logger.error(f"Error loading device tokens: {e}")

    def _save_tokens(self):
        """Save device tokens to disk."""
        try:
            self.tokens_file.parent.mkdir(exist_ok=True)
            with open(self.tokens_file, 'w') as f:
                json.dump({'tokens': list(self.device_tokens)}, f)
        except Exception as e:
            logger.error(f"Error saving device tokens: {e}")

    def register_device(self, token: str):
        """Register a new iOS device token."""
        if token not in self.device_tokens:
            self.device_tokens.add(token)
            self._save_tokens()
            logger.info(f"Registered new device token: {token[:8]}...")

    def _get_jwt_token(self) -> str:
        """Generate JWT token for APNs authentication."""
        if not (self.key_id and self.team_id and self.p8_path):
            return None

        now = time.time()
        if self.token and now < self.token_expiry - 60:
            return self.token

        try:
            with open(self.p8_path, 'r') as f:
                secret = f.read()

            payload = {
                'iss': self.team_id,
                'iat': int(now)
            }
            headers = {
                'alg': 'ES256',
                'kid': self.key_id
            }

            self.token = jwt.encode(payload, secret, algorithm='ES256', headers=headers)
            self.token_expiry = now + 3000  # Token valid for nearly 1 hour
            return self.token
        except Exception as e:
            logger.error(f"Error generating APNs JWT: {e}")
            return None

    async def send_notification(self, title: str, body: str, data: Dict = None):
        """Send push notification to all registered devices."""
        if not self.device_tokens:
            logger.warning("No device tokens registered, skipping notification")
            return

        jwt_token = self._get_jwt_token()
        if not jwt_token:
            logger.warning("APNs credentials not configured, skipping notification")
            return

        headers = {
            'authorization': f'bearer {jwt_token}',
            'apns-topic': self.bundle_id,
            'apns-push-type': 'alert',
            'apns-priority': '10'
        }

        payload = {
            'aps': {
                'alert': {
                    'title': title,
                    'body': body
                },
                'sound': 'default'
            }
        }
        if data:
            payload.update(data)

        async with httpx.AsyncClient(http2=True) as client:
            for device_token in self.device_tokens:
                try:
                    # Use production URL by default, fallback to sandbox if needed
                    url = f"https://api.push.apple.com/3/device/{device_token}"
                    response = await client.post(url, json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        logger.info(f"Notification sent to {device_token[:8]}...")
                    elif response.status_code == 410:
                        logger.info(f"Device token expired, removing: {device_token[:8]}...")
                        self.device_tokens.remove(device_token)
                        self._save_tokens()
                    else:
                        logger.error(f"APNs error {response.status_code}: {response.text}")
                except Exception as e:
                    logger.error(f"Error sending notification: {e}")
