import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import websockets
from deriv_api import DerivAPI
from app.settings import Settings

logger = logging.getLogger(__name__)

class DerivClient:
    """
    Async client for interacting with Deriv API via WebSockets.
    Handles authentication, historical data fetching, and real-time tick subscriptions.
    """
    
    def __init__(self):
        self.settings = Settings()
        self.app_id = self.settings.DERIV_APP_ID
        self.api_token = self.settings.DERIV_API_TOKEN
        self.api: Optional[DerivAPI] = None
        self.connection = None
        self.is_connected = False

    async def connect(self):
        """Establish WebSocket connection to Deriv."""
        if self.is_connected:
            return

        try:
            # Use the official deriv-api library or raw websockets if needed
            # Here we use raw websockets for better control over the event loop
            ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id={self.app_id}"
            self.connection = await websockets.connect(ws_url)
            self.is_connected = True
            logger.info("Connected to Deriv WebSocket")
            
            # Authenticate if token is present
            if self.api_token:
                await self.authenticate()
                
        except Exception as e:
            logger.error(f"Failed to connect to Deriv: {e}")
            self.is_connected = False
            raise

    async def authenticate(self):
        """Authenticate using the API token."""
        if not self.connection:
            return

        auth_req = {"authorize": self.api_token}
        await self.connection.send(json.dumps(auth_req))
        response = await self.connection.recv()
        data = json.loads(response)
        
        if "error" in data:
            logger.error(f"Deriv authentication failed: {data['error']['message']}")
            raise Exception(f"Auth failed: {data['error']['message']}")
            
        logger.info("Deriv authentication successful")

    async def get_history(self, symbol: str, style: str = "candles", 
                         interval: str = "1m", count: int = 1000) -> List[Dict]:
        """
        Fetch historical data (candles or ticks).
        
        Args:
            symbol: The asset symbol (e.g., "R_75", "frxXAUUSD")
            style: "candles" or "ticks"
            interval: Candle interval (e.g., "60" for 1m, "3600" for 1h) - Deriv uses seconds
            count: Number of data points
        """
        if not self.is_connected:
            await self.connect()

        # Map common intervals to Deriv seconds
        interval_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }
        granularity = interval_map.get(interval, 60)

        req = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "style": style,
            "granularity": granularity if style == "candles" else None
        }

        await self.connection.send(json.dumps(req))
        
        # Wait for response
        while True:
            response = await self.connection.recv()
            data = json.loads(response)
            
            if "error" in data:
                logger.error(f"Error fetching history for {symbol}: {data['error']['message']}")
                return []
                
            if "candles" in data:
                return data["candles"]
            elif "history" in data:
                # Format tick history to match candle structure roughly
                history = data["history"]
                return [
                    {"epoch": t, "quote": p} 
                    for t, p in zip(history["times"], history["prices"])
                ]

    async def get_active_symbols(self) -> List[Dict]:
        """Fetch list of active symbols."""
        if not self.is_connected:
            await self.connect()
            
        req = {"active_symbols": "brief", "product_type": "basic"}
        await self.connection.send(json.dumps(req))
        
        response = await self.connection.recv()
        data = json.loads(response)
        
        if "active_symbols" in data:
            return data["active_symbols"]
        return []

    async def disconnect(self):
        """Close the WebSocket connection."""
        if self.connection:
            await self.connection.close()
            self.is_connected = False
            logger.info("Disconnected from Deriv WebSocket")
