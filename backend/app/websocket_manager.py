#!/usr/bin/env python3
"""
WebSocket connection manager for real-time updates
"""

from typing import List, Dict
import logging
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections: List[WebSocket] = []
        self.connection_user_map: Dict[WebSocket, str] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept and store a WebSocket connection."""
        try:
            await websocket.accept()
            self.active_connections.append(websocket)
            if user_id:
                self.connection_user_map[websocket] = user_id
                logger.info(f"WebSocket connected for user: {user_id}")
            else:
                logger.info("WebSocket connected (anonymous)")
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {e}")
            raise
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        try:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            if websocket in self.connection_user_map:
                user_id = self.connection_user_map[websocket]
                del self.connection_user_map[websocket]
                logger.info(f"WebSocket disconnected for user: {user_id}")
            else:
                logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a personal message to a specific WebSocket connection."""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)
    
    async def send_personal_json(self, data: dict, websocket: WebSocket):
        """Send personal JSON data to a specific WebSocket."""
        try:
            import json
            await websocket.send_text(json.dumps(data))
        except Exception as e:
            logger.error(f"Error sending personal JSON: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: str):
        """Broadcast a message to all active WebSocket connections."""
        if not self.active_connections:
            return
        
        connections_to_remove = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                connections_to_remove.append(connection)
        
        # Remove failed connections
        for connection in connections_to_remove:
            self.disconnect(connection)
    
    async def broadcast_json(self, data: dict):
        """Broadcast JSON data to all active WebSocket connections."""
        if not self.active_connections:
            return
        
        connections_to_remove = []
        
        for connection in self.active_connections:
            try:
                import json
                await connection.send_text(json.dumps(data))
            except Exception as e:
                logger.error(f"Error broadcasting JSON: {e}")
                connections_to_remove.append(connection)
        
        # Remove failed connections
        for connection in connections_to_remove:
            self.disconnect(connection)
    
    async def get_connection_count(self) -> int:
        """Get number of active WebSocket connections."""
        return len(self.active_connections)
    
    async def get_connection_info(self) -> Dict:
        """Get information about active connections."""
        return {
            "connections": self.get_connection_count(),
            "users": list(self.connection_user_map.values())
        }