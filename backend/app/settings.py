#!/usr/bin/env python3
"""
Application settings and configuration for Eden Trading Bot API
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "Eden Trading Bot API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Security
    secret_key: str = os.getenv("SECRET_KEY", "EDEN_SECRET_CHANGE_FOR_PRODUCTION")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440  # 24 hours
    
    # Database
    database_url: str = os.getenv(
        "DATABASE_URL", 
        "sqlite:///./eden_trading.db"
    )
    
    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    # Optional direct TLS (only if terminating TLS in app instead of a reverse proxy)
    ssl_certfile: Optional[str] = os.getenv("SSL_CERTFILE")
    ssl_keyfile: Optional[str] = os.getenv("SSL_KEYFILE")
    
    # CORS
    allowed_origins: list = ["*"]  # Default in development, restrict in production
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Trading Bot
    @property
    def symbols(self) -> list:
        """Parse symbols from environment or use default."""
        symbols_str = os.getenv("SYMBOLS", "EURUSD,GBPUSD")
        if isinstance(symbols_str, str):
            return [s.strip() for s in symbols_str.split(",")]
        return symbols_str if isinstance(symbols_str, list) else ["EURUSD", "GBPUSD"]
    
    # AWS Configuration (for production)
    aws_region: Optional[str] = os.getenv("AWS_REGION")
    aws_cognito_user_pool_id: Optional[str] = os.getenv("AWS_COGNITO_USER_POOL_ID")
    aws_cognito_client_id: Optional[str] = os.getenv("AWS_COGNITO_CLIENT_ID")
    
    # External APIs
    mt5_server: Optional[str] = os.getenv("MT5_SERVER")
    mt5_account: Optional[int] = os.getenv("MT5_ACCOUNT")
    
    # Deriv API
    DERIV_APP_ID: int = int(os.getenv("DERIV_APP_ID", "1089"))  # Default to public app ID
    DERIV_API_TOKEN: Optional[str] = os.getenv("DERIV_API_TOKEN")
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env file

# Create global settings instance
settings = Settings()
