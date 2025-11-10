#!/usr/bin/env python3
"""
JWT authentication and authorization module for Eden Trading Bot API
"""

from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.settings import settings
from app.models import Token
from app.db_models import User
from app.database import get_db_session

# Constants
TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
ALGORITHM = "HS256"

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer scheme for token authentication
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    db_session = get_db_session()
    
    # Get user from database
    user = db_session.query(User).filter(User.email == email).first()
    
    if not user:
        return None
    
    if not verify_password(password, user.hashed_password):
        return None
    
    # Update last login
    user.last_login = datetime.utcnow()
    db_session.commit()
    
    return user

async def get_current_user_http(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Dependency to get current user from HTTP request."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(credentials.credentials)
        if payload is None:
            raise credentials_exception
        
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if email is None or user_id is None:
            raise credentials_exception
    
    except:
        raise credentials_exception
    
    # Get user from database
    db_session = get_db_session()
    user = db_session.query(User).filter(User.id == user_id).first()
    
    if user is None:
        raise credentials_exception
    
    return user

# Alias for most common usage pattern
get_current_user = get_current_user_http

async def get_current_user_ws(websocket: WebSocket, token: str) -> Optional[User]:
    """Get current user for WebSocket connections."""
    try:
        payload = verify_token(token)
        if payload is None:
            return None
        
        email: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if email is None or user_id is None:
            return None
        
        # Get user from database
        db_session = get_db_session()
        user = db_session.query(User).filter(User.id == user_id).first()
        
        return user
    
    except:
        return None

def check_permission(user: User, required_role: str = "user") -> bool:
    """Check if user has required role/permission."""
    # Simple implementation - can be extended for complex permission systems
    if required_role == "admin" and user.email != "admin@eden.com":
        return False
    
    return True

def require_role(required_role: str = "user"):
    """Decorator to require specific role."""
    def role_checker(current_user: User = Depends(get_current_user)):
        if not check_permission(current_user, required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f" insufficient permissions for {required_role} access"
            )
        return current_user
    
    return role_checker