#!/usr/bin/env python3
"""
Database configuration and session management for Eden Trading Bot API
"""

from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Database URL - defaults to SQLite for development
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./eden_trading.db"
)

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

def get_db_session() -> Session:
    """Get database session."""
    session = SessionLocal()
    try:
        return session
    except Exception as e:
        session.close()
        raise e

def init_db():
    """Initialize database with schema and default data."""
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
        # Create default admin user if needed
        session = SessionLocal()
        try:
            # Check if admin user exists
            from app.models import User
            from app.auth import get_password_hash
            
            admin = session.query(User).filter(User.email == "admin@eden.com").first()
            if not admin:
                # Create default admin user
                admin_user = User(
                    email="admin@eden.com",
                    full_name="System Administrator",
                    hashed_password=get_password_hash("admin123"),
                    is_active=True,
                    created_at=datetime.utcnow()
                )
                session.add(admin_user)
                session.commit()
                logger.info("Default admin user created: admin@eden.com / admin123")
            else:
                logger.info("Admin user already exists")
        
        except Exception as e:
            logger.error(f"Error during database initialization: {e}")
            session.rollback()
            raise
        
        finally:
            session.close()
    
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

def close_db_session(session: Session):
    """Close database session."""
    try:
        session.close()
    except Exception as e:
        logger.error(f"Error closing database session: {e}")