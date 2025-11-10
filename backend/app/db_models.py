#!/usr/bin/env python3
"""
SQLAlchemy database models for Eden Trading Bot API
"""

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from app.database import Base


class User(Base):
    """User database model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="user")
    bot_sessions = relationship("BotSession", back_populates="user")


class Trade(Base):
    """Trade history database model."""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    symbol = Column(String(50), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY or SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    volume = Column(Float, nullable=False)  # in lots
    pnl = Column(Float, default=0.0)
    commission = Column(Float, default=0.0)
    swap = Column(Float, default=0.0)
    r_value = Column(Float, nullable=True)
    bars_held = Column(Integer, default=0)
    timestamp_open = Column(DateTime, default=datetime.utcnow)
    timestamp_close = Column(DateTime, nullable=True)
    is_open = Column(Boolean, default=True)
    
    # Relationships
    user = relationship("User", back_populates="trades")


class BotSession(Base):
    """Bot trading session tracking."""
    __tablename__ = "bot_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    starting_balance = Column(Float, nullable=False)
    ending_balance = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    strategy_config = Column(Text, nullable=True)  # JSON string
    
    # Relationships
    user = relationship("User", back_populates="bot_sessions")


class StrategyConfiguration(Base):
    """Strategy configuration storage."""
    __tablename__ = "strategy_configurations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    strategy_type = Column(String(100), nullable=False)
    config_data = Column(Text, nullable=False)  # JSON string
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PerformanceMetric(Base):
    """Daily performance metrics."""
    __tablename__ = "performance_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    balance = Column(Float, nullable=False)
    daily_pnl = Column(Float, default=0.0)
    trades_count = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
