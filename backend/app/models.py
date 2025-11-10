#!/usr/bin/env python3
"""
Pydantic models for Eden Trading Bot API
"""

from datetime import datetime, date
from typing import List, Optional
from pydantic import BaseModel, Field, EmailStr
from enum import Enum

# ============================================================================
# BASE MODELS
# ============================================================================

class BaseSchema(BaseModel):
    """Base schema with common fields."""
    
    class Config:
        from_attributes = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# ============================================================================
# USER & AUTHENTICATION MODELS
# ============================================================================

class BaseUser(BaseModel):
    """Base user model."""
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True

class UserCreate(BaseUser):
    """User creation model."""
    password: str

class UserLogin(BaseModel):
    """User login credentials."""
    email: EmailStr
    password: str

class UserRegister(BaseUser):
    """User registration model (extends base with password)."""
    password: str

class User(BaseUser):
    """Complete user model as stored in database."""
    id: int
    created_at: datetime
    last_login: Optional[datetime] = None

class Token(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"

# ============================================================================
# TRADING MODELS
# ============================================================================

class DirectionEnum(str, Enum):
    """Position direction enum."""
    BUY = "BUY"
    SELL = "SELL"

class Position(BaseSchema):
    """Open position model."""
    id: int
    symbol: str
    direction: DirectionEnum
    entry: float
    current: float
    pnl: float
    confidence: float
    bars: int
    timestamp: datetime
    volume: float = Field(default=0.01)  # in lots

class Trade(BaseSchema):
    """Completed trade model."""
    id: int
    symbol: str
    direction: DirectionEnum
    entry: float
    exit: float
    pnl: float
    timestamp_open: datetime
    timestamp_close: datetime
    bars_held: int
    r_value: float
    commission: float = 0.0
    swap: float = 0.0

# ============================================================================
# BOT STATUS MODELS
# ============================================================================

class BotStatus(BaseSchema):
    """Current bot status including balance and performance."""
    is_running: bool
    balance: float
    daily_pnl: float
    active_positions: int
    win_rate: float
    risk_tier: str
    total_trades: Optional[int] = None
    profit_factor: Optional[float] = None
    peak_balance: Optional[float] = None
    current_drawdown: Optional[float] = None
    last_update: Optional[datetime] = None

class BotControl(BaseModel):
    """Bot control commands."""
    command: str  # "START", "STOP", "PAUSE", "RESUME"

# ============================================================================
# PERFORMANCE MODELS
# ============================================================================

class EquityPoint(BaseSchema):
    """Equity curve point."""
    time: datetime
    value: float

class DailyPerformance(BaseSchema):
    """Daily performance summary."""
    date: date
    trades: int
    pnl: float
    win_rate: float
    max_drawdown: float
    sharpe_ratio: Optional[float]

class PerformanceStats(BaseSchema):
    """Comprehensive performance statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    current_drawdown: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    total_pnl: float
    daily_pnl: float
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    calmar_ratio: Optional[float]
    period_start: datetime
    period_end: datetime

# ============================================================================
# STRATEGY MODELS
# ============================================================================

class StrategyConfig(BaseSchema):
    """Strategy configuration - MOST PROFITABLE: UltraSmall Mode V75."""
    name: str = "VolatilityBurst_UltraSmall_V75"
    enabled: bool = True
    timeframe: str = "M5"
    mode: str = "UltraSmall"
    
    # UltraSmall Parameters (172.5% Returns - MOST PROFITABLE)
    atr_period: int = 14
    confidence_threshold: float = 0.6
    tp_atr_multiplier: float = 2.0
    sl_atr_multiplier: float = 1.2
    trail_trigger_r: float = 0.8
    max_bars_in_trade: int = 30
    
    # Risk Management - UltraSmall Mode
    risk_percent: float = 2.0
    max_position_size: float = 1.0
    max_concurrent_positions: int = 1
    max_trades_per_day: int = 8
    
    # Trading Symbols - PRIMARY: V75 Index ONLY (172.5% return)
    symbols: List[str] = [
        "Volatility 75 Index"
    ]
    
    # Secondary symbols (optional, but V75 is primary focus)
    secondary_symbols: List[str] = [
        "Volatility 100 Index",
        "Crash 500 Index",
        "Boom 1000 Index"
    ]
    
    description: Optional[str] = "MOST PROFITABLE: UltraSmall Mode V75 - 172.5% return, Grid-search optimized"

# ============================================================================
# RISK MANAGEMENT MODELS
# ============================================================================

class RiskTier(str, Enum):
    """Risk tier enumeration."""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    HIGH_AGGRESSION = "HIGH_AGGRESSION"
    EXTREME = "EXTREME"

class RiskSettings(BaseSchema):
    """Risk management settings."""
    max_drawdown_percent: float
    max_daily_loss_percent: float
    max_concurrent_positions: int
    position_size_method: str  # "FIXED", "PERCENT", "RISK_LADDER"
    risk_tier: RiskTier
    enable_trailing_stop: bool
    use_volatility_sizing: bool

# ============================================================================
# NOTIFICATION MODELS
# ============================================================================

class NotificationLevel(str, Enum):
    """Notification level."""
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"

class Notification(BaseSchema):
    """System notification model."""
    id: int
    level: NotificationLevel
    title: str
    message: str
    timestamp: datetime
    read: bool = False
    data: Optional[dict] = None

# ============================================================================
# API RESPONSE MODELS
# ============================================================================

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool
    message: str
    data: Optional[dict] = None
    timestamp: datetime
    request_id: Optional[str] = None

class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    message: str
    code: int
    timestamp: datetime

# ============================================================================
# WEBHOOK MODELS
# ============================================================================

class WebhookEvent(BaseSchema):
    """Generic webhook event model."""
    event_type: str
    timestamp: datetime
    data: dict
    source: str
    signature: Optional[str] = None

class TradeEvent(WebhookEvent):
    """Trade-related webhook event."""
    event_type: str = "trade"
    action: str  # "OPEN", "CLOSE", "MODIFY"
    trade_data: dict