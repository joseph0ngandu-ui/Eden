#!/usr/bin/env python3
"""
Eden Trading Bot Backend API

REST API for the Eden iOS application with:
- JWT-based authentication
- Real-time bot status monitoring
- Trade history and performance analytics
- Strategy configuration management
- WebSocket support for real-time updates

STRATEGY: UltraSmall Mode - Volatility 75 Index (MOST PROFITABLE)
- Backtested Performance: 172.5% return (Jan-Oct 2025)
- Starting Capital: $50 → $136.25
- Grid-Search Score: 94.92 (Highest of all configurations)
- Configuration: confidence_threshold=0.6, tp_atr_multiplier=2.0, sl_atr_multiplier=1.2
- Mode: UltraSmall Risk Ladder for optimal capital preservation
"""

from fastapi import FastAPI, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from app.models import (
    Token, BotStatus, Trade, Position, PerformanceStats, 
    StrategyConfig, UserRegister, UserLogin,
    MT5Account, MT5AccountCreate, MT5AccountUpdate
)
from app.db_models import User, MT5Account as MT5AccountDB
from app.auth import (
    authenticate_user, create_access_token, get_current_user,
    verify_password, get_password_hash, TOKEN_EXPIRE_MINUTES
)
from app.database import init_db, get_db_session
from app.trading_service import TradingService
from app.websocket_manager import WebSocketManager
from app.database import Base, engine
from app.routers import strategies as strategies_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Eden Trading Bot API",
    description="REST API for Eden iOS trading application",
    version="1.0.0"
)

# Add CORS middleware for AWS API Gateway compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
trading_service = TradingService()
ws_manager = WebSocketManager()

# Routers
app.include_router(strategies_router.router)

# Strategy service (lazy import to avoid heavy deps at import time)
try:
    from ml_strategy_generator import MLStrategyGenerator
except Exception:
    MLStrategyGenerator = None

# Initialize database on startup
@app.on_event("startup")
async def startup():
    """Initialize database and services on startup."""
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)
    init_db()
    logger.info("✓ Database initialized")

    # Start background strategy discovery loop (non-blocking)
    if MLStrategyGenerator is not None:
        import asyncio
        generator = MLStrategyGenerator(data_dir=str((Path(__file__).resolve().parent.parent / 'data').resolve()))

        async def strategy_loop():
            while True:
                try:
                    # Discovery cycle
                    strat = generator.generate_and_test_strategy()
                    if strat and ws_manager:
                        await ws_manager.broadcast_json({
                            "type": "strategy_discovered",
                            "strategy": {
                                "id": strat.id,
                                "name": strat.name,
                                "type": strat.type,
                                "parameters": strat.parameters,
                                "performance": strat.backtest_results,
                                "validated": strat.is_validated,
                                "active": strat.is_active,
                            }
                        })

                    # Phase-out cycle
                    phased = generator.phase_out_strategies()
                    for sid, reason in phased.items():
                        await ws_manager.broadcast_json({
                            "type": "strategy_phased_out",
                            "strategy_id": sid,
                            "reason": reason,
                        })
                except Exception as e:
                    logger.error(f"Strategy discovery error: {e}")
                # Sleep 10 minutes between cycles
                await asyncio.sleep(600)

        asyncio.create_task(strategy_loop())
        logger.info("✓ Strategy discovery loop started")

    logger.info("✓ Eden API ready for deployment")

# ============================================================================
# HEALTH ENDPOINT
# ============================================================================

@app.get("/health")
async def health():
    return {"status": "ok"}

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/auth/register", response_model=Dict[str, str])
async def register(user_data: UserRegister):
    """Register a new user with email and password."""
    try:
        # Check if user already exists
        db_session = get_db_session()
        existing_user = db_session.query(User).filter(
            User.email == user_data.email
        ).first()
        
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        hashed_password = get_password_hash(user_data.password)
        new_user = User(
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password
        )
        db_session.add(new_user)
        db_session.commit()
        
        logger.info(f"✓ User registered: {user_data.email}")
        
        return {
            "message": "User registered successfully",
            "email": user_data.email
        }
    
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Authenticate user and return JWT token."""
    try:
        user = authenticate_user(credentials.email, credentials.password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create JWT token
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id},
            expires_delta=timedelta(minutes=TOKEN_EXPIRE_MINUTES)
        )
        
        logger.info(f"✓ User logged in: {user.email}")
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

# ============================================================================
# WEBSOCKET ENDPOINTS
# ============================================================================

@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# ============================================================================
# TRADING ENDPOINTS
# ============================================================================

@app.post("/orders/test")
async def place_test_order(payload: Dict[str, Any] = Body(...)):
    """Place a small test order via MT5 with adaptive filling."""
    symbol = payload.get('symbol', 'Volatility 75 Index')
    side = payload.get('side', 'BUY').upper()
    volume = float(payload.get('volume', 0.01))

    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail=f"MT5 init failed: {mt5.last_error()}")
        try:
            si = mt5.symbol_info(symbol)
            if si is None:
                raise HTTPException(status_code=400, detail="Symbol not found")
            if not si.visible:
                mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                raise HTTPException(status_code=400, detail="No tick")
            price = tick.ask if side == 'BUY' else tick.bid
            base = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY if side=='BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "deviation": 20,
                "comment": "Eden API test order",
            }
            tried = []
            for fm in [getattr(mt5, 'ORDER_FILLING_FOK', None), getattr(mt5, 'ORDER_FILLING_IOC', None), getattr(mt5, 'ORDER_FILLING_RETURN', None)]:
                if fm is None: continue
                req = dict(base)
                req['type_filling'] = fm
                res = mt5.order_send(req)
                tried.append({"fill": fm, "retcode": res.retcode, "comment": res.comment})
                if res.retcode == mt5.TRADE_RETCODE_DONE:
                    return {"status": "ok", "ticket": res.order, "tried": tried}
            raise HTTPException(status_code=500, detail={"message": "Order failed", "tried": tried})
        finally:
            mt5.shutdown()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/trades/logs")
async def get_trade_logs(limit: int = 100):
    import pandas as pd
    import os
    log_path = Path(__file__).resolve().parent.parent / 'logs' / 'trade_history.csv'
    if not log_path.exists():
        return []
    try:
        df = pd.read_csv(log_path)
        return df.tail(limit).to_dict('records')
    except Exception:
        # fallback simple read
        return []

# ============================================================================
# TRADING ENDPOINTS
# ============================================================================

@app.get("/trades/open", response_model=List[Position])
async def get_open_positions(current_user: User = Depends(get_current_user)):
    """Get current open positions."""
    try:
        positions = trading_service.get_open_positions()
        logger.info(f"Fetched {len(positions)} open positions for {current_user.email}")
        return positions
    except Exception as e:
        logger.error(f"Error fetching positions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch positions"
        )

@app.get("/trades/history", response_model=List[Trade])
async def get_trade_history(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get historical trade data with optional limit."""
    try:
        trades = trading_service.get_trade_history(limit=limit)
        logger.info(f"Fetched {len(trades)} trades for {current_user.email}")
        return trades
    except Exception as e:
        logger.error(f"Error fetching trade history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trade history"
        )

@app.get("/trades/recent", response_model=List[Trade])
async def get_recent_trades(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """Get recent trades from the last N days."""
    try:
        trades = trading_service.get_recent_trades(days=days)
        logger.info(f"Fetched {len(trades)} recent trades for {current_user.email}")
        return trades
    except Exception as e:
        logger.error(f"Error fetching recent trades: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recent trades"
        )

@app.post("/trades/close")
async def close_position(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """Close an open position by symbol."""
    try:
        result = trading_service.close_position(symbol)
        logger.info(f"Position closed: {symbol} by {current_user.email}")
        return {"status": "success", "message": f"Position {symbol} closed"}
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to close position"
        )

# ============================================================================
# PERFORMANCE ENDPOINTS
# ============================================================================

@app.get("/performance/stats", response_model=PerformanceStats)
async def get_performance_stats(current_user: User = Depends(get_current_user)):
    """Get comprehensive performance statistics."""
    try:
        stats = trading_service.calculate_performance_stats()
        logger.info(f"Performance stats retrieved for {current_user.email}")
        return stats
    except Exception as e:
        logger.error(f"Error calculating performance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate performance stats"
        )

@app.get("/performance/equity-curve")
async def get_equity_curve(current_user: User = Depends(get_current_user)):
    """Get equity curve data for charting."""
    try:
        equity_data = trading_service.get_equity_curve()
        logger.info(f"Equity curve retrieved for {current_user.email}")
        return {"equity_curve": equity_data}
    except Exception as e:
        logger.error(f"Error fetching equity curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch equity curve"
        )

@app.get("/performance/daily-summary")
async def get_daily_summary(current_user: User = Depends(get_current_user)):
    """Get daily PnL summary."""
    try:
        summary = trading_service.get_daily_summary()
        logger.info(f"Daily summary retrieved for {current_user.email}")
        return {"daily_summary": summary}
    except Exception as e:
        logger.error(f"Error fetching daily summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch daily summary"
        )

# ============================================================================
# BOT STATUS ENDPOINTS
# ============================================================================

@app.get("/bot/status", response_model=BotStatus)
async def get_bot_status(current_user: User = Depends(get_current_user)):
    """Get current bot status including balance, positions, and performance."""
    try:
        status_data = trading_service.get_bot_status()
        logger.info(f"Bot status retrieved for {current_user.email}")
        return status_data
    except Exception as e:
        logger.error(f"Error fetching bot status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch bot status"
        )

@app.post("/bot/start")
async def start_bot(current_user: User = Depends(get_current_user)):
    """Start the trading bot."""
    try:
        trading_service.start_bot()
        logger.info(f"Bot started by {current_user.email}")
        return {"status": "success", "message": "Trading bot started"}
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start bot"
        )

@app.post("/bot/stop")
async def stop_bot(current_user: User = Depends(get_current_user)):
    """Stop the trading bot."""
    try:
        trading_service.stop_bot()
        logger.info(f"Bot stopped by {current_user.email}")
        return {"status": "success", "message": "Trading bot stopped"}
    except Exception as e:
        logger.error(f"Error stopping bot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop bot"
        )

@app.post("/bot/pause")
async def pause_bot(current_user: User = Depends(get_current_user)):
    """Pause the trading bot."""
    try:
        trading_service.pause_bot()
        logger.info(f"Bot paused by {current_user.email}")
        return {"status": "success", "message": "Trading bot paused"}
    except Exception as e:
        logger.error(f"Error pausing bot: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to pause bot"
        )

# ============================================================================
# STRATEGY ENDPOINTS
# ============================================================================

@app.get("/strategy/config", response_model=StrategyConfig)
async def get_strategy_config(current_user: User = Depends(get_current_user)):
    """Get current strategy configuration."""
    try:
        config = trading_service.get_strategy_config()
        logger.info(f"Strategy config retrieved for {current_user.email}")
        return config
    except Exception as e:
        logger.error(f"Error fetching strategy config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch strategy config"
        )

@app.post("/strategy/config")
async def update_strategy_config(
    config: StrategyConfig,
    current_user: User = Depends(get_current_user)
):
    """Update strategy configuration."""
    try:
        updated_config = trading_service.update_strategy_config(config)
        logger.info(f"Strategy config updated by {current_user.email}")
        return {"status": "success", "config": updated_config}
    except Exception as e:
        logger.error(f"Error updating strategy config: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update strategy config"
        )

@app.get("/strategy/symbols")
async def get_trading_symbols(current_user: User = Depends(get_current_user)):
    """Get list of symbols being traded."""
    try:
        symbols = trading_service.get_trading_symbols()
        logger.info(f"Trading symbols retrieved for {current_user.email}")
        return {"symbols": symbols}
    except Exception as e:
        logger.error(f"Error fetching trading symbols: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch trading symbols"
        )

# ============================================================================
# HEALTH & DIAGNOSTIC ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "eden-trading-api"
    }

@app.get("/system/status")
async def system_status(current_user: User = Depends(get_current_user)):
    """Get system diagnostics and status."""
    try:
        status_data = {
            "api_version": "1.0.0",
            "bot_running": trading_service.is_running(),
            "last_heartbeat": trading_service.get_last_heartbeat(),
            "timestamp": datetime.utcnow().isoformat()
        }
        return status_data
    except Exception as e:
        logger.error(f"Error fetching system status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch system status"
        )

# ============================================================================
# MT5 ACCOUNT ENDPOINTS
# ============================================================================

@app.get("/account/mt5", response_model=List[MT5Account])
async def get_mt5_accounts(current_user: User = Depends(get_current_user)):
    """Get all MT5 accounts for the current user."""
    try:
        db_session = get_db_session()
        accounts = db_session.query(MT5AccountDB).filter(
            MT5AccountDB.user_id == current_user.id
        ).all()
        
        logger.info(f"Retrieved {len(accounts)} MT5 accounts for {current_user.email}")
        return accounts
    
    except Exception as e:
        logger.error(f"Error fetching MT5 accounts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch MT5 accounts"
        )

@app.get("/account/mt5/primary", response_model=MT5Account)
async def get_primary_mt5_account(current_user: User = Depends(get_current_user)):
    """Get the primary MT5 account for the current user."""
    try:
        db_session = get_db_session()
        account = db_session.query(MT5AccountDB).filter(
            MT5AccountDB.user_id == current_user.id,
            MT5AccountDB.is_primary == True,
            MT5AccountDB.is_active == True
        ).first()
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No primary MT5 account configured"
            )
        
        logger.info(f"Primary MT5 account retrieved for {current_user.email}")
        return account
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching primary MT5 account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch primary MT5 account"
        )

@app.post("/account/mt5", response_model=MT5Account)
async def create_mt5_account(
    account_data: MT5AccountCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new MT5 account configuration."""
    try:
        db_session = get_db_session()
        
        # If setting as primary, unset other primary accounts
        if account_data.is_primary:
            db_session.query(MT5AccountDB).filter(
                MT5AccountDB.user_id == current_user.id
            ).update({"is_primary": False})
        
        # Create new account (password should be encrypted in production)
        new_account = MT5AccountDB(
            user_id=current_user.id,
            account_number=account_data.account_number,
            account_name=account_data.account_name,
            broker=account_data.broker,
            server=account_data.server,
            password_encrypted=account_data.password,  # TODO: Encrypt in production
            is_primary=account_data.is_primary
        )
        
        db_session.add(new_account)
        db_session.commit()
        db_session.refresh(new_account)
        
        logger.info(f"✓ MT5 account created: {account_data.account_number} for {current_user.email}")
        
        return new_account
    
    except Exception as e:
        logger.error(f"Error creating MT5 account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create MT5 account"
        )

@app.put("/account/mt5/{account_id}", response_model=MT5Account)
async def update_mt5_account(
    account_id: int,
    account_data: MT5AccountUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update an existing MT5 account."""
    try:
        db_session = get_db_session()
        
        # Get account
        account = db_session.query(MT5AccountDB).filter(
            MT5AccountDB.id == account_id,
            MT5AccountDB.user_id == current_user.id
        ).first()
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MT5 account not found"
            )
        
        # If setting as primary, unset other primary accounts
        if account_data.is_primary:
            db_session.query(MT5AccountDB).filter(
                MT5AccountDB.user_id == current_user.id,
                MT5AccountDB.id != account_id
            ).update({"is_primary": False})
        
        # Update fields
        update_data = account_data.dict(exclude_unset=True)
        if "password" in update_data and update_data["password"]:
            update_data["password_encrypted"] = update_data.pop("password")  # TODO: Encrypt
        
        for key, value in update_data.items():
            if hasattr(account, key):
                setattr(account, key, value)
        
        account.updated_at = datetime.utcnow()
        db_session.commit()
        db_session.refresh(account)
        
        logger.info(f"✓ MT5 account updated: {account.account_number} for {current_user.email}")
        
        return account
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating MT5 account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update MT5 account"
        )

@app.delete("/account/mt5/{account_id}")
async def delete_mt5_account(
    account_id: int,
    current_user: User = Depends(get_current_user)
):
    """Delete an MT5 account (soft delete by setting is_active=False)."""
    try:
        db_session = get_db_session()
        
        # Get account
        account = db_session.query(MT5AccountDB).filter(
            MT5AccountDB.id == account_id,
            MT5AccountDB.user_id == current_user.id
        ).first()
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MT5 account not found"
            )
        
        # Soft delete
        account.is_active = False
        account.updated_at = datetime.utcnow()
        db_session.commit()
        
        logger.info(f"✓ MT5 account deleted: {account.account_number} for {current_user.email}")
        
        return {"status": "success", "message": "MT5 account deleted"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting MT5 account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete MT5 account"
        )

# ============================================================================
# WEBSOCKET ENDPOINTS (Real-time updates)
# ============================================================================

@app.websocket("/ws/updates/{token}")
async def websocket_endpoint(websocket: WebSocket, token: str):
    """WebSocket endpoint for real-time bot status updates."""
    try:
        # Verify token
        user = await get_current_user(token)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await ws_manager.connect(websocket)
        logger.info(f"WebSocket connected for {user.email}")
        
        try:
            while True:
                # Receive message (keep connection alive)
                data = await websocket.receive_text()
                
                # Send back current bot status
                bot_status = trading_service.get_bot_status()
                await websocket.send_json({
                    "type": "bot_status",
                    "data": bot_status.dict()
                })
        
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
            logger.info(f"WebSocket disconnected for {user.email}")
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_SERVER_ERROR)

@app.websocket("/ws/trades/{token}")
async def websocket_trades(websocket: WebSocket, token: str):
    """WebSocket endpoint for real-time trade updates."""
    try:
        user = await get_current_user(token)
        if not user:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        
        await ws_manager.connect(websocket)
        logger.info(f"Trade WebSocket connected for {user.email}")
        
        try:
            while True:
                data = await websocket.receive_text()
                
                # Send recent trades
                trades = trading_service.get_trade_history(limit=10)
                await websocket.send_json({
                    "type": "trade_update",
                    "data": [t.dict() for t in trades]
                })
        
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
            logger.info(f"Trade WebSocket disconnected for {user.email}")
    
    except Exception as e:
        logger.error(f"Trade WebSocket error: {e}")
        await websocket.close(code=status.WS_1011_SERVER_ERROR)

# ============================================================================
# ROOT & INFO ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Eden Trading Bot API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "openapi_schema": "/openapi.json"
    }

@app.get("/info")
async def api_info():
    """Get API information and available endpoints."""
    return {
        "name": "Eden Trading Bot Backend",
        "version": "1.0.0",
        "description": "REST API for Eden iOS trading application",
        "endpoints": {
            "auth": ["/auth/register", "/auth/login"],
            "trades": ["/trades/open", "/trades/history", "/trades/recent", "/trades/close"],
            "performance": ["/performance/stats", "/performance/equity-curve", "/performance/daily-summary"],
            "bot": ["/bot/status", "/bot/start", "/bot/stop", "/bot/pause"],
            "strategy": ["/strategy/config", "/strategy/symbols"],
            "websocket": ["/ws/updates/{token}", "/ws/trades/{token}"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    try:
        from app.settings import settings
    except Exception:
        settings = None

    run_kwargs = {
        "host": (settings.host if settings else "0.0.0.0"),
        "port": (settings.port if settings else 8000),
        "proxy_headers": True,
        "forwarded_allow_ips": "*",
    }

    # Enable HTTPS if cert and key are provided via env/.env (direct TLS mode)
    if settings and settings.ssl_certfile and settings.ssl_keyfile:
        run_kwargs["ssl_certfile"] = settings.ssl_certfile
        run_kwargs["ssl_keyfile"] = settings.ssl_keyfile

    uvicorn.run(app, **run_kwargs)
