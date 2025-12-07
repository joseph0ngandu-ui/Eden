#!/usr/bin/env python3
"""
Eden Bot Status Uploader to DynamoDB (Power Stack Core)

- Reads local bot state and MT5 balance/profit
- Updates DynamoDB tables eden-balances and eden-bot-status every 15 seconds by default
- Keeps serverless app in sync with live bot
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import MetaTrader5 as mt5
import boto3
from botocore.exceptions import ClientError

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
INTERVAL_SECONDS = 15          # How often to push status to DynamoDB
USER_ID = "demo-user"           # Fixed userId for mobile app demo
LOG_FILE = ROOT / "logs" / "eden-status-updater.log"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# DynamoDB wrapper
# ----------------------------------------------------------------------
 dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
balances_table = dynamodb.Table('eden-balances')
bot_status_table = dynamodb.Table('eden-bot-status')

def put_balance(balance: float, currency: str = "USD") -> bool:
    try:
        item = {
            "userId": USER_ID,
            "balance": balance,
            "currency": currency,
            "lastUpdated": datetime.utcnow().isoformat(),
            "source": "Eden-Status-Uploader"
        }
        balances_table.put_item(Item=item)
        logger.debug(f"Balance written: {balance} {currency}")
        return True
    except Exception as e:
        logger.error(f"Failed to write balance: {e}")
        return False

def put_bot_status(status: Dict[str, Any]) -> bool:
    try:
        payload = {
            "userId": USER_ID,
            "status": status.get("status", "unknown"),
            "uptime": status.get("uptime", "unknown"),
            "latencyMs": status.get("latencyMs", 42),
            "lastUpdate": datetime.utcnow().isoformat(),
        }
        if "account" in status:
            payload["account"] = status["account"]
        if "positions" in status:
            payload["positions"] = status["positions"]
        if "dailyPnl" in status:
            payload["dailyPnl"] = status["dailyPnl"]
        bot_status_table.put_item(Item=payload)
        logger.debug(f"Bot status written: {status.get('status')}")
        return True
    except Exception as e:
        logger.error(f"Failed to write bot status: {e}")
        return False

# ----------------------------------------------------------------------
# Eden local helpers
# ----------------------------------------------------------------------
TRADE_HISTORY_CSV = ROOT / "logs" / "trade_history.csv"

def load_recent_closed_trades(limit: int = 10) -> pd.DataFrame:
    try:
        if not TRADE_HISTORY_CSV.exists():
            logger.info("trade_history.csv not found yet (no closed trades).")
            return pd.DataFrame()
        df = pd.read_csv(TRADE_HISTORY_CSV)
        if df.empty:
            logger.debug("Empty trade_history.csv")
            return pd.DataFrame()
        if "exit_time" in df.columns:
            df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        df = df.sort_values("exit_time", ascending=False).head(limit)
        return df
    except Exception as e:
        logger.error(f"Error reading trade_history.csv: {e}")
        return pd.DataFrame()

def get_latest_balance() -> Optional[float]:
    """
    Compute latest equity from MT5 (preferred) or fallback to trade_history CSV.
    Returns balance as float.
    """
    # Try live MT5 first
    try:
        if mt5.initialize():
            acct = mt5.account_info()
            if acct:
                bal = acct.balance
                mt5.shutdown()
                logger.debug(f"Live MT5 balance read: {bal}")
                return bal
            else:
                logger.warning("MT5 initialized but account_info failed.")
                mt5.shutdown()
        else:
            logger.warning("MT5.initialize() failed; falling back to CSV.")
    except Exception as e:
        logger.warning(f"Exception while reading MT5 balance: {e}")

    # Fallback: use trade_history CSV cumulative PnL + a hardcoded starting balance
    try:
        df = load_recent_closed_trades()
        if not df.empty and "pnl" in df.columns:
            total_pnl = df["pnl"].sum()
        else:
            total_pnl = 0.0
        # Use a conservative starting-balance assumption
        base_balance = 1000.0
        latest_balance = base_balance + total_pnl
        logger.info(f"Computed balance from CSV (fallback): {latest_balance}")
        return latest_balance
    except Exception as e:
        logger.error(f"Error computing balance from CSV: {e}")
        return None

def get_bot_process_state() -> Dict[str, Any]:
    """
    Determine if the bot process is running or not.
    Simple windows process heuristic.
    Returns dict with minimal fields expected by API.
    """
    try:
        # Heuristic: check for python process matching watchdog or bot runner
        # On Windows, we can quickly count processes named python and check args or titles
        # For simplicity, fallback to assuming live if python processes are running Eden scripts
        
        # Attempt lightweight detection (optional; can be enhanced)
        import psutil
        python_pids = []
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.info['name'].lower() == 'python.exe' or p.info['name'].lower() == 'python':
                cmdline = p.info.get('cmdline') or []
                if any('watchdog' in str(arg) or 'bot_runner' in str(arg) or 'trading_bot.py' in str(arg) for arg in cmdline):
                    python_pids.append(p.info['pid'])
        if python_pids:
            logger.debug(f"Detected live Eden python processes: {python_pids}")
            return {
                "status": "online",
                "uptime": "stable",
                "latencyMs": 42,
                "account": {"balance": None},  # filled later
                "positions": 0
            }
        else:
            logger.debug("No Eden bot processes detected; marking as offline")
            return {
                "status": "offline",
                "uptime": "unavailable",
                "latencyMs": -1,
                "account": {"balance": None},
                "positions": 0
            }
    except Exception as e:
        logger.warning(f"Process detection failed: {e}")
        # Default to "unknown" to avoid false alarms
        return {
            "status": "unknown",
            "uptime": "unknown",
            "latencyMs": 42,
            "account": {"balance": None},
            "positions": 0
        }

# ----------------------------------------------------------------------
# Core loop
# ----------------------------------------------------------------------
def main_loop():
    while True:
        try:
            # 1. Get current bot balance
            balance = get_latest_balance()
            if balance is not None:
                put_balance(balance)
            else:
                logger.warning("Failed to determine balance this cycle")
            # 2. Get bot process state
            state = get_bot_process_state()
            if balance is not None:
                state["account"]["balance"] = balance
            # Optional: read closed trade count
            recent = load_recent_closed_trades(limit=100)
            if not recent.empty:
                state["dailyPnl"] = float(recent["pnl"].sum())
            # Publish state
            put_bot_status(state)
            logger.info(f"Updated balance={balance}, bot_status={state.get('status')}")
        except Exception as e:
            logger.error(f"Error in status uploader loop: {e}")
        finally:
            time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    logger.info("Eden Status Uploader starting...")
    logger.info(f"Pushing to DynamoDB tables: eden-balances, eden-bot-status for userId={USER_ID} every {INTERVAL_SECONDS}s")
    try:
        main_loop()
    except KeyboardInterrupt:
        logger.info("Eden Status Uploader stopped by user")
    except Exception as fatal:
        logger.critical(f"FATAL: {fatal}")
        sys.exit(1)