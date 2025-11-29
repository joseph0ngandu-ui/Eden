import MetaTrader5 as mt5
import time
import sys

def verify_trading():
    print("Initializing MetaTrader 5...")
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False

    symbol = "Volatility 75 Index"
    print(f"Checking symbol: {symbol}")
    
    # Ensure symbol is selected
    if not mt5.symbol_select(symbol, True):
        print(f"Failed to select {symbol}")
        # Try a fallback symbol just in case
        symbol = "EURUSD"
        print(f"Trying fallback: {symbol}")
        if not mt5.symbol_select(symbol, True):
            print(f"Failed to select {symbol}")
            return False

    # Prepare order
    lot = 0.001 # Minimum lot for V75 usually
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask
    deviation = 20
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": 0.0,
        "tp": 0.0,
        "deviation": deviation,
        "magic": 123456,
        "comment": "Eden Test Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }

    print(f"Sending BUY order for {symbol}...")
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.retcode} - {result.comment}")
        # Try with higher volume if min lot issue
        if "Invalid volume" in result.comment:
             request["volume"] = 0.01
             print(f"Retrying with volume 0.01...")
             result = mt5.order_send(request)
             if result.retcode != mt5.TRADE_RETCODE_DONE:
                 print(f"Retry failed: {result.retcode} - {result.comment}")
                 return False
        else:
            return False

    print(f"Order placed successfully! Ticket: {result.order}")
    
    # Wait a bit
    time.sleep(5)
    
    # Close position
    print("Closing position...")
    position_id = result.order
    
    # Get position details to close it
    positions = mt5.positions_get(ticket=position_id)
    if not positions:
        # Maybe it was a market execution and ticket changed? or closed by SL/TP?
        # Try finding by magic number
        positions = mt5.positions_get(symbol=symbol)
        target_pos = None
        for p in positions:
            if p.magic == 123456:
                target_pos = p
                break
        if not target_pos:
            print("Position not found to close!")
            return False
        position_id = target_pos.ticket
        lot = target_pos.volume
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "position": position_id,
        "price": mt5.symbol_info_tick(symbol).bid,
        "deviation": deviation,
        "magic": 123456,
        "comment": "Eden Test Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    close_result = mt5.order_send(close_request)
    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Close failed: {close_result.retcode} - {close_result.comment}")
        return False
        
    print(f"Position closed successfully! Ticket: {close_result.order}")
    mt5.shutdown()
    return True

if __name__ == "__main__":
    if verify_trading():
        print("\nSUCCESS: Trading capability verified!")
        sys.exit(0)
    else:
        print("\nFAILURE: Trading test failed.")
        sys.exit(1)
