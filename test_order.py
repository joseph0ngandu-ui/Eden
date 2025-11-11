import MetaTrader5 as mt5

symbol = "Volatility 75 Index"
volume = 0.01

if not mt5.initialize():
    print("INIT_FAIL", mt5.last_error())
    raise SystemExit(1)

info = mt5.terminal_info()
print("TERMINAL", info.path if info else None, "connected=", info.connected if info else None, "trade_allowed=", info.trade_allowed if info else None)

acc = mt5.account_info()
print("ACCOUNT", acc.login if acc else None, acc.server if acc else None)

# Ensure symbol is visible
si = mt5.symbol_info(symbol)
if si is None:
    print("SYMBOL_NOT_FOUND")
    mt5.shutdown(); raise SystemExit(2)
if not si.visible:
    mt5.symbol_select(symbol, True)

# Prepare order
price = mt5.symbol_info_tick(symbol).ask if mt5.symbol_info_tick(symbol) else None
if price is None:
    print("NO_TICK")
    mt5.shutdown(); raise SystemExit(3)

base_request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": volume,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "deviation": 20,
    "comment": "Eden test order",
}

# Try supported filling modes
fill_modes = [
    getattr(mt5, 'ORDER_FILLING_FOK', None),
    getattr(mt5, 'ORDER_FILLING_IOC', None),
    getattr(mt5, 'ORDER_FILLING_RETURN', None),
]

last = None
for fm in [m for m in fill_modes if m is not None]:
    req = dict(base_request)
    req["type_filling"] = fm
    result = mt5.order_send(req)
    print("TRY_FILL", fm, "RET", result.retcode, result.comment)
    last = result
    if result.retcode not in (10030, 10031):  # 10030 unsupported filling, 10031 unsupported type
        break

print("RESULT", last.retcode, last.comment)
mt5.shutdown()
