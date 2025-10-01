import ccxt
ex = ccxt.bybit({"enableRateLimit": True})
markets = ex.load_markets()
for k, m in markets.items():
    s = k
    if "XAU" in s or "GBP" in s or s.endswith("USD") or s.endswith("USDT"):
        print(s)
