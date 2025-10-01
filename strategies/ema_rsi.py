# strategies/ema_rsi.py
from dataclasses import dataclass
import pandas as pd
import ta

@dataclass
class StrategyParams:
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_len: int = 14
    rsi_long_gate: float = 55
    rsi_short_gate: float = 45
    take_both_sides: bool = False

NAME = "ema_rsi"

# Input: df with columns [timestamp, open, high, low, close, volume]
# Output: list of dict signals

def generate_signals(df: pd.DataFrame, params: dict) -> list:
    p = StrategyParams(**params)
    d = df.copy()
    d["ema_fast"] = d["close"].ewm(span=p.ema_fast, adjust=False).mean()
    d["ema_slow"] = d["close"].ewm(span=p.ema_slow, adjust=False).mean()
    d["rsi"] = ta.momentum.rsi(d["close"], window=p.rsi_len)
    d["atr"] = ta.volatility.average_true_range(d["high"], d["low"], d["close"], window=14)

    d["cross_up"] = (d["ema_fast"].shift(1) <= d["ema_slow"].shift(1)) & (d["ema_fast"] > d["ema_slow"])
    d["cross_dn"] = (d["ema_fast"].shift(1) >= d["ema_slow"].shift(1)) & (d["ema_fast"] < d["ema_slow"]) \
                     & (d["rsi"] <= p.rsi_short_gate)

    out = []
    start = max(p.ema_slow, p.rsi_len, 14) + 2
    for i in range(start, len(d)):
        row = d.iloc[i]
        price = float(row["close"])
        atr = float(row["atr"]) if row["atr"] == row["atr"] else None
        ts = pd.to_datetime(row["timestamp"], utc=True).isoformat()

        # LONG
        if row["cross_up"] and row["rsi"] >= p.rsi_long_gate:
            if atr is None:
                continue
            sl = price - 1.0 * atr
            r = price - sl
            out.append({
                "time": ts, "side": "LONG", "entry": price,
                "sl": sl, "tp1": price + 1.0 * r, "tp2": price + 2.0 * r,
                "meta": {"strategy": NAME}
            })

        # SHORT
        if p.take_both_sides and row["cross_dn"]:
            if atr is None:
                continue
            sl = price + 1.0 * atr
            r = sl - price
            out.append({
                "time": ts, "side": "SHORT", "entry": price,
                "sl": sl, "tp1": price - 1.0 * r, "tp2": price - 2.0 * r,
                "meta": {"strategy": NAME}
            })
    return out