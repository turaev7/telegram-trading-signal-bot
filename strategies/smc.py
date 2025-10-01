# strategies/smc.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

NAME = "smc"

@dataclass
class Params:
    min_confirmations: int = 4
    lookback_bars: int = 150
    wick_ratio: float = 0.6
    eq_high_tol_bps: int = 10
    fib_zones: Tuple[float, float] = (0.62, 0.79)
    gap_min_bps: int = 20
    atr_len: int = 14
    enable_elliott: bool = False

# --- Utility helpers ---

def atr(df, n):
    h,l,c = df['high'], df['low'], df['close']
    tr = np.maximum(h-l, np.maximum((h-c.shift()).abs(), (l-c.shift()).abs()))
    return tr.rolling(n).mean()

def bps(a, b):
    return 0.0 if b == 0 else (a/b - 1.0) * 1e4

# --- Detectors (heuristics; not academic definitions) ---

def detect_fvg(df, min_bps=20):
    # Returns list of tuples (index, direction) where direction=+1 for bullish, -1 for bearish
    out = []
    for i in range(2, len(df)):
        # bullish FVG: low[i] > high[i-2]
        if bps(df['low'].iloc[i], df['high'].iloc[i-2]) > min_bps:
            out.append((i, +1))
        # bearish FVG: high[i] < low[i-2]
        if bps(df['low'].iloc[i-2], df['high'].iloc[i]) > min_bps:
            out.append((i, -1))
    return out

def detect_gap(df, min_bps=20):
    out = []
    for i in range(1, len(df)):
        up = bps(df['open'].iloc[i], df['close'].iloc[i-1])
        if up > min_bps:
            out.append((i, +1))
        if up < -min_bps:
            out.append((i, -1))
    return out

def detect_order_block(df, atrv, impulse_mult=1.0):
    # Very simple: last opposite candle before range expansion > ATR
    out = []
    for i in range(2, len(df)):
        body = abs(df['close'].iloc[i] - df['open'].iloc[i])
        rng = df['high'].iloc[i] - df['low'].iloc[i]
        if rng < atrv.iloc[i]:
            continue
        # bullish impulse -> bearish OB at i-1 if red candle
        if df['close'].iloc[i] > df['open'].iloc[i] and df['close'].iloc[i] - df['open'].iloc[i] > impulse_mult * atrv.iloc[i]:
            if df['close'].iloc[i-1] < df['open'].iloc[i-1]:
                out.append((i-1, +1))
        # bearish impulse -> bullish OB at i-1 if green candle
        if df['close'].iloc[i] < df['open'].iloc[i] and df['open'].iloc[i] - df['close'].iloc[i] > impulse_mult * atrv.iloc[i]:
            if df['close'].iloc[i-1] > df['open'].iloc[i-1]:
                out.append((i-1, -1))
    return out

def detect_equal_levels(df, tol_bps=10):
    # equal highs/lows indicating liquidity
    eqh, eql = [], []
    for i in range(2, len(df)):
        ph = df['high'].iloc[i]
        for j in range(i-5, i):
            if j < 1: continue
            if abs(bps(ph, df['high'].iloc[j])) <= tol_bps:
                eqh.append(i); break
        pl = df['low'].iloc[i]
        for j in range(i-5, i):
            if j < 1: continue
            if abs(bps(pl, df['low'].iloc[j])) <= tol_bps:
                eql.append(i); break
    return eqh, eql

def detect_premium_discount(df):
    # swing over lookback window: 50% line
    hh = df['high'].rolling(50).max()
    ll = df['low'].rolling(50).min()
    mid = (hh + ll) / 2
    prem = df['close'] > mid
    disc = df['close'] < mid
    return prem, disc

def detect_wick(df, min_ratio=0.6):
    # long upper/lower wick
    u = (df['high'] - df[['open','close']].max(axis=1))
    l = (df[['open','close']].min(axis=1) - df['low'])
    body = (df['close'] - df['open']).abs() + 1e-9
    return (u / body > min_ratio), (l / body > min_ratio)

def detect_breaker(df, ob_list):
    # breaker: OB invalidated then retested
    br = []
    for idx, dirn in ob_list:
        if dirn == +1:  # bullish context from bearish OB
            # price later closes below OB low then returns above
            ob_low = df['low'].iloc[idx]
            broke = ((df['close'].iloc[idx+1:] < ob_low).any())
            if broke:
                # mark breaker direction with +1 (bullish after reclaim)
                br.append((idx, +1))
        else:
            ob_high = df['high'].iloc[idx]
            broke = ((df['close'].iloc[idx+1:] > ob_high).any())
            if broke:
                br.append((idx, -1))
    return br

def detect_mitigation(df, ob_list):
    # touch back into OB range within next N bars
    mit = []
    for idx, dirn in ob_list:
        hi, lo = df['high'].iloc[idx], df['low'].iloc[idx]
        for k in range(idx+1, min(idx+20, len(df))):
            if df['low'].iloc[k] <= hi and df['high'].iloc[k] >= lo:
                mit.append((k, dirn)); break
    return mit

def detect_inducement(df):
    # small fake HH/LL (deviation < 0.2% then reversal next bar)
    ind = []
    for i in range(2, len(df)):
        if df['high'].iloc[i] > df['high'].iloc[i-1] and bps(df['high'].iloc[i], df['high'].iloc[i-1]) < 20 and df['close'].iloc[i] < df['close'].iloc[i-1]:
            ind.append((i, -1))
        if df['low'].iloc[i] < df['low'].iloc[i-1] and bps(df['low'].iloc[i-1], df['low'].iloc[i]) < 20 and df['close'].iloc[i] > df['close'].iloc[i-1]:
            ind.append((i, +1))
    return ind

def detect_fib_zone(df, zone=(0.62, 0.79)):
    # Is price retracing into zone relative to last 50-bar swing?
    out = []
    for i in range(50, len(df)):
        hh = df['high'].iloc[i-50:i].max()
        ll = df['low'].iloc[i-50:i].min()
        if hh - ll <= 0: continue
        retr = (df['close'].iloc[i] - ll) / (hh - ll)
        if zone[0] <= retr <= zone[1]:
            # if in upper band, short bias; lower band, long bias (approx)
            bias = -1 if retr > 0.5 else +1
            out.append((i, bias))
    return out

def detect_bpr_from_fvg(fvg_list):
    # overlapping opposing FVGs → BPR
    out = []
    # Simplified marker near latest fvg
    if len(fvg_list) >= 2 and fvg_list[-1][1] != fvg_list[-2][1]:
        out.append((fvg_list[-1][0], fvg_list[-1][1]))
    return out

def detect_rdrb(df):
    # DBR/RBD pattern over last 7 bars
    out = []
    for i in range(7, len(df)):
        win = df.iloc[i-7:i]
        up = win['close'].iloc[-1] > win['close'].iloc[0]
        rng = win['high'].max() - win['low'].min()
        if rng == 0: continue
        base = (win['high'] - win['low']).mean() / rng < 0.35
        if base:
            out.append((i, +1 if up else -1))
    return out

def detect_volume_imbalance(df):
    # Proxy: sequence of wide-range candles with same direction & small lower/upper wicks
    out = []
    body = (df['close'] - df['open'])
    rng = (df['high'] - df['low']).replace(0, np.nan)
    dirn = np.sign(body)
    streak = (dirn == dirn.shift()).rolling(3).sum()
    wide = (rng > rng.rolling(20).mean())
    for i in range(2, len(df)):
        if wide.iloc[i] and streak.iloc[i] >= 3:
            out.append((i, int(dirn.iloc[i])))
    return out

def detect_ifvg(df, min_bps=20):
    # Inverse FVG = gap against current direction, then continuation
    fvg = detect_fvg(df, min_bps)
    out = []
    for i, dirn in fvg[-5:]:
        out.append((i, -dirn))
    return out

def detect_elliott(df):
    # Very light zigzag 5-swing; returns bias of last swing direction
    pivH = (df['high'] == df['high'].rolling(5, center=True).max())
    pivL = (df['low'] == df['low'].rolling(5, center=True).min())
    piv = []
    for i in range(len(df)):
        if pivH.iloc[i]: piv.append((i, 'H'))
        if pivL.iloc[i]: piv.append((i, 'L'))
    if len(piv) < 5:
        return []
    last = piv[-5:]
    dirn = +1 if last[-1][1] == 'H' else -1
    return [(last[-1][0], dirn)]

# --- Consensus aggregator ---

def smc_consensus_signals(df: pd.DataFrame, params: Dict) -> List[Dict]:
    p = Params(**params)
    d = df.copy().reset_index(drop=True)
    d['atr'] = atr(d, p.atr_len)

    fvg = detect_fvg(d, p.gap_min_bps)
    gap = detect_gap(d, p.gap_min_bps)
    ob = detect_order_block(d, d['atr'])
    brk = detect_breaker(d, ob)
    mit = detect_mitigation(d, ob)
    eqh, eql = detect_equal_levels(d, p.eq_high_tol_bps)
    prem, disc = detect_premium_discount(d)
    wickU, wickL = detect_wick(d, p.wick_ratio)
    fibz = detect_fib_zone(d, p.fib_zones)
    bpr = detect_bpr_from_fvg(fvg)
    rdrb = detect_rdrb(d)
    vimb = detect_volume_imbalance(d)
    ifvg = detect_ifvg(d, p.gap_min_bps)
    ell = detect_elliott(d) if p.enable_elliott else []

    # Build confirmations per bar index
    conf = {}
    def add(lst, name):
        for i, dirn in lst:
            conf.setdefault(i, []).append((name, dirn))
    add(fvg, 'FVG'); add(ifvg, 'IFVG'); add(gap, 'GAP')
    add(ob, 'OB'); add(brk, 'Breaker'); add(mit, 'Mitigation')
    for i in eqh: conf.setdefault(i, []).append(('LiquidityHigh', -1))
    for i in eql: conf.setdefault(i, []).append(('LiquidityLow', +1))
    for i, dirn in fibz: conf.setdefault(i, []).append(('Fibonacci', dirn))
    for i, dirn in bpr: conf.setdefault(i, []).append(('BPR', dirn))
    for i, dirn in rdrb: conf.setdefault(i, []).append(('RDRB', dirn))
    for i, dirn in vimb: conf.setdefault(i, []).append(('VolImb', dirn))
    for i in range(len(d)):
        if prem.iloc[i]: conf.setdefault(i, []).append(('Premium', -1))
        if disc.iloc[i]: conf.setdefault(i, []).append(('Discount', +1))
        if wickU.iloc[i]: conf.setdefault(i, []).append(('WickUp', -1))
        if wickL.iloc[i]: conf.setdefault(i, []).append(('WickDown', +1))
    add(ell, 'Elliott')

    # Build signals for latest bar only (or last 3 bars)
    out = []
    for i in range(max(0, len(d)-3), len(d)):
        if i not in conf: continue
        votes = conf[i]
        # count by direction and unique concept names
        names = {}
        long_votes = 0; short_votes = 0
        for name, dirn in votes:
            if name in names: continue
            names[name] = dirn
            if dirn > 0: long_votes += 1
            elif dirn < 0: short_votes += 1
        total_conf = max(long_votes, short_votes)
        if total_conf >= p.min_confirmations:
            side = 'LONG' if long_votes > short_votes else 'SHORT'
            price = float(d['close'].iloc[i])
            atrv = float(d['atr'].iloc[i] if pd.notna(d['atr'].iloc[i]) else (d['high'].iloc[i]-d['low'].iloc[i]))
            sl = price - p.atr_len/14 * atrv if side=='LONG' else price + p.atr_len/14 * atrv
            r = abs(price - sl)
            sig = {
                'time': pd.to_datetime(d['timestamp'].iloc[i], utc=True).isoformat(),
                'side': side,
                'entry': price,
                'sl': float(sl),
                'tp1': float(price + r if side=='LONG' else price - r),
                'tp2': float(price + 2*r if side=='LONG' else price - 2*r),
                'meta': {
                    'strategy': NAME,
                    'confirmations': list(names.keys()),
                    'long_votes': long_votes,
                    'short_votes': short_votes
                }
            }
            out.append(sig)
    return out
    # --- Helper: compact review of latest bar (for continuous analysis stream) ---
def smc_review_latest(df: pd.DataFrame, params: Dict) -> Dict:
    """
    Returns a compact review of the latest bar:
    - close price
    - long/short votes
    - unique confirmation names that fired
    - bar timestamp
    Does NOT enforce the min_confirmations gate.
    """
    # Reuse same parameter structure
    p = Params(**params) if not isinstance(params, Params) else params
    d = df.copy().reset_index(drop=True)
    if len(d) == 0:
        return {}
    # local ATR
    d['atr'] = atr(d, p.atr_len)

    # Run the same detectors we use for consensus
    fvg = detect_fvg(d, p.gap_min_bps)
    gap = detect_gap(d, p.gap_min_bps)
    ob = detect_order_block(d, d['atr'])
    brk = detect_breaker(d, ob)
    mit = detect_mitigation(d, ob)
    eqh, eql = detect_equal_levels(d, p.eq_high_tol_bps)
    prem, disc = detect_premium_discount(d)
    wickU, wickL = detect_wick(d, p.wick_ratio)
    fibz = detect_fib_zone(d, p.fib_zones)
    bpr = detect_bpr_from_fvg(fvg)
    rdrb = detect_rdrb(d)
    vimb = detect_volume_imbalance(d)
    ifvg = detect_ifvg(d, p.gap_min_bps)
    ell = detect_elliott(d) if getattr(p, "enable_elliott", False) else []

    conf = {}
    def add(lst, name):
        for i, dirn in lst:
            conf.setdefault(i, []).append((name, dirn))
    add(fvg, 'FVG'); add(ifvg, 'IFVG'); add(gap, 'GAP')
    add(ob, 'OB'); add(brk, 'Breaker'); add(mit, 'Mitigation')
    for i in eqh: conf.setdefault(i, []).append(('LiquidityHigh', -1))
    for i in eql: conf.setdefault(i, []).append(('LiquidityLow', +1))
    for i, dirn in fibz: conf.setdefault(i, []).append(('Fibonacci', dirn))
    for i, dirn in bpr: conf.setdefault(i, []).append(('BPR', dirn))
    for i, dirn in rdrb: conf.setdefault(i, []).append(('RDRB', dirn))
    for i, dirn in vimb: conf.setdefault(i, []).append(('VolImb', dirn))
    for i in range(len(d)):
        if prem.iloc[i]: conf.setdefault(i, []).append(('Premium', -1))
        if disc.iloc[i]: conf.setdefault(i, []).append(('Discount', +1))
        if wickU.iloc[i]: conf.setdefault(i, []).append(('WickUp', -1))
        if wickL.iloc[i]: conf.setdefault(i, []).append(('WickDown', +1))
    add(ell, 'Elliott')

    i = len(d) - 1  # latest bar
    votes = conf.get(i, [])
    names = {}
    long_votes = short_votes = 0
    for name, dirn in votes:
        if name in names:
            continue
        names[name] = dirn
        if dirn > 0: long_votes += 1
        elif dirn < 0: short_votes += 1

    return {
        'time': pd.to_datetime(d['timestamp'].iloc[i], utc=True).isoformat(),
        'close': float(d['close'].iloc[i]),
        'long_votes': long_votes,
        'short_votes': short_votes,
        'confirmations': sorted(names.keys()),
    }
