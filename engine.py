# engine.py
import json, time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml
import ccxt
from requests.exceptions import RequestException

# Optional Yahoo adapter (kept for future use; not used with prefixes)
try:
    import yfinance as yf
except Exception:
    yf = None

# Strategies
import strategies.ema_rsi as ema_rsi
try:
    from strategies.smc import smc_consensus_signals, smc_review_latest
except ImportError:
    from strategies.smc import smc_consensus_signals
    def smc_review_latest(df, params):  # noop if helper not present
        return {}

STATE_PATH = Path("state.json")


class Engine:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        ex_id = cfg["exchange"]["name"]
        self.ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
        self.rate_sleep = cfg["exchange"].get("rate_limit_sleep_ms", 350) / 1000.0

        # cache of ccxt exchange clients by lowercase id (e.g., "binance", "bybit")
        self._ccxt_exchanges: Dict[str, ccxt.Exchange] = { ex_id.lower(): self.ex }

        if STATE_PATH.exists():
            self.state = json.loads(STATE_PATH.read_text())
        else:
            self.state = {"sent": {}}

    def save_state(self):
        STATE_PATH.write_text(json.dumps(self.state, indent=2))

    # ---------- Data fetching ----------
    def _fetch_ccxt(self, exchange_name: str, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        ex_key = exchange_name.strip().lower()
        if ex_key not in self._ccxt_exchanges:
            self._ccxt_exchanges[ex_key] = getattr(ccxt, ex_key)({"enableRateLimit": True})
        ex = self._ccxt_exchanges[ex_key]
        o = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(o, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.sort_values("timestamp").reset_index(drop=True)

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV with routing:
        - 'EXCHANGE:SYMBOL' -> ccxt on that exchange (e.g., BYBIT:XAUUSD).
        - 'YF:SYMBOL'       -> yfinance (Yahoo).
        - else              -> ccxt on default exchange in config.
        """
        # 1) Prefixed ccxt exchange: "BYBIT:XAUUSD", "BINANCE:BTC/USDT"
        if ":" in symbol and not symbol.startswith("YF:"):
            ex_name, sym = symbol.split(":", 1)
            return self._fetch_ccxt(ex_name, sym, timeframe, limit)

        # 2) Yahoo route (kept for future, not used in your current config)
        if symbol.startswith("YF:"):
            if yf is None:
                raise RuntimeError("yfinance not installed. Add yfinance==0.2.43 to requirements and pip install.")
            ticker = symbol[3:]
            tf_map = {
                "1m":  ("1m",  "7d"),
                "5m":  ("5m",  "10d"),
                "15m": ("15m", "30d"),
                "30m": ("30m", "60d"),
                "1h":  ("60m", "60d"),
                "1d":  ("1d",  "max"),
            }
            if timeframe not in tf_map:
                raise ValueError(f"Timeframe {timeframe} not supported for Yahoo. Use one of {list(tf_map.keys())}.")
            interval, period = tf_map[timeframe]
            last_err = None
            for _ in range(3):
                try:
                    df = yf.download(
                        tickers=ticker,
                        interval=interval,
                        period=period,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                    )
                    if df is not None and not df.empty:
                        df = (
                            df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                              .reset_index()
                        )
                        ts_col = "Datetime" if "Datetime" in df.columns else "Date"
                        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)
                        df = df[["timestamp","open","high","low","close","volume"]].sort_values("timestamp").reset_index(drop=True)
                        k = max(200, limit)
                        return df.iloc[-k:].reset_index(drop=True)
                    last_err = RuntimeError(f"Yahoo returned empty for {ticker} {interval}/{period}")
                except (json.JSONDecodeError, RequestException, ValueError) as e:
                    last_err = e
                time.sleep(1.5)
            raise RuntimeError(f"Yahoo failed {ticker} {interval}/{period}: {last_err}")

        # 3) Default: ccxt on the configured exchange
        return self._fetch_ccxt(self.cfg["exchange"]["name"], symbol, timeframe, limit)

    # ---------- Dedupe ----------
    def dedupe_and_mark(self, key: str) -> bool:
        if key in self.state["sent"]:
            return False
        self.state["sent"][key] = int(time.time())
        self.save_state()
        return True

    # ---------- Strategy runs ----------
    def run_once(self) -> List[Dict]:
        results: List[Dict] = []
        for sym in self.cfg["symbols"]:
            for tf in self.cfg["timeframes"]:
                try:
                    df = self.fetch_ohlcv_df(sym, tf)
                except Exception as e:
                    print(f"[WARN] fetch failed for {sym} {tf}: {e}")
                    continue

                for s in self.cfg.get("strategies", []):
                    if not s.get("enabled", True):
                        continue
                    name = s.get("name")
                    try:
                        if name == "smc":
                            sigs = smc_consensus_signals(df, s.get("params", {}))
                        elif name == "ema_rsi":
                            sigs = ema_rsi.generate_signals(df, s.get("params", {}))
                        else:
                            continue
                    except Exception as e:
                        print(f"[WARN] strategy {name} failed on {sym} {tf}: {e}")
                        continue

                    for sig in sigs[-3:]:
                        key = f"{sym}|{tf}|{sig['time']}|{sig['side']}|{sig['meta']['strategy']}"
                        if self.dedupe_and_mark(key):
                            results.append({"symbol": sym, "timeframe": tf, **sig})

                time.sleep(self.rate_sleep)
        return results

    # ---------- Extras for snapshot & reviews ----------
    def snapshot_prices(self) -> List[Dict]:
        out: List[Dict] = []
        for sym in self.cfg["symbols"]:
            for tf in self.cfg["timeframes"]:
                try:
                    df = self.fetch_ohlcv_df(sym, tf, limit=3)
                    if len(df) == 0:
                        continue
                    last = df.iloc[-1]
                    out.append({
                        "symbol": sym,
                        "timeframe": tf,
                        "time": pd.to_datetime(last["timestamp"], utc=True).isoformat(),
                        "close": float(last["close"]),
                    })
                except Exception as e:
                    print(f"[WARN] snapshot fetch failed for {sym} {tf}: {e}")
        return out

    def analysis_reviews(self) -> List[Dict]:
        smc_params = None
        for s in self.cfg.get("strategies", []):
            if s.get("name") == "smc" and s.get("enabled", True):
                smc_params = s.get("params", {})
                break
        if smc_params is None:
            return []

        out: List[Dict] = []
        for sym in self.cfg["symbols"]:
            for tf in self.cfg["timeframes"]:
                try:
                    df = self.fetch_ohlcv_df(sym, tf, limit=200)
                    if len(df) == 0:
                        continue
                    review = smc_review_latest(df, smc_params)
                    if not review:
                        continue
                    review["symbol"] = sym
                    review["timeframe"] = tf
                    out.append(review)
                except Exception as e:
                    print(f"[WARN] review failed for {sym} {tf}: {e}")
        return out


def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
