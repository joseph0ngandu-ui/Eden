import hashlib
import io
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# Timeframe normalization
try:
    from eden.data.transforms import normalize_timeframe  # type: ignore
except Exception:
    def normalize_timeframe(x: str) -> str:  # fallback no-op
        return (x or "").strip().upper()


class DataLoader:
    def __init__(self, cache_dir: Path = Path("data/cache")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = logging.getLogger("eden.data.loader")

    def _cache_key(self, symbol: str, timeframe: str, start: str, end: str) -> Path:
        key = hashlib.md5(f"{symbol}-{timeframe}-{start}-{end}".encode()).hexdigest()
        return self.cache_dir / f"{key}.csv"

    def get_symbol_mapping(self, symbol: str, exchange_hint: Optional[str] = None) -> str:
        # Load optional broker mapping from config file
        try:
            import yaml  # type: ignore
            cfg_path = Path(__file__).resolve().parent.parent / "config" / "symbol_map.yaml"
            mapping_cfg = {}
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    mapping_cfg = yaml.safe_load(f) or {}
            # Broker hint key
            broker_key = (exchange_hint or mapping_cfg.get('default_broker') or 'default').lower()
            broker_map = mapping_cfg.get('brokers', {}).get(broker_key, {})
            # Try broker-specific mapping then default mapping
            mapped = broker_map.get(symbol.upper()) or mapping_cfg.get('default', {}).get(symbol.upper())
            if mapped:
                return mapped
        except Exception:
            pass
        # Built-in mapping for Yahoo Finance (yfinance)
        s = symbol.upper()
        mapping = {
            "US30": "^DJI",
            "NAS100": "^NDX",
            "XAUUSD": "GC=F",
        }
        if s in mapping:
            return mapping[s]
        # Generic FX mapping: EURUSD -> EURUSD=X
        if len(s) == 6 and s.isalpha():
            return f"{s}=X"
        return s

    def _stooq_symbol(self, symbol: str) -> str:
        # Stooq uses lowercase; indices often as ^dji, ^ndx
        smap = {
            "US30": "^dji",
            "NAS100": "^ndx",
        }
        return smap.get(symbol.upper(), symbol.lower())

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        # Expect columns: datetime, open, high, low, close, volume
        dt_col = None
        for c in ["datetime", "Date", "date", "time", "timestamp"]:
            if c in df.columns:
                dt_col = c
                break
        if dt_col is None:
            raise ValueError("CSV missing datetime column")
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors="coerce")
        df = df.dropna(subset=[dt_col])
        df = df.set_index(dt_col).sort_index()
        cols = {c.lower(): c for c in df.columns}
        rename = {cols.get("open", "open"): "open",
                  cols.get("high", "high"): "high",
                  cols.get("low", "low"): "low",
                  cols.get("close", "close"): "close",
                  cols.get("volume", "volume"): "volume"}
        df = df.rename(columns=rename)
        # Ensure all expected columns exist
        for col in ["open","high","low","close","volume"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df[["open", "high", "low", "close", "volume"]]

    def fetch_yfinance(self, symbol: str, timeframe: str, start: str, end: str, *, force_refresh: bool = False, allow_network: bool = True) -> Optional[pd.DataFrame]:
        mapped = self.get_symbol_mapping(symbol)
        cache_file = self._cache_key(symbol, timeframe, start, end)
        if cache_file.exists() and not force_refresh:
            try:
                return self.load_csv(cache_file)
            except Exception:
                pass
        if yf is None or not allow_network:
            self.log.warning("yfinance unavailable or network disabled; returning empty DataFrame for %s", symbol)
            return None
        try:
            tf_norm = normalize_timeframe(timeframe)
            interval_map = {"M1": "1m", "5M": "5m", "15M": "15m", "1H": "60m", "4H": "60m", "1D": "1d", "1W": "1wk", "1MO": "1mo"}
            interval = interval_map.get(tf_norm, "1d")
            # For intraday data, Yahoo limits to last ~730 days. Use period fallback.
            intraday = interval in ("1m", "5m", "15m", "60m")
            if intraday:
                df = yf.download(mapped, period="2y", interval=interval, progress=False, auto_adjust=False)
            else:
                df = yf.download(mapped, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                # Fallback: try shorter period for safety
                if intraday:
                    df = yf.download(mapped, period="1y", interval=interval, progress=False, auto_adjust=False)
                else:
                    df = yf.download(mapped, period="5y", interval=interval, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            if 'Adj Close' in df.columns and 'close' not in df.columns:
                df = df.rename(columns={"Adj Close": "close"})
            df.index = pd.to_datetime(df.index, utc=True)
            out = df[[c for c in ["open","high","low","close","volume"] if c in df.columns]].copy()
            if out.empty:
                return None
            # cache
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            out_reset = out.reset_index().rename(columns={out.index.name or "index": "datetime"})
            out_reset.to_csv(cache_file, index=False)
            return out
        except Exception as e:  # pragma: no cover (network)
            self.log.warning("yfinance fetch failed: %s", e)
            return None

    def fetch_dukascopy(self, symbol: str, timeframe: str, start: str, end: str, *, allow_network: bool = True) -> Optional[pd.DataFrame]:
        if not allow_network:
            return None
        try:
            from dukascopy import Downloader  # type: ignore
        except Exception:
            self.log.warning("dukascopy downloader not available; skipping")
            return None
        # Placeholder: implement as needed with the package
        return None

    def fetch_alpha_vantage_fx(self, symbol: str, timeframe: str, start: str, end: str, *, allow_network: bool = True) -> Optional[pd.DataFrame]:
        if not allow_network:
            return None
        api_key = os.getenv("ALPHAVANTAGE_API_KEY") or os.getenv("EDEN_ALPHA_VANTAGE_KEY")
        if not api_key:
            return None
        # Only FX pairs supported (e.g., EURUSD, GBPUSD)
        s = symbol.upper()
        if len(s) != 6:
            return None
        from_sym, to_sym = s[:3], s[3:]
        base_url = "https://www.alphavantage.co/query"
        try:
            if timeframe.upper() == "1D":
                params = {
                    "function": "FX_DAILY",
                    "from_symbol": from_sym,
                    "to_symbol": to_sym,
                    "outputsize": "full",
                    "apikey": api_key,
                }
                r = requests.get(base_url, params=params, timeout=20)
                r.raise_for_status()
                data = r.json().get("Time Series FX (Daily)", {})
                if not data:
                    return None
                recs = []
                for dt, row in data.items():
                    recs.append({
                        "datetime": pd.to_datetime(dt, utc=True),
                        "open": float(row.get("1. open", 0)),
                        "high": float(row.get("2. high", 0)),
                        "low": float(row.get("3. low", 0)),
                        "close": float(row.get("4. close", 0)),
                        "volume": 0.0,
                    })
                df = pd.DataFrame(recs).sort_values("datetime").set_index("datetime")
                return df
            else:
                # Intraday
                intr_map = {"1H": "60min", "15M": "15min", "5M": "5min"}
                interval = intr_map.get(timeframe.upper())
                if not interval:
                    return None
                params = {
                    "function": "FX_INTRADAY",
                    "from_symbol": from_sym,
                    "to_symbol": to_sym,
                    "interval": interval,
                    "outputsize": "full",
                    "apikey": api_key,
                }
                r = requests.get(base_url, params=params, timeout=20)
                r.raise_for_status()
                key = f"Time Series FX ({interval})"
                data = r.json().get(key, {})
                if not data:
                    return None
                recs = []
                for dt, row in data.items():
                    recs.append({
                        "datetime": pd.to_datetime(dt, utc=True),
                        "open": float(row.get("1. open", 0)),
                        "high": float(row.get("2. high", 0)),
                        "low": float(row.get("3. low", 0)),
                        "close": float(row.get("4. close", 0)),
                        "volume": 0.0,
                    })
                df = pd.DataFrame(recs).sort_values("datetime").set_index("datetime")
                return df
        except Exception as e:
            self.log.warning("AlphaVantage fetch failed for %s: %s", symbol, e)
            return None

    def fetch_stooq_daily(self, symbol: str, *, allow_network: bool = True) -> Optional[pd.DataFrame]:
        if not allow_network:
            return None
        stq = self._stooq_symbol(symbol)
        url = f"https://stooq.com/q/d/l/?s={stq}&i=d"
        try:
            r = requests.get(url, timeout=7)
            if r.status_code != 200 or not r.text or r.text.lower().startswith("brak danych"):
                return None
            df = pd.read_csv(io.StringIO(r.text))
            # stooq columns: Date, Open, High, Low, Close, Volume
            if "Date" not in df.columns:
                return None
            df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
            df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
            rename_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close"}
            if "Volume" in df.columns:
                rename_map["Volume"] = "volume"
            df = df.rename(columns=rename_map)
            # Ensure required columns exist
            for c in ["open","high","low","close","volume"]:
                if c not in df.columns:
                    df[c] = pd.NA
            # Coerce to standard types
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["open","high","low","close","volume"]].dropna(how="all")
            return df
        except Exception as e:
            self.log.warning("Stooq fetch failed for %s: %s", symbol, e)
            return None

    def _cache_store(self, symbol: str, timeframe: str, start: str, end: str, df: pd.DataFrame):
        """Store raw segment cache and merge into layered store."""
        try:
            cache_file = self._cache_key(symbol, timeframe, start, end)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            out = df.copy().reset_index()
            # normalize datetime column name
            if 'index' in out.columns:
                out = out.rename(columns={'index': 'datetime'})
            if 'Datetime' in out.columns:
                out = out.rename(columns={'Datetime': 'datetime'})
            out.to_csv(cache_file, index=False)
            # Also merge into layered store
            self._merge_layered_store(symbol, timeframe, out)
        except Exception:
            pass

    def _merge_layered_store(self, symbol: str, timeframe: str, new_out_with_dt: pd.DataFrame):
        """Merge a newly fetched segment into layered CSV so past runs accumulate.
        The layered file path is data/layered/{symbol}_{timeframe}.csv
        """
        try:
            layered_dir = self.cache_dir.parent / "layered"
            layered_dir.mkdir(parents=True, exist_ok=True)
            layered_file = layered_dir / f"{symbol.upper()}_{timeframe.upper()}.csv"

            # Normalize columns
            out = new_out_with_dt.rename(columns={new_out_with_dt.columns[0]: 'datetime'})
            out['datetime'] = pd.to_datetime(out['datetime'], utc=True, errors='coerce')
            out = out.dropna(subset=['datetime']).set_index('datetime').sort_index()
            out = out[[c for c in ['open','high','low','close','volume'] if c in out.columns]]

            if layered_file.exists():
                try:
                    existing = pd.read_csv(layered_file)
                    dt_col = None
                    for c in ["datetime","Date","date","timestamp","time"]:
                        if c in existing.columns:
                            dt_col = c; break
                    existing[dt_col] = pd.to_datetime(existing[dt_col], utc=True, errors='coerce')
                    existing = existing.dropna(subset=[dt_col]).set_index(dt_col).sort_index()
                    existing = existing[[c for c in ['open','high','low','close','volume'] if c in existing.columns]]
                    merged = pd.concat([existing, out], axis=0)
                    merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                except Exception:
                    merged = out
            else:
                merged = out

            merged_reset = merged.reset_index().rename(columns={'index':'datetime'})
            merged_reset.to_csv(layered_file, index=False)
        except Exception:
            pass

    def get_ohlcv(self, symbol: str, timeframe: str, start: str, end: str, *, allow_network: bool = True, force_refresh: bool = False, providers: Optional[list[str]] = None, prefer_mt5: bool = True, broker_hint: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Layered OHLCV retrieval across providers.
        Default Order: MT5 -> yfinance -> dukascopy -> alpha_vantage (FX) -> stooq (daily)
        Uses cache when available and stores successful fetches to cache.
        """
        # Try cache first
        cache_file = self._cache_key(symbol, timeframe, start, end)
        if cache_file.exists() and not force_refresh:
            try:
                return self.load_csv(cache_file)
            except Exception:
                pass
        # Decide provider order
        default_providers = ["yfinance", "dukascopy", "alpha_vantage", "stooq"]
        if providers is None:
            providers = default_providers
        # MT5 first if preferred
        if prefer_mt5:
            try:
                from eden.mt5_integration import mt5_fetch_ohlcv  # type: ignore
                df_mt5 = mt5_fetch_ohlcv(symbol, timeframe, start, end)
                if df_mt5 is not None and not df_mt5.empty:
                    # store layered but not raw segment (MT5 not keyed by exact dates consistently)
                    self._merge_layered_store(symbol, timeframe, df_mt5.reset_index())
                    return df_mt5
            except Exception:
                pass
        for prov in providers:
            df = None
            if prov == "yfinance":
                df = self.fetch_yfinance(symbol, timeframe, start, end, force_refresh=force_refresh, allow_network=allow_network)
            elif prov == "dukascopy":
                df = self.fetch_dukascopy(symbol, timeframe, start, end, allow_network=allow_network)
            elif prov == "alpha_vantage":
                df = self.fetch_alpha_vantage_fx(symbol, timeframe, start, end, allow_network=allow_network)
            elif prov == "stooq":
                df = self.fetch_stooq_daily(symbol, allow_network=allow_network)
            if df is not None and not df.empty:
                # write to cache
                try:
                    self._cache_store(symbol, timeframe, start, end, df)
                except Exception:
                    pass
                return df
        return None
