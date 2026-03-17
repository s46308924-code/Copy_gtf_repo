# DEMAND & SUPPLY TRADING – HOURLY SCANNER 

import pandas as pd
import os
from datetime import datetime, timedelta
from fyers_data import fetch_historical_data

# ===================== USER INPUT ===========================

OUTPUT_DIR   = "output/hourly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEFRAME = "60"          # 🔒 HOURLY
API_LIMIT = 90            # ✅ FIX-1 (hourly safe window)
HISTORY_YEARS = 2/12        # ✅ FIX-2 (7 → 2 years)

MAX_DISTANCE_PCT = 10

# button setting

HOURLY_BASE_BODY_PCT = 0.50
HOURLY_LEG_BODY_PCT  = 0.50
HOURLY_MAX_BASE = 3

HOURLY_LEG_IN_SINGLE_M  = 1.5
HOURLY_LEG_OUT_SINGLE_M = 2.0

HOURLY_CONT_MIN = 2
HOURLY_CONT_MAX = 3
HOURLY_CONT_BODY_PCT = 0.50
HOURLY_LEG_IN_CONT_M  = 1.5
HOURLY_LEG_OUT_CONT_M = 2.0

# ACCESS_TOKEN = (yaha se access token read hoga)

import json, os

def find_config():
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        config_path = os.path.join(current, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("❌ config.json nahi mila!")
        current = parent

ACCESS_TOKEN = find_config()["access_token"]

SYMBOL_CSV   = "symbols.csv"



# ============================================================
# ===================== HELPERS ==============================
# ============================================================

def fyers_symbol(sym):
    return f"NSE:{sym.strip().upper()}-EQ"

def body_pct(c):
    return abs(c["close"] - c["open"]) / (c["high"] - c["low"] + 1e-9)

def candle_color(c):
    return "GREEN" if c["close"] > c["open"] else "RED"

def candle_range(c):
    return c["high"] - c["low"]

def is_leg(c, pct):
    return body_pct(c) >= pct

def is_base(c, pct):
    return body_pct(c) < pct

# ============================================================
# ===================== DATA FETCH ===========================
# ============================================================

# ==================================================
# ============ CACHE HELPERS =======================
# ==================================================

def find_data_dir():
    """Find data/ directory by walking up parent directories."""
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        data_path = os.path.join(current, "data")
        if os.path.isdir(data_path):
            return data_path
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent

def safe_filename(symbol):
    """Convert FYERS symbol to safe filename: NSE:RELIANCE-EQ → NSE_RELIANCE-EQ"""
    return symbol.replace(":", "_")

def load_cached_data(symbol, timeframe):
    """Load cached parquet data if available."""
    data_dir = find_data_dir()
    if data_dir is None:
        return None
    tf_folder = "1D" if timeframe in ["1D", "D"] else "15m"
    parquet_path = os.path.join(data_dir, tf_folder, f"{safe_filename(symbol)}.parquet")
    if not os.path.exists(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return None

def save_cached_data(symbol, timeframe, df):
    """Save data to parquet cache."""
    data_dir = find_data_dir()
    if data_dir is None:
        return
    tf_folder = "1D" if timeframe in ["1D", "D"] else "15m"
    folder = os.path.join(data_dir, tf_folder)
    os.makedirs(folder, exist_ok=True)
    parquet_path = os.path.join(folder, f"{safe_filename(symbol)}.parquet")
    try:
        df.to_parquet(parquet_path)
    except Exception:
        pass

def fetch_data(symbol):
    end   = datetime.now().date()
    start = end - timedelta(days=HISTORY_YEARS * 365)

    # Step 1: Try 15m cache → resample to 75-minute candles
    try:
        cached_15m = load_cached_data(symbol, "15")
        if cached_15m is not None and len(cached_15m) > 0:
            first_date = cached_15m.index[0].date()
            last_date  = cached_15m.index[-1].date()
            if last_date >= end - timedelta(days=1):
                if first_date <= start + timedelta(days=7):
                    mask = (cached_15m.index.date >= start) & (cached_15m.index.date <= end)
                    df_15m = cached_15m.loc[mask].copy()
                    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
                else:
                    print(f"📥 Extending history: fetching {start} to {first_date}")
                    old_dfs = []
                    cur = start
                    while cur < first_date:
                        cur_end = min(cur + timedelta(days=API_LIMIT), first_date - timedelta(days=1))
                        try:
                            df_old_chunk = fetch_historical_data(
                                symbol, "15",
                                cur.strftime("%Y-%m-%d"),
                                cur_end.strftime("%Y-%m-%d"),
                                ACCESS_TOKEN
                            )
                            if df_old_chunk is not None and not df_old_chunk.empty:
                                old_dfs.append(df_old_chunk)
                        except Exception:
                            pass
                        cur = cur_end + timedelta(days=1)
                    if old_dfs:
                        df_old     = pd.concat(old_dfs)
                        df_merged  = pd.concat([df_old, cached_15m])
                        df_merged  = df_merged[~df_merged.index.duplicated()]
                        df_merged.sort_index(inplace=True)
                        save_cached_data(symbol, "15", df_merged)
                        mask  = (df_merged.index.date >= start) & (df_merged.index.date <= end)
                        df_15m = df_merged.loc[mask].copy()
                        df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
                    else:
                        mask  = (cached_15m.index.date >= start) & (cached_15m.index.date <= end)
                        df_15m = cached_15m.loc[mask].copy()
                        df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
            else:
                fetch_start = last_date + timedelta(days=1)
                new_dfs = []
                cur = fetch_start
                while cur <= end:
                    cur_end = min(cur + timedelta(days=API_LIMIT), end)
                    try:
                        df_new = fetch_historical_data(
                            symbol, "15",
                            cur.strftime("%Y-%m-%d"),
                            cur_end.strftime("%Y-%m-%d"),
                            ACCESS_TOKEN
                        )
                        if df_new is not None and not df_new.empty:
                            new_dfs.append(df_new)
                    except Exception:
                        pass
                    cur = cur_end + timedelta(days=1)
                if new_dfs:
                    df_new    = pd.concat(new_dfs)
                    df_merged = pd.concat([cached_15m, df_new])
                    df_merged = df_merged[~df_merged.index.duplicated()]
                    df_merged.sort_index(inplace=True)
                    save_cached_data(symbol, "15", df_merged)
                    mask  = (df_merged.index.date >= start) & (df_merged.index.date <= end)
                    df_15m = df_merged.loc[mask].copy()
                    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
                else:
                    mask  = (cached_15m.index.date >= start) & (cached_15m.index.date <= end)
                    df_15m = cached_15m.loc[mask].copy()
                    df_15m.index = pd.to_datetime(df_15m.index).tz_localize(None)
            if not df_15m.empty:
                # 75-min candles align with NSE trading sessions (9:15–15:30) producing
                # exactly 5 candles per day, avoiding the partial-candle artefacts of 60-min bars.
                df_75m = df_15m.resample("75min").agg({
                    "open":   "first",
                    "high":   "max",
                    "low":    "min",
                    "close":  "last",
                    "volume": "sum"
                }).dropna()
                if not df_75m.empty:
                    return df_75m
    except Exception:
        pass

    # Step 2: Fallback — fetch 60-minute data directly from API
    dfs = []
    cur = start
    while cur <= end:
        cur_end = min(cur + timedelta(days=API_LIMIT), end)
        try:
            df = fetch_historical_data(
                symbol,
                TIMEFRAME,
                cur.strftime("%Y-%m-%d"),
                cur_end.strftime("%Y-%m-%d"),
                ACCESS_TOKEN
            )
        except Exception:
            cur = cur_end + timedelta(days=1)
            continue

        if df is not None and not df.empty:
            dfs.append(df)

        cur = cur_end + timedelta(days=1)

    if not dfs:
        return None

    df = pd.concat(dfs)
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df

# ============================================================
# ===================== FRESHNESS ============================
# ============================================================

def is_fresh(df, idx, proximal):
    for i in range(idx + 1, len(df)):
        if df.iloc[i]["low"] <= proximal <= df.iloc[i]["high"]:
            return False
    return True

def near_cmp(proximal, cmp):
    return abs(proximal - cmp) / cmp * 100 <= MAX_DISTANCE_PCT

# ============================================================
# ===================== CONTINUOUS CHECK =====================
# ============================================================

def valid_continuous(df, start_idx, zone_range, multiplier):
    candles = []
    color = candle_color(df.iloc[start_idx])
    i = start_idx

    while (
        i >= 0
        and candle_color(df.iloc[i]) == color
        and len(candles) < HOURLY_CONT_MAX
    ):
        if body_pct(df.iloc[i]) < HOURLY_CONT_BODY_PCT:
            break
        candles.append(df.iloc[i])
        i -= 1

    if len(candles) < HOURLY_CONT_MIN:
        return False

    total_range = max(c["high"] for c in candles) - min(c["low"] for c in candles)
    return total_range >= zone_range * multiplier

# ============================================================
# ===================== ZONE SCAN ============================
# ============================================================

def scan_hourly_zones(df):
    zones = []
    i = len(df) - 1

    while i >= 3:
        legout = df.iloc[i]

        if not is_leg(legout, HOURLY_LEG_BODY_PCT):
            i -= 1
            continue

        base = []
        j = i - 1
        while j >= 0 and is_base(df.iloc[j], HOURLY_BASE_BODY_PCT) and len(base) < HOURLY_MAX_BASE:
            base.append(df.iloc[j])
            j -= 1

        if not base or j < 0:
            i -= 1
            continue

        legin = df.iloc[j]
        if not is_leg(legin, HOURLY_LEG_BODY_PCT):
            i -= 1
            continue

        lin, lout = candle_color(legin), candle_color(legout)

        if lin == "GREEN" and lout == "GREEN":
            pattern, ztype = "RBR", "DEMAND"
        elif lin == "RED" and lout == "GREEN":
            pattern, ztype = "DBR", "DEMAND"
        elif lin == "GREEN" and lout == "RED":
            pattern, ztype = "RBD", "SUPPLY"
        elif lin == "RED" and lout == "RED":
            pattern, ztype = "DBD", "SUPPLY"
        else:
            i -= 1
            continue

        base_df = pd.DataFrame(base)
        proximal = (
            base_df[["open","close"]].max(axis=1).max()
            if ztype == "DEMAND"
            else base_df[["open","close"]].min(axis=1).min()
        )

        wick_sources = []
        if pattern in ["DBR","RBD"]:
            wick_sources.append(legin)
        wick_sources += base + [legout]

        distal = (
            min(c["low"] for c in wick_sources)
            if ztype == "DEMAND"
            else max(c["high"] for c in wick_sources)
        )

        zone_range = abs(proximal - distal)

        legin_ok = (
            candle_range(legin) >= zone_range * HOURLY_LEG_IN_SINGLE_M or
            valid_continuous(df, j, zone_range, HOURLY_LEG_IN_CONT_M)
        )

        legout_ok = (
            candle_range(legout) >= zone_range * HOURLY_LEG_OUT_SINGLE_M or
            valid_continuous(df, i, zone_range, HOURLY_LEG_OUT_CONT_M)
        )

        if not (legin_ok and legout_ok):
            i -= 1
            continue

        if not is_fresh(df, i, proximal):
            i -= 1
            continue

        zones.append({
            "symbol": None,
            "zone_type": ztype,
            "pattern": pattern,
            "zone_create_date": df.index[i].strftime("%Y-%m-%d %H:%M"),
            "proximal": round(proximal, 2),
            "distal": round(distal, 2)
        })

        i -= 1

    return zones

# ============================================================
# ===================== MAIN RUN =============================
# ============================================================

def run():
    symbols = [s.strip() for s in open(SYMBOL_CSV) if s.strip()]
    results = []

    for sym in symbols:
        print("🔍 Hourly scanning:", sym)
        df = fetch_data(fyers_symbol(sym))
        if df is None:
            continue

        cmp = df["close"].iloc[-1]
        zones = scan_hourly_zones(df)

        for z in zones:
            if near_cmp(z["proximal"], cmp):
                z["symbol"] = sym
                results.append(z)

    if results:
        fname = f"hourly_zones_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print("✅ HOURLY CSV CREATED:", fname)
    else:
        print("⚠️ No valid hourly zones")

run()
