#
# DAILY DEMAND & SUPPLY SCANNER
# (MONTHLY + WEEKLY CSV BASED )

import pandas as pd
import os, glob
from datetime import datetime, timedelta
from fyers_data import fetch_historical_data


#================USER_INPUT============================================

MONTHLY_CSV_DIR = "output/monthly"
WEEKLY_CSV_DIR  = "output/weekly"
OUTPUT_DIR      = "output/daily"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TIMEFRAME = "1D"
API_LIMIT = 365
HISTORY_YEARS = 2
RRR = 5

# 🔒 CMP DISTANCE FILTER (PROXIMAL ONLY)
MAX_DISTANCE_PCT = 15

# 🔒 AREA MULTIPLIERS
MONTHLY_AREA_MULTIPLIER = 0.5
WEEKLY_AREA_MULTIPLIER  = 0.5

#  BUTTONS SETTING

DAILY_BASE_BODY_PCT = 0.50
DAILY_LEG_BODY_PCT  = 0.50
DAILY_MAX_BASE = 3

DAILY_LEG_IN_SINGLE_M  = 1.5
DAILY_LEG_OUT_SINGLE_M = 2.0
DAILY_CONT_MIN = 2
DAILY_CONT_MAX = 3
DAILY_CONT_BODY_PCT = 0.50
DAILY_LEG_IN_CONT_M  = 1.5
DAILY_LEG_OUT_CONT_M = 2.0



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



# ============================================================
# HELPERS
# ============================================================

def fyers_symbol(sym):
    return f"NSE:{sym}-EQ"

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
# DATA FETCH
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

    # Step 1: Try cache
    try:
        cached = load_cached_data(symbol, TIMEFRAME)
        if cached is not None and len(cached) > 0:
            first_date = cached.index[0].date()
            last_date  = cached.index[-1].date()
            if last_date >= end - timedelta(days=1):
                if first_date <= start + timedelta(days=7):
                    mask   = (cached.index.date >= start) & (cached.index.date <= end)
                    result = cached.loc[mask].copy()
                    if not result.empty:
                        result.index = pd.to_datetime(result.index).tz_localize(None)
                        return result
                else:
                    print(f"📥 Extending history: fetching {start} to {first_date}")
                    old_dfs = []
                    cur = start
                    while cur < first_date:
                        cur_end = min(cur + timedelta(days=API_LIMIT), first_date - timedelta(days=1))
                        try:
                            df_old_chunk = fetch_historical_data(
                                symbol, TIMEFRAME,
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
                        df_old    = pd.concat(old_dfs)
                        df_merged = pd.concat([df_old, cached])
                        df_merged = df_merged[~df_merged.index.duplicated()]
                        df_merged.sort_index(inplace=True)
                        save_cached_data(symbol, TIMEFRAME, df_merged)
                        mask   = (df_merged.index.date >= start) & (df_merged.index.date <= end)
                        result = df_merged.loc[mask].copy()
                        if not result.empty:
                            result.index = pd.to_datetime(result.index).tz_localize(None)
                            return result
                    else:
                        mask   = (cached.index.date >= start) & (cached.index.date <= end)
                        result = cached.loc[mask].copy()
                        if not result.empty:
                            result.index = pd.to_datetime(result.index).tz_localize(None)
                            return result
            else:
                fetch_start = last_date + timedelta(days=1)
                new_dfs = []
                cur = fetch_start
                while cur <= end:
                    cur_end = min(cur + timedelta(days=API_LIMIT), end)
                    try:
                        df_new = fetch_historical_data(
                            symbol, TIMEFRAME,
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
                    df_merged = pd.concat([cached, df_new])
                    df_merged = df_merged[~df_merged.index.duplicated()]
                    df_merged.sort_index(inplace=True)
                    save_cached_data(symbol, TIMEFRAME, df_merged)
                    mask   = (df_merged.index.date >= start) & (df_merged.index.date <= end)
                    result = df_merged.loc[mask].copy()
                    if not result.empty:
                        result.index = pd.to_datetime(result.index).tz_localize(None)
                        return result
    except Exception:
        pass

    # Step 2: Fallback — fetch from API
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
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception:
            cur = cur_end + timedelta(days=1)
            continue
        cur = cur_end + timedelta(days=1)

    if not dfs:
        return None

    df = pd.concat(dfs)
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    save_cached_data(symbol, TIMEFRAME, df)
    return df

# ============================================================
# FRESHNESS
# ============================================================

def is_fresh(df, idx, proximal):
    for i in range(idx + 1, len(df)):
        if df.iloc[i]["low"] <= proximal <= df.iloc[i]["high"]:
            return False
    return True

# ============================================================
# CMP FILTER (PROXIMAL)
# ============================================================

def near_cmp(proximal, cmp):
    return abs(proximal - cmp) / cmp * 100 <= MAX_DISTANCE_PCT

# ============================================================
# CONTINUOUS LEG CHECK
# ============================================================

def continuous_leg_ok(df, start_idx, direction, min_c, max_c, body_pct_min, zone_range, mult):
    ref_color = candle_color(df.iloc[start_idx])
    candles = []

    i = start_idx
    while 0 <= i < len(df) and len(candles) < max_c:
        c = df.iloc[i]
        if candle_color(c) != ref_color:
            break
        if body_pct(c) < body_pct_min:
            break
        candles.append(c)
        i += direction

    if len(candles) < min_c:
        return False

    high = max(c["high"] for c in candles)
    low  = min(c["low"] for c in candles)

    return (high - low) >= zone_range * mult

# ============================================================
# ZONE SCAN (SINGLE + CONTINUOUS)
# ============================================================

def scan_zones(df):
    zones = []
    i = len(df) - 1

    while i >= 3:
        legout = df.iloc[i]
        if not is_leg(legout, DAILY_LEG_BODY_PCT):
            i -= 1
            continue

        base = []
        j = i - 1
        while j >= 0 and is_base(df.iloc[j], DAILY_BASE_BODY_PCT) and len(base) < DAILY_MAX_BASE:
            base.append(df.iloc[j])
            j -= 1

        if not base or j < 0:
            i -= 1
            continue

        legin = df.iloc[j]
        if not is_leg(legin, DAILY_LEG_BODY_PCT):
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
            candle_range(legin) >= zone_range * DAILY_LEG_IN_SINGLE_M
            or continuous_leg_ok(
                df, j, -1,
                DAILY_CONT_MIN, DAILY_CONT_MAX,
                DAILY_CONT_BODY_PCT,
                zone_range,
                DAILY_LEG_IN_CONT_M
            )
        )

        if not legin_ok:
            i -= 1
            continue

        legout_ok = (
            candle_range(legout) >= zone_range * DAILY_LEG_OUT_SINGLE_M
            or continuous_leg_ok(
                df, i, +1,
                DAILY_CONT_MIN, DAILY_CONT_MAX,
                DAILY_CONT_BODY_PCT,
                zone_range,
                DAILY_LEG_OUT_CONT_M
            )
        )

        if not legout_ok:
            i -= 1
            continue

        if not is_fresh(df, i, proximal):
            i -= 1
            continue

        zones.append({
            "type": ztype,
            "pattern": pattern,
            "proximal": proximal,
            "distal": distal,
            "idx": i
        })

        i -= 1

    return zones

# ============================================================
# MAIN RUN
# ============================================================

def run():

    monthly_csv = max(glob.glob(f"{MONTHLY_CSV_DIR}/*.csv"), key=os.path.getmtime)
    weekly_csv  = max(glob.glob(f"{WEEKLY_CSV_DIR}/*.csv"), key=os.path.getmtime)

    mdf = pd.read_csv(monthly_csv)
    wdf = pd.read_csv(weekly_csv)

    symbols = sorted(set(mdf["symbol"]).union(set(wdf["symbol"])))
    out = []

    for sym in symbols:
        print("🔍 DAILY:", sym)
        df = fetch_data(fyers_symbol(sym))
        if df is None:
            continue

        cmp = df["close"].iloc[-1]
        daily_zones = scan_zones(df)

        m_zones = mdf[mdf["symbol"] == sym].to_dict("records")
        w_zones = wdf[wdf["symbol"] == sym].to_dict("records")

        for dz in daily_zones:

            # CMP distance filter
            if not near_cmp(dz["proximal"], cmp):
                continue

            # 🔒 FINAL PRICE SIDE CHECK (ONLY NEW CONDITION)
            if dz["type"] == "SUPPLY":
                if not (dz["proximal"] > cmp and dz["distal"] > cmp):
                    continue
            else:  # DEMAND
                if not (dz["proximal"] < cmp and dz["distal"] < cmp):
                    continue

            tags = []
            d_check = dz["distal"]  # STEP-4 FIX

            for mz in m_zones:
                h = abs(mz["proximal"] - mz["distal"])
                if mz["zone_type"] == "DEMAND":
                    if mz["distal"] <= d_check <= mz["proximal"]:
                        tags.append("MONTHLY")
                    elif mz["proximal"] < d_check <= mz["proximal"] + h * MONTHLY_AREA_MULTIPLIER:
                        tags.append("MONTHLY_AREA")
                else:
                    if mz["proximal"] >= d_check >= mz["distal"]:
                        tags.append("MONTHLY")
                    elif mz["proximal"] > d_check >= mz["proximal"] - h * MONTHLY_AREA_MULTIPLIER:
                        tags.append("MONTHLY_AREA")

            for wz in w_zones:
                h = abs(wz["proximal"] - wz["distal"])
                if wz["zone_type"] == "DEMAND":
                    if wz["distal"] <= d_check <= wz["proximal"]:
                        tags.append("WEEKLY")
                    elif wz["proximal"] < d_check <= wz["proximal"] + h * WEEKLY_AREA_MULTIPLIER:
                        tags.append("WEEKLY_AREA")
                else:
                    if wz["proximal"] >= d_check >= wz["distal"]:
                        tags.append("WEEKLY")
                    elif wz["proximal"] > d_check >= wz["proximal"] - h * WEEKLY_AREA_MULTIPLIER:
                        tags.append("WEEKLY_AREA")

            if not tags:
                continue

            entry = dz["proximal"]
            sl    = dz["distal"]
            tgt   = entry + (entry-sl)*RRR if dz["type"]=="DEMAND" else entry - (sl-entry)*RRR

            out.append({
                "symbol": sym,
                "zone_type": dz["type"],
                "pattern": dz["pattern"],
                "zone_create_date": df.index[dz["idx"]].strftime("%Y-%m-%d"),
                "proximal": round(entry,2),
                "distal": round(sl,2),
                "entry": round(entry,2),
                "stop_loss": round(sl,2),
                "target": round(tgt,2),
                "confluence_tag": "+".join(sorted(set(tags)))
            })

    if out:
        fname = f"daily_zones_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        pd.DataFrame(out).to_csv(os.path.join(OUTPUT_DIR, fname), index=False)
        print("✅ DONE:", fname)
    else:
        print("⚠️ No valid daily zones")

run()
