# ==================================================
# BULK DATA DOWNLOADER — GTF SMART CACHE SYSTEM
# ==================================================
# Run this script ONCE to pre-download all historical data.
# After that, scanners will read from cache instead of hitting API.
#
# Usage:
#   python download_data.py
#
# Data is saved to:
#   data/1D/{symbol}.parquet   ← Daily OHLCV (10 years)
#   data/15m/{symbol}.parquet  ← 15-minute OHLCV (100 days)
# ==================================================

import os
import json
import time
import unicodedata
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel


# ==================== CONFIG ====================

HISTORY_1D_YEARS  = 10    # Max needed by any scanner (Quarterly/HalfYearly use 10 years)
HISTORY_15M_DAYS  = 100   # ~3 months for intraday (scanners use HISTORY_YEARS=2/12)
API_LIMIT_1D      = 365   # Max days per API call for daily data
API_LIMIT_15M     = 90    # Max days per API call for intraday data


# ==================== FIND CONFIG ====================

def find_config():
    """Find config.json by walking up parent directories."""
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


# ==================== FYERS CLIENT ====================

def get_fyers_client(access_token):
    return fyersModel.FyersModel(
        client_id=access_token.split(":")[0],
        token=access_token,
        log_path=""
    )


# ==================== FETCH HISTORICAL DATA ====================

def fetch_historical_data(symbol, timeframe, start_date, end_date, access_token):
    """Fetch historical candle data from FYERS API."""
    fyers = get_fyers_client(access_token)

    data = {
        "symbol": symbol,
        "resolution": timeframe,
        "date_format": "1",
        "range_from": start_date,
        "range_to": end_date,
        "cont_flag": "1"
    }

    response = fyers.history(data=data)

    if response.get("s") != "ok":
        raise Exception(f"FYERS ERROR: {response}")

    candles = response["candles"]

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata")
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df.set_index("timestamp", inplace=True)

    return df


# ==================== FILE HELPERS ====================

def safe_filename(symbol):
    """Convert FYERS symbol to safe filename: NSE:RELIANCE-EQ → NSE_RELIANCE-EQ"""
    return symbol.replace(":", "_")


def fyers_symbol(sym):
    """Convert plain symbol to FYERS format."""
    sym = sym.strip().upper()
    return "NSE:NIFTY50-INDEX" if sym == "NIFTY50" else f"NSE:{sym}-EQ"


def get_data_dir():
    """Get the data/ directory next to this script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ==================== LOAD / SAVE PARQUET ====================

def load_parquet(parquet_path):
    """Load parquet file if it exists."""
    if not os.path.exists(parquet_path):
        return None
    try:
        df = pd.read_parquet(parquet_path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"    ⚠️  Could not read parquet: {e}")
        return None


def save_parquet(parquet_path, df):
    """Save dataframe to parquet."""
    try:
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path)
    except Exception as e:
        print(f"    ⚠️  Could not save parquet: {e}")


# ==================== DOWNLOAD SINGLE SYMBOL ====================

def download_symbol(symbol_plain, timeframe, history_days, api_limit, access_token, data_dir):
    """
    Download data for one symbol. Incremental update if cache exists.
    Returns (status, candle_count)
    """
    fyers_sym  = fyers_symbol(symbol_plain)
    tf_folder  = "1D" if timeframe == "1D" else "15m"
    parquet_path = os.path.join(data_dir, tf_folder, f"{safe_filename(fyers_sym)}.parquet")

    today = datetime.now().date()
    full_start = today - timedelta(days=history_days)

    # Check if cache exists
    cached = load_parquet(parquet_path)

    if cached is not None and len(cached) > 0:
        first_date = cached.index[0].date()
        last_date  = cached.index[-1].date()
        required_start = full_start

        if last_date >= today - timedelta(days=1):
            if first_date <= required_start + timedelta(days=7):
                print(f"    ✅ UP-TO-DATE (range: {first_date} → {last_date})")
                return "up-to-date", len(cached)
            else:
                # End is fresh but cache doesn't go back far enough — extend history
                print(f"    📥 Extending history: fetching {required_start} to {first_date}")
                old_dfs = []
                cur = required_start
                while cur < first_date:
                    cur_end = min(cur + timedelta(days=api_limit), first_date - timedelta(days=1))
                    try:
                        df_chunk = fetch_historical_data(
                            fyers_sym, timeframe,
                            cur.strftime("%Y-%m-%d"),
                            cur_end.strftime("%Y-%m-%d"),
                            access_token
                        )
                        if df_chunk is not None and not df_chunk.empty:
                            old_dfs.append(df_chunk)
                    except Exception as e:
                        print(f"    ⚠️  Chunk error {cur} → {cur_end}: {e}")
                    cur = cur_end + timedelta(days=1)
                if old_dfs:
                    df_old    = pd.concat(old_dfs)
                    df_merged = pd.concat([df_old, cached])
                    df_merged = df_merged[~df_merged.index.duplicated()]
                    df_merged.sort_index(inplace=True)
                    save_parquet(parquet_path, df_merged)
                    print(f"    ✅ History extended → {len(df_merged)} candles saved")
                    return "updated", len(df_merged)
                else:
                    print(f"    ✅ UP-TO-DATE (history extension returned no data)")
                    return "up-to-date", len(cached)

        # Incremental update — fetch only missing data
        fetch_start = last_date + timedelta(days=1)
        print(f"    📥 Incremental update from {fetch_start} to {today}")

        new_dfs = []
        cur = fetch_start
        while cur <= today:
            cur_end = min(cur + timedelta(days=api_limit), today)
            try:
                df_chunk = fetch_historical_data(
                    fyers_sym, timeframe,
                    cur.strftime("%Y-%m-%d"),
                    cur_end.strftime("%Y-%m-%d"),
                    access_token
                )
                if df_chunk is not None and not df_chunk.empty:
                    new_dfs.append(df_chunk)
            except Exception as e:
                print(f"    ⚠️  Chunk error {cur} → {cur_end}: {e}")
            cur = cur_end + timedelta(days=1)

        if new_dfs:
            df_new    = pd.concat(new_dfs)
            df_merged = pd.concat([cached, df_new])
            df_merged = df_merged[~df_merged.index.duplicated()]
            df_merged.sort_index(inplace=True)
            save_parquet(parquet_path, df_merged)
            print(f"    ✅ Merged → {len(df_merged)} candles saved")
            return "updated", len(df_merged)
        else:
            print(f"    ℹ️  No new data fetched (market may be closed)")
            return "no-new-data", len(cached)

    # Full download — no cache
    print(f"    📥 Full download from {full_start} to {today}")
    dfs = []
    cur = full_start
    while cur <= today:
        cur_end = min(cur + timedelta(days=api_limit), today)
        try:
            df_chunk = fetch_historical_data(
                fyers_sym, timeframe,
                cur.strftime("%Y-%m-%d"),
                cur_end.strftime("%Y-%m-%d"),
                access_token
            )
            if df_chunk is not None and not df_chunk.empty:
                dfs.append(df_chunk)
        except Exception as e:
            print(f"    ⚠️  Chunk error {cur} → {cur_end}: {e}")
        cur = cur_end + timedelta(days=1)

    if not dfs:
        print(f"    ❌ No data received")
        return "failed", 0

    df = pd.concat(dfs)
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)
    save_parquet(parquet_path, df)
    print(f"    ✅ Downloaded → {len(df)} candles saved")
    return "downloaded", len(df)


# ==================== MAIN ====================

def main():
    start_time = time.time()

    print("=" * 60)
    print("  📦 GTF BULK DATA DOWNLOADER")
    print("=" * 60)

    # Load access token
    config = find_config()
    access_token = config["access_token"]
    print(f"\n🔑 Access token loaded")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, "data")

    # Load symbol lists
    symbols_1d_path  = os.path.join(script_dir, "symbols_1D.csv")
    symbols_15m_path = os.path.join(script_dir, "symbols_15m.csv")

    with open(symbols_1d_path) as f:
        symbols_1d = [s.strip() for s in f if s.strip()]

    with open(symbols_15m_path) as f:
        symbols_15m = [s.strip() for s in f if s.strip()]

    print(f"\n📋 Symbols to download:")
    print(f"   1D  → {len(symbols_1d)} symbols ({HISTORY_1D_YEARS} years history)")
    print(f"   15m → {len(symbols_15m)} symbols ({HISTORY_15M_DAYS} days history)")

    history_1d_days = HISTORY_1D_YEARS * 365

    # ---- DOWNLOAD 1D DATA ----
    print(f"\n{'='*60}")
    print(f"  📊 DOWNLOADING 1D DATA ({len(symbols_1d)} symbols)")
    print(f"{'='*60}")

    stats_1d = {"up-to-date": 0, "updated": 0, "downloaded": 0, "no-new-data": 0, "failed": 0}
    candles_1d = 0

    for i, sym in enumerate(symbols_1d, 1):
        print(f"\n[{i}/{len(symbols_1d)}] {sym}")
        status, count = download_symbol(
            sym, "1D", history_1d_days, API_LIMIT_1D, access_token, data_dir
        )
        stats_1d[status] = stats_1d.get(status, 0) + 1
        candles_1d += count

    # ---- DOWNLOAD 15m DATA ----
    print(f"\n{'='*60}")
    print(f"  📊 DOWNLOADING 15m DATA ({len(symbols_15m)} symbols)")
    print(f"{'='*60}")

    stats_15m = {"up-to-date": 0, "updated": 0, "downloaded": 0, "no-new-data": 0, "failed": 0}
    candles_15m = 0

    for i, sym in enumerate(symbols_15m, 1):
        print(f"\n[{i}/{len(symbols_15m)}] {sym}")
        status, count = download_symbol(
            sym, "15", HISTORY_15M_DAYS, API_LIMIT_15M, access_token, data_dir
        )
        stats_15m[status] = stats_15m.get(status, 0) + 1
        candles_15m += count

    # ---- SUMMARY ----
    elapsed   = int(time.time() - start_time)
    mins, sec = divmod(elapsed, 60)
    time_str  = f"{mins}m {sec}s" if mins else f"{sec}s"

    cache_display = "..." + data_dir[-20:] if len(data_dir) > 23 else data_dir

    def vlen(s):
        """Visual terminal width: wide emoji/chars count as 2 columns."""
        w = 0
        for ch in s:
            eaw = unicodedata.east_asian_width(ch)
            w += 2 if eaw in ('W', 'F') else 1
        return w

    BOX_W  = 38  # total inner width between the two ║ borders
    CELL_W = 25  # inner cell width between ┌ ┐ / └ ┘ borders

    def outer_line(text=""):
        pad = BOX_W - vlen(text)
        return f"║  {text}{' ' * max(0, pad - 2)}║"

    def inner_row(label, value):
        val_str = str(value)
        content = f"{label}: {val_str}"
        pad = CELL_W - vlen(content)
        row = f"│ {content}{' ' * max(0, pad - 1)}│"
        opad = BOX_W - vlen(row)
        return f"║  {row}{' ' * max(0, opad - 2)}║"

    def inner_sep():
        bar = "┌" + "─" * CELL_W + "┐"
        opad = BOX_W - vlen(bar)
        return f"║  {bar}{' ' * max(0, opad - 2)}║"

    def inner_end():
        bar = "└" + "─" * CELL_W + "┘"
        opad = BOX_W - vlen(bar)
        return f"║  {bar}{' ' * max(0, opad - 2)}║"

    print()
    print(f"╔{'═' * BOX_W}╗")
    print(outer_line("🎉 DOWNLOAD COMPLETE!"))
    print(f"╠{'═' * BOX_W}╣")
    print(outer_line())
    print(outer_line(f"📊 1D DATA  ({len(symbols_1d)} symbols)"))
    print(inner_sep())
    print(inner_row("✅ Downloaded ", stats_1d['downloaded']))
    print(inner_row("🔄 Updated    ", stats_1d['updated']))
    print(inner_row("📦 Up-to-date ", stats_1d['up-to-date']))
    print(inner_row("⏸  No new data", stats_1d['no-new-data']))
    print(inner_row("❌ Failed     ", stats_1d['failed']))
    print(inner_row("📈 Candles    ", candles_1d))
    print(inner_end())
    print(outer_line())
    print(outer_line(f"📊 15m DATA ({len(symbols_15m)} symbols)"))
    print(inner_sep())
    print(inner_row("✅ Downloaded ", stats_15m['downloaded']))
    print(inner_row("🔄 Updated    ", stats_15m['updated']))
    print(inner_row("📦 Up-to-date ", stats_15m['up-to-date']))
    print(inner_row("⏸  No new data", stats_15m['no-new-data']))
    print(inner_row("❌ Failed     ", stats_15m['failed']))
    print(inner_row("📈 Candles    ", candles_15m))
    print(inner_end())
    print(outer_line())
    print(outer_line(f"⏱  Time  : {time_str}"))
    print(outer_line(f"📁 Cache : {cache_display}"))
    print(outer_line())
    print(f"╚{'═' * BOX_W}╝")


if __name__ == "__main__":
    main()
