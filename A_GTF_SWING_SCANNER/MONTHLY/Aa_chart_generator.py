import os
import glob
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from fyers_data import fetch_historical_data

# ==================================================
# USER BUTTONS
# ==================================================

ZONE_CSV_FOLDER = "output"
CHART_BASE_FOLDER = "charts"

CANDLES_BEFORE_ZONE = 15
CANDLES_AFTER_ZONE  = 1000

SAVE_CHARTS = True
DAILY_TF = "1D"

# ==================================================
# ACCESS TOKEN (config.json se securely padha)
# ==================================================

import json

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

MAX_DAYS_PER_CALL = 365

# ==================================================
# AUTO PICK LATEST CSV
# ==================================================

csv_files = glob.glob(os.path.join(ZONE_CSV_FOLDER, "*.csv"))
if not csv_files:
    raise Exception("❌ No zone scan CSV found")

LATEST_CSV = max(csv_files, key=os.path.getmtime)
print(f"📄 Using zone scan CSV: {LATEST_CSV}")

# ---------- TIMEFRAME DETECTION ----------
csv_name = os.path.basename(LATEST_CSV).lower()
if "halfyearly" in csv_name or "half_yearly" in csv_name or "half-yearly" in csv_name:
    CSV_TF = "HALFYEARLY"
elif "quarterly" in csv_name or "quarter" in csv_name:
    CSV_TF = "QUARTERLY"
elif "weekly" in csv_name:
    CSV_TF = "WEEKLY"
elif "monthly" in csv_name:
    CSV_TF = "MONTHLY"
else:
    CSV_TF = "DAILY"

print("🕒 Detected CSV Timeframe:", CSV_TF)
# -----------------------------------------

df_zones = pd.read_csv(LATEST_CSV)
print("Total zones:", len(df_zones))

# ==================================================
# CREATE RUN FOLDER (ONLY ONE FOLDER)
# ==================================================

run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_FOLDER = os.path.join(CHART_BASE_FOLDER, f"zone_charts_{run_time}")
os.makedirs(RUN_FOLDER, exist_ok=True)

# ==================================================
# SYMBOL MAP
# ==================================================

def fyers_symbol(sym):
    sym = sym.strip().upper()
    return "NSE:NIFTY50-INDEX" if sym == "NIFTY50" else f"NSE:{sym}-EQ"

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

# ==================================================
# FETCH PRICE DATA (DAILY ONLY)
# ==================================================

def fetch_price_data(symbol, anchor_date):
    today = date.today()
    start = anchor_date - timedelta(days=200)
    end   = today
    if start < date(2000, 1, 1):
        start = date(2000, 1, 1)

    # Step 1: Try cache
    try:
        cached = load_cached_data(symbol, DAILY_TF)
        if cached is not None and len(cached) > 0:
            first_date = cached.index[0].date()
            if first_date > start + timedelta(days=7):
                old_dfs = []
                cur = start
                while cur < first_date:
                    cur_end = min(cur + timedelta(days=MAX_DAYS_PER_CALL), first_date - timedelta(days=1))
                    try:
                        df_old_chunk = fetch_historical_data(
                            symbol, DAILY_TF,
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
                    save_cached_data(symbol, DAILY_TF, df_merged)
                    cached = df_merged
            mask   = (cached.index.date >= start) & (cached.index.date <= end)
            result = cached.loc[mask].copy()
            if not result.empty:
                result.index = pd.to_datetime(result.index).tz_localize(None)
                return result
    except Exception:
        pass

    # Step 2: Fallback — fetch from API
    dfs = []
    cur = start
    while cur <= end:
        cur_end = min(cur + timedelta(days=MAX_DAYS_PER_CALL), end)
        df = fetch_historical_data(
            symbol,
            DAILY_TF,
            cur.strftime("%Y-%m-%d"),
            cur_end.strftime("%Y-%m-%d"),
            ACCESS_TOKEN
        )
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

# ==================================================
# RESAMPLE HELPERS
# ==================================================

def resample_weekly(df):
    return df.resample("W").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

def resample_monthly(df):
    return df.resample("ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

def resample_quarterly(df):
    return df.resample("QE").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

def resample_halfyearly(df):
    return df.resample("6ME").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

# ==================================================
# MAIN LOOP — GROUP BY SYMBOL
# ==================================================

for symbol, df_symbol_zones in df_zones.groupby("symbol"):

    print(f"\n📊 Generating chart for symbol → {symbol}")

    # Earliest zone date for this symbol
    zone_dates_dt = pd.to_datetime(df_symbol_zones["zone_create_date"])
    earliest_zone_date = zone_dates_dt.min().date()

    fy_sym = fyers_symbol(symbol)

    price_df = fetch_price_data(fy_sym, earliest_zone_date)

    if price_df is None or price_df.empty:
        print("⚠️ Price data not available")
        continue

    # ---------- TIMEFRAME CONVERSION ----------
    if CSV_TF == "WEEKLY":
        price_df = resample_weekly(price_df)
    elif CSV_TF == "MONTHLY":
        price_df = resample_monthly(price_df)
    elif CSV_TF == "QUARTERLY":
        price_df = resample_quarterly(price_df)
    elif CSV_TF == "HALFYEARLY":
        price_df = resample_halfyearly(price_df)
    # ------------------------------------------

    # ---------- SMART ANCHOR ----------
    earliest_dt = zone_dates_dt.min()

    if earliest_dt not in price_df.index:
        earlier = price_df.index[price_df.index <= earliest_dt]
        if len(earlier) > 0:
            earliest_dt = earlier[-1]

    anchor_idx = price_df.index.get_loc(earliest_dt)

    start_idx = max(anchor_idx - CANDLES_BEFORE_ZONE, 0)
    end_idx   = min(anchor_idx + CANDLES_AFTER_ZONE, len(price_df) - 1)

    df_slice = price_df.iloc[start_idx:end_idx + 1].copy()
    df_slice["x"] = df_slice.index.strftime("%Y-%m-%d")

    # ==================================================
    # CHART
    # ==================================================

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_slice["x"],
        open=df_slice["open"],
        high=df_slice["high"],
        low=df_slice["low"],
        close=df_slice["close"],
        increasing=dict(fillcolor="green", line=dict(color="green")),
        decreasing=dict(fillcolor="black", line=dict(color="black")),
        name="Price"
    ))

    # ==================================================
    # DRAW ALL ZONES FOR THIS SYMBOL
    # ==================================================

    for _, row in df_symbol_zones.iterrows():

        zone_type      = row["zone_type"]
        confluence_tag = row.get("confluence_tag", "")
        zone_date      = pd.to_datetime(row["zone_create_date"])

        if zone_date not in df_slice.index:
            earlier = df_slice.index[df_slice.index <= zone_date]
            if len(earlier) == 0:
                continue
            zone_date = earlier[-1]

        proximal = row["proximal"]
        distal   = row["distal"]

        fig.add_shape(
            type="rect",
            x0=df_slice.loc[zone_date, "x"],
            x1=df_slice["x"].iloc[-1],
            y0=min(proximal, distal),
            y1=max(proximal, distal),
            fillcolor="rgba(0,150,0,0.18)" if zone_type == "DEMAND" else "rgba(255,0,0,0.18)",
            line_width=0
        )

        fig.add_hline(y=row["entry"], line_dash="dot", line_color="blue",
                      annotation_text=f"ENTRY {row['entry']}", annotation_position="right",
                      annotation_font_size=64)

        fig.add_hline(y=row["stop_loss"], line_color="red",
                      annotation_text=f"SL {row['stop_loss']}", annotation_position="right",
                      annotation_font_size=64)

        fig.add_hline(y=row["target"], line_color="green",
                      annotation_text=f"TARGET {row['target']}", annotation_position="right",
                      annotation_font_size=64)

        # ----- CONFLUENCE TAG -----
        if isinstance(confluence_tag, str) and confluence_tag.strip() != "":
            fig.add_annotation(
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                text=f"CONFLUENCE: {confluence_tag}",
                showarrow=False,
                align="left",
                font=dict(size=64, color="black"),
                bgcolor="rgba(255,255,0,0.85)",
                bordercolor="black",
                borderwidth=2
            )

    fig.update_layout(
        title=dict(
            text=f"{symbol} | ALL ZONES | {CSV_TF}",
            font=dict(size=64),
            x=0.01
        ),
        xaxis=dict(type="category", tickfont=dict(size=34)),
        yaxis=dict(tickfont=dict(size=34)),
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        width=3200,
        height=3200,
        margin=dict(t=260, l=250, r=300, b=150),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )

    # ==================================================
    # SAVE HTML + AUTO DOWNLOAD IMAGE
    # ==================================================

    if SAVE_CHARTS:

        fname = f"{symbol}_ALL_ZONES.html"

        chart_div = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            config={"displaylogo": False}
        )

        custom_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{symbol}_ALL_ZONES</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            background: white;
        }}
        #chart-container {{
            width: min(100vw, 100vh);
            height: min(100vw, 100vh);
            max-width: 100%;
            max-height: 100%;
            position: relative;
            overflow: hidden;
        }}
        #chart-wrapper {{
            width: 3200px;
            height: 3200px;
            transform-origin: top left;
        }}
    </style>
</head>
<body>
    <div id="chart-container">
        <div id="chart-wrapper">
            {chart_div}
        </div>
    </div>
    <script>
        function scaleChart() {{
            const container = document.getElementById('chart-container');
            const wrapper = document.getElementById('chart-wrapper');
            const containerWidth = container.offsetWidth;
            const scale = containerWidth / 3200;
            wrapper.style.transform = `scale(${{scale}})`;
        }}

        window.addEventListener('load', scaleChart);
        window.addEventListener('resize', scaleChart);

        document.addEventListener("DOMContentLoaded", function() {{
            setTimeout(function() {{
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                Plotly.downloadImage(gd, {{
                    format: 'png',
                    filename: '{symbol}_ALL_ZONES',
                    height: 3200,
                    width: 3200,
                    scale: 3
                }});
            }}, 1500);
        }});
    </script>
</body>
</html>"""

        with open(os.path.join(RUN_FOLDER, fname), 'w', encoding='utf-8') as f:
            f.write(custom_html)

        print(f"✅ Saved: {fname}")

print("\n✅ ALL ZONE CHARTS GENERATED SUCCESSFULLY\n")
