import os
import glob
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
from fyers_data import fetch_historical_data

# ==================================================
# USER BUTTONS
# ==================================================

ZONE_CSV_FOLDER   = "output/minutes"
CHART_BASE_FOLDER = "charts"

CANDLES_BEFORE_ZONE = 50
CANDLES_AFTER_ZONE  = 1000

SAVE_CHARTS = True

# -------- MINUTE TIMEFRAME BUTTON --------
INTRADAY_TF       = "15"   # 1,3,5,10,15,30,60
MAX_DAYS_PER_CALL = 90     # Intraday API limit

# ==================================================
# LOAD ACCESS TOKEN FROM config.json
# ==================================================

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

# ==================================================
# ANNOTATION OVERLAP PREVENTION
# ==================================================

# Minimum gap between two annotation labels (0.4% of visible price range)
ANNOTATION_MIN_GAP_PCT = 0.004

def resolve_overlaps(levels, price_range):
    """
    levels     : list of dicts  {y, label, color, dash}
    price_range: high - low of visible df_slice
    Returns    : same list with extra key 'ann_y' — adjusted label position.
                 The actual line is always drawn at real 'y'.
                 Only the annotation text shifts to avoid stacking.
    """
    if not levels:
        return levels

    min_gap = price_range * ANNOTATION_MIN_GAP_PCT

    # Sort by actual price level
    levels = sorted(levels, key=lambda d: d["y"])

    # Pass 1 — push labels UP if too close to previous
    levels[0]["ann_y"] = levels[0]["y"]
    for i in range(1, len(levels)):
        prev_ann  = levels[i - 1]["ann_y"]
        curr_real = levels[i]["y"]
        if curr_real - prev_ann < min_gap:
            levels[i]["ann_y"] = prev_ann + min_gap
        else:
            levels[i]["ann_y"] = curr_real

    return levels


def add_level_lines(fig, levels):
    """
    Draws each horizontal line at real price + annotation at shifted ann_y.
    A tiny arrow connects the label to the actual line if shifted.
    """
    for lvl in levels:
        real_y  = lvl["y"]
        ann_y   = lvl["ann_y"]
        color   = lvl["color"]
        dash    = lvl.get("dash", "solid")

        # Horizontal line spanning full chart width
        fig.add_shape(
            type="line",
            x0=0, x1=1, xref="paper",
            y0=real_y, y1=real_y,
            line=dict(color=color, width=1.5, dash=dash)
        )

        # Annotation at shifted position, arrow points to real line
        fig.add_annotation(
            x=1.0,
            xref="paper",
            xanchor="left",
            y=ann_y,
            yref="y",
            text=lvl["label"],
            showarrow=True,
            ax=0,       # arrow tip x = same as annotation x
            axref="pixel",  # fixed: paper invalid, only pixel allowed
            ay=real_y,  # arrow tip y = actual price
            ayref="y",
            arrowhead=0,
            arrowwidth=1,
            arrowcolor=color,
            font=dict(size=30, color=color),
            bgcolor="white",
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        )

# ==================================================
# AUTO PICK LATEST CSV
# ==================================================

csv_files = glob.glob(os.path.join(ZONE_CSV_FOLDER, "*.csv"))
if not csv_files:
    raise Exception("❌ No zone scan CSV found")

LATEST_CSV = max(csv_files, key=os.path.getmtime)
print(f"📄 Using zone scan CSV: {LATEST_CSV}")

df_zones = pd.read_csv(LATEST_CSV)
print("Total zones:", len(df_zones))

# ==================================================
# CREATE RUN FOLDER
# ==================================================

run_time   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
# FETCH INTRADAY DATA — end = today naturally handled
# ==================================================

def fetch_price_data(symbol, start_date, end_date):
    # Step 1: Try cache
    try:
        cached = load_cached_data(symbol, INTRADAY_TF)
        if cached is not None and len(cached) > 0:
            first_date = cached.index[0].date()
            if first_date > start_date + timedelta(days=7):
                old_dfs = []
                cur = start_date
                while cur < first_date:
                    cur_end = min(cur + timedelta(days=MAX_DAYS_PER_CALL), first_date - timedelta(days=1))
                    try:
                        df_old_chunk = fetch_historical_data(
                            symbol, INTRADAY_TF,
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
                    save_cached_data(symbol, INTRADAY_TF, df_merged)
                    cached = df_merged
            mask   = (cached.index.date >= start_date) & (cached.index.date <= end_date)
            result = cached.loc[mask].copy()
            if not result.empty:
                result.index = pd.to_datetime(result.index).tz_localize(None)
                return result
    except Exception:
        pass

    # Step 2: Fallback — fetch from API
    dfs = []
    cur = start_date

    while cur <= end_date:
        cur_end = min(cur + timedelta(days=MAX_DAYS_PER_CALL), end_date)

        df = fetch_historical_data(
            symbol,
            INTRADAY_TF,
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
# MAIN LOOP — GROUP BY SYMBOL
# ==================================================

for symbol, df_symbol_zones in df_zones.groupby("symbol"):

    print(f"\n📊 Generating chart for symbol → {symbol}")

    fy_sym = fyers_symbol(symbol)

    # Earliest zone date for this symbol
    zone_dates_dt      = pd.to_datetime(df_symbol_zones["zone_create_date"])
    earliest_zone_date = zone_dates_dt.min().date()

    # FETCH: zone start → TODAY  (no separate Phase 2 needed)
    zone_start = earliest_zone_date - timedelta(days=200)
    zone_end   = date.today()   # end = today naturally

    price_df = fetch_price_data(fy_sym, zone_start, zone_end)

    if price_df is None or price_df.empty:
        print("⚠️ Price data not available")
        continue

    # ---------- SMART ANCHOR ----------
    earliest_dt = zone_dates_dt.min()

    if earliest_dt not in price_df.index:
        earlier = price_df.index[price_df.index <= earliest_dt]
        if len(earlier) > 0:
            earliest_dt = earlier[-1]

    anchor_idx = price_df.index.get_loc(earliest_dt)
    start_idx  = max(anchor_idx - CANDLES_BEFORE_ZONE, 0)
    end_idx    = min(anchor_idx + CANDLES_AFTER_ZONE, len(price_df) - 1)

    df_slice = price_df.iloc[start_idx:end_idx + 1].copy()
    df_slice["x"] = df_slice.index.strftime("%Y-%m-%d %H:%M")

    # Price range for gap calculation
    price_range = df_slice["high"].max() - df_slice["low"].min()

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

    # Volume bars — color matches candle, 15% opacity
    vol_colors = [
        "rgba(0,128,0,0.15)" if c >= o else "rgba(0,0,0,0.15)"
        for o, c in zip(df_slice["open"], df_slice["close"])
    ]

    fig.add_trace(go.Bar(
        x=df_slice["x"],
        y=df_slice["volume"],
        marker_color=vol_colors,
        name="Volume",
        yaxis="y2"
    ))

    # ==================================================
    # COLLECT ALL LEVELS FIRST — then resolve overlaps
    # ==================================================

    all_levels         = []
    confluence_tags_seen = []   # unique tags collect — overlap avoid

    for _, row in df_symbol_zones.iterrows():

        zone_type      = row["zone_type"]
        confluence_tag = row.get("confluence_tag", "")
        zone_date      = pd.to_datetime(row["zone_create_date"])

        # Smart zone date match
        if zone_date not in df_slice.index:
            earlier = df_slice.index[df_slice.index <= zone_date]
            if len(earlier) == 0:
                continue
            zone_date = earlier[-1]

        proximal = row["proximal"]
        distal   = row["distal"]

        # Zone rectangle
        fig.add_shape(
            type="rect",
            x0=df_slice.loc[zone_date, "x"],
            x1=df_slice["x"].iloc[-1],
            y0=min(proximal, distal),
            y1=max(proximal, distal),
            fillcolor="rgba(0,150,0,0.18)" if zone_type == "DEMAND" else "rgba(255,0,0,0.18)",
            line_width=0
        )

        # Collect Entry / SL / Target
        all_levels.append({"y": row["entry"],     "label": f"ENTRY {row['entry']}",   "color": "blue",  "dash": "dot"})
        all_levels.append({"y": row["stop_loss"], "label": f"SL {row['stop_loss']}",  "color": "red",   "dash": "solid"})
        all_levels.append({"y": row["target"],    "label": f"TARGET {row['target']}", "color": "green", "dash": "solid"})

        # Collect confluence tag — only unique, no duplicates
        if isinstance(confluence_tag, str) and confluence_tag.strip() != "":
            tag = confluence_tag.strip()
            if tag not in confluence_tags_seen:
                confluence_tags_seen.append(tag)

    # Resolve overlaps across ALL zones, then draw lines + annotations
    resolved = resolve_overlaps(all_levels, price_range)
    add_level_lines(fig, resolved)

    # ----- DRAW CONFLUENCE TAGS — stacked vertically, no overlap -----
    # Each tag gets its own yellow box, spaced 0.06 apart
    for i, tag in enumerate(confluence_tags_seen):
        y_pos = 0.99 - i * 0.06
        fig.add_annotation(
            x=0.01,
            y=y_pos,
            xref="paper",
            yref="paper",
            text=f"CONFLUENCE: {tag}",
            showarrow=False,
            align="left",
            font=dict(size=64, color="black"),
            bgcolor="rgba(255,255,0,0.85)",
            bordercolor="black",
            borderwidth=2
        )

    fig.update_layout(
        title=dict(
            text=f"{symbol} | {INTRADAY_TF} MIN | ALL ZONES",
            font=dict(size=64),
            x=0.01
        ),
        xaxis=dict(type="category", tickfont=dict(size=34)),
        yaxis=dict(tickfont=dict(size=34), domain=[0.25, 1.0]),
        yaxis2=dict(
            tickfont=dict(size=24),
            domain=[0.0, 0.20],
            showgrid=False,
            zeroline=False,
        ),
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

        fname = f"{symbol}_{INTRADAY_TF}MIN_ALL_ZONES.html"

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
    <title>{symbol}_{INTRADAY_TF}MIN_ALL_ZONES</title>
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
            const wrapper   = document.getElementById('chart-wrapper');
            const scale     = container.offsetWidth / 3200;
            wrapper.style.transform = `scale(${{scale}})`;
        }}

        window.addEventListener('load', scaleChart);
        window.addEventListener('resize', scaleChart);

        document.addEventListener("DOMContentLoaded", function() {{
            setTimeout(function() {{
                var gd = document.getElementsByClassName('plotly-graph-div')[0];
                Plotly.downloadImage(gd, {{
                    format:   'png',
                    filename: '{symbol}_{INTRADAY_TF}MIN_ALL_ZONES',
                    height:   3200,
                    width:    3200,
                    scale:    3
                }});
            }}, 1500);
        }});
    </script>
</body>
</html>"""

        with open(os.path.join(RUN_FOLDER, fname), 'w', encoding='utf-8') as f:
            f.write(custom_html)

        print(f"✅ Saved: {fname}")

print("\n✅ ALL INTRADAY ZONE CHARTS GENERATED SUCCESSFULLY\n")
