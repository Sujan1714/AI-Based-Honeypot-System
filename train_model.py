"""
AI Honeypot Dashboard
======================
Interactive Streamlit dashboard for real-time honeypot monitoring.

Run: streamlit run dashboard/app.py
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Path setup ─────────────────────────────────────────────────────────────────
DASH_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR  = os.path.dirname(DASH_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

CSV_FILE     = os.path.join(BASE_DIR, "data", "processed_logs.csv")
LOG_FILE     = os.path.join(BASE_DIR, "data", "honeypot_logs.json")
MODEL_FILE   = os.path.join(BASE_DIR, "models", "rf_model.pkl")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "models", "metrics.json")

FEATURE_COLS = ["login_attempts", "command_count", "session_duration", "has_malicious_cmd"]

# ── Color palette ──────────────────────────────────────────────────────────────
COLORS = {
    "Normal"    : "#00d4aa",
    "Suspicious": "#ffc107",
    "Malicious" : "#ff4b6e",
    "bg"        : "#0d1117",
    "card"      : "#161b22",
    "border"    : "#30363d",
    "text"      : "#e6edf3",
    "accent"    : "#58a6ff",
}

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Honeypot Dashboard",
    page_icon="🍯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Space Grotesk', sans-serif;
      background-color: #0d1117;
      color: #e6edf3;
  }
  .main { background: #0d1117; padding-top: 0.5rem; }
  .block-container { padding: 1rem 2rem 2rem 2rem; max-width: 100%; }

  /* KPI Cards */
  .kpi-card {
      background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 1.2rem 1.4rem;
      margin-bottom: 1rem;
      position: relative;
      overflow: hidden;
  }
  .kpi-card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 3px;
  }
  .kpi-card.blue::before  { background: linear-gradient(90deg, #58a6ff, #1f6feb); }
  .kpi-card.red::before   { background: linear-gradient(90deg, #ff4b6e, #da3633); }
  .kpi-card.yellow::before{ background: linear-gradient(90deg, #ffc107, #e09f00); }
  .kpi-card.green::before { background: linear-gradient(90deg, #00d4aa, #26a641); }
  .kpi-card.purple::before{ background: linear-gradient(90deg, #bc8cff, #8957e5); }

  .kpi-value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 2.2rem;
      font-weight: 600;
      line-height: 1;
      margin: 0.3rem 0;
  }
  .kpi-label { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
  .kpi-delta { font-size: 0.85rem; margin-top: 0.3rem; }
  .kpi-icon  { font-size: 1.8rem; float: right; opacity: 0.6; margin-top: -0.2rem; }

  /* Section headers */
  .section-header {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #58a6ff;
      margin: 1.5rem 0 0.8rem 0;
      padding-bottom: 0.4rem;
      border-bottom: 1px solid #21262d;
  }

  /* Threat badge */
  .badge-malicious  { background:#3d1b1b; color:#ff4b6e; border:1px solid #ff4b6e44; border-radius:999px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .badge-suspicious { background:#2d2a00; color:#ffc107; border:1px solid #ffc10744; border-radius:999px; padding:2px 10px; font-size:0.75rem; font-weight:600; }
  .badge-normal     { background:#0d2018; color:#00d4aa; border:1px solid #00d4aa44; border-radius:999px; padding:2px 10px; font-size:0.75rem; font-weight:600; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: #161b22;
      border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
      color: #58a6ff;
      font-size: 0.9rem;
  }

  /* Streamlit table */
  .stDataFrame { border: 1px solid #30363d; border-radius: 8px; }

  /* Plotly charts bg */
  .js-plotly-plot .plotly { background: transparent !important; }

  /* Title */
  .dash-title {
      font-family: 'JetBrains Mono', monospace;
      font-size: 1.5rem;
      font-weight: 600;
      color: #e6edf3;
      display: flex;
      align-items: center;
      gap: 0.6rem;
  }
  .dash-subtitle {
      font-size: 0.85rem;
      color: #8b949e;
      margin-top: 0.2rem;
  }
  .live-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: #00d4aa;
      display: inline-block;
      animation: pulse 2s infinite;
      margin-right: 6px;
  }
  @keyframes pulse {
      0%, 100% { opacity: 1; }
      50%       { opacity: 0.3; }
  }
  .alert-bar {
      background: linear-gradient(90deg, #3d1b1b, #2d1515);
      border: 1px solid #ff4b6e44;
      border-left: 3px solid #ff4b6e;
      border-radius: 6px;
      padding: 0.6rem 1rem;
      font-size: 0.85rem;
      color: #ff4b6e;
      margin-bottom: 1rem;
  }
  /* Metrics grid */
  div[data-testid="column"] { padding: 0 0.4rem; }
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=10)   # refresh every 10 s
def load_data():
    if not os.path.exists(CSV_FILE):
        return pd.DataFrame()
    df = pd.read_csv(CSV_FILE)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
        return joblib.load(MODEL_FILE), joblib.load(ENCODER_FILE)
    return None, None


@st.cache_data(ttl=10)
def load_raw_logs():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        return json.load(f)


def load_metrics():
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return {}


# ── ML Prediction helper ───────────────────────────────────────────────────────
def run_predictions(df, model, le):
    """Add predicted_label and confidence columns if model is available."""
    if model is None or df.empty:
        if "label" in df.columns:
            df["predicted_label"] = df["label"]
        else:
            df["predicted_label"] = "Unknown"
        df["confidence"] = 0.0
        return df

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        df["predicted_label"] = df.get("label", "Unknown")
        df["confidence"] = 0.0
        return df

    X = df[FEATURE_COLS].fillna(0).values
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)
    df = df.copy()
    df["predicted_label"] = le.inverse_transform(y_pred)
    df["confidence"]      = (y_proba.max(axis=1) * 100).round(1)
    return df


# ── Plotly theme helper ────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    font_color   ="#8b949e",
    font_family  ="Space Grotesk, sans-serif",
    margin       =dict(l=10, r=10, t=30, b=10),
    showlegend   =True,
    legend       =dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8b949e")),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:1rem 0 0.5rem;'>
      <div style='font-size:2.5rem;'>🍯</div>
      <div style='font-size:1rem; font-weight:700; color:#e6edf3;'>Honeypot AI</div>
      <div style='font-size:0.72rem; color:#8b949e;'>Cyber Attack Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🔧 Filters")

    df_all = load_data()

    # Auto-refresh toggle
    auto_refresh = st.toggle("⚡ Auto-refresh (10s)", value=False)

    # Time range
    time_range = st.selectbox(
        "📅 Time Range",
        ["Last 1 hour", "Last 6 hours", "Last 24 hours", "Last 7 days", "All time"],
        index=3,
    )

    # IP filter
    ip_options = ["All IPs"]
    if not df_all.empty and "ip" in df_all.columns:
        ip_options += sorted(df_all["ip"].dropna().unique().tolist())
    selected_ip = st.selectbox("🌐 Filter by IP", ip_options)

    # Attack type filter
    type_options = ["All Types", "Normal", "Suspicious", "Malicious"]
    selected_type = st.multiselect(
        "🎯 Attack Type",
        ["Normal", "Suspicious", "Malicious"],
        default=["Normal", "Suspicious", "Malicious"],
    )

    st.markdown("---")
    st.markdown("## 🤖 Model Info")

    metrics = load_metrics()
    if metrics:
        acc = metrics.get("accuracy", 0) * 100
        color = "#00d4aa" if acc > 90 else "#ffc107"
        st.markdown(f"""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:8px; padding:0.8rem;'>
          <div style='font-size:0.7rem; color:#8b949e; text-transform:uppercase;'>Accuracy</div>
          <div style='font-size:1.6rem; font-family:JetBrains Mono; color:{color};'>{acc:.1f}%</div>
          <div style='font-size:0.7rem; color:#8b949e; margin-top:4px;'>
            CV: {metrics.get("cv_mean",0)*100:.1f}% ± {metrics.get("cv_std",0)*100:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)
        fi = metrics.get("feature_importance", {})
        if fi:
            st.markdown("<div style='font-size:0.72rem; color:#8b949e; margin-top:0.8rem;'>Feature Importance</div>", unsafe_allow_html=True)
            for feat, val in sorted(fi.items(), key=lambda x: -x[1]):
                pct = val * 100
                st.markdown(f"""
                <div style='display:flex; justify-content:space-between; font-size:0.72rem; margin-bottom:2px;'>
                  <span style='color:#c9d1d9;'>{feat}</span>
                  <span style='color:#58a6ff; font-family:JetBrains Mono;'>{pct:.0f}%</span>
                </div>
                <div style='background:#21262d; border-radius:3px; height:4px; margin-bottom:6px;'>
                  <div style='background:#58a6ff; width:{pct}%; height:4px; border-radius:3px;'></div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Train model first:\n`python src/train_model.py`")

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════

# Header
col_title, col_time = st.columns([3, 1])
with col_title:
    st.markdown("""
    <div class='dash-title'>
      <span class='live-dot'></span>
      AI Honeypot — Cyber Attack Detection
    </div>
    <div class='dash-subtitle'>Real-time threat intelligence powered by Random Forest ML</div>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style='text-align:right; padding-top:0.4rem;'>
      <div style='font-family:JetBrains Mono; font-size:0.85rem; color:#58a6ff;'>{datetime.now().strftime("%H:%M:%S")}</div>
      <div style='font-size:0.72rem; color:#8b949e;'>{datetime.now().strftime("%d %b %Y")}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ── Load and filter data ───────────────────────────────────────────────────────
df_raw = load_data()
model, le = load_model()

if df_raw.empty:
    st.markdown("""
    <div class='alert-bar'>
      ⚠ No data found. Generate sample data first:<br>
      <code>python src/generate_sample_data.py</code>  then  <code>python src/train_model.py</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Apply ML predictions
df = run_predictions(df_raw.copy(), model, le)

# Time filter
now = datetime.utcnow()
if "timestamp" in df.columns and df["timestamp"].notna().any():
    time_map = {
        "Last 1 hour" : timedelta(hours=1),
        "Last 6 hours": timedelta(hours=6),
        "Last 24 hours": timedelta(hours=24),
        "Last 7 days" : timedelta(days=7),
        "All time"    : timedelta(days=36500),
    }
    cutoff = pd.Timestamp(now - time_map[time_range])
    df = df[df["timestamp"] >= cutoff]

# IP filter
if selected_ip != "All IPs":
    df = df[df["ip"] == selected_ip]

# Type filter
if selected_type:
    df = df[df["predicted_label"].isin(selected_type)]

# ── Alert bar for recent malicious ────────────────────────────────────────────
recent_mal = df[df["predicted_label"] == "Malicious"]
if not recent_mal.empty:
    last_mal_ip = recent_mal.iloc[-1].get("ip", "unknown")
    st.markdown(f"""
    <div class='alert-bar'>
      🚨 <strong>THREAT DETECTED</strong> — {len(recent_mal)} malicious session(s) in selected range.
      Last seen from IP: <strong>{last_mal_ip}</strong>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📊 Key Metrics</div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)

total_attacks  = len(df)
unique_ips     = df["ip"].nunique() if "ip" in df.columns else 0
malicious_cnt  = (df["predicted_label"] == "Malicious").sum()
suspicious_cnt = (df["predicted_label"] == "Suspicious").sum()
avg_attempts   = df["login_attempts"].mean() if "login_attempts" in df.columns else 0

def kpi_card(label, value, icon, color_class, delta=""):
    return f"""
    <div class='kpi-card {color_class}'>
      <span class='kpi-icon'>{icon}</span>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      {f"<div class='kpi-delta' style='color:#8b949e;'>{delta}</div>" if delta else ""}
    </div>
    """

with k1:
    st.markdown(kpi_card("Total Sessions", total_attacks, "🍯", "blue", f"{time_range}"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Unique IPs", unique_ips, "🌐", "accent" if True else "blue", "Source addresses"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("Malicious", malicious_cnt, "🔴", "red",
                          f"{malicious_cnt/max(total_attacks,1)*100:.0f}% of total"), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card("Suspicious", suspicious_cnt, "🟡", "yellow",
                          f"{suspicious_cnt/max(total_attacks,1)*100:.0f}% of total"), unsafe_allow_html=True)
with k5:
    st.markdown(kpi_card("Avg Login Attempts", f"{avg_attempts:.1f}", "🔐", "green", "Per session"), unsafe_allow_html=True)


# ── Charts Row 1: Bar + Pie ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📈 Attack Distribution</div>", unsafe_allow_html=True)
col_bar, col_pie = st.columns([3, 2])

with col_bar:
    st.markdown("**Attack Frequency by IP**")
    if "ip" in df.columns and not df.empty:
        ip_counts = (
            df.groupby(["ip", "predicted_label"])
              .size()
              .reset_index(name="count")
        )
        top_ips = df["ip"].value_counts().head(12).index
        ip_counts = ip_counts[ip_counts["ip"].isin(top_ips)]

        color_map = {k: v for k, v in COLORS.items() if k in ["Normal","Suspicious","Malicious"]}
        fig_bar = px.bar(
            ip_counts,
            x="ip", y="count", color="predicted_label",
            color_discrete_map=color_map,
            barmode="stack",
        )
        fig_bar.update_layout(
            **PLOT_LAYOUT,
            xaxis=dict(tickangle=-30, gridcolor="#21262d", title="IP Address"),
            yaxis=dict(gridcolor="#21262d", title="Sessions"),
            height=300,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No data for chart.")

with col_pie:
    st.markdown("**Attack Categories**")
    if "predicted_label" in df.columns and not df.empty:
        pie_data = df["predicted_label"].value_counts().reset_index()
        pie_data.columns = ["label", "count"]
        fig_pie = px.pie(
            pie_data, names="label", values="count",
            color="label",
            color_discrete_map={"Normal": COLORS["Normal"],
                                  "Suspicious": COLORS["Suspicious"],
                                  "Malicious": COLORS["Malicious"]},
            hole=0.55,
        )
        fig_pie.update_layout(**PLOT_LAYOUT, height=300)
        fig_pie.update_traces(textfont_color="#e6edf3", textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})


# ── Charts Row 2: Timeline ──────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⏱ Attack Timeline</div>", unsafe_allow_html=True)

if "timestamp" in df.columns and df["timestamp"].notna().any():
    df_time = df.copy()
    df_time["hour"] = df_time["timestamp"].dt.floor("H")
    timeline = (
        df_time.groupby(["hour", "predicted_label"])
               .size()
               .reset_index(name="count")
    )
    color_map = {"Normal": COLORS["Normal"], "Suspicious": COLORS["Suspicious"], "Malicious": COLORS["Malicious"]}
    fig_time = px.area(
        timeline, x="hour", y="count", color="predicted_label",
        color_discrete_map=color_map,
        line_shape="spline",
    )
    fig_time.update_layout(
        **PLOT_LAYOUT,
        xaxis=dict(gridcolor="#21262d", title="Time"),
        yaxis=dict(gridcolor="#21262d", title="Events per Hour"),
        height=260,
    )
    fig_time.update_traces(opacity=0.7)
    st.plotly_chart(fig_time, use_container_width=True, config={"displayModeBar": False})
else:
    st.info("Timestamp data not available for timeline.")


# ── Charts Row 3: Heatmap + Login dist ────────────────────────────────────────
st.markdown("<div class='section-header'>🔬 Deep Dive Analysis</div>", unsafe_allow_html=True)
col_heat, col_hist = st.columns(2)

with col_heat:
    st.markdown("**Session Duration vs Login Attempts**")
    if all(c in df.columns for c in ["login_attempts","session_duration","predicted_label"]):
        fig_scatter = px.scatter(
            df.head(500),
            x="login_attempts", y="session_duration",
            color="predicted_label",
            size="command_count" if "command_count" in df.columns else None,
            size_max=20,
            color_discrete_map={"Normal": COLORS["Normal"],
                                  "Suspicious": COLORS["Suspicious"],
                                  "Malicious": COLORS["Malicious"]},
            opacity=0.8,
            hover_data=["ip","username"] if "ip" in df.columns else None,
        )
        fig_scatter.update_layout(
            **PLOT_LAYOUT,
            xaxis=dict(gridcolor="#21262d", title="Login Attempts"),
            yaxis=dict(gridcolor="#21262d", title="Session Duration (s)"),
            height=280,
        )
        st.plotly_chart(fig_scatter, use_container_width=True, config={"displayModeBar": False})

with col_hist:
    st.markdown("**Login Attempts Distribution**")
    if "login_attempts" in df.columns:
        fig_hist = px.histogram(
            df, x="login_attempts", color="predicted_label",
            nbins=15, barmode="overlay",
            color_discrete_map={"Normal": COLORS["Normal"],
                                  "Suspicious": COLORS["Suspicious"],
                                  "Malicious": COLORS["Malicious"]},
            opacity=0.8,
        )
        fig_hist.update_layout(
            **PLOT_LAYOUT,
            xaxis=dict(gridcolor="#21262d", title="Login Attempts"),
            yaxis=dict(gridcolor="#21262d", title="Count"),
            height=280,
        )
        st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})


# ── Top Attackers ──────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🎯 Top Threat Actors</div>", unsafe_allow_html=True)

if "ip" in df.columns and not df.empty:
    top_attackers = (
        df[df["predicted_label"].isin(["Malicious","Suspicious"])]
          .groupby("ip")
          .agg(
              sessions    =("ip", "count"),
              avg_attempts=("login_attempts", "mean"),
              total_cmds  =("command_count", "sum"),
              threat_type =("predicted_label", lambda x: x.value_counts().index[0]),
          )
          .reset_index()
          .sort_values("sessions", ascending=False)
          .head(10)
    )

    if not top_attackers.empty:
        fig_top = px.bar(
            top_attackers, x="sessions", y="ip",
            color="threat_type", orientation="h",
            color_discrete_map={"Malicious": COLORS["Malicious"], "Suspicious": COLORS["Suspicious"]},
        )
        fig_top.update_layout(
            **PLOT_LAYOUT,
            yaxis=dict(gridcolor="#21262d", autorange="reversed", title=""),
            xaxis=dict(gridcolor="#21262d", title="Attack Sessions"),
            height=min(40 * len(top_attackers) + 60, 380),
        )
        st.plotly_chart(fig_top, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("No threat actors found in selected time range / type filter.")


# ── Log Table ──────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Session Log</div>", unsafe_allow_html=True)

display_cols = [c for c in ["timestamp","ip","username","password","login_attempts",
                              "command_count","session_duration","has_malicious_cmd",
                              "predicted_label","confidence"] if c in df.columns]

df_display = df[display_cols].copy()

# Colour-code threat column
def style_threat(val):
    colors = {"Malicious": "#3d1b1b", "Suspicious": "#2d2a00", "Normal": "#0d2018"}
    fg     = {"Malicious": "#ff4b6e", "Suspicious": "#ffc107", "Normal": "#00d4aa"}
    return f"background-color:{colors.get(val,'')};color:{fg.get(val,'')};font-weight:600;"

# Search box
search = st.text_input("🔍 Search logs (IP / username / command type)", "")
if search:
    mask = df_display.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
    df_display = df_display[mask]

st.markdown(f"<div style='font-size:0.78rem; color:#8b949e; margin-bottom:0.5rem;'>Showing {len(df_display)} of {len(df)} records</div>", unsafe_allow_html=True)

if "predicted_label" in df_display.columns:
    styled = df_display.head(200).style.applymap(style_threat, subset=["predicted_label"])
    st.dataframe(styled, use_container_width=True, height=380)
else:
    st.dataframe(df_display.head(200), use_container_width=True, height=380)

# Download button
csv_dl = df_display.to_csv(index=False).encode()
st.download_button("⬇ Download filtered logs as CSV", csv_dl, "honeypot_filtered.csv", "text/csv")


# ── Raw Logs Tab ────────────────────────────────────────────────────────────────
with st.expander("🗂 Raw JSON Logs (last 20 entries)"):
    raw = load_raw_logs()
    if raw:
        st.json(raw[-20:])
    else:
        st.info("No raw logs found.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; border-top:1px solid #21262d; margin-top:2rem;'>
  <div style='font-size:0.75rem; color:#484f58;'>
    🍯 AI Honeypot Dashboard — MCA Final Year Project
    &nbsp;|&nbsp; Random Forest ML Model
    &nbsp;|&nbsp; Built with Streamlit + Plotly
  </div>
</div>
""", unsafe_allow_html=True)

# Auto-refresh
if auto_refresh:
    time.sleep(10)
    st.rerun()
