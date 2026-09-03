"""
Free-Cooling Operator Dashboard (v2)
------------------------------------
Streamlit dashboard for Thames Valley data centre operators.

Adds a "prediction vs what actually happened" proof panel to v1: for the chosen
forecast origin, it shows the model's 3h-ahead forecast next to the real value
that occurred 3h later — demonstrating the model catching a "new ball" it never
saw in training.

Loads the two saved models produced by main.py:
    models/stage1_model.joblib   -> MultiOutputRegressor, predicts [Td, Tw] 3h ahead
    models/stage2_model.joblib   -> HistGradientBoostingRegressor, predicts optimised PUE

Run:
    streamlit run dashboard2.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------
# Constants (kept in sync with the training scripts)
# ----------------------------------------------------------------------
DATA_PATH = Path("2010-2023-cleaned.csv")
STAGE1_MODEL_PATH = Path("models/stage1_model.joblib")
STAGE2_MODEL_PATH = Path("models/stage2_model.joblib")

FORECAST_STEPS = 36              # 36 * 5 min = 3 hours
DATA_INTERVAL_HOURS = 5.0 / 60.0
FORECAST_HORIZON_HOURS = FORECAST_STEPS * DATA_INTERVAL_HOURS

INDUSTRY_AVERAGE_PUE = 1.56

# Stage-2 feature order — MUST match train_stage2_pue exactly.
STAGE2_FEATURES = [
    "pred_Td", "pred_Tw", "current_Td", "current_Tw",
    "RH", "P", "tod_sin", "tod_cos", "doy_sin", "doy_cos",
]

# ----------------------------------------------------------------------
# Palette — dark base, green highlight
# ----------------------------------------------------------------------
BG = "#0d1512"
PANEL = "#141f1a"
INK = "#e8f0ec"
MUTED = "#7d938a"
GREEN = "#3ddc84"          # free cooling / primary highlight
BLUE = "#5fb0e0"           # wet-bulb line
AMBER = "#e6b450"          # evaporative
RED = "#e0665f"            # mechanical
GRID = "#22322b"

MODE_STYLE = {
    "free":        {"label": "Free cooling",  "colour": GREEN, "note": "Chillers off — ride the weather"},
    "evaporative": {"label": "Evaporative",   "colour": AMBER, "note": "Assisted cooling, some water use"},
    "mechanical":  {"label": "Mechanical",    "colour": RED,   "note": "Full chillers — highest energy"},
}

# ----------------------------------------------------------------------
# Cooling logic (mirrors train_sklearn_pue_stage2 / stage3)
# ----------------------------------------------------------------------
SERVER_INLET_SETPOINT_C = 24.0
EVAP_WET_BULB_LIMIT_C = 20.0


def cooling_mode(td, tw, setpoint, evap_limit):
    if td <= setpoint:
        return "free"
    if tw <= evap_limit:
        return "evaporative"
    return "mechanical"


# ----------------------------------------------------------------------
# Feature engineering (mirrors add_features in stage 1)
# ----------------------------------------------------------------------
def add_features(df):
    out = df[["Td", "Tw", "RH", "P"]].copy()

    for col in ["Td", "Tw", "RH", "P"]:
        for lag in [1, 3, 6, 12]:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for window in [3, 6, 12]:
            out[f"{col}_mean_{window}"] = out[col].rolling(window).mean()
        out[f"{col}_trend_12"] = out[col] - out[col].shift(12)

    minutes_in_day = 24 * 60
    day_minutes = out.index.hour * 60 + out.index.minute
    out["tod_sin"] = np.sin(2 * np.pi * day_minutes / minutes_in_day)
    out["tod_cos"] = np.cos(2 * np.pi * day_minutes / minutes_in_day)
    out["doy_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 365.25)

    # Actual future values — the "truth" the forecast is checked against.
    out["actual_Td"] = df["Td"].shift(-FORECAST_STEPS)
    out["actual_Tw"] = df["Tw"].shift(-FORECAST_STEPS)
    return out.dropna()


# ----------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    stage1 = joblib.load(STAGE1_MODEL_PATH)
    stage2 = joblib.load(STAGE2_MODEL_PATH)
    return stage1, stage2


@st.cache_data(show_spinner=False)
def load_features():
    raw = pd.read_csv(DATA_PATH)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["TimeStamp"]).sort_values("TimeStamp").set_index("TimeStamp")
    weather = raw[["Td", "Tw", "RH", "P"]].apply(pd.to_numeric, errors="coerce")
    weather = weather.dropna(subset=["Td", "Tw", "RH", "P"])
    feats = add_features(weather)
    feature_cols = [c for c in feats.columns if not c.startswith("actual_")]
    return feats, feature_cols


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def predict_window(stage1, stage2, feats, feature_cols, start_idx, n_steps):
    """Run both models across n_steps consecutive rows starting at start_idx."""
    window = feats.iloc[start_idx:start_idx + n_steps]

    stage1_pred = stage1.predict(window[feature_cols])
    pred_td = stage1_pred[:, 0]
    pred_tw = stage1_pred[:, 1]

    stage2_input = pd.DataFrame(
        {
            "pred_Td": pred_td,
            "pred_Tw": pred_tw,
            "current_Td": window["Td"].values,
            "current_Tw": window["Tw"].values,
            "RH": window["RH"].values,
            "P": window["P"].values,
            "tod_sin": window["tod_sin"].values,
            "tod_cos": window["tod_cos"].values,
            "doy_sin": window["doy_sin"].values,
            "doy_cos": window["doy_cos"].values,
        },
        index=window.index,
    )
    predicted_pue = stage2.predict(stage2_input[STAGE2_FEATURES])

    out = pd.DataFrame(
        {
            "pred_Td": pred_td,
            "pred_Tw": pred_tw,
            "current_Td": window["Td"].values,
            "current_Tw": window["Tw"].values,
            "actual_Td": window["actual_Td"].values,   # truth, carried through
            "actual_Tw": window["actual_Tw"].values,
            "predicted_pue": predicted_pue,
        },
        index=window.index,
    )
    out["valid_for"] = out.index + pd.Timedelta(hours=FORECAST_HORIZON_HOURS)
    return out


# ----------------------------------------------------------------------
# Page setup + CSS
# ----------------------------------------------------------------------
st.set_page_config(page_title="Free-Cooling Operator Dashboard",
                   page_icon="❄", layout="wide")

st.markdown(f"""
<style>
    .stApp {{ background: {BG}; color: {INK}; }}
    section[data-testid="stSidebar"] {{ background: {PANEL}; }}
    section[data-testid="stSidebar"] * {{ color: {INK}; }}
    h1, h2, h3, h4 {{ color: {INK}; font-weight: 650; letter-spacing: -0.01em; }}
    .block-container {{ padding-top: 2.2rem; }}

    .headline {{
        background: {PANEL};
        border-left: 3px solid {GREEN};
        border-radius: 4px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.4rem;
    }}
    .headline .mode {{ font-size: 1.7rem; font-weight: 700; }}
    .headline .sub  {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.15rem; }}

    .tile {{
        background: {PANEL};
        border: 1px solid {GRID};
        border-radius: 6px;
        padding: 1rem 1.1rem;
    }}
    .tile .k {{ color: {MUTED}; font-size: 0.8rem; margin-bottom: 0.35rem; }}
    .tile .v {{ font-size: 1.55rem; font-weight: 700; line-height: 1; }}
    .tile .u {{ color: {MUTED}; font-size: 0.85rem; margin-left: 0.2rem; }}

    .proof {{
        background: {PANEL};
        border: 1px solid {GRID};
        border-left: 3px solid {GREEN};
        border-radius: 6px;
        padding: 1rem 1.2rem;
    }}
    .proof .row {{ display: flex; justify-content: space-between; padding: 0.3rem 0; }}
    .proof .lbl {{ color: {MUTED}; }}
    .proof .val {{ font-weight: 700; }}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Guard: models / data present?
# ----------------------------------------------------------------------
missing = [str(p) for p in (STAGE1_MODEL_PATH, STAGE2_MODEL_PATH, DATA_PATH) if not p.exists()]
if missing:
    st.title("Free-Cooling Operator Dashboard")
    st.error(
        "Missing required files: " + ", ".join(missing)
        + "\n\nRun `python main.py` first to train and save the models, and make "
        "sure the cleaned CSV sits next to this script."
    )
    st.stop()

stage1, stage2 = load_models()
feats, feature_cols = load_features()

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("Controls")

if st.sidebar.button("↺ Standard Reading facility"):
    for k in ("setpoint", "evap", "load", "carbon", "horizon"):
        st.session_state.pop(k, None)
    st.rerun()

setpoint = st.sidebar.slider("Cold-aisle setpoint (°C)", 18.0, 30.0,
                             SERVER_INLET_SETPOINT_C, 0.5, key="setpoint")
evap_limit = st.sidebar.slider("Evaporative limit (°C)", 14.0, 26.0,
                               EVAP_WET_BULB_LIMIT_C, 0.5, key="evap")
it_load_mw = st.sidebar.number_input("IT load (MW)", min_value=0.1, value=10.0,
                                     step=1.0, key="load")
carbon = st.sidebar.number_input("Grid carbon intensity (kgCO₂/kWh)",
                                 min_value=0.0, value=0.20, step=0.01, key="carbon")

st.sidebar.divider()

n_positions = len(feats) - FORECAST_STEPS - 1
default_pos = int(n_positions * 0.85)
pos = st.sidebar.slider("Forecast origin (position in record)",
                        0, n_positions, default_pos)
now_time = feats.index[pos]
st.sidebar.caption(f"'Now' = {now_time:%d %b %Y  %H:%M}")

horizon_hours = st.sidebar.select_slider("Timeline horizon (hours)",
                                         options=[12, 24, 48, 72], value=72,
                                         key="horizon")
n_steps = int(horizon_hours / DATA_INTERVAL_HOURS)
n_steps = min(n_steps, len(feats) - pos)

# ----------------------------------------------------------------------
# Run models
# ----------------------------------------------------------------------
result = predict_window(stage1, stage2, feats, feature_cols, pos, n_steps)
result["mode"] = [
    cooling_mode(td, tw, setpoint, evap_limit)
    for td, tw in zip(result["pred_Td"], result["pred_Tw"])
]

head = result.iloc[0]
head_mode = head["mode"]
style = MODE_STYLE[head_mode]

# The actual mode that occurred (truth), for the proof panel.
actual_head_mode = cooling_mode(head["actual_Td"], head["actual_Tw"], setpoint, evap_limit)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.title("Free-Cooling Operator Dashboard")
st.caption("Reading Atmospheric Observatory · 3-hour-ahead cooling decision")

st.markdown(f"""
<div class="headline">
  <div class="mode" style="color:{style['colour']};">▊ {style['label']}</div>
  <div class="sub">Recommended for {head['valid_for']:%H:%M} (3h ahead) — {style['note']}</div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Metric tiles
# ----------------------------------------------------------------------
def tile(col, k, v, u=""):
    col.markdown(
        f'<div class="tile"><div class="k">{k}</div>'
        f'<div class="v">{v}<span class="u">{u}</span></div></div>',
        unsafe_allow_html=True,
    )

c1, c2, c3, c4 = st.columns(4)
tile(c1, "Predicted dry-bulb (3h)", f"{head['pred_Td']:.1f}", " °C")
tile(c2, "Predicted wet-bulb (3h)", f"{head['pred_Tw']:.1f}", " °C")
tile(c3, "Predicted PUE",           f"{head['predicted_pue']:.3f}")
saving_kw_now = (INDUSTRY_AVERAGE_PUE - head["predicted_pue"]) * it_load_mw * 1000.0
tile(c4, "Saving vs 1.56 PUE",      f"{saving_kw_now:,.0f}", " kW")

st.write("")

# ----------------------------------------------------------------------
# PROOF PANEL — prediction vs what actually happened
# ----------------------------------------------------------------------
st.subheader("Did the forecast come true?")
st.caption(
    f"The model forecast from {now_time:%d %b %Y %H:%M} — a moment it never saw "
    f"in training — for 3 hours later. Here is that forecast against what the "
    f"observatory actually recorded."
)

err_td = abs(head["pred_Td"] - head["actual_Td"])
err_tw = abs(head["pred_Tw"] - head["actual_Tw"])
mode_match = (head_mode == actual_head_mode)

pc1, pc2 = st.columns([1, 1])

with pc1:
    st.markdown(f"""
    <div class="proof">
      <div class="row"><span class="lbl">Dry-bulb — forecast</span>
        <span class="val" style="color:{GREEN};">{head['pred_Td']:.1f} °C</span></div>
      <div class="row"><span class="lbl">Dry-bulb — actual</span>
        <span class="val">{head['actual_Td']:.1f} °C</span></div>
      <div class="row"><span class="lbl">Miss by</span>
        <span class="val">{err_td:.1f} °C</span></div>
      <hr style="border-color:{GRID};">
      <div class="row"><span class="lbl">Wet-bulb — forecast</span>
        <span class="val" style="color:{BLUE};">{head['pred_Tw']:.1f} °C</span></div>
      <div class="row"><span class="lbl">Wet-bulb — actual</span>
        <span class="val">{head['actual_Tw']:.1f} °C</span></div>
      <div class="row"><span class="lbl">Miss by</span>
        <span class="val">{err_tw:.1f} °C</span></div>
    </div>
    """, unsafe_allow_html=True)

with pc2:
    verdict_col = GREEN if mode_match else AMBER
    verdict_txt = "Correct call" if mode_match else "Missed the call"
    st.markdown(f"""
    <div class="proof" style="border-left-color:{verdict_col};">
      <div class="row"><span class="lbl">Recommended mode</span>
        <span class="val" style="color:{MODE_STYLE[head_mode]['colour']};">
        {MODE_STYLE[head_mode]['label']}</span></div>
      <div class="row"><span class="lbl">Actual mode that occurred</span>
        <span class="val" style="color:{MODE_STYLE[actual_head_mode]['colour']};">
        {MODE_STYLE[actual_head_mode]['label']}</span></div>
      <hr style="border-color:{GRID};">
      <div class="row"><span class="lbl">Verdict</span>
        <span class="val" style="color:{verdict_col};">{verdict_txt}</span></div>
    </div>
    """, unsafe_allow_html=True)

# Accuracy across the whole window, so one lucky point isn't the whole story.
win_td_mae = (result["pred_Td"] - result["actual_Td"]).abs().mean()
win_tw_mae = (result["pred_Tw"] - result["actual_Tw"]).abs().mean()
win_modes_pred = result["mode"]
win_modes_actual = [
    cooling_mode(td, tw, setpoint, evap_limit)
    for td, tw in zip(result["actual_Td"], result["actual_Tw"])
]
win_mode_acc = np.mean([p == a for p, a in zip(win_modes_pred, win_modes_actual)])

m1, m2, m3 = st.columns(3)
tile(m1, f"Avg dry-bulb error ({horizon_hours}h)", f"{win_td_mae:.2f}", " °C")
tile(m2, f"Avg wet-bulb error ({horizon_hours}h)", f"{win_tw_mae:.2f}", " °C")
tile(m3, f"Mode accuracy ({horizon_hours}h)",       f"{win_mode_acc*100:.0f}", " %")

st.write("")

# ----------------------------------------------------------------------
# Forecast vs actual — line chart
# ----------------------------------------------------------------------
st.subheader("Forecast vs actual, across the window")

figv = go.Figure()
figv.add_trace(go.Scatter(x=result["valid_for"], y=result["pred_Td"],
                          name="Dry-bulb (forecast)", line=dict(color=GREEN, width=2)))
figv.add_trace(go.Scatter(x=result["valid_for"], y=result["actual_Td"],
                          name="Dry-bulb (actual)",
                          line=dict(color=GREEN, width=1.5, dash="dot")))
figv.add_trace(go.Scatter(x=result["valid_for"], y=result["pred_Tw"],
                          name="Wet-bulb (forecast)", line=dict(color=BLUE, width=2)))
figv.add_trace(go.Scatter(x=result["valid_for"], y=result["actual_Tw"],
                          name="Wet-bulb (actual)",
                          line=dict(color=BLUE, width=1.5, dash="dot")))
figv.add_hline(y=setpoint, line=dict(color=INK, dash="dash", width=1),
               annotation_text="setpoint", annotation_font_color=MUTED)
figv.add_hline(y=evap_limit, line=dict(color=AMBER, dash="dot", width=1),
               annotation_text="evap limit", annotation_font_color=MUTED)
figv.update_layout(
    height=340, paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=INK),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID, title="°C"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(figv, use_container_width=True)

# ----------------------------------------------------------------------
# Cooling-mode timeline
# ----------------------------------------------------------------------
st.subheader(f"Next {horizon_hours}h cooling-mode timeline")

fig = go.Figure()
for mode, spec in MODE_STYLE.items():
    sub = result[result["mode"] == mode]
    if sub.empty:
        continue
    fig.add_trace(go.Scatter(
        x=sub["valid_for"], y=[1] * len(sub),
        mode="markers", marker=dict(color=spec["colour"], size=9, symbol="square"),
        name=spec["label"], hovertemplate="%{x|%d %b %H:%M}<br>" + spec["label"] + "<extra></extra>",
    ))
fig.update_layout(
    height=140, paper_bgcolor=BG, plot_bgcolor=BG,
    font=dict(color=INK), margin=dict(l=10, r=10, t=10, b=10),
    yaxis=dict(visible=False, range=[0.5, 1.5]),
    xaxis=dict(gridcolor=GRID, showline=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.05, x=0, bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------
# Sustainability readouts over the window
# ----------------------------------------------------------------------
st.subheader("Impact over the window")

mode_hours = result["mode"].value_counts() * DATA_INTERVAL_HOURS
free_h = mode_hours.get("free", 0.0)
evap_h = mode_hours.get("evaporative", 0.0)
mech_h = mode_hours.get("mechanical", 0.0)

pue_gap = (INDUSTRY_AVERAGE_PUE - result["predicted_pue"]).clip(lower=0)
saving_kwh = (pue_gap * it_load_mw * 1000.0 * DATA_INTERVAL_HOURS).sum()
avoided_co2 = saving_kwh * carbon

s1, s2, s3, s4 = st.columns(4)
tile(s1, "Free-cooling hours",   f"{free_h:.0f}", " h")
tile(s2, "Mechanical hours",     f"{mech_h:.0f}", " h")
tile(s3, "Avoided energy",       f"{saving_kwh:,.0f}", " kWh")
tile(s4, "Avoided CO₂",          f"{avoided_co2:,.0f}", " kg")

st.caption(
    f"Per {it_load_mw:.1f} MW IT load over {horizon_hours}h, versus the "
    f"{INDUSTRY_AVERAGE_PUE} industry-average PUE. Evaporative hours: {evap_h:.0f}."
)

st.divider()
st.caption(
    "Trained and validated on the observatory record to 2023. This view backtests "
    "from a chosen point in held-out data — the forecast is checked against what "
    "actually happened, and the same models run unchanged on a live feed."
)
