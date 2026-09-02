# Free-Cooling Sustainability Dashboard for Reading Data Centres

A hackathon project for **Theme 1 — Hyperlocal forecasting tools**.

## The pitch in one line

A sustainability dashboard for Thames Valley data centre operators that forecasts free-cooling availability over the next 72 hours and reports it as avoided energy, avoided carbon, and cooling efficiency — turning a weather forecast into a decarbonisation decision.

## Why this angle

The Reading/Slough corridor is the UK's largest data centre cluster, so this is a genuinely local, economically significant use case. Free cooling (pulling cool, dry outside air instead of running mechanical chillers) is the single biggest lever weather has on both a facility's operating cost **and** its carbon footprint. Every chiller-hour avoided is grid electricity not drawn and CO₂ not emitted — so a good free-cooling forecast is directly a sustainability instrument, not just a cost one. The observatory's high-frequency record has exactly the variables needed to model it.

## Who it's for

A data centre facilities/sustainability engineer who has to decide, hour by hour, whether to run mechanical cooling or ride the weather — and who has to report PUE and Scope 2 emissions to management and regulators. The dashboard answers "how much cooling can I get from the weather in the next 72h, and what does that save in energy and carbon?"

## What makes it "innovative" rather than "another forecast app"

- The output is a **sustainability decision**, not a temperature: a green/amber/red cooling schedule tied to avoided kWh and kgCO₂.
- **Uncertainty is the product.** The amber bands are where the quantile interval straddles the free-cooling cutoff — precisely the windows an operator must plan a contingency for.
- The cutoff is **operator-adjustable**, turning a generic model into a facility-specific tool.
- **Carbon framing throughout** — avoided emissions computed from a grid carbon-intensity factor, with a running "cooling PUE contribution" style efficiency readout.
- A **long-record climatology panel** frames the sustainability trend: how the free-cooling resource itself is shrinking as the local climate warms — a CAPEX and net-zero planning point.

---

## Dataset

**Primary: UoR climate data 1908 to date - CSV (as at 28 Aug 2026 SB)**
- This dataset is given alongside the hackathon briefing. 

This is the right engine because free cooling is an hour-by-hour decision — a window opens overnight and closes mid-morning. Only the high-frequency automatic record can resolve that:

- Sub-hourly wet-bulb behaviour, so you model the actual overnight free-cooling window, not a daily proxy.
- ~26 years of data — plenty to train quantile models and validate properly with `TimeSeriesSplit`.
- Temperature and humidity at the same fine cadence, so wet-bulb can be computed directly.

Caveat: the series ends in 2023, so it's a historical/backtest dataset, not a live feed. For the hackathon, frame the tool as "trained and validated on the observatory record, ready to run on a live feed," and demo on held-out 2022–2023 data.

**Optional secondary: Reading manual record (1901–present).**
One daily 0900 UTC observation — too coarse to drive the forecast, but ideal for a single climatology panel showing how hot-day frequency (and therefore the free-cooling resource) has shifted over the century. Use it only for the sustainability-trend framing, not the core engine.

---

## The stack

| Layer | Choice | Why |
|---|---|---|
| Data | pandas | CSV load, resample, clean once, cache it. No database. |
| Model | `HistGradientBoostingRegressor` (scikit-learn) ×3 | One median + two quantile models (`loss="quantile"`, `quantile=0.1` and `0.9`) for the interval band. Handles missing values natively. |
| Validation | `TimeSeriesSplit` | Never random-split time series — it leaks the future into training. |
| GUI | Streamlit | Pure Python, dataframe-to-dashboard in an afternoon, free deploy on Community Cloud. |
| Charts | Plotly | Shaded confidence band (`fill='tonexty'`) and the coloured timeline. |

Dependencies:

```
pandas
scikit-learn
streamlit
plotly
```

No FastAPI, no Flask, no React, no torch. For a map (optional polish), use `st.map` or `folium` — not Google Maps, which needs billing setup and doesn't embed cleanly in Streamlit.

---

## Weather data needed

### Core inputs — required to compute the free-cooling metric

The free-cooling decision hinges on **wet-bulb temperature**, because humidity determines how much cooling outside air can actually provide.

| Variable | Role |
|---|---|
| Dry-bulb (air) temperature | Input to wet-bulb derivation and a predictor. |
| Relative humidity | Input to wet-bulb derivation and a predictor. |
| Wet-bulb temperature (derived) | The forecast target. Compute from dry-bulb + RH via **Stull's formula** if not provided directly. |
| Pressure | Improves wet-bulb accuracy; secondary predictor. |

### Engineered features — built from the core inputs

| Feature | What it captures |
|---|---|
| Lag features (e.g. `t-1`, `t-24h`) | Recent and same-time-yesterday persistence. |
| Rolling mean / std (e.g. 24h) | Trend and volatility. |
| Cyclical hour-of-day (`sin`/`cos`) | So 23:00 and 00:00 aren't treated as far apart. |
| Cyclical day-of-year (`sin`/`cos`) | Seasonality — the free-cooling resource is strongly seasonal. |

### Sustainability layer — the "so what"

| Input | Role |
|---|---|
| Grid carbon-intensity factor (kgCO₂/kWh) | Converts avoided chiller-hours into avoided emissions. Editable assumption; can cite a UK grid average. |
| Facility IT load + chiller COP | Convert avoided chiller-hours into avoided kWh. Editable assumptions. |
| Hot-day frequency (from long record) | Drives the "free-cooling resource is shrinking" climate-trend panel. |

---

## Step by step (2 days)

### Day 1 AM — data
1. Load the automatic 5-minute CSV with pandas. Handle gaps and encoding quirks.
2. Parse timestamps, sort chronologically, set a datetime index. Resample to a consistent cadence (e.g. hourly) if 5-minute is noisier than needed.
3. Derive wet-bulb from dry-bulb + RH via Stull's formula.
4. Build features: lags, rolling mean/std, sin/cos hour-of-day and day-of-year.

### Day 1 PM — model
5. Set the target to wet-bulb temperature.
6. Fit three `HistGradientBoostingRegressor` models: the median, plus `quantile=0.1` and `quantile=0.9`.
7. Validate with `TimeSeriesSplit` (train on earlier data, test on later). Sanity-check the error.
8. Produce the forecast as a dataframe with three columns — median (`wb`), low (`lo`), high (`hi`) — over the next 72 hours.

### Day 2 AM — dashboard
9. Streamlit layout: sidebar sliders for the free-cooling wet-bulb cutoff and sustainability assumptions (IT load, COP, grid carbon intensity).
10. Compute the banding per forecast hour:
    - **green** = whole interval below cutoff (confident free cooling)
    - **red** = whole interval above cutoff (confident mechanical)
    - **amber** = interval straddles the cutoff (marginal — needs contingency)
11. Render the 72h cooling-mode timeline (Plotly horizontal bar or coloured columns).

### Day 2 PM — sustainability readouts and polish
12. Headline sentence: where the median crosses the cutoff → "Free-cooling window opens ~HH:00, holds ~Nh."
13. Sustainability metrics: free-cooling hours, avoided chiller-hours, **avoided kWh**, **avoided kgCO₂**, and a cooling-efficiency readout (share of hours on free cooling).
14. Heatwave risk flag from consecutive forecast hot hours — sustained mechanical load hurts PUE.
15. Climatology panel: hot-day frequency by decade from the long record → "the free-cooling resource is shrinking."
16. Buffer for the demo story: lead with the decision, show the slider recompute live, close on the climate trend and the net-zero framing.

---

## Judging hooks to hit

- **Interesting approach** → uncertainty-aware quantile forecast, correct time-series validation.
- **Innovative use of the observatory's data** → high-frequency record reframed as a free-cooling and carbon-avoidance signal; long record reframed as a decarbonisation-planning trend.
- **Community/business decision-making** → a real sustainability decision (chiller staging, PUE, Scope 2) for a real local industry.
