# Free-Cooling Sustainability Dashboard for Reading Data Centres

A hackathon project for **Theme 1 — Hyperlocal forecasting tools**.

## The pitch in one line

A sustainability dashboard for Thames Valley data centre operators that forecasts free-cooling availability over the next 72 hours and reports it as avoided energy, avoided carbon, and cooling efficiency (PUE) — turning a weather forecast into a decarbonisation decision.

## Why this angle

The Reading/Slough corridor is the UK's largest data centre cluster, so this is a genuinely local, economically significant use case. Free cooling (pulling cool, dry outside air instead of running mechanical chillers) is the single biggest lever weather has on both a facility's operating cost **and** its carbon footprint. Every chiller-hour avoided is grid electricity not drawn and CO₂ not emitted — so a good free-cooling forecast is directly a sustainability instrument. The observatory's high-frequency record has exactly the variables needed to model it.

## Who it's for

A data centre facilities/sustainability engineer who has to decide, hour by hour, whether to run mechanical cooling or ride the weather — and who reports PUE and Scope 2 emissions to management and regulators. The dashboard answers "how much cooling can I get from the weather in the next 72h, and what does that save in energy and carbon?"

## What makes it "innovative" rather than "another forecast app"

- The output is a **sustainability decision**, not a temperature: a green/amber/red cooling schedule tied to avoided kWh and kgCO₂.
- **Uncertainty is the product.** The amber bands are where the quantile interval straddles a cooling-mode threshold — precisely the windows an operator must plan a contingency for.
- The thresholds are **operator-adjustable**, turning a generic model into a facility-specific tool.
- **Carbon framing throughout** — avoided emissions computed from a grid carbon-intensity factor, with a running cooling-PUE efficiency readout.
- A **long-record climatology panel** frames the sustainability trend: how the free-cooling resource itself is shrinking as the local climate warms — a CAPEX and net-zero planning point.

---

## The physics: the cooling-mode "switch" (from the Tetra Tech study)

The core of the whole project is a simple transfer function borrowed from the Tetra Tech data-centre cooling study. It compares the outdoor weather to two operator-set thresholds and returns which cooling mode the facility is in. Two weather numbers in, one cooling mode out.

- **Dry-bulb temperature** decides free vs not-free.
- **Wet-bulb temperature** decides "a bit of water" vs "full mechanical cooling" (wet-bulb is the lowest temperature evaporative cooling can reach).

```
if dry-bulb <= cold-aisle setpoint:       -> free cooling   (green)
elif wet-bulb <= evaporative limit:       -> evaporative    (amber)
else:                                     -> mechanical     (red)
```

| Mode | Condition | Relative power (PUE) | Water |
|---|---|---|---|
| Free cooling | dry-bulb ≤ setpoint | lowest | none |
| Evaporative | dry-bulb > setpoint AND wet-bulb ≤ limit | low–moderate | moderate |
| Mechanical | wet-bulb > limit | highest | high |

Once the mode is known, look up its PUE, multiply by IT load to get kW, and multiply by a grid carbon factor to get kgCO₂. The two thresholds the operator sets — cold-aisle **setpoint** (e.g. 24°C) and **evaporative limit** (e.g. 21°C) — are the dashboard's sliders, because different facilities configure them differently.

### Anchors reusable from the study
- Model per **1 MW IT load** so results scale linearly, then multiply by facility size.
- Add **7% electrical losses** (UPS/transformers) as extra heat the cooling must remove.
- Benchmark references: US average PUE ≈ 1.40, WUE ≈ 0.38 L/kWh (LBNL 2023).
- PUE-vs-temperature curves come from the Tetra Tech figures / the underlying LBNL 2024 US Data Center Energy Usage Report — cite these rather than inventing PUE values.

Note the difference in scope: Tetra Tech used a synthetic "typical year" for Houston and El Paso. You have decades of real Reading observations — so you can do what they couldn't: show how free-cooling availability and PUE have actually shifted decade by decade as the climate warmed.

---

## Dataset

**Primary: Reading Atmospheric Observatory automatic measurements (1997–2023).**
5-minute surface observations (temperature, humidity, pressure, wind, rainfall) in CSV.
DOI: https://doi.org/10.17864/1947.000490

This is the right engine because free cooling is an hour-by-hour decision — a window opens overnight and closes mid-morning. Only the high-frequency automatic record can resolve that:

- Sub-hourly wet-bulb behaviour, so you model the actual overnight free-cooling window, not a daily proxy.
- ~26 years of data — plenty to train quantile models and validate properly with `TimeSeriesSplit`.
- Temperature and humidity at the same fine cadence, so wet-bulb can be computed directly.

Caveat: the series ends in 2023, so it's a historical/backtest dataset, not a live feed. For the hackathon, frame the tool as "trained and validated on the observatory record, ready to run on a live feed," and demo on held-out 2022–2023 data.

**Optional secondary: Reading manual record (1901–present).**
One daily 0900 UTC observation — too coarse to drive the forecast, but ideal for a single climatology panel showing how free-cooling-day frequency has shifted over the century. Use it only for the sustainability-trend framing, not the core engine.

---

## Weather data needed

### Core inputs — required to run the cooling-mode switch

| Variable | Role |
|---|---|
| Dry-bulb (air) temperature | Decides free vs not-free; input to wet-bulb derivation. |
| Relative humidity | Input to wet-bulb derivation. |
| Wet-bulb temperature (derived) | Decides evaporative vs mechanical; the forecast target. Compute from dry-bulb + RH via **Stull's formula** if not measured directly. |
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
| Grid carbon-intensity factor (kgCO₂/kWh) | Converts avoided chiller-hours into avoided emissions. Editable; cite a UK grid average. |
| Facility IT load (per MW) + PUE per mode | Convert mode → kW. Per-MW so it scales linearly. |
| Cold-aisle setpoint + evaporative limit | The two operator sliders that set the mode thresholds. |
| Hot-day / mechanical-day frequency (long record) | Drives the "free-cooling resource is shrinking" climate-trend panel. |

---

## The stack

| Layer | Choice | Why |
|---|---|---|
| Data | pandas | CSV load, resample, clean once, cache it. No database. |
| Model | `HistGradientBoostingRegressor` (scikit-learn) ×3 | One median + two quantile models (`loss="quantile"`, `quantile=0.1` and `0.9`) for the interval band. Handles missing values natively. |
| Cooling logic | plain Python function | Takes a weather reading, returns `(mode, PUE, power, water, CO₂)`. The Tetra Tech switch. |
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

## Step by step (2 days)

### Day 1 AM — data
1. Load the automatic 5-minute CSV with pandas. Handle gaps and encoding quirks.
2. Parse timestamps, sort chronologically, set a datetime index. Resample to a consistent cadence (e.g. hourly) if 5-minute is noisier than needed.
3. Derive wet-bulb from dry-bulb + RH via Stull's formula.
4. Build features: lags, rolling mean/std, sin/cos hour-of-day and day-of-year.

### Day 1 PM — model + cooling logic
5. Set the forecast target to wet-bulb temperature.
6. Fit three `HistGradientBoostingRegressor` models: the median, plus `quantile=0.1` and `quantile=0.9`.
7. Validate with `TimeSeriesSplit` (train earlier, test later). Sanity-check the error.
8. Write the cooling-mode function: weather reading → mode → PUE → power → CO₂/water. Run it over history to confirm mode counts look sane.
9. Produce the forecast as a dataframe: median (`wb`), low (`lo`), high (`hi`) over the next 72 hours.

### Day 2 AM — dashboard
10. Streamlit layout: sidebar sliders for the cold-aisle setpoint, evaporative limit, IT load, and grid carbon intensity.
11. Band each forecast hour by running the mode switch on the interval:
    - **green** = whole interval stays in free cooling
    - **red** = whole interval in mechanical
    - **amber** = interval crosses a mode threshold (marginal — needs contingency)
12. Render the 72h cooling-mode timeline (Plotly horizontal bar or coloured columns).

### Day 2 PM — sustainability readouts and polish
13. Headline sentence: where the median crosses the setpoint → "Free-cooling window opens ~HH:00, holds ~Nh."
14. Sustainability metrics: free-cooling hours, avoided chiller-hours, **avoided kWh**, **avoided kgCO₂**, and a cooling-efficiency readout (share of hours on free cooling / effective PUE).
15. Heatwave risk flag from consecutive forecast mechanical hours — sustained mechanical load hurts PUE.
16. Climatology panel: free-cooling vs mechanical days per year from the long record → "the free-cooling resource is shrinking."
17. Buffer for the demo story: lead with the decision, show the slider recompute live, close on the climate trend and the net-zero framing.

---

## Reference trail

- **LBNL 2024 US Data Center Energy Usage Report** (Shehabi et al., LBNL-2001637) — source of the PUE/WUE benchmarks and cooling-system definitions. The primary open reference.
- **The Green Grid** — original definitions of PUE (2012) and WUE (2011).
- **Tetra Tech modelling study** — the cooling-mode switch and PUE-vs-temperature curves you implement in simplified form.
- **TMYx / climate.onebuilding.org** — the synthetic weather files the study used, which your Reading CSV replaces.

---

## Judging hooks to hit

- **Interesting approach** → uncertainty-aware quantile forecast feeding a physics-based cooling-mode switch, correct time-series validation.
- **Innovative use of the observatory's data** → high-frequency record reframed as a free-cooling and carbon-avoidance signal; long record reframed as a decarbonisation-planning trend the Tetra Tech study couldn't produce.
- **Community/business decision-making** → a real sustainability decision (chiller staging, PUE, Scope 2) for a real local industry.
