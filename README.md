# Free-Cooling Forecast for Data Centre Cooling Optimisation

A simple ML prediction model that turns localised Reading weather data into a
**3-hour-ahead cooling decision** for Thames Valley data centre operators:
when to ride the weather with *free* or *evaporative* cooling instead of running
mechanical chillers — and what that saves in energy, carbon, and PUE. The model
feeds an interactive Streamlit operator dashboard.

Built for a hackathon under **Theme 1 — Hyperlocal forecasting tools**.

---

## What it does

Free cooling means pulling cool, dry outside air instead of running mechanical
chillers. Whether a facility can use it depends on two weather numbers:

- **Dry-bulb temperature (`Td`)** — decides *free* vs *not free*.
- **Wet-bulb temperature (`Tw`)** — decides *evaporative* vs *full mechanical*.

The pipeline forecasts those two temperatures 3 hours ahead from the local
observatory record, then maps the forecast onto a cooling mode and its relative
energy cost (PUE):

```
if dry-bulb <= cold-aisle setpoint (24°C):   -> free cooling   (green,  lowest PUE)
elif wet-bulb <= evaporative limit (20°C):   -> evaporative    (amber,  low–moderate)
else:                                        -> mechanical     (red,    highest)
```

The output is a **decision**, not a temperature: a green/amber/red cooling
schedule tied to avoided kWh and avoided CO₂, benchmarked against an
industry-average PUE of 1.56.

---

## Architecture

The pipeline runs in three stages, wired together in `main.py`. Training the
first two stages also saves them to disk (`models/*.joblib`) so the dashboard
can load them and predict instantly without retraining.

**Stage 1 — weather forecaster** (`train_sklearn_pue_stage1.py`)
A `MultiOutputRegressor` wrapping `HistGradientBoostingRegressor` predicts
`Td` and `Tw` 36 steps (3 hours) ahead from engineered features: lags, rolling
means, short-term trends, and cyclical time-of-day / day-of-year terms. Reported
against a persistence baseline using MAE in °C.

**Stage 2 — PUE model** (`train_sklearn_pue_stage2.py`)
A second `HistGradientBoostingRegressor` learns the optimised PUE a
weather-responsive control strategy could achieve, trained on the stage-1
forecasts plus current conditions. It's compared against two baselines:
physics-only (feed stage-1 forecasts straight into the cooling curve) and
persistence.

**Stage 3 — operational impact** (`stage3.py`)
Translates the stage-2 PUE prediction into stakeholder-facing numbers:
recommended cooling mode, PUE improvement vs the 1.56 baseline, and — given an
IT load, electricity price, and grid carbon intensity — the energy saved, cost
saved, and CO₂ avoided. When no IT-load telemetry is supplied it reports
normalised per MW of IT load. Savings follow from the PUE gap:
`saving = IT_load × (industry_average_PUE − predicted_PUE)`.

`train_sklearn.py` is an earlier standalone script that does stage-1 forecasting
plus cooling-mode accuracy in a single file — kept for reference.

---

## Datasets

| Dataset | Coverage | Role |
|---|---|---|
| Reading Atmospheric Observatory automatic measurements | 1997–2023, 5-min cadence | Primary — trains and validates the forecaster |
| Reading manual weather observations | 1901–present, daily 0900 UTC | Secondary — long-record climatology / decarbonisation-trend framing only |

The model reads a cleaned CSV (`2010-2023-cleaned.csv`, 400k+ rows) and uses four
fields known at forecast time: `Td`, `Tw`, `RH`, `P`. Column names and units for
the full observatory record are documented in `2012-2023-cleaned_csv_units.csv`
(e.g. `Td`, `Tw` in °C, `RH` in %, `P` in hPa). Stage 1 caches the parsed weather
to a Parquet file next to the CSV to speed up repeat runs.

> Note: the automatic series ends in 2023, so this is a historical/backtest
> dataset, not a live feed. The tool is framed as "trained and validated on the
> observatory record, ready to run on a live feed." The dashboard demonstrates
> this honestly by checking each forecast against what actually happened.

---

## Stack

- **Python** — scikit-learn (`HistGradientBoostingRegressor`), pandas, NumPy, joblib
- **Frontend** — Streamlit dashboard with Plotly charts

```
pandas
numpy
scikit-learn
joblib
streamlit
plotly
```

---

## Running it

### 1. Train the models

Train and evaluate all three stages by running `main.py` directly:

```bash
python main.py
```

This runs stage 1, stage 2, and stage 3, saves the fitted stage-1 and stage-2
models to `models/`, and prints:

- Stage 1: rows used, train/test split, and Td/Tw forecast MAE
- Stage 2: baseline-PUE MAE for the ML model vs physics-only vs persistence,
  plus cooling-mode accuracy carried over from stage 1
- Stage 3: average PUE improvement, normalised kW/MW-IT saving, and recommended
  mode-hours

You can also run stage 1 or 2 on their own:

```bash
python train_sklearn_pue_stage1.py   # forecaster only
python train_sklearn_pue_stage2.py   # forecaster + PUE model
python train_sklearn.py              # standalone reference script
```

Make sure the cleaned CSV referenced by `DATA_PATH` sits alongside the scripts.

### 2. Launch the dashboard

Once `main.py` has saved the models, start the operator dashboard:

```bash
streamlit run dashboard2.py
```

It opens in the browser at `http://localhost:8501`. It expects, relative to
where you run it: `models/stage1_model.joblib`, `models/stage2_model.joblib`,
and `2010-2023-cleaned.csv`. If any are missing it shows a clear message
instead of crashing.

---

## The dashboard

A dark-themed Streamlit dashboard (green = free cooling) for a facilities /
sustainability engineer deciding, hour by hour, whether to run mechanical cooling
or ride the weather. It loads the saved models and forecasts from a chosen point
in the held-out record.

- **Headline decision** — the recommended cooling mode 3 hours ahead, colour-coded
  green / amber / red.
- **Metric tiles** — predicted dry-bulb, predicted wet-bulb, predicted PUE, and
  kW saved versus the 1.56 baseline.
- **"Did the forecast come true?" proof panel** — the model's 3h forecast shown
  against what the observatory actually recorded, with the miss in °C, the
  recommended vs actual cooling mode, and a correct/missed verdict. Backed by
  average error and mode accuracy across the whole window, so a single lucky
  point isn't the whole story.
- **Forecast-vs-actual chart** — predicted (solid) and actual (dotted) dry- and
  wet-bulb lines with the setpoint and evaporative-limit thresholds drawn in.
- **Cooling-mode timeline** — each forecast step coloured by mode across a
  12/24/48/72-hour horizon.
- **Impact readouts** — free-cooling hours, mechanical hours, avoided kWh, and
  avoided CO₂ over the window.
- **Sidebar controls** — cold-aisle setpoint, evaporative limit, IT load (MW),
  grid carbon intensity, forecast origin, and timeline horizon, plus a
  one-click "Standard Reading facility" preset.

> The proof panel works because these are historical points where the future is
> known — that is exactly what a backtest is. For a genuinely live forecast there
> is no "actual" yet; the same models run unchanged on a live weather feed.

---

## Configuration

Key constants you can tune (in the stage scripts and the dashboard):

| Constant | Default | Meaning |
|---|---|---|
| `SERVER_INLET_SETPOINT_C` | 24.0 | Cold-aisle setpoint — free-cooling threshold |
| `EVAP_WET_BULB_LIMIT_C` | 20.0 | Evaporative limit — evaporative vs mechanical threshold |
| `INDUSTRY_AVERAGE_PUE` | 1.56 | Unoptimised reference PUE savings are measured against |
| `FORECAST_STEPS` | 36 | Forecast horizon (36 × 5 min = 3 h) |

In the dashboard, the setpoint and evaporative limit are operator sliders, since
different facilities configure them differently. Note that the stage-2 PUE model
was trained at the fixed 24 / 20 thresholds, so moving those two sliders
recomputes the mode decision and mode-hours but not the trained PUE value itself.

---

## Repository layout

```
main.py                        # trains 3 stages end to end, saves models/*.joblib
train_sklearn_pue_stage1.py    # stage 1: 3h-ahead Td/Tw forecaster
train_sklearn_pue_stage2.py    # stage 2: PUE model
stage3.py                      # stage 3: operational impact (energy, cost, CO2)
train_sklearn.py               # earlier single-file reference script
dashboard2.py                  # Streamlit operator dashboard
models/                        # saved stage1_model.joblib + stage2_model.joblib
brief.md                       # project brief and design notes
2010-2023-cleaned.csv          # primary training/backtest data (not in repo)
2012-2023-cleaned_csv_units.csv# column/units schema for the observatory data
```

---

## Roadmap

- **Live feed** — swap the static CSV for a weather API (e.g. Open-Meteo, or the
  Met Office DataHub observations feed for a UK production story), derive wet-bulb
  via Stull's formula, and run the same saved models for a real-time 3h decision.
- **Uncertainty bands** — add quantile forecasts (`loss="quantile"`, 0.1 / 0.9)
  so the timeline can flag *amber* hours where the interval straddles a mode
  threshold — the windows an operator must plan a contingency for.
- **Climatology panel** — use the long manual record to show how the free-cooling
  resource is shrinking as the local climate warms, for net-zero and CAPEX planning.
