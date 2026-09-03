# Free-Cooling Forecast for Data Centre Cooling Optimisation

A simple ML prediction model that turns localised Reading weather data into a
**3-hour-ahead cooling decision** for Thames Valley data centre operators:
when to ride the weather with *free* or *evaporative* cooling instead of running
mechanical chillers — and what that saves in energy, carbon, and PUE.

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

The eventual output is a **decision**, not a temperature: a green/amber/red
cooling schedule tied to avoided kWh and avoided CO₂, benchmarked against an
industry-average PUE of 1.56.

---

## Architecture

The model runs in two stages, wired together in `main.py`.

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
persistence. Savings are estimated as `IT_load × (industry_average_PUE − predicted_PUE)`.

`train_sklearn.py` is an earlier standalone script that does stage-1 forecasting
plus cooling-mode accuracy in a single file — kept for reference.

---

## Datasets

| Dataset | Coverage | Role |
|---|---|---|
| Reading Atmospheric Observatory automatic measurements | 1997–2023, 5-min cadence | Primary — trains and validates the forecaster |
| Reading manual weather observations | 1901–present, daily 0900 UTC | Secondary — long-record climatology / decarbonisation-trend framing only |

The model reads a cleaned CSV (e.g. `2010-2023-cleaned.csv`) and uses four
fields known at forecast time: `Td`, `Tw`, `RH`, `P`. Column names and units for
the full observatory record are documented in `2012-2023-cleaned_csv_units.csv`
(e.g. `Td`, `Tw` in °C, `RH` in %, `P` in hPa).

> Note: the automatic series ends in 2023, so this is a historical/backtest
> dataset, not a live feed. The tool is framed as "trained and validated on the
> observatory record, ready to run on a live feed."

---

## Stack

- **Python** — scikit-learn (`HistGradientBoostingRegressor`), pandas, NumPy
- **Frontend (planned)** — Streamlit dashboard for operators
  *(not yet built)*

```
pandas
numpy
scikit-learn
streamlit
```

---

## Running it

Train and evaluate both stages by running `main.py` directly:

```bash
python main.py
```

This runs stage 1, then stage 2, and prints:

- Stage 1: rows used, train/test split, and Td/Tw forecast MAE
- Stage 2: baseline-PUE MAE for the ML model vs physics-only vs persistence,
  plus cooling-mode accuracy carried over from stage 1

You can also run each stage on its own:

```bash
python train_sklearn_pue_stage1.py   # forecaster only
python train_sklearn_pue_stage2.py   # forecaster + PUE model
python train_sklearn.py              # standalone reference script
```

Make sure the cleaned CSV referenced by `DATA_PATH` sits alongside the scripts.

---

## Configuration

Key constants you can tune (in the stage scripts):

| Constant | Default | Meaning |
|---|---|---|
| `SERVER_INLET_SETPOINT_C` | 24.0 | Cold-aisle setpoint — free-cooling threshold |
| `EVAP_WET_BULB_LIMIT_C` | 20.0 | Evaporative limit — evaporative vs mechanical threshold |
| `INDUSTRY_AVERAGE_PUE` | 1.56 | Unoptimised reference PUE savings are measured against |
| `FORECAST_STEPS` | 36 | Forecast horizon (36 × 5 min = 3 h) |

In the planned dashboard, the setpoint and evaporative limit become operator
sliders, since different facilities configure them differently.

---

## Frontend (planned)

A Streamlit dashboard for a facilities/sustainability engineer who decides, hour
by hour, whether to run mechanical cooling or ride the weather. Intended panels:

- A 72-hour green/amber/red cooling-mode timeline, with **amber** marking hours
  where the forecast interval straddles a mode threshold (needs a contingency).
- Sidebar sliders for setpoint, evaporative limit, IT load, and grid carbon intensity.
- Sustainability readouts: free-cooling hours, avoided kWh, avoided kgCO₂, effective PUE.
- A long-record climatology panel showing how the free-cooling resource is
  shrinking as the local climate warms.

---

## Repository layout

```
main.py                        # runs stage 1 + stage 2 end to end
train_sklearn_pue_stage1.py    # 3h-ahead Td/Tw forecaster
train_sklearn_pue_stage2.py    # PUE model + savings estimate
train_sklearn.py               # earlier single-file reference script
brief.md                       # project brief and design notes
2012-2023-cleaned_csv_units.csv# column/units schema for the observatory data
```
