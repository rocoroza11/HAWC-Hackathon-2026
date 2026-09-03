import numpy as np
import pandas as pd

from train_sklearn_pue_stage1 import FORECAST_STEPS
from train_sklearn_pue_stage2 import cooling_mode

INDUSTRY_AVERAGE_PUE = 1.56
DATA_INTERVAL_HOURS = 5.0 / 60.0
FORECAST_HORIZON_HOURS = FORECAST_STEPS * DATA_INTERVAL_HOURS

def _as_aligned_series(value, index, name):

    """Turn a scalar, Series, array, or callable into an aligned Series."""

    if callable(value):
        value = value(index)

    if np.isscalar(value):
        return pd.Series(float(value), index=index, name=name)

    if isinstance(value, pd.Series):
        return value.reindex(index).astype(float).rename(name)

    return pd.Series(value, index=index, dtype=float, name=name)


def build_stage3_impact_frame(
    test,
    stage2_predictions,
    baseline_pue=INDUSTRY_AVERAGE_PUE,
    it_load_kw=None,
    interval_hours=DATA_INTERVAL_HOURS,
    electricity_price_per_kwh=None,
    carbon_intensity_kg_per_kwh=None,
):
    """Translate Stage 2 predictions into operational impact metrics.

    IT load, electricity price, and carbon intensity may each be a scalar,
    an index-aligned Series/array, or a callable receiving the test index.
    When IT load is omitted, the output is normalized per MW of IT load.
    """
    result = test.copy()
    result.attrs["interval_hours"] = interval_hours
    result["stage2_predicted_pue"] = np.asarray(stage2_predictions, dtype=float)
    result["recommended_mode"] = [
        cooling_mode(td, tw) for td, tw in zip(result["pred_Td"], result["pred_Tw"])
    ]
    result["baseline_pue"] = _as_aligned_series(
        baseline_pue, result.index, "baseline_pue"
    )
    result["pue_improvement"] = (
        result["baseline_pue"] - result["stage2_predicted_pue"]
    )

    # PUE is kW facility power divided by kW IT power. Therefore a PUE delta
    # directly gives the savings per unit of IT load.
    result["saving_kw_per_mw_it"] = result["pue_improvement"] * 1000.0
    result["saving_kwh_per_mw_it"] = (
        result["saving_kw_per_mw_it"] * interval_hours
    )

    if it_load_kw is not None:
        load = _as_aligned_series(it_load_kw, result.index, "it_load_kw")
        result["it_load_kw"] = load
        result["baseline_facility_kw"] = load * result["baseline_pue"]
        result["recommended_facility_kw"] = load * result["stage2_predicted_pue"]
        result["saving_kw"] = (
            result["baseline_facility_kw"] - result["recommended_facility_kw"]
        )
        result["saving_kwh"] = result["saving_kw"] * interval_hours

        if electricity_price_per_kwh is not None:
            price = _as_aligned_series(
                electricity_price_per_kwh, result.index, "electricity_price_per_kwh"
            )
            result["cost_saved"] = result["saving_kwh"] * price

        if carbon_intensity_kg_per_kwh is not None:
            carbon = _as_aligned_series(
                carbon_intensity_kg_per_kwh,
                result.index,
                "carbon_intensity_kg_per_kwh",
            )
            result["co2e_avoided_kg"] = result["saving_kwh"] * carbon

    return result


def print_stage3_report(stage3_frame):

    """Print stakeholder-facing totals from the Stage 3 impact frame."""

    interval_hours = stage3_frame.attrs.get("interval_hours", DATA_INTERVAL_HOURS)
    mode_hours = stage3_frame["recommended_mode"].value_counts() * interval_hours
    print("\n=== Stage 3: operational impact translation ===")
    print(f"Average baseline PUE             : {stage3_frame['baseline_pue'].mean():.3f}")
    print(f"Average recommended PUE          : {stage3_frame['stage2_predicted_pue'].mean():.3f}")
    print(f"Average PUE improvement           : {stage3_frame['pue_improvement'].mean():.3f}")
    print(f"Normalized saving                : {stage3_frame['saving_kw_per_mw_it'].mean():.1f} kW/MW IT")
    print(f"Normalized interval saving       : {stage3_frame['saving_kwh_per_mw_it'].sum():.1f} kWh/MW IT")
    print("Recommended mode-hours:")
    for mode in ("free", "evaporative", "mechanical"):
        print(f"  {mode:12s}: {mode_hours.get(mode, 0):.1f} h")

    for column, label, unit in (
        ("saving_kwh", "Estimated energy saving", "kWh"),
        ("cost_saved", "Estimated cost saving", "currency units"),
        ("co2e_avoided_kg", "Estimated carbon avoided", "kg CO2e"),
    ):
        if column in stage3_frame:
            print(f"{label:32s}: {stage3_frame[column].sum():.1f} {unit}")
