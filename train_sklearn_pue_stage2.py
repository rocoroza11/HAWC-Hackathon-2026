import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from train_sklearn_pue_stage1 import (
    DATA_PATH,
    Stage1Result,
    print_stage1_report,
    train_stage1_forecaster,
)


SERVER_INLET_SETPOINT_C = 24.0
EVAP_WET_BULB_LIMIT_C = 20.0

# Unoptimized reference: what a facility gets by default, with no
# weather-responsive control. Source: Uptime Institute Global Data Center
# Survey 2024 -- industry-average PUE, flat at 1.56 for the 5th straight
# year. This is what "saving" is measured against, NOT what the curve below
# predicts.
INDUSTRY_AVERAGE_PUE = 1.56


def cooling_mode(td, tw):
    if td <= SERVER_INLET_SETPOINT_C:
        return "free"
    if tw <= EVAP_WET_BULB_LIMIT_C:
        return "evaporative"
    return "mechanical"


def optimized_pue_series(td, tw):
    """PUE an aggressively weather-responsive control strategy could achieve.

    This is the stage-2 model's target, not an unoptimized baseline. The
    unoptimized comparison point is INDUSTRY_AVERAGE_PUE.
    """
    td = np.asarray(td, dtype=float)
    tw = np.asarray(tw, dtype=float)
    pue = np.empty_like(td)

    free_mask = td <= SERVER_INLET_SETPOINT_C
    evap_mask = (~free_mask) & (tw <= EVAP_WET_BULB_LIMIT_C)
    mech_mask = (~free_mask) & (~evap_mask)

    pue[free_mask] = 1.07 + 0.02 * np.clip(td[free_mask] / SERVER_INLET_SETPOINT_C, 0, 1)
    pue[evap_mask] = 1.10 + 0.11 * np.clip(tw[evap_mask] / EVAP_WET_BULB_LIMIT_C, 0, 1)
    excess = np.clip(tw[mech_mask] - EVAP_WET_BULB_LIMIT_C, 0, 10)
    pue[mech_mask] = np.minimum(1.30 + 0.03 * excess, INDUSTRY_AVERAGE_PUE)

    return pue


def estimate_saving_kw(it_load_kw, predicted_pue):
    """Estimate cooling-control savings against industry-average PUE."""
    it_load_kw = np.asarray(it_load_kw, dtype=float)
    predicted_pue = np.asarray(predicted_pue, dtype=float)
    return it_load_kw * (INDUSTRY_AVERAGE_PUE - predicted_pue)


def build_stage2_eval_frame(stage1_result: Stage1Result):
    X_test = stage1_result.X_test
    y_test = stage1_result.y_test
    pred = stage1_result.predictions

    eval_df = pd.DataFrame(
        {
            "pred_Td": pred[:, 0],
            "pred_Tw": pred[:, 1],
            "actual_Td": y_test["target_Td"].values,
            "actual_Tw": y_test["target_Tw"].values,
            "current_Td": X_test["Td"].values,
            "current_Tw": X_test["Tw"].values,
            "RH": X_test["RH"].values,
            "P": X_test["P"].values,
            "tod_sin": X_test["tod_sin"].values,
            "tod_cos": X_test["tod_cos"].values,
            "doy_sin": X_test["doy_sin"].values,
            "doy_cos": X_test["doy_cos"].values,
        },
        index=X_test.index,
    )

    # Ground-truth label: optimized PUE evaluated on actual future Td/Tw.
    eval_df["true_pue"] = optimized_pue_series(eval_df["actual_Td"], eval_df["actual_Tw"])
    return eval_df


def build_stage2_model():
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )


def train_stage2_pue(stage1_result: Stage1Result):
    eval_df = build_stage2_eval_frame(stage1_result)
    feature_cols = [
        "pred_Td",
        "pred_Tw",
        "current_Td",
        "current_Tw",
        "RH",
        "P",
        "tod_sin",
        "tod_cos",
        "doy_sin",
        "doy_cos",
    ]

    split_at = int(len(eval_df) * 0.8)
    train, test = eval_df.iloc[:split_at], eval_df.iloc[split_at:]

    model = build_stage2_model()
    model.fit(train[feature_cols], train["true_pue"])
    predictions = model.predict(test[feature_cols])

    return model, eval_df, train, test, predictions


def print_stage2_report(eval_df, train, test, predictions):
    physics_only_pred = optimized_pue_series(test["pred_Td"], test["pred_Tw"])
    persistence_pred = optimized_pue_series(test["current_Td"], test["current_Tw"])

    print("\n=== Stage 2: baseline PUE model ===")
    print(f"Stage 2 rows: {len(eval_df):,}  Train: {len(train):,}  Test: {len(test):,}")
    print("3-hour baseline-PUE MAE")
    print(f"  ML stage-2 model                    : {mean_absolute_error(test['true_pue'], predictions):.4f}")
    print(f"  Physics-only (stage-1 fcst -> curve) : {mean_absolute_error(test['true_pue'], physics_only_pred):.4f}")
    print(f"  Persistence (now -> curve)           : {mean_absolute_error(test['true_pue'], persistence_pred):.4f}")

    actual_modes = [cooling_mode(td, tw) for td, tw in zip(test["actual_Td"], test["actual_Tw"])]
    physics_modes = [cooling_mode(td, tw) for td, tw in zip(test["pred_Td"], test["pred_Tw"])]
    persistence_modes = [cooling_mode(td, tw) for td, tw in zip(test["current_Td"], test["current_Tw"])]

    print("\n3-hour cooling-mode accuracy (context, carried over from stage 1)")
    print(f"  Physics-only (stage-1 fcst) : {accuracy_score(actual_modes, physics_modes):.3f}")
    print(f"  Persistence                 : {accuracy_score(actual_modes, persistence_modes):.3f}")


def main(data_path=DATA_PATH):
    stage1_result = train_stage1_forecaster(data_path)
    print_stage1_report(stage1_result)

    _, eval_df, train, test, predictions = train_stage2_pue(stage1_result)
    print_stage2_report(eval_df, train, test, predictions)


if __name__ == "__main__":
    main()
