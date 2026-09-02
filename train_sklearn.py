import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor


DATA_PATH = "reading_5min_all_columns.csv"
FORECAST_STEPS = 36  # 36 * 5 minutes = 3 hours
SERVER_INLET_SETPOINT_C = 24.0
EVAP_WET_BULB_LIMIT_C = 20.0


def cooling_mode(td, tw):
    if td <= SERVER_INLET_SETPOINT_C:
        return "free"
    if tw <= EVAP_WET_BULB_LIMIT_C:
        return "evaporative"
    return "mechanical"


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

    out["target_Td"] = df["Td"].shift(-FORECAST_STEPS)
    out["target_Tw"] = df["Tw"].shift(-FORECAST_STEPS)
    return out.dropna()


def main():
    raw = pd.read_csv(DATA_PATH)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["TimeStamp"])
    raw = raw.sort_values("TimeStamp").set_index("TimeStamp")

    # Keep only the weather fields the model can know at forecast time.
    weather = raw[["Td", "Tw", "RH", "P"]].apply(pd.to_numeric, errors="coerce")
    weather = weather.dropna(subset=["Td", "Tw", "RH", "P"])

    model_df = add_features(weather)
    feature_cols = [c for c in model_df.columns if not c.startswith("target_")]
    target_cols = ["target_Td", "target_Tw"]

    split_at = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_at]
    test = model_df.iloc[split_at:]

    X_train = train[feature_cols]
    y_train = train[target_cols]
    X_test = test[feature_cols]
    y_test = test[target_cols]

    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss = "squared_error",
            max_iter=300, 
            learning_rate=0.05,
            max_leaf_nodes = 31, 
            max_depth = None, 
            min_samples_leaf = 20, 
            l2_regularization = 0.0,
            max_features = 1.0, 
            early_stopping= False, 
            random_state=42
            )
    )
    model.fit(X_train, y_train)

    predictions = pd.DataFrame(
        model.predict(X_test),
        columns=["pred_Td", "pred_Tw"],
        index=X_test.index,
    )

    persistence = X_test[["Td", "Tw"]].rename(columns={"Td": "pred_Td", "Tw": "pred_Tw"})

    print(f"Rows used: {len(model_df):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print()
    print("3-hour temperature MAE")
    print(f"Model Td: {mean_absolute_error(y_test['target_Td'], predictions['pred_Td']):.3f} C")
    print(f"Model Tw: {mean_absolute_error(y_test['target_Tw'], predictions['pred_Tw']):.3f} C")
    print(f"Persistence Td: {mean_absolute_error(y_test['target_Td'], persistence['pred_Td']):.3f} C")
    print(f"Persistence Tw: {mean_absolute_error(y_test['target_Tw'], persistence['pred_Tw']):.3f} C")
    print()

    actual_modes = [
        cooling_mode(td, tw)
        for td, tw in zip(y_test["target_Td"], y_test["target_Tw"])
    ]
    predicted_modes = [
        cooling_mode(td, tw)
        for td, tw in zip(predictions["pred_Td"], predictions["pred_Tw"])
    ]
    persistence_modes = [
        cooling_mode(td, tw)
        for td, tw in zip(persistence["pred_Td"], persistence["pred_Tw"])
    ]

    print("3-hour cooling-mode accuracy")
    print(f"Model: {accuracy_score(actual_modes, predicted_modes):.3f}")
    print(f"Persistence: {accuracy_score(actual_modes, persistence_modes):.3f}")


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor


DATA_PATH = "reading_5min_all_columns.csv"
FORECAST_STEPS = 36  # 36 * 5 minutes = 3 hours
SERVER_INLET_SETPOINT_C = 24.0
EVAP_WET_BULB_LIMIT_C = 20.0


def cooling_mode(td, tw):
    if td <= SERVER_INLET_SETPOINT_C:
        return "free"
    if tw <= EVAP_WET_BULB_LIMIT_C:
        return "evaporative"
    return "mechanical"


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

    out["target_Td"] = df["Td"].shift(-FORECAST_STEPS)
    out["target_Tw"] = df["Tw"].shift(-FORECAST_STEPS)
    return out.dropna()


def main():
    raw = pd.read_csv(DATA_PATH)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["TimeStamp"])
    raw = raw.sort_values("TimeStamp").set_index("TimeStamp")

    # Keep only the weather fields the model can know at forecast time.
    weather = raw[["Td", "Tw", "RH", "P"]].apply(pd.to_numeric, errors="coerce")
    weather = weather.dropna(subset=["Td", "Tw", "RH", "P"])

    model_df = add_features(weather)
    feature_cols = [c for c in model_df.columns if not c.startswith("target_")]
    target_cols = ["target_Td", "target_Tw"]

    split_at = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_at]
    test = model_df.iloc[split_at:]

    X_train = train[feature_cols]
    y_train = train[target_cols]
    X_test = test[feature_cols]
    y_test = test[target_cols]

    model = MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss = "squared_error",
            max_iter=300, 
            learning_rate=0.05,
            max_leaf_nodes = 31, 
            max_depth = None, 
            min_samples_leaf = 20, 
            l2_regularization = 0.0,
            max_features = 1.0, 
            early_stopping= False, 
            random_state=42
            )
    )
    model.fit(X_train, y_train)

    predictions = pd.DataFrame(
        model.predict(X_test),
        columns=["pred_Td", "pred_Tw"],
        index=X_test.index,
    )

    persistence = X_test[["Td", "Tw"]].rename(columns={"Td": "pred_Td", "Tw": "pred_Tw"})

    print(f"Rows used: {len(model_df):,}")
    print(f"Train rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print()
    print("3-hour temperature MAE")
    print(f"Model Td: {mean_absolute_error(y_test['target_Td'], predictions['pred_Td']):.3f} C")
    print(f"Model Tw: {mean_absolute_error(y_test['target_Tw'], predictions['pred_Tw']):.3f} C")
    print(f"Persistence Td: {mean_absolute_error(y_test['target_Td'], persistence['pred_Td']):.3f} C")
    print(f"Persistence Tw: {mean_absolute_error(y_test['target_Tw'], persistence['pred_Tw']):.3f} C")
    print()

    actual_modes = [
        cooling_mode(td, tw)
        for td, tw in zip(y_test["target_Td"], y_test["target_Tw"])
    ]
    predicted_modes = [
        cooling_mode(td, tw)
        for td, tw in zip(predictions["pred_Td"], predictions["pred_Tw"])
    ]
    persistence_modes = [
        cooling_mode(td, tw)
        for td, tw in zip(persistence["pred_Td"], persistence["pred_Tw"])
    ]

    print("3-hour cooling-mode accuracy")
    print(f"Model: {accuracy_score(actual_modes, predicted_modes):.3f}")
    print(f"Persistence: {accuracy_score(actual_modes, persistence_modes):.3f}")


if __name__ == "__main__":
    main()
