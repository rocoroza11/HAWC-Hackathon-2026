from dataclasses import dataclass
from pathlib import Path

import joblib 
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor


DATA_PATH = Path(__file__).resolve().parent / "2010-2023-cleaned.csv"
FORECAST_STEPS = 36  # 36 * 5 minutes = 3 hours

def load_model():
    return joblib.load("stage1")

def default_cache_path(data_path):
    return Path(data_path).with_suffix(".weather.parquet")


@dataclass(frozen=True)
class Stage1Result:
    model: MultiOutputRegressor
    model_df: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    predictions: np.ndarray


def _read_weather_from_csv(data_path):
    raw = pd.read_csv(data_path)
    raw["TimeStamp"] = pd.to_datetime(raw["TimeStamp"], dayfirst=True, errors="coerce")
    raw = raw.dropna(subset=["TimeStamp"])
    raw = raw.sort_values("TimeStamp").set_index("TimeStamp")

    weather = raw[["Td", "Tw", "RH", "P"]].apply(pd.to_numeric, errors="coerce")
    return weather.dropna(subset=["Td", "Tw", "RH", "P"])


def load_weather_data(data_path=DATA_PATH, cache_path=None, use_cache=True):
    data_path = Path(data_path)
    cache_path = default_cache_path(data_path) if cache_path is None else Path(cache_path)

    if use_cache and cache_path.exists():
        csv_mtime = data_path.stat().st_mtime if data_path.exists() else 0
        if cache_path.stat().st_mtime >= csv_mtime:
            try:
                return pd.read_parquet(cache_path)
            except (ImportError, ValueError, OSError) as exc:
                print(f"Could not read Parquet cache ({exc}); rebuilding from CSV.", flush=True)

    weather = _read_weather_from_csv(data_path)

    if use_cache:
        try:
            weather.to_parquet(cache_path)
            print(f"Cached weather data at {cache_path.resolve()}", flush=True)
        except (ImportError, ValueError, OSError) as exc:
            print(f"Could not write Parquet cache ({exc}); continuing without cache.", flush=True)

    return weather


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


def build_stage1_model():
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=300,
            learning_rate=0.75,
            max_leaf_nodes=50,
            max_depth=None,
            min_samples_leaf=20,
            l2_regularization=0.01,
            max_features=1.0,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        )
    )


def train_stage1_forecaster(data_path=DATA_PATH):
    print(f"Loading weather data from {Path(data_path).resolve()}...", flush=True)
    weather = load_weather_data(data_path)
    model_df = add_features(weather)
    print(f"Prepared {len(model_df):,} training rows. Fitting models...", flush=True)
    feature_cols = [c for c in model_df.columns if not c.startswith("target_")]
    target_cols = ["target_Td", "target_Tw"]

    split_at = int(len(model_df) * 0.8)
    train, test = model_df.iloc[:split_at], model_df.iloc[split_at:]
    X_train, y_train = train[feature_cols], train[target_cols]
    X_test, y_test = test[feature_cols], test[target_cols]

    model = build_stage1_model()
    model.fit(X_train, y_train)
    print("Generating test predictions...", flush=True)
    predictions = model.predict(X_test)

    return Stage1Result(
        model=model,
        model_df=model_df,
        train=train,
        test=test,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )


def print_stage1_report(result):
    print("=== Stage 1: 3h-ahead Td/Tw forecaster ===")
    print(
        f"Rows used: {len(result.model_df):,}  "
        f"Train: {len(result.train):,}  Test: {len(result.test):,}"
    )
    print(
        f"Td MAE: {mean_absolute_error(result.y_test['target_Td'], result.predictions[:, 0]):.3f} C   "
        f"Tw MAE: {mean_absolute_error(result.y_test['target_Tw'], result.predictions[:, 1]):.3f} C"
    )


def main():
    result = train_stage1_forecaster()
    print_stage1_report(result)

if __name__ == "__main__":
    main()
