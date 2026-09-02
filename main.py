import pandas as pd 
import numpy as np

filepath = str(input("Enter filepath"))

def file_write(result, filepath):
    try:
        if isinstance(result, pd.DataFrame):
            result.to_csv(filepath, index=False)
            print(f"File has been saved to {filepath}")
    except OSError as e:
        print(f"File write failed: {e}")


dataset = pd.read_csv(
    "maindata.csv",
    encoding = "latin1",
    low_memory = False
)

dataset["timestamp"] = pd.to_datetime(
    dataset[["YYYY", "MM", "DD"]].rename(columns={"YYYY" : "year", "MM": "month", "DD" : "day"})
)

dataset = dataset.sort_values("timestamp").set_index("timestamp")

# Drop exact duplicate timestamps, keep first
dataset = dataset[~dataset.index.duplicated(keep="first")]

# Reindex to a full regular hourly grid so lags/rolling windows are honest
full_range = pd.date_range(dataset.index.min(), dataset.index.max(), freq="D")
df = dataset.reindex(full_range)

# Sanity-clip physically impossible values rather than dropping rows
df["RH09h %"] = df["RH09h %"].clip(0, 100)
df.loc[df["Tdry 09h, °C"] < -40, "Tdry 09h, °C"] = pd.NA   # implausible for UK data, treat as missing
df.loc[df["Tdry 09h, °C"] > 45,"Tdry 09h, °C"] = pd.NA

# Light interpolation for short gaps only (don't paper over long outages)
df["Tdry 09h, °C"] = df["Tdry 09h, °C"].interpolate(limit=3)
df["RH09h %"]   = df["RH09h %"].interpolate(limit=3)

file_write(df, "dftest.csv")




#def wet_bulb_stull(df):
