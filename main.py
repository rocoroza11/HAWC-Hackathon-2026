import pandas as pd 

dataset = pd.read_csv(
    "maindata.csv",
    encoding = "latin1"
)

dataset["timestamp"] = pd.to_datetime(
    dataset[["YYYY", "MM", "DD"]].rename(columns={"YYYY" : "year", "MM": "month", "DD" : "day"})
)

dataset = dataset.sort_values("timestamp").set_index("timestamp")

print(dataset.dtypes)