import os
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import fire_and_forget
import pandas as pd

from forecaster import app
from calcs import make_forecast


BUCKET = "insulator-citi-bikecaster"
USERNAME = os.environ["USERNAME"]
API_KEY = os.environ["API_KEY"]

@app.route("/", methods=["POST",])
def post():
    df = dd.read_csv(f"s3://{BUCKET}/status_cache.csv").compute()
    df["last_reported"] = pd.to_datetime(df["last_reported"])
    MIN_DATE = "2016-01-01"
    df = df[df.last_reported >= MIN_DATE]

    df_future = app.dask_client.scatter(df)
    app.dask_client.upload_file(Path(__file__).absolute().parent.parent / "calcs.py")
    for station_id in sorted(df["station_id"].unique().tolist()):
        fire_and_forget(app.dask_client.submit(make_forecast, df_future, station_id, USERNAME, API_KEY))

    return "OK"


@app.route("/", methods=["GET",])
def get():
    return "OK"
