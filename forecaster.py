from datetime import datetime as dt
from datetime import timedelta
import logging
import os
from pathlib import Path
import pickle
import time

import dask.dataframe as dd
from dask.distributed import Client, fire_and_forget
from flask import Flask
import numpy as np
import pandas as pd
import requests
import s3fs


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

BUCKET = "insulator-citi-bikecaster"
USERNAME = os.environ["USERNAME"]
API_KEY = os.environ["API_KEY"]
INSULATOR_URL = "https://api-dev.insulator.ai/v1/time_series"

s3 = s3fs.S3FileSystem()
app = Flask(__name__)


def model_key(station_id):
    return f"models/station_{station_id}.pkl"


def load_model(station_id):
    with s3.open(f"{BUCKET}/{model_key(station_id)}", "rb") as f:
        return pickle.loads(f.read())


def load_local_model(station_id):
    with open(f"models/station_{station_id}.pkl", "rb") as f:
        return pickle.load(f)


def ts_to_unixtime(series):
    return series.astype(np.int64) // 10 ** 9


def post_outcome(df, station_id):
    two_hours_ago = dt.now() - timedelta(hours=2)
    past_two_hours = df[df["last_reported"] >= two_hours_ago]
    past_two_hours = past_two_hours.sort_values("last_reported")

    series_timestamps = ts_to_unixtime(past_two_hours["last_reported"]).tolist()
    series_values = past_two_hours["num_bikes_available"].astype("int").tolist()
    post_event(station_id, series_timestamps, series_values, "outcome")


def post_event(station_id, series_timestamps, series_values, event_type):
    payload = {
        "service_name": "bikecaster",
        "model_name": "lin_reg",
        "model_version": "0.1.0",
        "timestamp": time.time(),
        "entities": {"station_id": station_id},
        "series_timestamps": series_timestamps,
        "series_values": series_values
    }
    assert event_type in ("prediction", "outcome")
    url = f"{INSULATOR_URL}/{event_type}"
    response = requests.post(url, auth=(USERNAME, API_KEY), json=payload)
    if not response:
        logger.error(f"Error posting to insulator ingest API: {response.text}")


def handler(event, context):
    logger.info("Download data cache")
    df = dd.read_csv(f"s3://{BUCKET}/status_cache.csv").compute()
    try:
        station_id = event["station_id"]
        logger.info(f"Generating forecast for station {station_id}")
    except KeyError:
        msg = "`station_id` must be included in event body"
        logger.exception(msg)
        raise ValueError("`station_id` must be included in event body")

    df = df[df["station_id"] == station_id]
    df["last_reported"] = pd.to_datetime(df["last_reported"])

    post_outcome(df, station_id)

    df = (
        df
        .set_index("last_reported")
        .sort_index()
        .resample("5T", label="right", closed="right")
        .last()
        .fillna(method="ffill")
    )
    y = df["num_bikes_available"].values.copy()
    X = y.reshape(-1, 1).copy()

    logger.info("Loading model")
    try:
        model = load_model(station_id)
    except:
        logger.exception(f"There's no model for station {station_id}")
        return

    logger.info("Predicting with model")
    series_values = np.squeeze(model.predict(X, start_idx=len(X) - 1))

    logger.info("Sending prediction event")
    series_values = np.clip(series_values.astype(int), 0, None).astype("int").tolist()
    series_timestamps = pd.date_range(
        df.index[-1], periods=len(series_values) + 1, freq="5T"
    )
    # Remove the first value because it's the last value in the original data.
    series_timestamps = series_timestamps[1:]
    series_timestamps = ts_to_unixtime(series_timestamps).astype("int").tolist()
    post_event(station_id, series_timestamps, series_values, "prediction")
    return "OK"


def make_forecast(s3, df, station_id, job_run):
    station_df = df[df["station_id"] == station_id]

    post_outcome(station_df, station_id)

    logger.info("start sync")
    station_df = (
        station_df
        .set_index("last_reported")
        .sort_index()
        .resample("5T", label="right", closed="right")
        .last()
        .fillna(method="ffill")
    )
    y = station_df["num_bikes_available"].values.copy()
    X = y.reshape(-1, 1).copy()

    logger.info("start load model")
    try:
        model = load_local_model(station_id)
    except:
        logger.exception(f"There's no model for station {station_id}")
        return False

    logger.info("Predicting with model")
    try:
        series_values = np.squeeze(model.predict(X, start_idx=len(X) - 1))
    except:
        logger.exception(f"Error predicting for station {station_id}")
        return False

    logger.info("Sending prediction event")
    series_values = np.clip(series_values.astype(int), 0, None).astype("int").tolist()
    series_timestamps = pd.date_range(
        station_df.index[-1], periods=len(series_values) + 1, freq="5T"
    )
    # Remove the first value because it's the last value in the original data.
    series_timestamps = series_timestamps[1:]
    series_timestamps = ts_to_unixtime(series_timestamps).astype("int").tolist()
    post_event(station_id, series_timestamps, series_values, "prediction")
    with open(f"job_runs/{job_run}/{station_id}.txt", "w") as f:
        f.write("OK\n")
    return True


@app.route("/", methods=["POST",])
def post():
    logger.info("Loading data cache")
    # TODO: Have all the models on disk, ready to go (rather than downloading form S3 each time).
    df = dd.read_csv(f"s3://{BUCKET}/status_cache.csv").compute()
    logger.info("Data cache loaded")
    df["last_reported"] = pd.to_datetime(df["last_reported"])
    MIN_DATE = "2016-01-01"
    df = df[df.last_reported >= MIN_DATE]

    logger.info("Scattering dataframe")
    df_future = app.dask_client.scatter(df)
    job_run = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
    (Path("./job_runs") / job_run).mkdir()
    for station_id in sorted(df["station_id"].unique().tolist()):
        # station_df = df[df["station_id"] == station_id]
        fire_and_forget(app.dask_client.submit(make_forecast, s3, df_future, station_id, job_run))

    return "OK"


if __name__ == "__main__":
    app.dask_client = Client("192.168.0.140:8786")
    app.run(host="0.0.0.0", port=8000)
