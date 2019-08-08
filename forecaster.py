try:
  import unzip_requirements
except ImportError:
  pass

import gc
from io import StringIO
import logging
import os
import pickle
import time

import boto3
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import pandas as pd
import requests


logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.resource("s3")
BUCKET = "insulator-citi-bikecaster"
USERNAME = os.environ["USERNAME"]
API_KEY = os.environ["API_KEY"]
INSULATOR_URL = "https://api-dev.insulator.ai/v1/time_series"


def model_key(station_id):
    return f"models/station_{station_id}.pkl"


def load_model(station_id):
    response = s3.Object(BUCKET, model_key(station_id)).get()
    return pickle.loads(response["Body"].read())


def load_data_cache():
    response = s3.Object(BUCKET, "status_cache.csv").get()
    return pd.read_csv(
        StringIO(
            response["Body"]
            .read()
            .decode("utf-8")
        )
    )


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
    df = load_data_cache()
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
    del df
    gc.collect()

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
