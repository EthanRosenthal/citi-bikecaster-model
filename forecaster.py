try:
  import unzip_requirements
except ImportError:
  pass

from io import StringIO
import logging
import pickle

import boto3
import numpy as np
import pandas as pd


logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.resource("s3")
BUCKET = "insulator-citi-bikecaster"


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


def handler(event, context):
    logger.info("Download data cache")
    df = load_data_cache()
    try:
        station_id = event["station_id"]
    except KeyError:
        msg = "`station_id` must be included in event body"
        logger.exception(msg)
        raise ValueError("`station_id` must be included in event body")

    df = df[df["station_id"] == station_id]
    df["last_reported"] = pd.to_datetime(df["last_reported"])
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

    model = load_model(station_id)

    series_values = np.squeeze(model.predict(X, start_idx=len(X) - 1))
    series_values = np.clip(series_values.astype(int), 0, None)
    series_timestamps = pd.date_range(
        df.index[-1], periods=len(series_values) + 1, freq="5T"
    )
    # Remove the first value because it's the last value in the original data.
    series_timestamps = series_timestamps[1:]
    series_timestamps = ts_to_unixtime(series_timestamps)
    logger.info({"series_values": series_values, "series_timestamps": series_timestamps})
    return "OK"
