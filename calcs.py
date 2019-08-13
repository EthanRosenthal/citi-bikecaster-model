from datetime import datetime as dt
from datetime import timedelta
import pickle
import time

import numpy as np
import pandas as pd
import requests
import s3fs


BUCKET = "insulator-citi-bikecaster"
INSULATOR_URL = "https://api-dev.insulator.ai/v1/time_series"

s3 = s3fs.S3FileSystem()


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


def post_outcome(df, station_id, username, api_key):
    two_hours_ago = dt.now() - timedelta(hours=2)
    past_two_hours = df[df["last_reported"] >= two_hours_ago]
    past_two_hours = past_two_hours.sort_values("last_reported")

    series_timestamps = ts_to_unixtime(past_two_hours["last_reported"]).tolist()
    series_values = past_two_hours["num_bikes_available"].astype("int").tolist()
    post_event(station_id, series_timestamps, series_values, "outcome", username, api_key)


def post_event(station_id, series_timestamps, series_values, event_type, username, api_key):
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
    response = requests.post(url, auth=(username, api_key), json=payload)
    if not response:
        print(f"Error posting to insulator ingest API: {response.text}")


def make_forecast(df, station_id, username, api_key):
    station_df = df[df["station_id"] == station_id]

    post_outcome(station_df, station_id, username, api_key)

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

    try:
        model = load_local_model(station_id)
    except:
        print(f"There's no model for station {station_id}")
        return False

    try:
        series_values = np.squeeze(model.predict(X, start_idx=len(X) - 1))
    except:
        print(f"Error predicting for station {station_id}")
        return False

    series_values = np.clip(series_values.astype(int), 0, None).astype("int").tolist()
    series_timestamps = pd.date_range(
        station_df.index[-1], periods=len(series_values) + 1, freq="5T"
    )
    # Remove the first value because it's the last value in the original data.
    series_timestamps = series_timestamps[1:]
    series_timestamps = ts_to_unixtime(series_timestamps).astype("int").tolist()
    post_event(station_id, series_timestamps, series_values, "prediction", username, api_key)
    return True
