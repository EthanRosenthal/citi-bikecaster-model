from datetime import datetime as dt
from datetime import timedelta
import pickle
import time

import dask.dataframe as dd
from dask.distributed import as_completed, worker_client
import numpy as np
import pandas as pd
import requests
import s3fs


BUCKET = "insulator-citi-bikecaster"
INSULATOR_URLS = [
    "https://api-dev.insulator.ai/v1/time_series",
    "https://ybcbwoz3w6.execute-api.us-east-1.amazonaws.com/staging/v1/time_series"
]

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


def post_outcome(df, station_id, usernames, api_keys):
    two_hours_ago = dt.now() - timedelta(hours=2)
    past_two_hours = df[df["last_reported"] >= two_hours_ago]
    past_two_hours = past_two_hours.sort_values("last_reported")

    series_timestamps = ts_to_unixtime(past_two_hours["last_reported"]).tolist()
    series_values = past_two_hours["num_bikes_available"].astype("int").tolist()
    post_event(station_id, series_timestamps, series_values, "outcome", usernames, api_keys)


def post_event(station_id, series_timestamps, series_values, event_type, usernames, api_keys):
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
    for username, api_key, insulator_url in zip(usernames, api_keys, INSULATOR_URLS):
        url = f"{insulator_url}/{event_type}"
        try:
            response = requests.post(url, auth=(username, api_key), json=payload)
            if not response:
                print(f"Error posting to insulator ingest API: {response.text}")
        except Exception as e:
            print(e)


def make_forecast(df, station_id, usernames, api_keys):
    station_df = df[df["station_id"] == station_id]

    post_outcome(station_df, station_id, usernames, api_keys)

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
    post_event(station_id, series_timestamps, series_values, "prediction", usernames, api_keys)
    return True


def pipeline(s3_path, usernames, api_keys):
    df = dd.read_csv(s3_path).compute()
    df["last_reported"] = pd.to_datetime(df["last_reported"])
    MIN_DATE = "2016-01-01"
    df = df[df.last_reported >= MIN_DATE]
    with worker_client() as client:
        df_future = client.scatter(df)
        futures = []
        for station_id in sorted(df["station_id"].unique().tolist()):
            futures.append(client.submit(make_forecast, df_future, station_id, usernames, api_keys))
        total = len(futures)
        success = 0
        for result in as_completed(futures):
            if result.result():
                success += 1
                if success % 50 == 0:
                    print(f"{success} / {total} tasks successfully completed")
    print(f"Done. Final tally: {success} / {total} tasks successfully completed")
    return True
