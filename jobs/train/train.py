import logging
import pickle

import boto3
import dask.dataframe as dd
import fire
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion

from skits.feature_extraction import AutoregressiveTransformer
from skits.pipeline import ForecasterPipeline
from skits.preprocessing import HorizonTransformer, ReversibleImputer

BUCKET = "insulator-citi-bikecaster"
TRAINING_DATA_PATH = f"s3://{BUCKET}/models/training_data.csv"


logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def load_training_data(station_id):
    df = dd.read_csv(TRAINING_DATA_PATH)
    df.columns = ["station_id", "last_reported", "num_bikes_available"]
    df = df[df["station_id"] == station_id].compute()
    df = df.drop(columns=["station_id"])
    df["last_reported"] = pd.to_datetime(df["last_reported"])
    df = df.set_index("last_reported").sort_index()
    df = df.resample("5T", label="right", closed="right").last().fillna(method="ffill")
    return df


def make_model():
    period_minutes = 5
    samples_per_hour = int(60 / period_minutes)
    samples_per_day = int(24 * samples_per_hour)
    samples_per_week = int(7 * samples_per_day)

    pipeline = ForecasterPipeline([
        # Convert the `y` target into a horizon
        ('pre_horizon', HorizonTransformer(horizon=samples_per_hour * 2)),
        ('pre_reversible_imputer', ReversibleImputer(y_only=True)),
        ('features', FeatureUnion([
            # Generate a week's worth of autoregressive features
            ('ar_features', AutoregressiveTransformer(num_lags=samples_per_week)),
        ])),
        ('post_feature_imputer', ReversibleImputer()),
        ('regressor', LinearRegression(n_jobs=-1))
    ])
    return pipeline


def upload_model(model, station_id):
    key = f"models/station_{station_id}.pkl"

    model_bytes = pickle.dumps(model)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(BUCKET, key).put(Body=model_bytes)


def train(station_id):
    logger.info("Loading training data")
    df = load_training_data(station_id)
    y = df["num_bikes_available"].values.copy()
    X = y.reshape(-1, 1).copy()

    logger.info("Fitting model")
    model = make_model()
    model = model.fit(X, y)

    logger.info("Uploading model to S3")
    upload_model(model, station_id)


if __name__ == "__main__":
    fire.Fire(train)
