import os
from pathlib import Path

import dask.dataframe as dd
from dask.distributed import fire_and_forget
import pandas as pd

from forecaster import app
from calcs import pipeline


BUCKET = "insulator-citi-bikecaster"
USERNAME = os.environ["USERNAME"]
API_KEY = os.environ["API_KEY"]


@app.route("/", methods=["POST",])
def post():
    app.dask_client.upload_file(Path(__file__).absolute().parent.parent / "calcs.py")
    s3_path = f"s3://{BUCKET}/status_cache.csv"
    fire_and_forget(app.dask_client.submit(pipeline, s3_path, USERNAME, API_KEY))
    return "OK"


@app.route("/", methods=["GET",])
def get():
    return "OK"
