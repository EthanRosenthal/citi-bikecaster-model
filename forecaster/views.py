import os
from pathlib import Path

from dask.distributed import fire_and_forget

from forecaster import app
from calcs import pipeline


BUCKET = "insulator-citi-bikecaster"
USERNAMES = os.environ["USERNAMES"].split(",")
API_KEYS = os.environ["API_KEYS"].split(",")


@app.route("/", methods=["POST",])
def post():
    app.dask_client.upload_file(Path(__file__).absolute().parent.parent / "calcs.py")
    s3_path = f"s3://{BUCKET}/status_cache.csv"
    fire_and_forget(app.dask_client.submit(pipeline, s3_path, USERNAMES, API_KEYS))
    return "OK"


@app.route("/", methods=["GET",])
def get():
    return "OK"
