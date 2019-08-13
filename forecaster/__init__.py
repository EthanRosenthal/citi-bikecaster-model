import os

from dask.distributed import Client
from flask import Flask

app = Flask(__name__)
app.dask_client = Client(os.environ["DASK_SCHEDULER"])

from forecaster import views
