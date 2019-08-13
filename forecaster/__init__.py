import os

from dask.distributed import Client
from flask import Flask

app = Flask(__name__)
app.dask_client = Client("127.0.0.1:8786")

from forecaster import views
