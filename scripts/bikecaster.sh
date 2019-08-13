#!/usr/bin/env bash

conda activate citi_bikecaster_model


USERNAME=bikecaster API_KEY=#FILL DASK_SCHEDULER="192.168.0.140:8786" gunicorn --workers 1 --bind unix:bikecaster.sock -m 007 wsgi:app