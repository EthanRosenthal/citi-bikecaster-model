#!/usr/bin/env bash

PATH=/home/ubuntu/miniconda3/envs/citi_bikecaster_model/bin/:$PATH

dask-scheduler --host localhost --port 8786
