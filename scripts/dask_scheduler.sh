#!/usr/bin/env bash

PATH=/home/ubuntu/miniconda3/envs/citi_bikecaster_model/bin/:$PATH

dask-scheduler --host 192.168.0.140 --port 8786
