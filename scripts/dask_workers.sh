#!/usr/bin/env bash

PATH=/home/ubuntu/miniconda3/envs/citi_bikecaster_model/bin/:$PATH

dask-worker --nthreads 1 --nprocs 2 --memory-limit "1.7GB" tcp://127.0.0.1:8786
