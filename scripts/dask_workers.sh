#!/usr/bin/env bash

PATH=/home/ubuntu/miniconda3/envs/citi_bikecaster_model/bin/:$PATH

dask-worker --nthreads 1 --nprocs 2 --memory-limit "1.7GB" tcp://192.168.0.140:8786
