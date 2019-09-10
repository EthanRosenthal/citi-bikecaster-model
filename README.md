# citi-bikecaster-model

## Hacky Server Deployment

Run `ssh-add -K` on local machine to get ssh key into keychain. This 
allows for forwarding it when logging into the cloud instance so that 
you can git pull, git clone, etc...

### Create environment

```commandline
conda create -n citi_bikecaster_model python=3.6 -y -q
conda activate citi_bikecaster_model
# Need to install pandas from conda so you pickup gcc
conda install pandas==0.24.2
pip install -r requirements.txt
```

### Download models

```python
import boto3
import dask.dataframe as dd

df = dd.read_csv(f"s3://insulator-citi-bikecaster/status_cache.csv").compute()
station_ids = df["station_id"].unique().tolist()

bucket = boto3.resource("s3").Bucket("insulator-citi-bikecaster")

for ctr, station_id in enumerate(station_ids, start=1):
    if ctr % 50 == 0:
        print(ctr)
    try:
         bucket.download_file(f"models/station_{station_id}.pkl", f"models/station_{station_id}.pkl")
    except:
        print(f"No model for station {station_id}")
```

### Setup nginx

```commandline
sudo cp nginx.conf /etc/nginx/sites-available/bikecaster
sudo ln -s /etc/nginx/sites-available/bikecaster /etc/nginx/sites-enabled
sudo nginx -t
sudo service nginx restart
```


### Run the dask scheduler, workers, and flask app

```commandline
nohup ./scripts/dask_scheduler.sh > logs/scheduler.log &
nohup ./scripts/dask_workers.sh > logs/workers.log &
nohup ./scripts/bikecaster.sh > logs/bikecaster.log &
```
