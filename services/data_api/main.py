import fsspec
import xarray as xr
import pandas as pd
import numpy as np
from typing import List
from pathlib import Path
from scipy.signal import savgol_filter, stft
from fastapi import FastAPI

from client import MongoDBClient
from model import Shot
from annotators import AnnotatorType
from data_pool import DataPool
from model_runner import run_training, run_inference

DATA_PATH = Path('/data/elms')

def get_saddle_coil_channel_fft(data, nperseg=128):
    ds = data
    # Compute the Short-Time Fourier Transform (STFT)
    sample_rate = 1 / (ds.time[1] - ds.time[0])
    f, t, Zxx = stft(ds, fs=int(sample_rate), nperseg=nperseg, noverlap=nperseg // 5)

    t = t + ds.time.values[0]
    x = xr.DataArray(np.abs(Zxx), coords=dict(frequency=f, time=t))
    phi = xr.DataArray(np.angle(Zxx, deg=True), coords=dict(frequency=f, time=t))
    dataset = xr.Dataset(dict(amplitude=x, phase=phi))

    return dataset

class S3DataReader:
    def __init__(self):
        self.file_url = "s3://mast/level2/shots/{shot_id}.zarr"
        self.endpoint_url = "https://s3.echo.stfc.ac.uk"
        self.fs = fsspec.filesystem(
            **dict(
                protocol="filecache",
                target_protocol="s3",
                cache_storage=".cache",
                target_options=dict(anon=True, endpoint_url=self.endpoint_url),
            )
        )

    def get_remote_store(self, path: str):
        return self.fs.get_mapper(path)

    def get_data(self, shot_id: int):
        df_alpha = pd.read_parquet(DATA_PATH / f"{shot_id}.parquet")
        df_alpha.fillna(0, inplace=True)
        df_alpha = df_alpha.reset_index()
        df_alpha["time"] = pd.to_timedelta(df_alpha.time, "s")
        df_alpha = df_alpha.set_index("time")
        df_alpha = df_alpha.resample("0.1ms").max()
        df_alpha = df_alpha.reset_index()
        df_alpha.density_gradient = savgol_filter(df_alpha.density_gradient, 100, 2)
        df_alpha.rename(columns={"dalpha": "value"}, inplace=True)
        data = df_alpha.to_dict(orient="records")
        return data
    
    def get_disruption_data(self, shot_id: int):
        store = self.get_remote_store(self.file_url.format(shot_id=shot_id))

        magnetics = xr.open_zarr(store, group="magnetics")
        ip: xr.DataArray = magnetics['ip']
        ip = ip.dropna(dim="time")

        df_ip = ip.to_dataframe().reset_index()
        df_ip.fillna(0, inplace=True)
        df_ip.rename(columns={"ip": "value"}, inplace=True)
        data = df_ip.to_dict(orient="records")
        return data

    def get_locked_mode_data(self, shot_id: int, transform: bool = True):
        store = self.get_remote_store(self.file_url.format(shot_id=shot_id))

        magnetics = xr.open_zarr(store, group="magnetics")
        saddle_coils = magnetics["b_field_tor_probe_saddle_voltage"]
        saddle_coils = saddle_coils.rename(
            dict(time_saddle="time", b_field_tor_probe_saddle_voltage_channel="channel")
        )

        amps = []
        for i in range(len(saddle_coils.channel)):
            saddle_coil = saddle_coils.isel(channel=i)
            spec = get_saddle_coil_channel_fft(saddle_coil, nperseg=256)
            amps.append(spec.amplitude)

        coil_fft = sum([coil_fft for coil_fft in amps])
        coil_fft = coil_fft.sel(frequency=coil_fft.frequency > 2000)
        coil_fft = coil_fft.sel(
            time=slice(coil_fft.time.min() + 0.02, coil_fft.time.max() - 0.02)
        )
        if transform:
            coil_fft = xr.ufuncs.log10(coil_fft)
            coil_fft = (coil_fft - coil_fft.min()) / (coil_fft.max() - coil_fft.min())

        df = coil_fft.to_dataframe().reset_index()
        return df.to_dict(orient="records")


        


app = FastAPI()
db = MongoDBClient()
data_pool = DataPool(DATA_PATH)
data_reader = S3DataReader()


@app.post("/annotations/{shot_id}")
async def create_item(
    shot_id: int, item: Shot, method: AnnotatorType = AnnotatorType.UNET
):
    print(f"Updating annotations for {shot_id}")
    # Insert or update annotation in database
    new_annotation: bool = await db.upsert(item)

    # Get validated shots and update data pool
    shot_ids = await db.get_validated_shot_ids()
    data_pool.set_validated(shot_ids)

    # Check if we have met the criteria for retraining
    if not (data_pool.retrain() and new_annotation):
        return

    # Retrain the model
    print("Retraining model")
    annotations = await db.get_validated_annotations()
    unlabelled_shots = data_pool.unlabelled_shots
    run_model(method, shot_ids, annotations, unlabelled_shots)


@app.get("/annotations", response_model=List[Shot])
async def get_items():
    return await db.list()


@app.get("/annotations/{shot_id}", response_model=Shot)
async def get_item(
    shot_id: str,
    method: AnnotatorType = AnnotatorType.UNET,
    prominence: float = 0.1,
    distance: int = 1,
    force: bool = False,
):
    annotation = await db.find(shot_id)

    if annotation is None or force:
        print(f"Using annotator {method}")
        params = {
            "prominence": prominence,
            "distance": distance,
            "height": 0.01,
        }
        future = run_inference.delay(method, shot_id, **params)
        peaks = future.get(timeout=45)
        regions = []  # We currently do not support regions
        annotation = Shot(shot_id=shot_id, elms=peaks, regions=regions)
    else:
        print("Using annotation from database")

    return annotation


@app.get("/models/train")
async def train_model(method: AnnotatorType = AnnotatorType.UNET):
    labelled_shot_ids = await db.get_validated_shot_ids()
    annotations = await db.get_validated_annotations()
    unlabelled_shot_ids = data_pool.unlabelled_shots
    data_pool.set_validated(labelled_shot_ids)
    return run_model(method, labelled_shot_ids, annotations, unlabelled_shot_ids)


def run_model(method, labelld_shot_ids, annotations, unlabelled_shot_ids):
    if len(labelld_shot_ids) == 0:
        return {"status": "No labelled data."}
    if not data_pool.currently_training:
        future = run_training.delay(
            method, labelld_shot_ids, annotations, unlabelled_shot_ids
        )
        data_pool.training_future = future
        return {"status": "Training model"}
    else:
        print(data_pool.training_future.state)
        return {"status": "Model already training"}


@app.get("/next")
def get_next_shot_id():
    print("Querying next shot")
    shot_id = data_pool.query()
    return {"shot_id": shot_id}


@app.get("/data/{shot_id}")
async def get_data(shot_id: int):
    data = data_reader.get_data(shot_id)

    payload = {
        "dalpha": data,
        "shot_id": shot_id,
    }

    return payload

@app.get("/data/disruption/{shot_id}")
async def get_disruption_data(shot_id: int):
    return {
        "ip": data_reader.get_disruption_data(shot_id),
        "shot_id": shot_id,
    }

@app.get("/data/locked-mode/{shot_id}")
async def get_locked_mode_data(shot_id: int):
    return {
        "saddle_coil_fft": data_reader.get_locked_mode_data(shot_id),
        "shot_id": shot_id,
    }

@app.get("/data/locked-mode-raw/{shot_id}")
async def get_locked_mode_data(shot_id: int):
    return {
        "saddle_coil_fft": data_reader.get_locked_mode_data(shot_id, False),
        "shot_id": shot_id,
    }
