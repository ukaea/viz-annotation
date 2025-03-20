from typing import List
import fsspec
import xarray as xr
from fastapi import FastAPI, HTTPException

from client import MongoDBClient
from model import Shot
from annotators import AnnotatorType
from data_pool import DataPool
from model_runner import run_training, run_inference


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
        store = self.get_remote_store(self.file_url.format(shot_id=shot_id))

        dataset = xr.open_zarr(store, group="spectrometer_visible")
        dalpha: xr.DataArray = dataset.filter_spectrometer_dalpha_voltage
        dalpha = dalpha.isel(dalpha_channel=2)
        dalpha = dalpha.dropna(dim="time")

        df_alpha = dalpha.to_dataframe().reset_index()
        df_alpha.fillna(0, inplace=True)
        df_alpha.rename(
            columns={"filter_spectrometer_dalpha_voltage": "value"}, inplace=True
        )
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
        

app = FastAPI()

db = MongoDBClient()
data_pool = DataPool("/data/elms")
data_reader = S3DataReader()


@app.post("/annotations")
async def create_item(item: Shot, method: AnnotatorType = AnnotatorType.UNET):
    print(f"Updating annotations for {item.shot_id}")
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


@app.get("/annotations", response_model=List[Shot])
async def get_items():
    return await db.list()


@app.get("/annotations/{shot_id}", response_model=Shot)
async def get_item(shot_id: str, method: AnnotatorType = AnnotatorType.UNET):
    annotation = await db.find(shot_id)

    if annotation is None:
        print(f"Using annotator {method}")
        future = run_inference.delay(method, shot_id)
        peaks = future.get(timeout=30)
        regions = []  # We currently do not support regions
        annotation = Shot(shot_id=shot_id, elms=peaks, regions=regions)
    else:
        print("Using annotation from database")

    return annotation


@app.delete("/annotations/{shot_id}")
async def delete_item(shot_id: str):
    try:
        return db.delete(shot_id)
    except RuntimeError:
        raise HTTPException(status_code=404, detail="Item not found")


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