from enum import Enum
from typing import List
import fsspec
import pandas as pd
import xarray as xr
import numpy as np
from fastapi import FastAPI, HTTPException

from client import MongoDBClient
from model import Shot
from annotators import ANNOTATORS, AnnotatorType, DataAnnotator


class DataPool:
    def __init__(self):
        self.db = MongoDBClient()
        sources = pd.read_parquet("https://mastapp.site/parquet/level2/sources")
        sources = sources.loc[sources.name == "spectrometer_visible"]
        self.shots = sources.shot_id.values.tolist()

    @property
    def size(self) -> int:
        return len(self.shots)

    def query(self) -> int:
        return int(np.random.choice(self.shots))


class ELMDataReader:
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


app = FastAPI()

db = MongoDBClient()
data_pool = DataPool()
data_reader = ELMDataReader()


@app.post("/annotations")
async def create_item(item: Shot):
    return await db.upsert(item)


@app.get("/annotations", response_model=List[Shot])
async def get_items():
    return await db.list()


@app.get("/annotations/{shot_id}", response_model=Shot)
async def get_item(shot_id: str, method: AnnotatorType = AnnotatorType.UNET):
    annotation = await db.find(shot_id)

    if annotation is None:
        print(f"Using annotator {method}")
        annotator: DataAnnotator = ANNOTATORS[method]
        peaks = annotator.get_annotations(shot_id)
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
