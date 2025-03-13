from typing import List
import fsspec
import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from fastapi import FastAPI, HTTPException

from client import MongoDBClient
from model import Shot, ShotInDB


class DataPool:
    def __init__(self):
        sources = pd.read_parquet("https://mastapp.site/parquet/level2/sources")
        sources = sources.loc[sources.name == "spectrometer_visible"]
        self.shots = sources.shot_id.values.tolist()

    def query(self) -> int:
        return int(np.random.choice(self.shots))


class ELMDataAnnotator:
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

    def background_subtract(
        self, signal: xr.DataArray, moving_av_length: float
    ) -> xr.DataArray:
        dtime = signal.time.values
        values = signal.values
        dt = dtime[1] - dtime[0]
        n = int(moving_av_length / dt)
        ret = np.cumsum(values, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1 :] / n
        values[n - 1 :] -= ret
        signal.values = values
        return signal

    def get_annotations(self, shot_id: int):
        store = self.get_remote_store(self.file_url.format(shot_id=shot_id))

        dataset = xr.open_zarr(store, group="spectrometer_visible")
        dalpha: xr.DataArray = dataset.filter_spectrometer_dalpha_voltage
        dalpha = dalpha.isel(dalpha_channel=2)
        dalpha = dalpha.dropna(dim="time")

        signal = self.background_subtract(dalpha.copy(), 0.001)

        trend = uniform_filter1d(signal, 1000)
        dalpha_detrend = signal - trend

        peak_idx, params = find_peaks(
            dalpha_detrend, prominence=0.2, width=[1, 150], distance=200, height=0.1
        )

        peaks = pd.DataFrame(params)
        peaks["time"] = dalpha_detrend.time.values[peak_idx]
        peaks["height"] = dalpha.values[peak_idx]
        peaks["valid"] = True
        peaks = peaks[["time", "height", "valid"]].to_dict(orient="records")
        return peaks


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
data_annotator = ELMDataAnnotator()


@app.post("/annotations", response_model=ShotInDB)
async def create_item(item: Shot):
    return await db.upsert(item)


@app.get("/annotations", response_model=List[ShotInDB])
async def get_items():
    return await db.list()


@app.get("/annotations/{shot_id}", response_model=Shot)
async def get_item(shot_id: str):
    annotations = await db.find(shot_id)

    if annotations is None:
        peaks = data_annotator.get_annotations(shot_id)
    else:
        peaks = annotations["elms"]

    return Shot(shot_id=shot_id, elms=peaks, regions=[])


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
