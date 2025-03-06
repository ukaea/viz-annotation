import fsspec
import xarray as xr
import zarr
from typing import Callable, Optional

import zarr.storage

END_POINT_URL = "https://s3.echo.stfc.ac.uk"

def get_dataset(shot_id: str, group: str) -> xr.Dataset:
    url = f"s3://mast/level2/shots/{shot_id}.zarr"
    fs = fsspec.filesystem(
        **dict(
            protocol="simplecache",
            target_protocol="s3",
            target_options=dict(anon=True, endpoint_url=END_POINT_URL)
        )
    )

    store = zarr.storage.FSStore(fs=fs, url=url)
    dataset = xr.open_zarr(store, group=group)
    return dataset

    