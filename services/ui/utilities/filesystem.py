import fsspec
import xarray as xr
import zarr
from typing import Callable, Optional

import zarr.storage

END_POINT_URL = "https://s3.echo.stfc.ac.uk"

def get_dataset_loader(shot_id: str, location: str, metadata: bool = True) -> Callable[[str], Optional[xr.Dataset]]:
    """
    Function used to generate a callback to fetch datasets from the S3 server based on shot ID

    Parameters
    ----------
    shot_id : str
        The ID of the shot that data should be fetched from

    location : str
        Storage location ('level1' or 'test/level2' for example)

    metadata : bool
        Can consolidated metadata be used (preferable unless dataset doesn't allow)

    Returns
    -------
    Callable
        Callable used to return the dataset provided as a parameter
    """
    url = f"s3://mast/{location}/shots/{shot_id}.zarr"
    fs = fsspec.filesystem(
        **dict(
            protocol="simplecache",
            target_protocol="s3",
            target_options=dict(anon=True, endpoint_url=END_POINT_URL)
        )
    )

    store = zarr.storage.FSStore(fs=fs, url=url)

    # This function is returned so that it can be called for the different groups
    def get_dataset(group: str) -> Optional[xr.DataArray]:
        # Some data can be fetched using the consolidated metadata (faster), but some cannot
        try:
            if metadata:
                data = xr.open_zarr(fs.get_mapper(f"{url}/{group}"))
            else:
                data = xr.open_zarr(store, group=group)
            return data
        except:
            return None
    
    return get_dataset
    