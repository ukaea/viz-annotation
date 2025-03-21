from pathlib import Path
import fsspec
from joblib import Parallel, delayed
import pandas as pd
import xarray as xr
import time


def build_cache_file(shot, file_url, endpoint_url, cache_storage, output_dir):
    fs = fsspec.filesystem(
        **dict(
            protocol="filecache",
            target_protocol="s3",
            cache_storage=cache_storage,
            target_options=dict(anon=True, endpoint_url=endpoint_url),
        )
    )
    try:
        store = fs.get_mapper(file_url.format(shot_id=shot))

        # Dalpha dataset
        dataset = xr.open_zarr(store, group="spectrometer_visible")
        dalpha: xr.DataArray = dataset.filter_spectrometer_dalpha_voltage
        dalpha = dalpha.isel(dalpha_channel=2)

        dataset = xr.open_zarr(store, group="summary")
        ip: xr.DataArray = dataset.ip
        ip = ip.interp_like(dalpha)

        df = dalpha.to_dataframe()
        df["ip"] = ip.values
        df = df.drop("dalpha_channel", axis=1)
        df = df.rename({"filter_spectrometer_dalpha_voltage": "dalpha"}, axis=1)
        df = df.reset_index()

        df.to_parquet(output_dir / f"{shot}.parquet")
    except Exception as e:
        print(e)
        return None
    return shot


def main():
    endpoint_url = "https://s3.echo.stfc.ac.uk"
    file_url = "s3://mast/level2/shots/{shot_id}.zarr"
    cache_storage = ".cache"

    sources = pd.read_parquet("https://mastapp.site/parquet/level2/sources")
    sources = sources.loc[sources.name == "spectrometer_visible"]
    shots = sources.shot_id.values.tolist()

    output_dir = Path("./data/elms1")
    output_dir.mkdir(exist_ok=True, parents=True)

    tasks = [
        delayed(build_cache_file)(
            shot, file_url, endpoint_url, cache_storage, output_dir
        )
        for shot in reversed(shots)
    ]
    pool = Parallel(n_jobs=16, return_as="generator")
    results = pool(tasks)
    s = time.time()
    for result in results:
        print(result)
        pass
    e = time.time()
    print("total", e - s)


if __name__ == "__main__":
    main()
