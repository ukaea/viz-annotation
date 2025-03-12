import fsspec
import xarray as xr
import numpy as np
from fastapi import FastAPI


def get_remote_store(path: str, endpoint_url: str):
    fs = fsspec.filesystem(
        **dict(
            protocol="filecache",
            target_protocol="s3",
            cache_storage=".cache",
            target_options=dict(anon=True, endpoint_url=endpoint_url),
        )
    )
    return fs.get_mapper(path)


def background_subtract(signal: xr.DataArray, moving_av_length: float) -> xr.DataArray:
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


app = FastAPI()


@app.get("/data")
def get_data(shot_id: int = 29495):
    store = get_remote_store(
        f"s3://mast/test/level2/shots/{shot_id}.zarr", "https://s3.echo.stfc.ac.uk"
    )

    dalpha_dataset = xr.open_zarr(store, group="dalpha")
    signal: xr.DataArray = dalpha_dataset.dalpha_mid_plane_wide.copy()
    signal = signal.dropna(dim="time")
    signal = background_subtract(signal, 0.001)

    df_alpha = signal.to_dataframe().reset_index()  # Convert Xarray to Pandas DataFrame
    df_alpha.fillna(0, inplace=True)
    df_alpha.rename(columns={"dalpha_mid_plane_wide": "value"}, inplace=True)

    payload = {
        "dalpha": df_alpha.to_dict(orient="records"),
    }
    return payload


if __name__ == "__main__":
    app.run(debug=True)
