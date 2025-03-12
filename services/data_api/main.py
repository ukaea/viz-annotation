import fsspec
import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
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
    dalpha: xr.DataArray = dalpha_dataset.dalpha_mid_plane_wide.copy()
    dalpha = dalpha.dropna(dim="time")
    signal = background_subtract(dalpha.copy(), 0.001)

    trend = uniform_filter1d(signal, 1000)
    dalpha_detrend = signal - trend

    peak_idx, params = find_peaks(
        dalpha_detrend, prominence=0.2, width=[1, 150], distance=200, height=0.1
    )

    peaks = pd.DataFrame(params)
    peaks["time"] = dalpha_detrend.time.values[peak_idx]
    peaks = peaks[["time"]].to_dict(orient="records")

    # reduced_time = np.arange(dalpha.time.min(), dalpha.time.max(), 0.001)
    # dalpha = dalpha.interp(time=reduced_time)
    df_alpha = dalpha.to_dataframe().reset_index()  # Convert Xarray to Pandas DataFrame
    df_alpha.fillna(0, inplace=True)
    df_alpha.rename(columns={"dalpha_mid_plane_wide": "value"}, inplace=True)

    payload = {
        "dalpha": df_alpha.to_dict(orient="records"),
        "elms": peaks,
        "shot_id": shot_id,
    }
    return payload


if __name__ == "__main__":
    app.run(debug=True)
