import fsspec
import pandas as pd
import xarray as xr
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


class ClassicELMModel:
    def __init__(self):
        pass

    def predict(self, shot_id: int):
        store = self.get_remote_store(
            f"s3://mast/test/level2/shots/{shot_id}.zarr", "https://s3.echo.stfc.ac.uk"
        )

        dalpha_dataset = xr.open_zarr(store, group="dalpha")
        dalpha: xr.DataArray = dalpha_dataset.dalpha_mid_plane_wide.copy()
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

        predictions = {
            "elms": peaks,
            "shot_id": shot_id,
        }
        return predictions

    def get_remote_store(self, path: str, endpoint_url: str):
        fs = fsspec.filesystem(
            **dict(
                protocol="filecache",
                target_protocol="s3",
                cache_storage=".cache",
                target_options=dict(anon=True, endpoint_url=endpoint_url),
            )
        )
        return fs.get_mapper(path)

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
