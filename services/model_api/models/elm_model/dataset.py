import torch
import fsspec
import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, annotations):
        self.elms = [item['elms'] for item in annotations]
        self.shots = [shot['shot_id'] for shot in annotations]
        self.endpoint_url = "https://s3.echo.stfc.ac.uk"
        self.window_size = 512
        self.step_size = 256

    def __len__(self):
        return len(self.shots)

    def __getitem__(self, idx):
        shot = self.shots[idx]
        elms = self.elms[idx]

        url = f"s3://mast/level2/shots/{shot}.zarr"
        store = get_remote_store(url, self.endpoint_url)

        dataset = xr.open_zarr(store, group="spectrometer_visible")
        dalpha = dataset.filter_spectrometer_dalpha_voltage.isel(dalpha_channel=2)
        dalpha = dalpha.fillna(0)
        dalpha = background_subtract(dalpha)

        class_ = np.zeros_like(dalpha.time.values)
        class_ = xr.DataArray(class_, coords=dict(time=dalpha.time))

        for item in elms:
            buffer = 0.001
            class_.loc[
                dict(time=slice(item['time'] - buffer, item['time'] + buffer))
            ] = 1

        windows = generate_windows(dalpha.values, self.window_size, self.step_size)
        labels = generate_windows(class_.values, self.window_size, self.step_size)

        windows = torch.tensor(windows, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)

        windows.unsqueeze_(1)
        labels.unsqueeze_(1)

        return windows, labels


def generate_windows(data, window_size, step_size):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = [
        data[i * step_size : i * step_size + window_size] for i in range(num_windows)
    ]
    return windows


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


def background_subtract(
    dalpha: xr.DataArray, moving_av_length: float = 0.001
) -> xr.DataArray:
    dtime = dalpha.time.values
    values = dalpha.values
    dt = dtime[1] - dtime[0]
    n = int(moving_av_length / dt)
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1 :] / n
    values[n - 1 :] -= ret
    dalpha.values = values
    return dalpha