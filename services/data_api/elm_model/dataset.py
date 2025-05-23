from pathlib import Path
import pandas as pd
import torch
import fsspec
import numpy as np
import xarray as xr
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        shots,
        elms=None,
        window_size: int = 1024,
        step_size: int = 512,
        data_path: str = "/data/elms",
    ):
        self.shots = shots
        self.elms = elms
        self.window_size = window_size
        self.step_size = step_size
        self.data_path = data_path

    def read_shot(self, shot: int):
        store = Path(self.data_path) / f"{shot}.parquet"
        dalpha = pd.read_parquet(store)
        dalpha = dalpha.reset_index()
        dalpha = dalpha.fillna(0)
        dalpha = dalpha.loc[dalpha.time > 0]
        return dalpha

    def __len__(self):
        return len(self.shots)

    def __getitem__(self, idx):
        shot = self.shots[idx]

        dalpha = self.read_shot(shot)
        dalpha = background_subtract(dalpha)

        dalpha.ip = dalpha.ip / 1e6

        windows = generate_windows(
            dalpha.dalpha.values, self.window_size, self.step_size
        )
        windows = torch.tensor(windows, dtype=torch.float)
        windows.unsqueeze_(1)

        ip_windows = generate_windows(
            dalpha.ip.values, self.window_size, self.step_size
        )
        ip_windows = torch.tensor(ip_windows, dtype=torch.float)
        ip_windows.unsqueeze_(1)

        windows = torch.cat([windows, ip_windows], dim=1)

        time_windows = generate_windows(
            dalpha.time.values, self.window_size, self.step_size
        )
        time_windows = torch.tensor(time_windows, dtype=torch.float)
        time_windows.unsqueeze_(1)

        if self.elms is not None:
            class_ = np.zeros_like(dalpha.time.values)
            class_ = xr.DataArray(class_, coords=dict(time=dalpha.time))

            # create labels
            elms = self.elms[idx]
            for item in elms:
                if item["valid"]:
                    # buffer = item["widths"] * 1e-5
                    buffer = 14 * 1e-5
                    class_.loc[
                        dict(time=slice(item["time"] - buffer, item["time"] + buffer))
                    ] = 1

            labels = generate_windows(class_.values, self.window_size, self.step_size)
            labels = torch.tensor(labels, dtype=torch.float)
            labels.unsqueeze_(1)

            return windows, time_windows, labels
        else:
            return windows, time_windows


def generate_windows(data, window_size, step_size):
    if len(data) % window_size != 0:
        data = np.pad(data, (0, window_size - len(data) % window_size))

    num_windows = (len(data) - window_size) // step_size + 1
    windows = [
        data[i * step_size : i * step_size + window_size] for i in range(num_windows)
    ]
    return np.stack(windows)


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
    values = dalpha.dalpha.values
    dt = dtime[1] - dtime[0]
    n = int(moving_av_length / dt)
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1 :] / n
    values[n - 1 :] -= ret
    dalpha.dalpha = values
    return dalpha
