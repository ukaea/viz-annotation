from abc import ABC, abstractmethod
from enum import Enum
import torch
import numpy as np
import fsspec
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from elm_model.model import UNet1D
from elm_model.dataset import TimeSeriesDataset


class AnnotatorType(str, Enum):  # noqa: F821
    CLASSIC = "classic"
    UNET = "unet"


class DataAnnotator(ABC):
    @abstractmethod
    def get_annotations(self, shot_id: int):
        pass


class ClassicELMDataAnnotator(DataAnnotator):
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


class UnetELMDataAnnotator(DataAnnotator):
    def __init__(self):
        self.network = UNet1D()
        self.network.load_state_dict(torch.load("elm_model/model.pth"))

    @torch.no_grad
    def get_annotations(self, shot_id: int):
        dataset = TimeSeriesDataset([shot_id], window_size=1024, step_size=1024)
        batch, time = dataset[0]
        loss, probs = self.network(batch)

        df = pd.DataFrame(
            dict(probs=probs.flatten(), time=time.flatten(), values=batch.flatten())
        )

        peaks = self.segment_max_values(df["values"], df["time"], df["probs"] > 0.5)
        return peaks

    def segment_max_values(self, data, time, segmentation):
        # Ensure input is numpy array
        data = np.array(data)
        time = np.array(time)
        segmentation = np.array(segmentation)

        # Find indices where segmentation changes
        change_indices = np.where(np.diff(segmentation) != 0)[0] + 1
        segment_starts = np.concatenate(([0], change_indices))
        segment_ends = np.concatenate((change_indices, [len(segmentation)]))

        results = []

        # Iterate through each segment and find max value
        for start, end in zip(segment_starts, segment_ends):
            max_index = np.argmax(data[start:end])
            segment_max = data[start:end][max_index]
            time_max = time[start:end][max_index]

            segment_value = segmentation[start]  # 0 or 1
            if segment_value > 0:
                results.append({"time": time_max, "height": segment_max, "valid": True})

        return results


ANNOTATORS = {
    AnnotatorType.CLASSIC: ClassicELMDataAnnotator(),
    AnnotatorType.UNET: UnetELMDataAnnotator(),
}
