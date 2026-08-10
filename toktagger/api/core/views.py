import numpy as np
import xarray as xr

from toktagger.api.core.annotators import compute_stft
from toktagger.api.schemas.data import (
    CompositeData,
    Data,
    MultiVariateTimeSeriesData,
    SpectrogramData,
    TimeSeriesData,
)
from toktagger.api.schemas.views import SpectrogramViewParams, ViewParams, ViewType


class IdentityView:
    def __init__(self, params: ViewParams):
        self.params = params

    def __call__(self, data: Data):
        return data


class SpectrogramView:
    def __init__(self, params: SpectrogramViewParams):
        self.params = params

    def __call__(self, data: Data) -> Data:
        if isinstance(data, MultiVariateTimeSeriesData):
            response = {}
            for key, value in data.values.items():
                response[key] = self.convert_timeseries_to_spectrogram(value)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        return CompositeData(values=response)

    def convert_timeseries_to_spectrogram(
        self, data: TimeSeriesData
    ) -> SpectrogramData:
        time = np.array(data.time)
        values = np.array(data.values)

        # Compute the Short-Time Fourier Transform (STFT)
        freq, ts, values = compute_stft(data)
        freq /= 1000
        time = ts + time[0]

        # Clip to time/frequency range
        time_min = (
            self.params.time_min if self.params.time_min is not None else time.min()
        )
        time_max = (
            self.params.time_max if self.params.time_max is not None else time.max()
        )

        frequency_min = (
            self.params.frequency_min
            if self.params.frequency_min is not None
            else freq.min()
        )
        frequency_max = (
            self.params.frequency_max
            if self.params.frequency_max is not None
            else freq.max()
        )

        amplitude_min = (
            self.params.amplitude_min
            if self.params.amplitude_min is not None
            else values.min()
        )
        amplitude_max = (
            self.params.amplitude_max
            if self.params.amplitude_max is not None
            else values.max()
        )

        ds = xr.DataArray(values, coords={"frequency": freq, "time": time})
        ds = ds.sel(time=slice(time_min, time_max))
        ds = ds.sel(frequency=slice(frequency_min, frequency_max))
        ds = ds.clip(amplitude_min, amplitude_max)

        return SpectrogramData(
            time=ds.time.values.tolist(),
            frequency=ds.frequency.values.tolist(),
            amplitude=ds.values.tolist(),
        )


DATA_VIEWS = {
    ViewType.IDENTITY: IdentityView,
    ViewType.SPECTROGRAM: SpectrogramView,
}
