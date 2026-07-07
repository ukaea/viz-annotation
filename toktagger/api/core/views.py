from toktagger.api.core.annotators import compute_stft
import numpy as np
import xarray as xr
from toktagger.api.schemas.data import (
    Profile2DData,
    Data,
    MultiProfile2DData,
    MultiVariateTimeSeriesData,
    TimeSeriesData,
)
from toktagger.api.schemas.views import Profile2DViewParams, ViewParams, ViewType


class IdentityView:
    def __init__(self, params: ViewParams):
        self.params = params

    def __call__(self, data: Data):
        return data


class Profile2DView:
    def __init__(self, params: Profile2DViewParams):
        if not isinstance(params, Profile2DViewParams):
            raise RuntimeError(f"Invalid params type for Profile2DView: {type(params)}")

        self.params = params

    def __call__(
        self, data: MultiProfile2DData | MultiVariateTimeSeriesData
    ) -> Profile2DData:
        if self.params.signal_name not in data.values:
            raise RuntimeError("Signal name not found in data")

        profile_data = data.values[self.params.signal_name]

        if profile_data is None:
            raise RuntimeError(
                f"Profile data for {self.params.signal_name} does not exist."
            )

        return self.convert_profile_to_view(profile_data)

    def convert_profile_to_view(
        self, data: Profile2DData | TimeSeriesData
    ) -> Profile2DData:
        if isinstance(data, TimeSeriesData):
            dim_1, time, values = compute_stft(data)  # shape (dim_1, time)
        elif isinstance(data, Profile2DData):
            time = np.array(data.time)
            dim_1 = np.array(data.dim_1)
            values = np.array(data.values).T  # (time, dim_1) -> (dim_1, time)
        else:
            raise RuntimeError(f"Unsupported data type for Profile2DView: {type(data)}")

        # Clip to time/frequency range
        time_min = (
            self.params.time_min if self.params.time_min is not None else time.min()
        )
        time_max = (
            self.params.time_max if self.params.time_max is not None else time.max()
        )

        dim_1_min = (
            self.params.dim_1_min if self.params.dim_1_min is not None else dim_1.min()
        )
        dim_1_max = (
            self.params.dim_1_max if self.params.dim_1_max is not None else dim_1.max()
        )

        values_min = (
            self.params.values_min
            if self.params.values_min is not None
            else np.nanmin(values)
        )
        values_max = (
            self.params.values_max
            if self.params.values_max is not None
            else np.nanmax(values)
        )

        ds = xr.DataArray(
            values, coords=dict(dim_1=dim_1, time=time), dims=["dim_1", "time"]
        )
        ds = ds.sel(time=slice(time_min, time_max))
        ds = ds.sel(dim_1=slice(dim_1_min, dim_1_max))
        ds = ds.clip(values_min, values_max)

        return Profile2DData(
            time=ds.time.values.tolist(),
            dim_1=ds.dim_1.values.tolist(),
            values=ds.values.tolist(),
        )


DATA_VIEWS = {
    ViewType.IDENTITY: IdentityView,
    ViewType.PROFILE_2D: Profile2DView,
}
