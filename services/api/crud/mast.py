import fsspec
import xarray as xr
import pandas as pd
from scipy.signal import savgol_filter

class S3DataReader:
    def __init__(self, data_path):
        self.data_path = data_path
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

    def get_remote_store(self, path: str):
        return self.fs.get_mapper(path)

    def get_data(self, shot_id: int):
        df_alpha = pd.read_parquet(self.data_path / f"{shot_id}.parquet")
        df_alpha.fillna(0, inplace=True)
        df_alpha = df_alpha.reset_index()
        df_alpha["time"] = pd.to_timedelta(df_alpha.time, "s")
        df_alpha = df_alpha.set_index("time")
        df_alpha = df_alpha.resample("0.1ms").max()
        df_alpha = df_alpha.reset_index()
        df_alpha.density_gradient = savgol_filter(df_alpha.density_gradient, 100, 2)
        df_alpha.rename(columns={"dalpha": "value"}, inplace=True)
        data = df_alpha.to_dict(orient="records")
        return data
    
    def get_disruption_data(self, shot_id: int):
        store = self.get_remote_store(self.file_url.format(shot_id=shot_id))

        magnetics = xr.open_zarr(store, group="magnetics")
        ip: xr.DataArray = magnetics['ip']
        ip = ip.dropna(dim="time")

        df_ip = ip.to_dataframe().reset_index()
        df_ip.fillna(0, inplace=True)
        df_ip.rename(columns={"ip": "value"}, inplace=True)
        data = df_ip.to_dict(orient="records")
        return data