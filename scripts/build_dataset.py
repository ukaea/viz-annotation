from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import xarray as xr
import fsspec
from joblib import Parallel, delayed

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

def process_shot(shot_id: int):
    store = get_remote_store(
        f"s3://mast/level2/shots/{shot_id}.zarr",
        endpoint_url="https://s3.echo.stfc.ac.uk",
    )

    spec = xr.open_zarr(store, group="spectrometer_visible")
    dalpha = spec.filter_spectrometer_dalpha_voltage
    dalpha = dalpha.isel(dalpha_channel=2)
    dalpha = dalpha.dropna(dim="time")
    dalpha = dalpha.squeeze(drop=True)
    dalpha = dalpha.drop_vars("dalpha_channel")
    dalpha.name = "D$\\alpha$"

    density_gradient = spec["density_gradient"]
    density_gradient = density_gradient.interp_like(dalpha)

    ds = xr.open_zarr(store, group="summary")
    power_nbi = ds["power_nbi"]
    power_nbi = power_nbi.interp_like(dalpha)

    ip = ds["ip"]
    ip = ip.interp_like(dalpha)

    ts = xr.open_zarr(store, group="thomson_scattering")
    t_e_core = ts["t_e_core"]
    t_e_core = t_e_core.interp_like(dalpha)
    n_e_core = ts["n_e_core"]
    n_e_core = n_e_core.interp_like(dalpha)

    ds = xr.Dataset(
        dict(
            ip=ip,
            power_nbi=power_nbi,
            density_gradient=density_gradient,
            t_e_core=t_e_core,
            n_e_core=n_e_core,
            dalpha=dalpha,
        )
    )
    file_name = f"data/elms/{shot_id}.nc"
    ds.to_netcdf(file_name, group="summary", mode="a")
    ts.to_netcdf(file_name, group="thomson_scattering", mode="a")

    frame = ds.to_dataframe()
    frame.to_parquet(Path(file_name).with_suffix(".parquet"))
    return shot_id


def safe_process_shot(shot_id: int):
    try:
        return process_shot(shot_id)
    except Exception as e:
        print(f"Skipping {shot_id}", e)
        return shot_id


def main():
    parser = ArgumentParser()
    parser.add_argument("shot_file", type=str)
    args = parser.parse_args()
    df = pd.read_csv(args.shot_file, index_col=0)
    df = df.sort_index(ascending=False)
    shots = df.index.values

    Path('data/elms').mkdir(exist_ok=True, parents=True)

    tasks = [delayed(safe_process_shot)(shot) for shot in shots]
    pool = Parallel(n_jobs=16, return_as="generator")
    results = pool(tasks)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
