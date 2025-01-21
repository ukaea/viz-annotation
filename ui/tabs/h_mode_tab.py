import streamlit as st
import zarr
import s3fs
import fsspec
import numpy as np
import xarray as xr
from utilities.h_mode_analysis import HModeSignals, HModeParams, HModeAnalysis
from utilities.custom_components import slider_with_input

@st.cache_data
def get_signals(shot_id: str, min_pasma_current: float) -> HModeSignals:
    end_point_url = "https://s3.echo.stfc.ac.uk"
    url = f"s3://mast/level1/shots/{shot_id}.zarr"

    fs = fsspec.filesystem(
        **dict(
            protocol="simplecache",
            target_protocol="s3",
            target_options=dict(anon=True, endpoint_url=end_point_url)
        ))

    adg = xr.open_zarr(fs.get_mapper(url + "/adg"))
    amc = xr.open_zarr(fs.get_mapper(url + "/amc"))
    xim = xr.open_zarr(fs.get_mapper(url + "/xim"))
    ane = xr.open_zarr(fs.get_mapper(url + "/ane"))

    limit_signal = adg.density_gradient
    gradient_series = adg.density_gradient.to_pandas().copy()

    ip: xr.DataArray = amc.plasma_current
    ip = ip.sel(time=ip>min_pasma_current)

    return HModeSignals(
        adg=adg,
        amc=amc,
        ane=ane,
        xim=xim,
        gradient_series=gradient_series,
        analyzed_current=ip,
        min_time=float(min(limit_signal.time.values)),
        max_time=float(max(limit_signal.time.values))
    )

@st.fragment
def generate_h_mode_tab(shot_id: str):
    signals: HModeSignals = get_signals(shot_id, 200)
    
    config_col, graph_col = st.columns([2,3])

    with config_col:
        threshold = slider_with_input("H-Mode Threshold", default=1, max=10.0)

        with st.expander(label="Advanced Configuration"):
            tmin, tmax = st.slider(
                label="Analysis Time Range",
                min_value=signals.min_time,
                max_value=signals.max_time,
                value=(signals.min_time, signals.max_time),
                step=0.001,
                format="%0.3f"
            )
            min_current = slider_with_input("Maximum plasma current", default=100, max=200.0)

        params = HModeParams(
            threshold=threshold,
            tmin=tmin,
            tmax=tmax,
            min_pasma_current=min_current
        )
        h_mode_analysis = HModeAnalysis(signals, params)

        @st.fragment
        def download_data():
            zone_data = h_mode_analysis.get_zone_data()
            csv_data = zone_data.to_csv(index=False)
            st.download_button(label="Download Data", data=csv_data, file_name="HMode_Data.csv")

        download_data()

    with graph_col:
        width = 900
        height = 250
        h_mode_figure = h_mode_analysis.generate_graphs(width, height)
        st.bokeh_chart(h_mode_figure)