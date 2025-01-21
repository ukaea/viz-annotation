import streamlit as st
import xarray as xr

from utilities import elms_analysis
from utilities.custom_components import slider_with_input, elm_zone_controls

def generate_elm_tab(data_file_name: str):
    if 'zones' not in st.session_state:
        st.session_state['zones'] = []
    # Load the d-alpha data set from local storage
    dataset = xr.open_zarr(data_file_name, group='dalpha')
    signal: xr.DataArray = dataset.dalpha_mid_plane_wide.copy()
    
    config_col, graph_col = st.columns([2, 3])

    with config_col:
        threshold = slider_with_input("ELM Threshold", default=0.15)
        
        # These options may be tweaked less frequently so hide them to save space
        with st.expander(label="Advanced Configuration"):
            min_time = min(signal.time.values)
            max_time = max(signal.time.values)
            tmin, tmax = st.slider(
                label="Analysis Time Range", 
                min_value=min_time, 
                max_value=max_time, 
                value=(min_time, max_time),
                step=0.001,
                format="%0.3f"
                )
            duration_min, duration_max = st.slider(
                label="Elm Duration (ms)",
                min_value=0.01,
                max_value=100.0,
                value=(0.05, 50.0),
                step=0.01,
                format="%0.2f"
            )
            min_elm_seperation = slider_with_input("Minimum ELM Seperation (ms)", default=1.5, max=10.0)
            moving_av_length = slider_with_input("Moving Average Length", default=0.001, max=0.1)
            elm_interval = slider_with_input("ELM Interval", default=0.01, max=0.1)

        # Structure to feed into elm analysis algorithm
        params = elms_analysis.ELMParams(
            threshold = threshold,
            moving_av_length = moving_av_length,
            min_elm_duration = duration_min * 1e-3,
            max_elm_duration = duration_max * 1e-3,
            min_elm_seperation = 1.5 * 1e-3,
            tmin= tmin,
            tmax = tmax,
            elm_interval = elm_interval,
            zones = st.session_state["zones"]
        )
        elm_analysis = elms_analysis.ELMAnalysis(signal, params)

        # ELMs located by the algorithm
        with st.expander(label="ELM Location Data"):
            st.dataframe(elm_analysis.get_elm_data(), width=500)

        with st.expander(label="ELM Region Selection"):
            def add_zone():
                st.session_state['zones'].append((params.tmin, params.tmax, "Type I"))
            st.button(label="Add ELM Region", on_click=add_zone)
            zones = st.session_state["zones"]
            zone_id = 0
            for zone in zones:
                 elm_zone_controls(id=zone_id, min=params.tmin, max=params.tmax)
                 zone_id += 1

    with graph_col:
        width = 900
        height = 250
        elm_figure = elm_analysis.generate_elm_graph(width, height)
        st.write(elm_figure)
        frequency_figure = elm_analysis.generate_elm_frequency_graph(width, height)
        st.write(frequency_figure)