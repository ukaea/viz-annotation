import streamlit as st
import xarray as xr
from bokeh.plotting import figure
import os
from utilities import elms_analysis

st.set_page_config(layout="wide")

def slider_with_input(key: str, min=0.0, max=1.0, default=0.0):
    if f'{key}_slider' not in st.session_state:
        st.session_state[f'{key}_slider'] = default
    if f'{key}_input' not in st.session_state:
        st.session_state[f'{key}_input'] = default
    def update_slider():
        st.session_state[f'{key}_slider'] = st.session_state[f'{key}_input']
    def update_input():
        st.session_state[f'{key}_input'] = st.session_state[f'{key}_slider']
    slider_col, input_col = st.columns([3,1])
    with slider_col:
        slider_value = st.slider(label=key, label_visibility="visible", key=f'{key}_slider', step=0.001, min_value=min, max_value=max, on_change=update_input, format="%0.3f")
    with input_col:
        input_value = st.number_input(label=key, label_visibility="hidden", step=0.001, key=f'{key}_input', on_change=update_slider, format="%0.3f")
    if slider_value != input_value:
        update_slider()
    return input_value

width = 900
height = 250

st.title('MAST Data Viewer')

st.sidebar.title("Shot Selector")
st.sidebar.text_input("Shot ID: ", key="shot_id")

if st.session_state.shot_id:
    st.subheader(f'MAST Shot #{st.session_state.shot_id}')

    file_name=f'./data/{st.session_state.shot_id}.zarr'
    if os.path.exists(file_name):
        h_mode_tab, elms_tab = st.tabs(["H Mode Analysis", "ELMs Analysis"])
        with h_mode_tab:
            st.write("Nothing yet")
        with elms_tab:
            dataset = xr.open_zarr(file_name, group='dalpha')
            signal: xr.DataArray = dataset.dalpha_mid_plane_wide.copy()
            
            config_col, graph_col = st.columns([2, 3])

            with config_col:
                threshold = slider_with_input("ELM Threshold", default=0.15)
                
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
                    moving_av_length = slider_with_input("Moving Average Length", default=0.001, max=0.1)
                    elm_interval = slider_with_input("ELM Interval", default=0.01, max=0.1)

                    params = elms_analysis.ELMParams(
                    threshold = threshold,
                    moving_av_length = moving_av_length,
                    min_elm_duration = duration_min * 1e-3,
                    max_elm_duration = duration_max * 1e-3,
                    min_elm_seperation = 1.5 * 1e-3,
                    tmin= tmin,
                    tmax = tmax,
                    elm_interval = elm_interval,
                )
                    
                elm_analysis = elms_analysis.ELMAnalysis(signal, params)

                with st.expander(label="ELM Location Data"):
                    st.dataframe(elm_analysis.get_elm_data(), width=500)

            with graph_col:
                elm_figure = elm_analysis.generate_elm_graph(width, height)
                st.write(elm_figure)
                frequency_figure = elm_analysis.generate_elm_frequency_graph(width, height)
                st.write(frequency_figure)
            
    else:
        with st.expander("Invalid shot number"):
            st.write('The shot ID provided does not exist')
else:
    with st.expander("Input required"):
        st.write('Please provide a shot selection input')