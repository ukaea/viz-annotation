import streamlit as st
import xarray as xr

from utilities.elm_analysis import ELMAnalysis, ELMParams, pull_data
from utilities.custom_components import slider_with_input, elm_zone_controls

@st.fragment
def generate_elm_tab(shot_id: str):
    # Zones must be stored as session state to persist through updates
    if 'zones' not in st.session_state:
        st.session_state['zones'] = []

    # Pull data from server - this is cached based on shot ID
    dalpha_signal = pull_data(shot_id)
    if dalpha_signal is None:
        st.write("Invalid shot ID")
        return
    
    config_col, graph_col = st.columns([2, 3])

    with config_col:        
        threshold = slider_with_input("ELM Threshold", default=0.15)
        
        # These options may be tweaked less frequently so hide them to save space
        with st.expander(label="Advanced Configuration"):
            min_time = min(dalpha_signal.time.values)
            max_time = max(dalpha_signal.time.values)
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
            elm_interval = slider_with_input("ELM Interval", default=0.01, max=0.1)

        # Structure to feed into elm analysis algorithm
        params = ELMParams(
            threshold = threshold,
            min_elm_duration = duration_min * 1e-3,
            max_elm_duration = duration_max * 1e-3,
            min_elm_seperation = min_elm_seperation * 1e-3,
            tmin= tmin,
            tmax = tmax,
            elm_interval = elm_interval,
            zones = st.session_state["zones"]
        )
        elm_analysis = ELMAnalysis(dalpha_signal, params)

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

        # Download the data created by the analysis tool - fragmented to avoid update
        @st.fragment
        def download_data():
            elm_data = elm_analysis.get_elm_data()
            zone_data = elm_analysis.get_zone_data()
            csv_data = elm_data.to_csv(index=False)
            csv_data += "\n" + zone_data.to_csv(index=False)
            st.download_button(label="Download Data", data=csv_data, file_name=f"ELM_Data_{shot_id}.csv")

        download_data()

    with graph_col:
        width = 900
        height = 250
        elm_figure = elm_analysis.generate_elm_graph(width, height)
        st.write(elm_figure)
        frequency_figure = elm_analysis.generate_elm_frequency_graph(width, height)
        st.write(frequency_figure)