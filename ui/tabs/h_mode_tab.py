import streamlit as st
from utilities.h_mode_anlaysis import HModeParams, HModeAnalysis, pull_data
from utilities.custom_components import slider_with_input

@st.fragment
def generate_h_mode_tab(shot_id: str):
    signals = pull_data(shot_id, 100)
    if signals == None:
        st.write("Invalid shot ID")
        return
    
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