import streamlit as st
import xarray as xr
import os
from tabs import elm_tab as elm
from tabs import h_mode_tab as h_mode
st.set_page_config(layout="wide")

st.title('MAST Data Viewer')

def clear_state():
    if 'zones' in st.session_state:
        del st.session_state['zones']

st.sidebar.title("Shot Selector")
if 'shot_id' not in st.session_state:
    st.session_state.shot_id = 30421
# st.sidebar.text_input("Shot ID: ", key="shot_id", on_change=clear_state)

# Improvements needed for handling initial state with no input - maybe just improve look
if st.session_state.shot_id:
    st.subheader(f'MAST Shot #{st.session_state.shot_id}')

    # Some input validation is done, but this needs to be worked on to make it nicer looking
    file_name=f'./data/{st.session_state.shot_id}.zarr'
    shot_id = elm.generate_elm_tab(st.session_state.shot_id) 
else:
    with st.expander("Input required"):
        st.write('Please provide a shot selection input')