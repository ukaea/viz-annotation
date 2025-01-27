import streamlit as st

def slider_with_input(key: str, min=0.0, max=1.0, default=0.0):
    """
        Component consisting of a slider with a linked number input

        Parameters
        ----------
        key : str
            A unique key for identifying this input group
        min : float, optional
            The minimum value of the slider (default is 0.0)
        max : float, optional
            The maximum value of the slider (default is 1.0)
        default : float, optional
            The default value set for this input group (default is 0.0)
    """
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

def elm_zone_controls(id: int, min: float, max: float):
    """
        Component allowing for the creation of ELM zones

        Parameters
        ----------
        id : int
            A unique id for identifying the zone
        min : float, optional
            The minimum value of the slider
        max : float, optional
            The maximum value of the slider
    """

    def update_input():
        lower = st.session_state[f'lower_{id}']
        upper = st.session_state[f'upper_{id}']
        type = st.session_state[f'zone_type_{id}']
        st.session_state['zones'][id] = (lower, upper, type)

    def remove():
        st.session_state['zones'].pop(id)

    lower_col, upper_col, type_col, rem_col = st.columns([1, 1, 1, 1])
    zone = st.session_state['zones'][id]
    with lower_col:
        st.number_input(label="Lower bound:", key=f'lower_{id}', value=zone[0], label_visibility="collapsed", on_change=update_input)
    with upper_col:
        st.number_input(label="Upper bound:", key=f'upper_{id}', value=zone[1], label_visibility="collapsed", on_change=update_input)
    with type_col:
        st.selectbox(key=f'zone_type_{id}', label="ELM type", label_visibility="collapsed", options=(
            "Type I",
            "Type II",
            "Type III"
        ), on_change=update_input)
    with rem_col:
        st.button("❌", key=f'rem_{id}', on_click=remove)