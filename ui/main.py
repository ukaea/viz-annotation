import streamlit as st
import xarray as xr
from bokeh.plotting import figure, column
from bokeh.models import ColumnDataSource, BoxAnnotation, CustomJS
st.set_page_config(layout="wide")


width = 1500
height = 250

shot = 30462
st.title(f'MAST Shot #{shot}')

file_name =f'./data/{shot}.zarr' 
dataset = xr.open_zarr(file_name, group='core_profiles')

signal = dataset.plasma_current
p1 = figure(title=signal.name, x_axis_label="x", y_axis_label="y", width=width, height=height)
p1.line(signal.time.values, signal.values, legend_label="Trend", line_width=2)

dataset = xr.open_zarr(file_name, group='dalpha')
signal = dataset.dalpha_mid_plane_wide
p2 = figure(title=signal.name, x_axis_label="x", y_axis_label="y", x_range=p1.x_range, width=width, height=height)
p2.line(signal.time.values, signal.values, legend_label="Trend", line_width=2)

box = BoxAnnotation(left=0.1, right=0.2, fill_alpha=0.3, fill_color='lightgreen')
p2.add_layout(box)

# Add a callback for updating the annotation dynamically
callback = CustomJS(args=dict(box=box), code="""
    var start = cb_obj.start;
    var end = cb_obj.end;
    box.left = start;
    box.right = end;
""")

# Add the range tool for user selection
p2.x_range.js_on_change('start', callback)
p2.x_range.js_on_change('end', callback)


dataset = xr.open_zarr(file_name, group='mirnov')
signal = dataset.omv_210
p3 = figure(title=signal.name, x_axis_label="x", y_axis_label="y", x_range=p1.x_range, width=width, height=height)
p3.line(signal.time.values, signal.values, legend_label="Trend", line_width=2)

p = column(p1, p2, p3)
st.bokeh_chart(p, use_container_width=False)