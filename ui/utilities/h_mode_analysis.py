from dataclasses import dataclass
import xarray as xr
import pandas as pd
from pandas import Series
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import Range1d, BoxAnnotation
import streamlit as st

@dataclass
class HModeSignals:
    adg: xr.Dataset
    amc: xr.Dataset
    xim: xr.Dataset
    ane: xr.Dataset
    gradient_series: Series
    analyzed_current: xr.DataArray
    min_time: float
    max_time: float

@dataclass
class HModeParams:
    threshold: float
    min_pasma_current: float
    tmin: float
    tmax: float

class HModeAnalysis():
    def __init__(self, signals: HModeSignals, params: HModeParams):
        self.signals = signals
        self.params = params
        self.find_h_mode_window()

    def find_h_mode_window(self):
        threshold = self.params.threshold
        tmin = self.params.tmin
        tmax = self.params.tmax
        series: Series = self.signals.gradient_series
        series = series.loc[tmin:tmax]
        series /= 1e20
        series = series.rolling(window=10, center=True).mean()
        series.loc[series>threshold] = threshold
        max_ip = self.signals.analyzed_current.sel(time=slice(tmin, tmax)).time.values.max()
        series.loc[series.index > max_ip] = 0

        h_mode_times = series.index.values[series >= threshold]
        if len(h_mode_times) > 0:
            h_mode_start, h_mode_end = h_mode_times.min(), h_mode_times.max()
        else:
            h_mode_start, h_mode_end = None, None

        self.h_mode_start = h_mode_start
        self.h_mode_end = h_mode_end
        self.density_gradient_thresholded = series
        print("end")

    def generate_graphs(self, width, height):
        adg = self.signals.adg.density_gradient.sel(time=slice(self.params.tmin, self.params.tmax))
        density_gradient_figure = figure(title=adg.name, x_axis_label="time", y_axis_label="Density Gradient")
        density_gradient_figure.width = width
        density_gradient_figure.height = height
        density_gradient_figure.line(adg.time.values, adg.values, line_width=2)
        density_gradient_figure.add_layout(
            BoxAnnotation(left=self.h_mode_start, right=self.h_mode_end, fill_alpha=0.3, fill_color="lightblue")
        )

        thresholded = self.density_gradient_thresholded
        threshold_figure = figure(title=thresholded.name, x_axis_label="time", y_axis_label="Density Gradient")
        threshold_figure.x_range = Range1d(self.params.tmin, self.params.tmax)
        threshold_figure.width = width
        threshold_figure.height = height
        threshold_figure.line(thresholded.index, thresholded.values, line_width=2)
        threshold_figure.add_layout(
            BoxAnnotation(left=self.h_mode_start, right=self.h_mode_end, fill_alpha=0.3, fill_color="lightblue")
        )

        amc = self.signals.amc.plasma_current.sel(time=slice(self.params.tmin, self.params.tmax))
        plasma_current_figure = figure(title=amc.name, x_axis_label="time", y_axis_label="Density Gradient")
        plasma_current_figure.width = width
        plasma_current_figure.height = height
        plasma_current_figure.line(amc.time.values, amc.values, line_width=2)
        plasma_current_figure.add_layout(
            BoxAnnotation(left=self.h_mode_start, right=self.h_mode_end, fill_alpha=0.3, fill_color="lightblue")
        )

        xim = self.signals.xim.da_hm10_t.sel(time=slice(self.params.tmin, self.params.tmax))
        delta_alpha_figure = figure(title=xim.name, x_axis_label="time", y_axis_label="Density Gradient")
        delta_alpha_figure.width = width
        delta_alpha_figure.height = height
        delta_alpha_figure.line(xim.time.values, xim.values, line_width=2)
        delta_alpha_figure.add_layout(
            BoxAnnotation(left=self.h_mode_start, right=self.h_mode_end, fill_alpha=0.3, fill_color="lightblue")
        )

        return column(density_gradient_figure, threshold_figure, plasma_current_figure, delta_alpha_figure)
    
    def get_zone_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "start": [self.h_mode_start],
                "end": [self.h_mode_end]
            }
        )