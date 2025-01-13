import numpy as np
import xarray as xr
from bokeh.plotting import figure, Figure
from bokeh.models import Span, Range1d
from dataclasses import dataclass
import pandas as pd

@dataclass
class ELMParams:
   threshold: float
   moving_av_length: float
   min_elm_duration: float
   max_elm_duration: float
   min_elm_seperation: float
   tmin: float
   tmax: float
   elm_interval: float

@dataclass
class ELMData:
   elm_times: xr.DataArray
   elm_heights: xr.DataArray
   elm_frequency: xr.DataArray
   elm_frequency_times: xr.DataArray
   too_short: xr.DataArray
   too_short_t: xr.DataArray
   too_long: xr.DataArray
   too_long_t: xr.DataArray
   freq_filtered: xr.DataArray
   freq_filtered_t: xr.DataArray

class ELMAnalysis():
  def __init__(self, dalpha_series: xr.DataArray, params: ELMParams):
    self.params = params
    self.dalpha_series = self.__preprocess_signal(dalpha_series)
    self.windows = self.__identify_bursts(analyse=False)
    self.elm_data = self.__analyze_bursts()

  def generate_elm_graph(self, width, height) -> Figure:
    series = self.dalpha_series
    fig = figure(title=series.name, x_axis_label="time", y_axis_label="d-alpha")
    fig.width = width
    fig.height = height
    fig.line(series.time.values, series.values, line_width=2)
    fig.scatter(self.elm_data.elm_times, self.elm_data.elm_heights, marker="x", size=15, color="red", line_width=2)
    fig.scatter(self.elm_data.too_short_t, self.elm_data.too_short, marker="x", size=15, color="blue", line_width=2)
    fig.scatter(self.elm_data.freq_filtered_t, self.elm_data.freq_filtered, marker="x", size=15, color="orange", line_width=2)
    fig.scatter(self.elm_data.too_long_t, self.elm_data.too_long, marker="x", size=15, color="green", line_width=2)
    span = Span(location=self.params.threshold, dimension='width', line_color='red', line_width=1.5, line_dash='dashed')
    fig.add_layout(span)
    return fig
  
  def generate_elm_frequency_graph(self, width, height, range=None) -> Figure:
    fig = figure(title="ELM Frequency Plot", x_axis_label="time", y_axis_label="ELM Frequency")
    fig.width = width
    fig.height = height
    if range == None:
      fig.x_range = Range1d(self.params.tmin, self.params.tmax)
    else:
      fig.x_range = range
    fig.scatter(self.elm_data.elm_times[1:], 1/(self.elm_data.elm_times[1:]-self.elm_data.elm_times[:-1]), marker="+", size=15, color="red", line_width=2)
    fig.line(self.elm_data.elm_frequency_times, self.elm_data.elm_frequency, color="red", line_width=2)
    return fig

  def get_elm_data(self) -> pd.DataFrame:
     df = pd.DataFrame(
        {
           "Time": self.elm_data.elm_times,
           "Height": self.elm_data.elm_heights,
        }
     )
     return df

  def __preprocess_signal(self, signal: xr.DataArray) -> xr.DataArray:
    time_period = self.params.tmax - self.params.tmin
    signal = signal.dropna(dim='time').sel(time=slice(self.params.tmin - time_period*0.05, self.params.tmax))
    signal = self.__background_subtract(signal)
    signal = signal.sel(time=slice(self.params.tmin, self.params.tmax))
    return signal

  def __background_subtract(self, signal) -> xr.DataArray:
    dtime = signal.time.values
    values = signal.values
    dt = dtime[1]-dtime[0]
    n = int(self.params.moving_av_length/dt)
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    values[n - 1:]-=ret
    signal.values = values
    return signal
  
  def __identify_bursts(self, analyse=True):
    # a handy function from Nick Walkdens time series tools. Thanks Nick!
    series = self.dalpha_series
    thresh = self.params.threshold
    inds_above_thresh = np.where(series>thresh)
    inds_below_thresh = np.where(series<=thresh)
    windows = []
    leng = len(inds_above_thresh[0])+0.0
    for i, ind in enumerate(inds_above_thresh[0]):
        try:
            #Extract highest possible index that occurs before a filement,
            ind_low = np.extract(inds_below_thresh < ind,inds_below_thresh)[-1]
            #Extract lowest possible index that occurs after a filemant
            ind_up = np.extract(inds_below_thresh > ind,inds_below_thresh)[0]
            if (ind_low,ind_up) not in windows:
                #Make sure that there is no double counting
                windows.append((ind_low,ind_up))
        except:
            pass
    if analyse:
        N_bursts = len(windows)
        burst_ratio = len(list(series))/N_bursts
        av_window = np.mean([y - x for x,y in windows])
        return windows,N_bursts,burst_ratio,av_window
    else:
        return windows
    
  def __analyze_bursts(self) -> ELMData:
    too_short_windows = []
    too_long_windows = []

    num_too_short=0
    num_too_long=0

    dtime = self.dalpha_series.time.values
    dalpha = self.dalpha_series.values

    # filter spikes too short to be elms
    for i in range(len(self.windows)):
      if (dtime[self.windows[i][1]]-dtime[self.windows[i][0]])<self.params.min_elm_duration:
        too_short_windows.append(self.windows[i])
        self.windows[i]=(0,0)
        num_too_short+=1
      if (dtime[self.windows[i][1]]-dtime[self.windows[i][0]])>self.params.max_elm_duration:
        too_long_windows.append(self.windows[i])
        self.windows[i]=(0,0)
        num_too_long+=1

    self.windows = [x for x in self.windows if x!=(0,0)]

    too_short_inds = [np.argmax(dalpha[too_short_windows[x][0]:too_short_windows[x][1]]) + too_short_windows[x][0] for x in range(len(too_short_windows))]
    too_long_inds = [np.argmax(dalpha[too_long_windows[x][0]:too_long_windows[x][1]]) + too_long_windows[x][0] for x in range(len(too_long_windows))]

    too_short = dalpha[too_short_inds]
    too_short_t = dtime[too_short_inds]

    too_long = dalpha[too_long_inds]
    too_long_t = dtime[too_long_inds]

    elm_inds = [np.argmax(dalpha[self.windows[x][0]:self.windows[x][1]]) + self.windows[x][0] for x in range(len(self.windows))]
    elm_heights = dalpha[elm_inds]
    elm_times = dtime[elm_inds]


    freq_filtered_inds = np.zeros(0,dtype=int)

    for i in range(len(elm_inds)-1):
      if elm_heights[i]!=0 and i not in freq_filtered_inds:
        close_inds = np.where(elm_times[i+1:]-elm_times[i]<self.params.min_elm_seperation)[0]+i+1
        freq_filtered_inds=np.concatenate((freq_filtered_inds,close_inds))

    freq_filtered_inds = np.unique(freq_filtered_inds)
    freq_filtered_t = elm_times[freq_filtered_inds]
    freq_filtered = elm_heights[freq_filtered_inds]

    elm_times[freq_filtered_inds]=0
    elm_heights[freq_filtered_inds]=0

    elm_times=elm_times[np.where(elm_heights!=0)]
    elm_heights=elm_heights[np.where(elm_heights!=0)]

    elm_frequency_times, elm_frequency = self.__calculate_elm_frequency(elm_times)

    return ELMData(
      elm_heights=elm_heights,
      elm_times=elm_times,
      freq_filtered=freq_filtered,
      freq_filtered_t=freq_filtered_t,
      too_long=too_long,
      too_long_t=too_long_t,
      too_short=too_short,
      too_short_t=too_short_t,
      elm_frequency=elm_frequency,
      elm_frequency_times=elm_frequency_times
    )
  
  def __calculate_elm_frequency(self, elm_times):
    elm_interval = self.params.elm_interval
    freq_t = np.arange(min(elm_times),max(elm_times),self.params.elm_interval)
    freq = np.zeros(len(freq_t))

    for index, time in enumerate(freq_t):
      num = len(np.where((elm_times>time-elm_interval/2)&(elm_times<time+elm_interval/2))[0])
      freq[index] = num/elm_interval

    return freq_t, freq

def calculate_elm_frequency(elm_times, elm_interval):
  freq_t = np.arange(min(elm_times),max(elm_times),elm_interval)
  freq = np.zeros(len(freq_t))

  for index, time in enumerate(freq_t):
    num = len(np.where((elm_times>time-elm_interval/2)&(elm_times<time+elm_interval/2))[0])
    freq[index] = num/elm_interval

  return freq_t, freq