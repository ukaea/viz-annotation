import random
import torch
import numpy as np
import fsspec
import pandas as pd
import xarray as xr
from enum import Enum
from abc import ABC, abstractmethod
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


class AnnotatorType(str, Enum):  # noqa: F821
    CLASSIC = "classic"
    UNET = "unet"


class DataAnnotator(ABC):
    @abstractmethod
    def get_annotations(self, shot_id: int, **kwargs):
        pass

    @abstractmethod
    def train(self, shot_ids: list[int]):
        pass

    @abstractmethod
    def score(self, shot_ids: list[int]):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass
