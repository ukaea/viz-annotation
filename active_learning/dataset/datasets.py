"""
@authors: 
    Niraj Bhujel, SciML-STFC-UKRI (niraj.bhujel@stfc.ac.uk)
    Samuel Jackson, UKAEA (samuel.jackson@ukaea.uk)

"""
import os
import glob
import json
import fsspec
import random
from copy import copy
import numpy as np
import pandas as pd
import xarray as xr

import torch
from torch.utils.data import Dataset

from utils.misc import TensorDictStruct

def split_data(data, train_split=0.8, n_folds=1, seed=None):
    """
    Splits the common data and label files into training and validation sets.
    If n_folds > 1, performs manual cross-validation splits.
    
    :param data: List of files to split
    :param train_split: Fraction of data to use for training (default 0.8)
    :param n_folds: Number of folds for cross-validation (if >1)
    :param seed: Random seed for reproducibility
    :return: List of (train_files, val_files) tuples
    """

    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(data)
        
    if n_folds > 1:
        fold_size = len(data) // n_folds
        folds = []
        for i in range(n_folds):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < n_folds - 1 else len(data)
            val_files = data[val_start:val_end]
            train_files = data[:val_start] + data[val_end:]
            folds.append((train_files, val_files))
        return folds
    else:
        split_idx = int(len(data) * train_split)
        train_files, val_files = data[:split_idx], data[split_idx:]
        return train_files, val_files

def generate_windows(data, window_size=512, step_size=256):
    num_windows = (len(data) - window_size) // step_size + 1
    windows = [
        data[i * step_size : i * step_size + window_size] for i in range(num_windows)
    ]
    return np.array(windows)

def segment_time_series(time_series, segment_length=512):
    # Compute the number of full segments
    total_length = len(time_series)
    num_segments = int(np.ceil(total_length / segment_length))
    # num_segments = total_length//segment_length
    
    # Initialize an array filled with NaN for all segments
    segmented_ts = np.full((num_segments, segment_length), np.nan, dtype=np.float32)
    for i in range(num_segments):
        start_idx = i * segment_length
        end_idx = min(start_idx + segment_length, total_length)
        segment = time_series[start_idx:end_idx]
        segmented_ts[i, :len(segment)] = segment

    return segmented_ts

def get_remote_store(path: str, endpoint_url: str):
    fs = fsspec.filesystem(
        **dict(
            protocol="filecache",
            target_protocol="s3",
            cache_storage=".cache",
            target_options=dict(anon=True, endpoint_url=endpoint_url),
        )
    )
    return fs.get_mapper(path)

def background_subtract(dalpha, dtime, moving_av_length=0.001):
    values = dalpha.copy()
    
    dt = dtime[1]-dtime[0]
    n = int(moving_av_length/dt)
    ret = np.cumsum(values, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    values[n - 1:] -= ret
    
    return values

def map_elm_loc(label, dtime, start_buffer=0.00006, end_buffer=0.0002):
    """
    Assign elm label [0, 1] to detected location in dalpha
    """
    mapped_loc = np.zeros((len(dtime)), dtype=np.int32)
    for idx, row in label.iterrows():
        start_idx = np.argmin(np.abs(dtime - (float(row['Time'])-start_buffer)))
        end_idx = np.argmin(np.abs(dtime - (float(row['Time'])+end_buffer)))
        mapped_loc[start_idx:end_idx] = 1
    return mapped_loc

def map_elm_types(label, dtime, n_classes=3, elm_types=None):
    """
    Assigns type labels [0,1,2] based on the time interval of each ELM types.
    :param label: DataFrame with 'Start', 'End' and 'Type' columns
    :param dtime: Numpy array containing time values
    :return: mapped_label: Numpy array with label for each data points
    """
    # Map Type to numerical labels
    if elm_types is None:
        elm_types = ["Type I", "Type II", "Type III"]

    label_map = {elm_types[i]:i for i in range(n_classes)}
    label['mapped_label'] = label['Type'].map(label_map)
    
    # Assign labels to each data time point
    mapped_label = np.zeros((len(dtime)))
    for idx, row in label.iterrows():
        start_idx = np.argmin(np.abs(dtime - float(row['Start'])))
        end_idx = np.argmin(np.abs(dtime - float(row['End'])))
        mapped_label[start_idx:end_idx] = int(row['mapped_label'])
    
    return mapped_label

def map_labels(labels, dtime, n_classes=3, elm_types=None):
    
    split_idx = labels.index[labels.Time=='Start'][0]
    
    # Map elm class [None, Type I, Type II] to  [0, 1, 2]
    cls_labels = labels.iloc[split_idx+1:].reset_index(drop=True)
    cls_labels.columns = labels.iloc[split_idx] # replace header
    cls_labels = map_elm_types(cls_labels, dtime, n_classes, elm_types)
    
    # elm label [0, 1]
    elm_labels = labels.iloc[:split_idx].reset_index(drop=True)
    elm_labels = elm_labels[elm_labels.Valid==1]
    elm_labels = map_elm_loc(elm_labels, dtime)
    
    return cls_labels, elm_labels
        
        
def collate_fn(batch):
    keys = batch[0].keys()
    collated_batch = {}
    for k in keys:
        values = [sample[k] for sample in batch]
        if isinstance(values[0], torch.Tensor):
            collated_batch[k] = torch.cat(values)
        elif isinstance(values[0], str):
            collated_batch[k] = values
        elif isinstance(values[0], list):
            collated_batch[k] = [v for val in values for v in val]
            
    return TensorDictStruct(collated_batch)

    
class ELMDataset(Dataset):
    """
    A simple implementation of the dataloader class for time series data.

    Attributes:
        cfg (DictConfig): Hydra configuration object containing dataset parameters.
        label_files (list of str): List of file paths containing corresponding labels.
        mode (str): Specifies the dataset mode, must be one of ['train', 'test', 'val'].
    """

    def __init__(self, data_cfg, label_files, mode='train'):
        
        self.data_cfg = data_cfg
        self.label_files = label_files
        self.mode = mode
        self.n_classes = data_cfg.n_classes
        self.endpoint_url = 'https://s3.echo.stfc.ac.uk'
        self.data_dir = data_cfg.data_dir

    @property
    def training(self):
        return 1 if self.mode=='train' else 0
        
    def __len__(self):
        return len(self.label_files)

    def sample_uniform(self, dalpha, indices, window_size=256, n_samples=1, buffer=None):
        """
        Sample a signal around the indices location
        Args:
            dalpha (np.array): 1D signal of length N
            indices (list or np.array): List of indices corresponding to ELM type. 
            window_size (int): Length of the sampled signal
            num_samples (int): Number of signal to be sampled.
            buffer (int): Buffer length for overlapp/outside the indices, usually window_size//2
    
        Return:
            sampled_indices (np.ndarray): Array of shape (num_samples, len(indices))
            sampled_signal (np.ndarray): Sampled array of shape (num_samples, len(indices), sample_len)
        """
        
        if buffer is None:
            buffer = window_size//2

        if len(indices) < window_size + buffer:
            return None

        # Set valid indices
        valid_start = np.isin(indices-buffer, indices)
        valid_end = np.isin(indices+buffer, indices)
        indices = indices[valid_start & valid_end]
            
        # Random sample indices
        sampled_indices = np.random.choice(indices, size=(n_samples), replace=True if len(indices)<n_samples else False)
        
        # Select window from each indices
        offsets = np.arange(-window_size//2, window_size//2)[None, :]  # Shape: (1, sample_len)
        sampled_indices = sampled_indices[:, None]  + offsets  # Shape: (num_samples, sample_len)
        sampled_indices = np.clip(sampled_indices, 0, len(dalpha)-1)
        
        return sampled_indices
    
    def __getitem__(self, idx):
        shot_id = self.label_files[idx]
        
        if self.data_dir.startswith('s3://'):
            store = get_remote_store(f"{self.data_dir}/{shot_id}.zarr", self.endpoint_url)
            profiles = xr.open_zarr(store, group='dalpha')
            dalpha = profiles.dalpha_mid_plane_wide.copy()
            dalpha = dalpha.dropna(dim='time')
            dtime = dalpha.time.values
            dalpha = dalpha.values
        else:
            data_df = pd.read_csv(f"{self.data_dir}/{shot_id}.csv")
            dtime = data_df['time'].values
            dalpha = data_df['dalpha_mid_plane_wide'].values
        
        dalpha = background_subtract(dalpha, dtime)
                                        
        labels = pd.read_csv(f"{self.data_cfg.label_dir}/{shot_id}.csv")
        cls_labels, elm_labels = map_labels(labels, 
                                            dtime, 
                                            n_classes=self.data_cfg.n_classes,
                                            elm_types=self.data_cfg.class_types,
                                            )
        
        sampled_indices = []
        for l in np.unique(cls_labels):
            sampled_class_indices = self.sample_uniform(
                dalpha, 
                np.where(cls_labels==l)[0],
                window_size=self.data_cfg.context_len,
                n_samples=int(self.data_cfg.train_samples * self.data_cfg.sampling_factor[int(l)]) if self.training else self.data_cfg.val_samples
            )
            if sampled_class_indices is not None:
                sampled_indices.append(sampled_class_indices)
            
        sampled_indices = np.concatenate(sampled_indices) # (n_samples x class, context_len)
        
        dalpha = dalpha[sampled_indices]
        dtime = dtime[sampled_indices]
        cls_labels = cls_labels[sampled_indices][:, 0]
        elm_labels = elm_labels[sampled_indices]
        files = [shot_id] * len(sampled_indices)
        
        # print(f"{dalpha.shape=}, {class_label.shape=}, {elm_label.shape=}")
        
        dalpha = torch.tensor(dalpha, dtype=torch.float32).unsqueeze(1)
        dtime = torch.tensor(dtime, dtype=torch.float32).unsqueeze(1)
        cls_labels = torch.tensor(cls_labels, dtype=torch.long) # xent class label
        elm_labels = torch.tensor(elm_labels, dtype=torch.long) # binary class label

        return dict(dalpha=dalpha, dtime=dtime, cls_labels=cls_labels, elm_labels=elm_labels, files=files)
        
        