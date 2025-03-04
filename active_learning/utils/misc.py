import os
import io
import sys
import csv
import json
import math
import importlib
import random
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig
from PIL import Image

import torch
import torchvision.transforms as T

from collections import UserDict

class TensorDictStruct(UserDict):
    def __init__(self, dictionary={}):
        super().__init__(dictionary)
        # Convert nested dictionaries to TensorDictStruct recursively
        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.data[key] = TensorDictStruct(value)

    def __getattr__(self, key):
        try:
            return super().__getattribute__("data")[key] # Avoid recursion issue
        except KeyError:
            raise AttributeError(f"'TensorDictStruct' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        if key == "data":  # Prevent infinite recursion in __init__
            super().__setattr__(key, value)
        else:
            self.data[key] = value  # Attribute assignment

    def __delattr__(self, key):
        try:
            del self.data[key]  # Attribute deletion
        except KeyError:
            raise AttributeError(f"'TensorDictStruct' object has no attribute '{key}'")
            
    def to(self, device):
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                self.data[k] = v.to(device)
                
        return self

    def to_numpy(self, ):
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor):
                self.data[k] = v.cpu().numpy()
        return self

    def shape(self, ):
        shapes = {}
        for k, v in self.data.items():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                shapes[k] = v.shape
            elif isinstance(v, list):
                shapes[k] = f"List[{type(v[0])}]"
                
        return shapes
        
def set_random_seed(seed):
    #setup seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
def model_parameters(model, verbose=0):
    if verbose>0:
        print('{:<30} {:<10} {:}'.format('Parame Name', 'Total Param', 'Param Shape'))
    total_params=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            if verbose>0:
                print('{:<30} {:<10} {:}'.format(name, param.numel(), tuple(param.shape)))
            total_params+=param.numel()
    print('Total Trainable Parameters :{:<10}'.format(total_params))
    return total_params


def contains_nan(item):
    """
    Check if a list contains any NaN values, supporting nested lists, numpy arrays, and numbers.
    Parameters:
    item (list or array): List of numbers, arrays, or file names.

    Returns:
    bool: True if any NaN is found, False otherwise.
    """
    # Check if item is a list or array; if not, return False (no NaN)
    if isinstance(item, (list, np.ndarray)):
        for element in item:
            if isinstance(element, (float, int)) and math.isnan(element):
                return True
            elif isinstance(element, np.ndarray) and np.isnan(element).any():
                return True
            elif isinstance(element, list) and contains_nan(element):  # Recursive check for nested lists
                return True
    return False

def get_best_val(val, mode='min'):

    if mode=='max':
        return np.argmax(val), np.max(val)
    elif mode=='min':
        return np.argmin(val), np.min(val)
    else:
        raise Excpetion(f"{mode} is not implemented")
            
def get_files_in_dir(
    root: str, 
    file_end: tuple | str,
) -> list:
    """
    Returns an array of all the files in the root dir with the given file
    ending.

    Inputs:
        root (str): root directory
        file_end(str) | tuple[str, ...]: file ending, i.e. 'h5' or tuple
            of file endings

    Returns:
        ndarray of the various files in alphanumeric order
    """
    return sorted([
            f
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and f.endswith(file_end)
        ])
    
    
def create_new_dir(new_dir, clean=False):
    if clean:
        shutil.rmtree(new_dir, ignore_errors=True)
    #if not os.path.exists(new_dir):
    os.makedirs(new_dir, exist_ok=True)
    return new_dir
    
def save_to_csv(file_path, row_data, header, overwrite=False):
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    if overwrite:
        shutil.rmtree(file_path, ignore_errors=True)
        
    # Write the row to the CSV file
    with open(file_path, "a", newline="\n") as csv_file:
        writer = csv.writer(csv_file)

        # Write header if the file is empty
        if not file_exists:
            # header = ["Column1", "Column2", "Column3"]
            # # Replace with your actual column names
            writer.writerow(header)

        # Write the row data
        writer.writerow(row_data)

