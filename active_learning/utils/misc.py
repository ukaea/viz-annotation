import os
import csv
import json
import math
import random
import numpy as np
from omegaconf import DictConfig, ListConfig

import torch

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

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]
    
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

# use computation graph to find all contributing tensors
def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
            print(f.shape)
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

def non_contributing_params(model, outputs=[]):
    
    contributing_parameters = set.union(*[set(get_contributing_params(out)) for out in outputs])
    all_parameters = set(model.parameters())
    non_contributing_parameters = all_parameters - contributing_parameters

    if len(non_contributing_parameters)>0:

        print(f"{len(non_contributing_parameters)} non-contributing parameters:")
        
        # Map parameters back to their names in the model
        for name, param in model.named_parameters():
            if param in non_contributing_parameters:
                print(f"{name}, Shape: {tuple(param.shape)}")
         
        # print([tuple(p.shape) for p in non_contributing_parameters])
        
    total_non_contributing = sum([np.product(p.shape) for p  in non_contributing_parameters])

    return total_non_contributing


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
    file_end: str,
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
    return [
            f
            for f in os.listdir(root)
            if os.path.isfile(os.path.join(root, f))
            and f.endswith(file_end)
        ]
    
    
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


def dictconf_to_dict(config, prefix=None, cfg_dict={}):
    """
    Recursively get fields in config and convert to primitive dictionary.
    """
    for key, value in config.items():
        k = f"{prefix}.{key}" if prefix else key
        # print(k, value, type(value))
        if isinstance(value, DictConfig):
            cfg_dict[k] = dictconf_to_dict(value, prefix=k, cfg_dict=cfg_dict)
        elif isinstance(value, ListConfig):
            cfg_dict[k] = ",".join([str(v) for v in value])
        else:
            cfg_dict[k] = value

    # remove field with value '{...}'
    cfg_dict = {k: v for k, v in cfg_dict.items() if not isinstance(v, dict)}

    return cfg_dict
