"""
Created on Sat Aug 22 15:37:55 2020

@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk) SciML, SCD-STFC

Imported from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
"""

import numpy as np


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        mode='min',
        patience=7,
        warmup_steps=0,
        delta=0,
        metric_name="val/loss",
        last_epoch=0,
        trace_func=print,
    ):
        self.mode = mode
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.counter = 0
        self.best_score = None
        self.best_metric_val = np.inf
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        self.metric_name = metric_name
        self.last_epoch = last_epoch

    def __call__(self, metric_val, epoch=None):
        
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        if self.mode=='min':
            score = -metric_val
        else:
            score = metric_val

        if self.best_score is None:
            self.trace_func(f"Early Stopping: {self.metric_name} improved from {self.best_metric_val:.6f} to {metric_val:.6f}.")
            self.best_metric_val = metric_val
            self.best_score = score

        elif score > (self.best_score + self.delta):
            self.trace_func(f"Early Stopping: {self.metric_name} improved from {self.best_metric_val:.6f} to {metric_val:.6f}.")
            self.best_score = score
            self.best_metric_val = metric_val
            self.counter = 0

        else:
            if self.last_epoch<self.warmup_steps:
                self.trace_func(f"Early Stopping: {self.metric_name} didn't improve from {self.best_metric_val:.6f}. Counter will start after {self.warmup_steps} warmup steps") 
            else:
                if epoch is None:
                    self.counter += 1 
                else:
                    epoch - self.last_epoch
                    
                self.trace_func(f"Early Stopping: {self.metric_name} didn't improve from {self.best_metric_val:.6f}. Early stopping counter: {self.counter}/{self.patience}")
                
                if (self.counter >= self.patience):
                    self.early_stop = True
