"""
Created on Sat June 10 15:40:55 2024
@author: Niraj Bhujel (niraj.bhujel@stf.ac.uk)
"""

import os
import numpy as np
import torch


class ModelCheckpoint:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self,
        model,
        ckpt_dir,
        ckpt_name=None,
        monitor=None,
        mode="min",
        trace_func=print,
        verbose=True,
        debug=False,
        logger=None,
    ):
        self.model = model
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name or f"best_{monitor}"
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.best_val = None
        self.metric_val_min = np.inf
        self.delta = 1e-9
        self.trace_func = trace_func
        self.debug = debug
        self.best_model = False
        self.count = 0

    def __call__(self, metric_val, epoch=None):
        self.count += 1
        current_val = metric_val
        if self.mode != "min":
            current_val *= -1

        # save on first call
        if self.best_val is None:
            self.trace_func("Modelcheckpoint: {} improved from {:.6f} to {:.6f}.".format(self.monitor, self.metric_val_min, metric_val))
            self.best_val = current_val
            self.metric_val_min = metric_val
            self.save_checkpoint()
            self.best_model = True

        elif current_val < (self.best_val + self.delta):
            self.trace_func("Modelcheckpoint: {} improved from {:.6f} to {:.6f}.".format(self.monitor, self.metric_val_min, metric_val))
            self.best_val = current_val
            self.metric_val_min = metric_val
            self.save_checkpoint()
            self.best_model = True
        else:
            self.best_model = False
            self.trace_func("Modelcheckpoint: {} didn't improved from {:.6f}.".format(self.monitor, self.metric_val_min))

    def save_checkpoint(
        self,
    ):
        """Saves model when metric improved."""

        if not self.debug:
            save_path = os.path.join(self.ckpt_dir, self.ckpt_name + ".pth")
            self.trace_func("Saving model checkpoint ... ")
            try:
                torch.save(self.model.module.state_dict(), save_path)
            except:
                torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self,):
        load_path = os.path.join(self.ckpt_dir, self.ckpt_name + ".pth")
        self.trace_func(f"Loading model checkpoint from {load_path}")
        ckpt = torch.load(load_path)
        if self.ckpt_name=='model_states':
            ckpt = ckpt['model_state']
        self.model.load_state_dict(ckpt, strict=True)
