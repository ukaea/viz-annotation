"""
Created on Wed Aug  5 21:20:35 2020

@author: Niraj Bhujel (niraj.bhujel@stfc.ac.uk) SciML, SCD-STFC
"""
import torch
import math
import numpy as np
from functools import partial

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def frange_cycle_linear(start, stop, epochs, n_cycle=4, ratio=0.5):
    L = np.ones(epochs)
    period = epochs / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):

        v, i = start, 0
        while v <= stop and (int(i + c * period) < epochs):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def cosine_scheduler(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


class CosineScheduler(_LRScheduler):
    """A simple wrapper class for cosine annealing.
    In case of updating learning rate per epoch, set steps_per_epoch to 1.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        target_lr,
        epochs,
        steps_per_epoch=1,
        warmup_epochs=0,
        last_epoch=-1,
        verbose=False,
        **kwargs
    ):
        self.optimizer = optimizer

        self.lrs_groups = []
        for group in self.optimizer.param_groups:
            lrs = cosine_scheduler(
                group["lr"], target_lr, epochs, steps_per_epoch, warmup_epochs
            )
            self.lrs_groups.append(lrs)

        super(CosineScheduler, self).__init__(optimizer, last_epoch, verbose, **kwargs)

    def get_lr(self):
        return [group[self.last_epoch] for group in self.lrs_groups]


class ReduceLROnPlateauWithWarmup(ReduceLROnPlateau):
    """ReduceLROnPlateau but with a linear warm-up period.

    Args:
        optimizer (:obj:`torch.optim.Optimizer`): an optimizer for the given model.
        init_lr (float): LR at beginning of warm-up
        max_lr (float): LR at end of warm-up
        warmup_epochs (int): Number of epochs for warm-up
        batches_per_epoch (int, optional): Number of batches per epoch if we want a warm-up per batch
        **kwargs: Arguments for ReduceLROnPlateau
    """

    def __init__(self, optimizer, init_lr, max_lr, warmup_epochs, **kwargs):

        self.warmup_epochs = warmup_epochs
        self.init_lr = init_lr

        self.max_lr = max_lr
        self.optimizer = optimizer

        self.base_lr = max_lr if warmup_epochs <= 0 else init_lr

        self._set_lr(self.base_lr)

        super(ReduceLROnPlateauWithWarmup, self).__init__(optimizer, **kwargs)

    def step(self, metrics, epoch=None):
        """Scheduler step at end of epoch.

        This function will pass the arguments to ReduceLROnPlateau if the warmup is done, and call
        `self.batch_step` if the warm-up is per epoch, to update the LR.

        Args:
            metrics (float): Current loss

        """
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch if epoch != 0 else 1

        if self.last_epoch <= self.warmup_epochs:
            progress = self.last_epoch / self.warmup_epochs
            new_lr = progress * self.max_lr + (1 - progress) * self.init_lr
            self._set_lr(new_lr)
        else:
            super(ReduceLROnPlateauWithWarmup, self).step(metrics, epoch=None)

    def _set_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr


class LinearScheduler:
    def __init__(self, start_value, target_value=None, epochs=None):
        self.start_value = start_value
        self.target_value = target_value
        assert (
            start_value != target_value
        ), "start_value and target_value should be different"
        self.mode = min if target_value > start_value else max
        self.per_step = (target_value - start_value) / epochs

    def step(self, step_num):
        return self.mode(self.start_value + step_num * self.per_step, self.target_value)


class CyclicScheduler:
    def __init__(
        self,
        start_value,
        target_value=None,
        step_size_up=2000,
        step_size_down=None,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
    ):

        self.start_value = start_value
        self.target_value = target_value

        assert (
            start_value != target_value
        ), "start_value and target_value should be different"

        step_size_up = float(step_size_up)

        step_size_down = (
            float(step_size_down) if step_size_down is not None else step_size_up
        )

        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        self.mode = mode

        if scale_fn == None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

    def step(self, last_epoch):
        cycle = math.floor(1 + last_epoch / self.total_size)
        x = 1.0 + last_epoch / self.total_size - cycle

        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        base_height = (self.target_value - self.start_value) * scale_factor

        if self.scale_mode == "cycle":
            lr = self.start_value + base_height * self.scale_fn(cycle)
        else:
            lr = self.start_value + base_height * self.scale_fn(last_epoch)

        return lr


class CyclicLinearScheduler:
    """
    https://github.com/haofuml/cyclical_annealing
    """

    def __init__(
        self,
        start_value,
        target_value,
        epochs=100,
        n_cycle=5,
        step_ratio=0.5,
    ):

        self.start_value = float(start_value)
        self.target_value = float(target_value)

        assert (
            start_value != target_value
        ), "start_value and target_value should be different"

        self.lrs = frange_cycle_linear(
            start_value, target_value, epochs, n_cycle, step_ratio
        )

    def step(self, epoch):

        return self.lrs[epoch]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    epochs = 100
    num_batches = 100

    lr = 0.0003 * 256 * 2 / 64
    target_ratio = (1e-3, 10)
    max_lr = target_ratio[1] * lr
    base_lr = target_ratio[0] * lr

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # scheduler = LinearScheduler(0, 1, epochs=50)
    # scheduler = CyclicScheduler(start_value=5, target_value=0, step_size_up=5, mode='triangular')
    # scheduler = CyclicLinearScheduler(start_value=0, target_value=1)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=20, mode='triangular2', cycle_momentum=False)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=1, pct_start=10/epochs, div_factor=10)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
    # scheduler = ReduceLROnPlateauWithWarmup(optimizer, lr*1e-3, lr, warmup_epochs=20, mode='min', factor=0.9, patience=5, verbose=True)

    lrs = cosine_scheduler(
        0.0005, 1e-6, 1000, 1, warmup_epochs=10, start_warmup_value=0
    )

    # lrs = []
    # for e in range(epochs):
    #     # for b in range(num_batches):
    #     # optimizer.step()
    #     # scheduler.step()
    #     # lrs.append(optimizer.param_groups[0]['lr'])
    #     lrs.append(scheduler.step(e))
    #     # break

    plt.plot(lrs)
    # plt.plot(moms)
    # plt.show()
    # plt.plot(moms)
    plt.show()
    print(lrs)
    print(np.max(lrs), np.min(lrs))
