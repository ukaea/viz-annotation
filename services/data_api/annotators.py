import time
import random
import torch
import numpy as np
import fsspec
import pandas as pd
import xarray as xr
from enum import Enum
from abc import ABC, abstractmethod
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from torch.utils.data import DataLoader

from utils import RedisModelCache
from elm_model.model import UNet1D
from elm_model.dataset import TimeSeriesDataset


def entropy(probs):
    """Compute the entropy of a probability distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    # Check for GPU availability
    if torch.backends.mps.is_available():  # Check for Apple MPS
        device = torch.device("mps")
    elif torch.cuda.is_available():  # Check for NVIDIA CUDA
        device = torch.device("cuda")
    else:  # Default to CPU
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


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


class ClassicELMDataAnnotator(DataAnnotator):
    def __init__(self):
        self.file_url = "s3://mast/level2/shots/{shot_id}.zarr"
        self.endpoint_url = "https://s3.echo.stfc.ac.uk"
        self.fs = fsspec.filesystem(
            **dict(
                protocol="filecache",
                target_protocol="s3",
                cache_storage=".cache",
                target_options=dict(anon=True, endpoint_url=self.endpoint_url),
            )
        )

    def train(self, shot_ids: list[int], annotations: dict[int, list[dict]]):
        # No training required for classic annotator
        pass

    def score(self, shot_ids: list[int]):
        pass

    def get_annotations(
        self,
        shot_id: int,
        prominence: float = 0.2,
        distance: int = 200,
        height: float = 0.1,
    ):
        df_alpha = pd.read_parquet(f"/data/elms/{shot_id}.parquet")
        dalpha = df_alpha.to_xarray()
        # store = self.get_remote_store(self.file_url.format(shot_id=shot_id))

        # dataset = xr.open_zarr(store, group="spectrometer_visible")
        # dalpha: xr.DataArray = dataset.filter_spectrometer_dalpha_voltage
        # dalpha = dalpha.isel(dalpha_channel=2)
        dalpha = dalpha.dropna(dim="time")

        signal = self.background_subtract(dalpha.copy(), 0.001)

        trend = uniform_filter1d(signal, 1000)
        dalpha_detrend = signal - trend

        peak_idx, params = find_peaks(
            dalpha_detrend,
            prominence=prominence,
            width=[1, 150],
            distance=distance,
            height=height,
        )

        peaks = pd.DataFrame(params)
        peaks["time"] = dalpha_detrend.time.values[peak_idx]
        peaks["height"] = dalpha.values[peak_idx]
        peaks["valid"] = True
        peaks = peaks.loc[peaks.time > 0]
        peaks = peaks[["time", "height", "valid"]].to_dict(orient="records")
        return peaks

    def get_remote_store(self, path: str):
        return self.fs.get_mapper(path)

    def background_subtract(
        self, signal: xr.DataArray, moving_av_length: float
    ) -> xr.DataArray:
        dtime = signal.time.values
        values = signal.values
        dt = dtime[1] - dtime[0]
        n = int(moving_av_length / dt)
        ret = np.cumsum(values, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1 :] / n
        values[n - 1 :] -= ret
        signal.values = values
        return signal


class UnetELMDataAnnotator(DataAnnotator):
    def __init__(self):
        self.network = UNet1D()
        self.network.load_state_dict(torch.load("elm_model/model.pth"))
        self.learning_rate = 0.003
        self.epochs = 5
        self.device = get_device()
        self.seed = 42
        set_random_seed(self.seed)

    def save_model(self):
        cache = RedisModelCache()
        cache.save_state("current_model", self.network.state_dict())

    def load_model(self):
        cache = RedisModelCache()
        if cache.exists("current_model"):
            print("Loading model from cache")
            state = cache.load_state("current_model")
            self.network.load_state_dict(state)

    def train(self, shot_ids: list[int], annotations):
        annotations = [item["elms"] for item in annotations]
        train_dataset = TimeSeriesDataset(shot_ids, annotations)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self._train(self.network, train_dataloader)

    @torch.no_grad
    def score(self, shot_ids: list[int]) -> list[float]:
        shot_ids = np.random.choice(shot_ids, 1000)
        print(f"Scoring {len(shot_ids)} samples")
        test_dataset = TimeSeriesDataset(shot_ids)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )

        entropy_scores = self._score(self.network, test_dataloader)
        entropy_scores = entropy_scores.tolist()
        entropy_scores = [float(score) for score in entropy_scores]
        scores = [
            {"shot_id": shot, "score": score}
            for shot, score in zip(shot_ids, entropy_scores)
        ]
        return scores

    @torch.no_grad
    def _score(self, network, dataloader):
        self.network.eval()

        scores = []
        for batch in dataloader:
            x, t = batch
            x = x.to(self.device)
            _, prob = network(x)
            score = entropy(prob)
            scores.append(score)

        scores = torch.stack(scores).cpu().numpy()
        return scores

    def _train(self, network, train_dataloader):
        optim = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)

        print("Beginning training...")
        network.train()

        loss_hist = defaultdict(list)
        for epoch in range(self.epochs):
            epoch_loss = defaultdict(int)

            time_start = time.time()
            for i, batch in enumerate(train_dataloader):
                x, t, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)

                loss_dict, probs = network(x, labels)

                loss = 0
                for k, v in loss_dict.items():
                    loss += v
                    epoch_loss[k] += v
                    loss_hist[k].append(v.detach().item())

                epoch_loss["total_loss"] += loss
                string = ", ".join(
                    [f"{k}:{v / (i + 1):.6f}" for k, v in epoch_loss.items()]
                )
                optim.zero_grad()

                loss.backward()
                optim.step()

            print(f"\n{epoch=}, etc={time.time() - time_start:.3f}secs, {string}")
            # break
        print("Done!!!")

    @torch.no_grad
    def get_annotations(self, shot_id: int, **kwargs):
        cache = RedisModelCache()
        if cache.exists("current_model"):
            print("Loading model from cache")
            state = cache.load_state("current_model")
            self.network.load_state_dict(state)

        dataset = TimeSeriesDataset([shot_id], window_size=1024, step_size=1024)
        batch, time = dataset[0]
        loss, probs = self.network(batch)

        df = pd.DataFrame(
            dict(probs=probs.flatten(), time=time.flatten(), values=batch.flatten())
        )

        peaks = self.segment_max_values(df["values"], df["time"], df["probs"] > 0.5)
        return peaks

    def segment_max_values(self, data, time, segmentation):
        # Ensure input is numpy array
        data = np.array(data)
        time = np.array(time)
        segmentation = np.array(segmentation)

        # Find indices where segmentation changes
        change_indices = np.where(np.diff(segmentation) != 0)[0] + 1
        segment_starts = np.concatenate(([0], change_indices))
        segment_ends = np.concatenate((change_indices, [len(segmentation)]))

        results = []

        # Iterate through each segment and find max value
        for start, end in zip(segment_starts, segment_ends):
            max_index = np.argmax(data[start:end])
            segment_max = data[start:end][max_index]
            time_max = time[start:end][max_index]

            segment_value = segmentation[start]  # 0 or 1
            if segment_value > 0:
                results.append(
                    {
                        "time": float(time_max),
                        "height": float(segment_max),
                        "valid": True,
                    }
                )

        return results


ANNOTATORS = {
    AnnotatorType.CLASSIC: ClassicELMDataAnnotator,
    AnnotatorType.UNET: UnetELMDataAnnotator,
}
