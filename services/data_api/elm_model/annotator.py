from pathlib import Path
import torch
import time
import fsspec
import random
import numpy as np
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from collections import defaultdict
from utils import RedisModelCache
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryF1Score

from annotators import DataAnnotator
from elm_model.model import UNet1D
from elm_model.dataset import TimeSeriesDataset

DATA_PATH = Path("/data/elms")


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


class UnetELMDataAnnotator(DataAnnotator):
    def __init__(self, epochs: int = 5, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.network = UNet1D(in_channels=3)
        self.learning_rate = 0.003
        self.epochs = epochs
        self.device = get_device()
        self.seed = 42
        set_random_seed(self.seed)

    def save_model(self):
        cache = RedisModelCache()
        cache.save_state("current_model", self.network.state_dict())

    def load_model(self):
        # cache = RedisModelCache()
        # if cache.exists("current_model"):
        #     print("Loading model from cache")
        #     state = cache.load_state("current_model")
        #     self.network.load_state_dict(state)
        # else:
        # Default model initialization
        state = torch.load("elm_model/model.pth")
        self.network.load_state_dict(state)

    def train(self, shot_ids: list[int], annotations):
        annotations = [item["elms"] for item in annotations]
        train_dataset = TimeSeriesDataset(
            shot_ids, annotations, data_path=self.data_path
        )
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
    def evaluate(self, shot_ids: list[int], annotations):
        annotations = [item["elms"] for item in annotations]
        dataset = TimeSeriesDataset(shot_ids, annotations, data_path=self.data_path)
        dataloader = DataLoader(
            dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        return self._evaluate(dataloader)

    def _evaluate(self, dataloader):
        scorer = BinaryF1Score()
        scorer.to(self.device)
        self.network.eval()

        for batch in dataloader:
            x, t, labels = batch
            x = x.to(self.device)
            t = t.to(self.device)
            x = torch.cat([x, t], dim=1)
            labels = labels.to(self.device)
            _, probs = self.network(x)
            scorer(probs, labels)

        return {"f1": scorer.compute()}

    @torch.no_grad
    def score(self, shot_ids: list[int]) -> list[float]:
        shot_ids = np.random.choice(shot_ids, min(100, len(shot_ids)))
        print(f"Scoring {len(shot_ids)} samples")
        test_dataset = TimeSeriesDataset(shot_ids, data_path=self.data_path)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )

        entropy_scores = self._score(test_dataloader)
        entropy_scores = entropy_scores.tolist()
        entropy_scores = [float(score) for score in entropy_scores]
        scores = [
            {"shot_id": shot, "score": score}
            for shot, score in zip(shot_ids, entropy_scores)
        ]
        return scores

    @torch.no_grad
    def _score(self, dataloader):
        self.network.eval()

        scores = []
        for batch in dataloader:
            x, t = batch
            x = x.to(self.device)
            t = t.to(self.device)
            x = torch.cat([x, t], dim=1)
            _, prob = self.network(x)
            score = entropy(prob)
            scores.append(score)

        scores = torch.stack(scores).cpu().numpy()
        return scores

    def _train(self, network, train_dataloader):
        optim = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)

        print("Beginning training...")
        network.train()
        network.to(self.device)

        loss_hist = defaultdict(list)
        for epoch in range(self.epochs):
            epoch_loss = defaultdict(int)

            time_start = time.time()
            for i, batch in enumerate(train_dataloader):
                x, t, labels = batch
                t = t.to(self.device)
                x = x.to(self.device)
                x = torch.cat([x, t], dim=1)
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
        batch = torch.cat([batch, time], dim=1)
        _, probs = self.network(batch)

        df_data = dataset.read_shot(shot_id)
        values = df_data.dalpha.values
        n = len(values)
        probs = probs.flatten()[:n]
        time = time.flatten()[:n]

        print(probs.shape, time.shape, values.shape)
        df = pd.DataFrame(dict(probs=probs, time=time, values=values))
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

    @torch.no_grad
    def evaluate(self, shot_ids: list[int], annotations):
        # No evaluation required for classic annotator
        pass

    def score(self, shot_ids: list[int]):
        # No scoring available for classic annotator
        pass

    def get_annotations(
        self,
        shot_id: int,
        prominence: float = 0.2,
        distance: int = 200,
        height: float = 0.1,
    ):
        dalpha = pd.read_parquet(DATA_PATH / f"{shot_id}.parquet")
        dalpha = dalpha.reset_index()
        dalpha = dalpha.dropna()

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
        peaks["time"] = dalpha.time.values[peak_idx]
        peaks["height"] = dalpha.dalpha.values[peak_idx]
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
        values = signal.dalpha.values
        # dtime = signal.time.values
        # values = signal.values
        dt = dtime[1] - dtime[0]
        n = int(moving_av_length / dt)
        ret = np.cumsum(values, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = ret[n - 1 :] / n
        values[n - 1 :] -= ret
        signal = values
        return signal
