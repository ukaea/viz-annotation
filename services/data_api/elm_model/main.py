from sklearn.model_selection import train_test_split
import torch
import time
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from model import UNet1D
from dataset import TimeSeriesDataset
from torchmetrics.classification import BinaryF1Score


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


def entropy(probs):
    """Compute the entropy of a probability distribution."""
    return -torch.sum(probs * torch.log(probs + 1e-9), dim=1).mean()


class ELMModel:
    def __init__(self, all_shots):
        self.learning_rate = 0.003
        self.epochs = 30
        self.device = get_device()
        self.seed = 42
        set_random_seed(self.seed)
        # self.network = UNet1D()
        self.network = UNet1D()
        self.network = self.network.to(self.device)
        self.all_shots = all_shots[:100]  # for testing

    def train(self, annotations):
        train_dataset = TimeSeriesDataset(self.all_shots, annotations)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        self._train(self.network, train_dataloader)

    def query(self, n_samples: int = 1) -> int:
        test_shots = [
            shot for shot in self.all_shots if shot not in self.labelled_shots
        ]
        test_shots = np.array(test_shots)

        test_dataset = TimeSeriesDataset(test_shots)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
        entropy_scores = self.inference(self.network, test_dataloader)
        idx = np.argsort(entropy_scores)
        next_shots = test_shots[idx][-n_samples:]
        return next_shots

    @torch.no_grad
    def inference(self, network, dataloader):
        f1_score = BinaryF1Score()
        self.network.eval()

        scores = []
        probs = []
        for batch in dataloader:
            x, t, y = batch
            x = x.to(self.device)
            _, prob = network(x)
            f1_score.update(prob.cpu(), y)
            probs.append(prob.cpu().numpy())
            score = entropy(prob)
            scores.append(score)

        print(f"F1 Score: {f1_score.compute()}")
        scores = torch.stack(scores).cpu().numpy()
        return scores, probs

    def _train(self, network, train_dataloader):
        optim = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)

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


def main():
    elms = pd.read_parquet("elm_events.parquet")
    elms = elms.rename({"shot": "shot_id"}, axis=1)
    general = pd.read_parquet("general.parquet")
    general = general.rename({"shot": "shot_id"}, axis=1)

    all_shots = general.shot_id.values

    annotations = {
        shot: elms.loc[elms.shot_id == shot].to_dict("records") for shot in all_shots
    }

    train_shots, test_shots = train_test_split(all_shots, random_state=42)

    train_annotations = [annotations[shot] for shot in train_shots]
    test_annotations = [annotations[shot] for shot in test_shots]

    model = ELMModel(train_shots)
    model.train(train_annotations)

    s = time.time()
    test_dataset = TimeSeriesDataset(test_shots, test_annotations)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=None,
        batch_sampler=None,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )
    entropy_scores, probs = model.inference(model.network, test_dataloader)
    e = time.time()
    print(e - s)

    model.network.to("cpu")
    torch.save(model.network.state_dict(), "model.pth")
    import matplotlib.pyplot as plt

    x, t, y = next(iter(test_dataloader))
    print(x.shape)
    print(probs[0].shape)

    index = 45
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(x[index].squeeze())
    axes[1].plot(y[index].squeeze())
    axes[2].plot(probs[0][index].squeeze())
    plt.savefig("debug.png")


if __name__ == "__main__":
    main()
