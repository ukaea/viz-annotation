import multiprocessing
import torch
import time
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from models.elm_model.model import Network, UNet1D
from models.elm_model.dataset import TimeSeriesDataset
from model import Model

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

class ELMModel(Model):
    def __init__(self, all_shots):
        self.epochs = 10
        self.device = get_device()
        self.seed = 42
        set_random_seed(self.seed)
        self.network = UNet1D()
        self.network = self.network.to(self.device)
        self.all_shots = all_shots[:100] # for testing

    def train(self, annotations):
        elms = [item['elms'] for item in annotations]
        self.labelled_shots = [shot['shot_id'] for shot in annotations]
        train_dataset = TimeSeriesDataset(self.labelled_shots, elms)
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
        test_shots = [shot for shot in self.all_shots if shot not in self.labelled_shots]
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
        self.network.eval()
        scores = []
        for batch in dataloader:
            
            x = batch
            x = x.to(self.device)
            _, probs = network(x)
            score = entropy(probs)
            scores.append(score)

        scores = torch.stack(scores).cpu().numpy()
        return scores


    def _train(self, network, train_dataloader):
        optim = torch.optim.AdamW(network.parameters(), lr=0.003)
    
        network.train()
    
        loss_hist = defaultdict(list)
        for epoch in range(self.epochs):
            epoch_loss = defaultdict(int)
    
            time_start = time.time()
            for i, batch in enumerate(train_dataloader):
                x, labels = batch
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
