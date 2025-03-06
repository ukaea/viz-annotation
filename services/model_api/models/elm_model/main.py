import torch
import time
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from models.elm_model.model import Network, UNet1D
from models.elm_model.dataset import TimeSeriesDataset
from model import Model

def set_random_seed(seed):
    # setup seeds
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


class ELMModel(Model):
    def __init__(self):
        self.epochs = 30
        self.device = get_device()
        self.seed = 42
    
    def run(self, annotations):
        set_random_seed(self.seed)
    
        train_dataset = TimeSeriesDataset(annotations)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            batch_sampler=None,
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
    
        network = UNet1D()
        network = network.to(self.device)
        self.train(network, train_dataloader)

    def train(self, network, train_dataloader):
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
    





if __name__ == "__main__":
    main()
