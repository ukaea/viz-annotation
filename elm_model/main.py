import torch
import time
import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, Subset
from elm_model.model import Network, UNet1D
from elm_model.dataset import TimeSeriesDataset


def set_random_seed(seed):
    # setup seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    hidden_dim = 128
    code_dim = 64
    n_clusters = 2
    model_type = "cnn"

    epochs = 30
    # device = "cuda:0"
    device = "mps:0"

    set_random_seed(42)

    # create dataset
    full_dataset = TimeSeriesDataset()
    train_dataset = Subset(TimeSeriesDataset(), np.arange(100, 200))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=None,
        batch_sampler=None,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    print("Testing dataloader")
    batch = next(iter(train_dataloader))
    print(f"{batch[0].shape=}, {batch[1].shape=}")

    if model_type == "cnn":
        network = Network(
            input_dim=1,
            hidden_dim=hidden_dim,
            code_dim=code_dim,
            output_dim=512,
        )
    elif model_type == "unet":
        network = UNet1D()
    else:
        raise RuntimeError(f"Unknown model type {model_type}")

    network = network.to(device)
    # print(network)

    optim = torch.optim.AdamW(network.parameters(), lr=0.003)

    network.train()
    pbar = tqdm(len(train_dataloader), position=0, leave=True)

    loss_hist = defaultdict(list)
    for epoch in range(epochs):
        epoch_loss = defaultdict(int)

        time_start = time.time()
        for i, batch in enumerate(train_dataloader):
            pbar.update(1)
            x, labels = batch
            x = x.to(device)
            labels = labels.to(device)

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
            pbar.set_postfix_str(string)

            optim.zero_grad()

            loss.backward()
            optim.step()

        print(f"\n{epoch=}, etc={time.time() - time_start:.3f}secs, {string}")
        # break

    pbar.close()
    print("Done!!!")

    fig, axes = plt.subplots(len(loss_hist), 1)
    for k, v in loss_hist.items():
        axes.plot(v, label=k)
        axes.legend()
    plt.savefig("loss.png")

    x, label = full_dataset[10]
    network.to("cpu")

    with torch.no_grad():
        probs = network(x)[1].cpu().numpy()

    fig, axes = plt.subplots(5, 2)
    for i, ax in enumerate(axes):
        i = i + 25
        ax[0].plot(x.squeeze()[i])
        ax[1].plot(label.squeeze()[i])
        ax[1].plot(probs.squeeze()[i])
        # ax[0].set_ylim(-0.1, 1.1)
        ax[1].set_ylim(-0.1, 1.1)

    plt.savefig("output.png")


if __name__ == "__main__":
    main()
