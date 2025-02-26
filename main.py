from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from model import Net
from train import train_model
from plot import plot_loss, plot_res
from dataloader import dataloader
import os


def main():

    config = OmegaConf.load(os.path.join("config", "config.yaml"))

    DATASET_DIR = config.paths.data.dataset_dir
    ZIP_PATH = config.paths.data.zip_path
    DOWNLOAD_URL = config.paths.data.download_url
    weights_save_path = config.paths.weights_root
    results_save_path = config.paths.results_root

    os.makedirs(weights_save_path, exist_ok=True)
    os.makedirs(results_save_path, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    device = torch.device(config.training.device)
    batch_size = config.training.batch_size
    dataloaders, dataset_sizes = dataloader(
        DATASET_DIR, ZIP_PATH, DOWNLOAD_URL, batch_size
    )
    net = Net()
    net.to(device)
    criterion = nn.MSELoss()

    optimizer = optim.SGD(
        net.parameters(),
        lr=config.optimizer.lr,
        momentum=config.optimizer.momentum,
        weight_decay=config.optimizer.weight_decay,
        nesterov=config.optimizer.nesterov,
    )

    lr_sched = lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_scheduler.step_size,
        gamma=config.lr_scheduler.gamma,
    )

    model_ft, train_epoch_losses, test_epoch_losses = train_model(
        net,
        criterion,
        optimizer,
        lr_sched,
        dataloaders,
        dataset_sizes,
        device,
        weights_save_path,
        num_epochs=config.training.num_epochs,
    )

    plot_loss(
        train_epoch_losses,
        test_epoch_losses,
        save_path=os.path.join(results_save_path, "loss_plot.png"),
    )
    plot_res(
        model_ft,
        dataloaders["train"],
        save_path=os.path.join(results_save_path, "predictions.png"),
    )


if __name__ == "__main__":
    main()
