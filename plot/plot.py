import matplotlib.pyplot as plt
import torch
from .utils import imshow


def plot_loss(train_epoch_losses, test_epoch_losses, save_path=None):
    """Plots and optionally saves training and test loss curves."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(train_epoch_losses) + 1), train_epoch_losses, label="Train")
    ax.plot(range(1, len(test_epoch_losses) + 1), test_epoch_losses, "r", label="Test")
    ax.legend()
    ax.set_title("Epoch Losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved at: {save_path}")
    else:
        plt.show()

    plt.close(fig)  # Close to free memory


def plot_res(model, dataloader, save_path=None):
    """Plots predictions of a model on sample batch images."""
    model.to("cpu")
    model.eval()

    inputs, labels = next(iter(dataloader))  # Fetch one batch

    with torch.no_grad():  # No need to track gradients
        preds = model(inputs)

    # Select images to visualize
    datas = [inputs[0], labels[0], preds[0], inputs[1], labels[1], preds[1]]
    titles = [
        "Input Image",
        "True Output",
        "Network Output",
        "Input Image",
        "True Output",
        "Network Output",
    ]

    imshow(datas, titles, save_path)
