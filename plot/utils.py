import matplotlib.pyplot as plt
import numpy as np


def imshow(inps, titles, save_path=None):
    """Displays or saves a batch of images."""
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    fig, axs = plt.subplots(1, len(inps), figsize=(20, 3))

    if len(inps) == 1:
        axs = [axs]  # Ensure axs is iterable when there's only one image

    for i, img in enumerate(inps):
        img = img.numpy().transpose(
            (1, 2, 0)
        )  # Convert from tensor format (C, H, W) to (H, W, C)
        img = std * img + mean  # Denormalize
        img = np.clip(img, 0, 1)

        if titles[i] == "Input Image":
            axs[i].imshow(img)
        else:
            axs[i].imshow(img[:, :, 0], cmap="gray")

        axs[i].set_title(titles[i])
        axs[i].axis("off")  # Remove axis for a cleaner display

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Images saved at: {save_path}")
    else:
        plt.show()

    plt.close(fig)
