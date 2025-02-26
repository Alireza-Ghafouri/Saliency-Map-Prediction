import os
from .utils import download_dataset, unzip_dataset, data_preprocessing
from .dataset import SaliencyDataset
from torch.utils.data import DataLoader
from omegaconf import OmegaConf


def dataloader(DATASET_DIR, ZIP_PATH, DOWNLOAD_URL, batch_size):
    """Main function to ensure dataset is ready and return dataloaders."""

    # Ensure dataset is downloaded and extracted only if necessary
    download_dataset(DOWNLOAD_URL, ZIP_PATH)
    unzip_dataset(ZIP_PATH, DATASET_DIR)

    # Preprocess data
    (
        train_data_paths_train,
        train_data_paths_test,
        label_data_paths_train,
        label_data_paths_test,
    ) = data_preprocessing(DATASET_DIR)

    # Create datasets and dataloaders
    transformed_dataset_train = SaliencyDataset(
        train_data_paths_train, label_data_paths_train, True
    )
    transformed_dataset_test = SaliencyDataset(
        train_data_paths_test, label_data_paths_test, True
    )

    dataloader_train = DataLoader(
        transformed_dataset_train, batch_size=batch_size, shuffle=True, num_workers=2
    )
    dataloader_test = DataLoader(
        transformed_dataset_test, batch_size=batch_size, shuffle=True, num_workers=2
    )

    dataloaders = {"train": dataloader_train, "val": dataloader_test}
    dataset_sizes = {
        "train": len(transformed_dataset_train),
        "val": len(transformed_dataset_test),
    }

    return dataloaders, dataset_sizes
