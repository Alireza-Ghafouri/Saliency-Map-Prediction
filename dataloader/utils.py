import requests
import zipfile
import os
import glob
import random
from tqdm import tqdm


def download_dataset(url, save_path):
    """Download dataset from a given URL if it does not exist locally, with a progress bar."""
    if os.path.exists(save_path):
        print(f"✅ Dataset already downloaded: {save_path}")
        return

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))
        with open(save_path, "wb") as file, tqdm(
            desc="📥 Downloading dataset", total=total_size, unit="B", unit_scale=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)
                pbar.update(len(chunk))  # Update tqdm progress bar
        print("✅ Download complete.")
    else:
        print("❌ Failed to download the file.")


def unzip_dataset(zip_path, extract_to):
    """Extract dataset only if it hasn't been extracted already, with progress bar."""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"✅ Dataset already extracted: {extract_to}")
        return

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        file_list = zip_ref.namelist()
        with tqdm(
            total=len(file_list), desc="📂 Extracting dataset", unit="files"
        ) as pbar:
            for file in file_list:
                zip_ref.extract(file, extract_to)
                pbar.update(1)  # Update tqdm progress bar
    print("✅ Extraction complete.")


def data_preprocessing(dataset_dir):
    """Process dataset paths only if not already processed, with a progress bar."""
    dataset_dir = os.path.join(dataset_dir, "trainSet", "Stimuli")
    preprocessed_files = os.path.join(dataset_dir, "preprocessed_done.txt")

    if os.path.exists(preprocessed_files):
        print("✅ Dataset already preprocessed.")
        with open(preprocessed_files, "r") as f:
            data_paths = eval(f.read())  # Load stored paths
        return data_paths

    print("🔄 Preprocessing dataset...")
    cat_list = [
        d
        for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]
    if "Output" in cat_list:
        cat_list.remove("Output")

    train_data_paths_train, train_data_paths_test = [], []
    label_data_paths_train, label_data_paths_test = [], []

    with tqdm(
        total=len(cat_list), desc="📊 Processing categories", unit="category"
    ) as pbar:
        for cat in cat_list:
            images = glob.glob(os.path.join(dataset_dir, cat, "*.jpg"))
            random.shuffle(images)
            n = int(0.15 * len(images))  # 15% test split

            train_data_paths_train.extend(images[n:])
            train_data_paths_test.extend(images[:n])
            pbar.update(1)  # Update tqdm progress bar

    # Generate label paths
    label_data_paths_train = [
        p.replace("Stimuli", "FIXATIONMAPS") for p in train_data_paths_train
    ]
    label_data_paths_test = [
        p.replace("Stimuli", "FIXATIONMAPS") for p in train_data_paths_test
    ]

    # Save preprocessed paths
    data_paths = (
        train_data_paths_train,
        train_data_paths_test,
        label_data_paths_train,
        label_data_paths_test,
    )
    with open(preprocessed_files, "w") as f:
        f.write(str(data_paths))

    print("✅ Preprocessing complete.")
    return data_paths
