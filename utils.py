import os
import zipfile

import requests
from tqdm import tqdm


def download_mind(variant="small", dataset_dir="./data"):
    BASE_URL = "https://mind201910small.blob.core.windows.net/release"

    os.makedirs(dataset_dir, exist_ok=True)

    splits = {
        "small": ["train", "dev"],
        "large": ["train", "dev", "test"],
    }
    assert variant in splits.keys()

    for split in splits[variant]:
        filename = f"MIND{variant}_{split}.zip"
        filepath = os.path.join(dataset_dir, filename)
        response = requests.get(f"{BASE_URL}/{filename}", stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024
        with tqdm(
            total=total_size,
            unit="iB",
            unit_scale=True,
            desc=f"Downloading {filename}",
        ) as pbar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(chunk_size):
                    pbar.update(len(data))
                    file.write(data)

        with zipfile.ZipFile(filepath) as zip:
            zip.extractall(os.path.join(dataset_dir, f"mind_{variant}", split))

        os.remove(filepath)
