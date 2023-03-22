import os
import pandas as pd
import zipfile

import requests
from tqdm import tqdm


splits = ["train", "dev"]


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

def load_behaviors(path):
    behaviors = pd.concat(
        [
            pd.read_table(
                os.path.join(path, split, "behaviors.tsv"),
                names=["impression_id", "user", "time", "clicked_news", "impressions"],
            )
            for split in splits
        ],
        ignore_index=True,
    )
    behaviors.clicked_news = behaviors.clicked_news.fillna("").str.split()
    return behaviors


def combine_history(histories):
    return histories[histories.apply(len).idxmax()]


def convert_behaviors_to_users(behaviors):
    grouped = behaviors.groupby("user")
    users = grouped.agg({"clicked_news": combine_history})
    users = users.rename(columns={"clicked_news": "history"})
    return users


def load_news(path):
    news = pd.concat(
        [
            pd.read_table(
                os.path.join(path, split, "news.tsv"),
                usecols=[0, 1, 2],
                names=["id", "category", "subcategory"],
            )
            for split in splits
        ]
    )
    news = news.drop_duplicates(subset="id")
    assert news is not None
    news = news.set_index("id")
    return news
