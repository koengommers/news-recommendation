import os
import zipfile

import pandas as pd
import requests
from tqdm import tqdm

SPLITS = {
    "small": ["train", "dev"],
    "large": ["train", "dev", "test"],
}


def _get_mind_path(variant, split=None, dataset_dir="./data"):
    path = os.path.join(dataset_dir, f"mind_{variant}")

    if split:
        path = os.path.join(path, split)

    return path


def _get_mind_file(variant, split, filename, dataset_dir="./data"):
    return os.path.join(_get_mind_path(variant, split, dataset_dir), filename)


def download_mind(variant="small", dataset_dir="./data"):
    BASE_URL = "https://mind201910small.blob.core.windows.net/release"

    os.makedirs(dataset_dir, exist_ok=True)

    assert variant in SPLITS.keys()

    for split in SPLITS[variant]:
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
            zip.extractall(_get_mind_path(variant, split, dataset_dir))

        os.remove(filepath)


def _load_mind_data(filename, variant, splits, column_names, column_indices):
    return pd.concat(
        [
            pd.read_table(
                _get_mind_file(variant, split, filename),
                names=column_names,
                usecols=column_indices,
            )
            for split in splits
        ],
        ignore_index=True,
    )


def load_behaviors(variant="small", splits=None, columns=None):
    if splits is None:
        splits = SPLITS[variant]

    column_names = ["impression_id", "user", "time", "history", "impressions"]
    if columns is None:
        columns = column_names
    column_indices = [column_names.index(col) for col in columns]

    behaviors = _load_mind_data(
        "behaviors.tsv", variant, splits, columns, column_indices
    )
    behaviors.history = behaviors.history.fillna("").str.split()
    return behaviors


def load_news(variant="small", splits=None, columns=None):
    if splits is None:
        splits = SPLITS[variant]

    column_names = [
        "id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    if columns is None:
        columns = column_names
    column_indices = [column_names.index(col) for col in columns]

    news = _load_mind_data("news.tsv", variant, splits, columns, column_indices)
    news = news.drop_duplicates(subset="id")
    assert news is not None
    news = news.set_index("id")
    return news


def _combine_history(histories):
    return histories[histories.apply(len).idxmax()]


def convert_behaviors_to_users(behaviors):
    grouped = behaviors.groupby("user")
    users = grouped.agg({"history": _combine_history})
    return users
