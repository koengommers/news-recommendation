import os
import zipfile
from urllib.parse import urlparse

import pandas as pd
import requests
import torch
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


def download_zip(name, url, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    filename = os.path.basename(urlparse(url).path)
    filepath = os.path.join(target_dir, filename)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024
    with tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        desc=f"Downloading {name}",
    ) as pbar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(chunk_size):
                pbar.update(len(data))
                file.write(data)

    with zipfile.ZipFile(filepath) as zip:
        zip.extractall(target_dir)

    os.remove(filepath)


def download_mind(variant="small", dataset_dir="./data"):
    BASE_URL = "https://mind201910small.blob.core.windows.net/release"

    assert variant in SPLITS.keys()

    for split in SPLITS[variant]:
        filename = f"MIND{variant}_{split}.zip"
        url = f"{BASE_URL}/{filename}"
        download_zip(filename, url, _get_mind_path(variant, split, dataset_dir))


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

    available_columns = [
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
        columns = available_columns
    else:
        columns = ["id"] + columns
    column_indices = sorted([available_columns.index(col) for col in columns])
    column_names = [available_columns[i] for i in column_indices]

    news = _load_mind_data("news.tsv", variant, splits, column_names, column_indices)
    news = news.drop_duplicates(subset="id")
    assert news is not None
    news = news.set_index("id")
    return news


def load_users(variant="small", splits=None):
    behaviors = load_behaviors(variant, splits)
    users = convert_behaviors_to_users(behaviors)
    return users


def _combine_history(histories):
    return histories[histories.apply(len).idxmax()]


def convert_behaviors_to_users(behaviors):
    grouped = behaviors.groupby("user")
    users = grouped.agg({"history": _combine_history})
    return users


def _download_glove(dataset_dir="./data"):
    url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    download_zip("GloVe Embeddings", url, os.path.join(dataset_dir, "glove"))


def load_pretrained_embeddings(token2int, dataset_dir="./data"):
    glove_path = os.path.join(dataset_dir, "glove", "glove.840B.300d.txt")
    if not os.path.exists(glove_path):
        _download_glove(dataset_dir)

    print("Creating word embedding matrix...")
    embeddings = torch.empty((len(token2int) + 1, 300))
    torch.nn.init.normal_(embeddings)

    hits = 0
    with open(glove_path, "rb") as file:
        for line in file:
            values = line.split()
            token = values[0].decode()
            if token in token2int:
                i = token2int[token]
                embeddings[i] = torch.tensor([float(x) for x in values[1:]])
                hits += 1
            if hits == len(token2int):
                break

    print(f"Missed {len(token2int) - hits} / {len(token2int)} words")
    return embeddings
