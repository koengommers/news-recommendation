import os
import zipfile
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import requests
import torch
from hydra.utils import to_absolute_path
from tqdm import tqdm

SPLITS = {
    "small": ["train", "dev"],
    "large": ["train", "dev", "test"],
}
DEFAULT_DATA_DIR = to_absolute_path("data")


def get_mind_path(
    variant: str, split: Optional[str] = None, data_dir: str = DEFAULT_DATA_DIR
) -> str:
    path = os.path.join(data_dir, f"mind_{variant}")

    if split:
        path = os.path.join(path, split)

    return path


def get_mind_file(
    variant: str, split: str, filename: str, data_dir: str = DEFAULT_DATA_DIR
) -> str:
    return os.path.join(get_mind_path(variant, split, data_dir), filename)


def download_zip(name: str, url: str, target_dir: str) -> None:
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


def download_mind(variant: str = "small", data_dir: str = DEFAULT_DATA_DIR) -> None:
    BASE_URL = "https://mind201910small.blob.core.windows.net/release"

    assert variant in SPLITS.keys()

    for split in SPLITS[variant]:
        filename = f"MIND{variant}_{split}.zip"
        url = f"{BASE_URL}/{filename}"
        download_zip(filename, url, get_mind_path(variant, split, data_dir))


def _load_mind_data(
    filename: str,
    variant: str,
    splits: list[str],
    column_names: list[str],
    column_indices: list[int],
    data_dir: str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    if not os.path.exists(get_mind_path(variant, data_dir=data_dir)):
        download_mind(variant, data_dir)

    return pd.concat(
        [
            pd.read_table(
                get_mind_file(variant, split, filename, data_dir),
                names=column_names,
                usecols=column_indices,
            ).assign(split=split)
            for split in splits
        ],
        ignore_index=True,
    )


def _get_available_mind_splits(variant, data_dir=DEFAULT_DATA_DIR):
    mind_path = get_mind_path(variant, data_dir=data_dir)
    return [f.name for f in os.scandir(mind_path) if f.is_dir()]


def load_behaviors(
    variant: str = "small",
    splits: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    if splits is None:
        splits = _get_available_mind_splits(variant, data_dir)

    column_names = ["log_id", "user", "time", "history", "impressions"]
    if columns is None:
        columns = column_names
    column_indices = [column_names.index(col) for col in columns]

    behaviors = _load_mind_data(
        "behaviors.tsv", variant, splits, columns, column_indices, data_dir
    )
    if "history" in columns:
        behaviors.history = behaviors.history.fillna("").str.split()
    if "impressions" in columns:
        behaviors.impressions = behaviors.impressions.str.split()
    return behaviors


def load_news(
    variant: str = "small",
    splits: Optional[list[str]] = None,
    columns: Optional[list[str]] = None,
    drop_duplicates: bool = True,
    data_dir: str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    if splits is None:
        splits = _get_available_mind_splits(variant, data_dir)

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

    news = _load_mind_data(
        "news.tsv", variant, splits, column_names, column_indices, data_dir
    )
    if drop_duplicates:
        news = news.drop_duplicates(subset="id")
        assert news is not None
    news = news.set_index("id")
    return news


def load_users(
    variant: str = "small",
    splits: Optional[list[str]] = None,
    data_dir: str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    behaviors = load_behaviors(
        variant, splits, columns=["user", "history"], data_dir=data_dir
    )
    users = behaviors.drop_duplicates(subset="user").set_index("user")
    return users


def _download_glove(data_dir: str = DEFAULT_DATA_DIR) -> None:
    url = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    download_zip("GloVe Embeddings", url, os.path.join(data_dir, "glove"))


def load_pretrained_embeddings(
    token2int: dict[str, int], data_dir: str = DEFAULT_DATA_DIR
) -> torch.Tensor:
    glove_path = os.path.join(data_dir, "glove", "glove.840B.300d.txt")
    if not os.path.exists(glove_path):
        _download_glove(data_dir)

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
