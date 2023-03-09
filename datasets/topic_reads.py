import os

import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset

from utils import download_mind

splits = ["train", "dev"]


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
                usecols=[0, 2, 3],
                names=["id", "category", "title"],
            )
            for split in splits
        ]
    )
    news = news.drop_duplicates(subset="id")
    assert news is not None
    news = news.set_index("id")
    return news


class TopicReadsDataset(Dataset):
    def __init__(self, variant="small", dataset_dir="./data"):
        path = os.path.join(dataset_dir, f"mind_{variant}")
        if not os.path.exists(path):
            download_mind(variant, dataset_dir)

        print("Loading behaviors...")
        behaviors = load_behaviors(path)
        print("Processing users...")
        users = convert_behaviors_to_users(behaviors)

        news = load_news(path)

        reads = users.explode("history").dropna()
        reads = pd.merge(reads, news, left_on="history", right_index=True)
        reads = reads.reset_index()
        self.topic_encoder = preprocessing.OrdinalEncoder()
        self.contexts = self.topic_encoder.fit_transform(
            reads["category"].values.reshape(-1, 1)
        )
        self.user_encoder = preprocessing.OrdinalEncoder()
        self.targets = self.user_encoder.fit_transform(
            reads["user"].values.reshape(-1, 1)
        )

    @property
    def number_of_topics(self):
        return len(self.topic_encoder.categories_[0])

    @property
    def number_of_users(self):
        return len(self.user_encoder.categories_[0])

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return self.contexts[idx], self.targets[idx]
