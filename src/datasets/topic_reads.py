import os
import pickle

import pandas as pd
from sklearn import preprocessing
from torch.utils.data import Dataset

from utils.data import download_mind, load_news, load_users


def news_to_topics(news):
    categories = news[["category", "subcategory"]]
    categories = categories.drop_duplicates()
    assert categories is not None
    categories = categories.reset_index(drop=True)
    return categories


class TopicReadsDataset(Dataset):
    def __init__(self, variant="small", dataset_dir="./data"):
        prepared_dir = os.path.join(dataset_dir, "prepared")
        prepared_path = os.path.join(
            prepared_dir,
            f"mind_{variant}_topic_reads.pickle",
        )
        if os.path.exists(prepared_path):
            with open(prepared_path, "rb") as f:
                (
                    self.topic_encoder,
                    self.topics,
                    self.user_encoder,
                    self.contexts,
                    self.all_topics,
                ) = pickle.load(f)
        else:
            os.makedirs(prepared_dir, exist_ok=True)

            # Download MIND if it is not downloaded yet
            path = os.path.join(dataset_dir, f"mind_{variant}")
            if not os.path.exists(path):
                download_mind(variant, dataset_dir)

            users = load_users(variant)
            news = load_news(variant, columns=["category", "subcategory"])
            self.all_topics = news_to_topics(news)

            reads = users.explode("history").dropna()
            reads = pd.merge(reads, news, left_on="history", right_index=True)
            reads = reads.drop(columns=["history"])
            reads = reads.reset_index()

            # Topic name to ordinal number
            self.topic_encoder = preprocessing.OrdinalEncoder()
            self.topics = self.topic_encoder.fit_transform(
                reads["subcategory"].values.reshape(-1, 1)
            )

            # User ID to ordinal number
            self.user_encoder = preprocessing.OrdinalEncoder()
            self.contexts = self.user_encoder.fit_transform(
                reads["user"].values.reshape(-1, 1)
            )

            # Save preprocessed data
            with open(prepared_path, "wb") as f:
                pickle.dump(
                    (
                        self.topic_encoder,
                        self.topics,
                        self.user_encoder,
                        self.contexts,
                        self.all_topics,
                    ),
                    f,
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
        return self.topics[idx], self.contexts[idx]
