import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from src.utils.data import DEFAULT_DATA_DIR, load_news, load_users


def news_to_topics(news: pd.DataFrame) -> pd.DataFrame:
    categories = news[["category", "subcategory"]]
    categories = categories.drop_duplicates()
    assert categories is not None
    categories = categories.reset_index(drop=True)
    return categories


class TopicReadsDataset(Dataset):
    def __init__(self, variant: str = "small", data_dir: str = DEFAULT_DATA_DIR):
        users = load_users(variant, data_dir=data_dir)
        news = load_news(
            variant, columns=["category", "subcategory"], data_dir=data_dir
        )
        self.all_topics = news_to_topics(news)

        reads = users.explode("history").dropna()
        reads = pd.merge(reads, news, left_on="history", right_index=True)
        reads = reads.drop(columns=["history"])
        reads = reads.reset_index()

        # Topic name to ordinal number
        self.topic_encoder = LabelEncoder()
        self.topics = self.topic_encoder.fit_transform(reads["subcategory"].values)

        # User ID to ordinal number
        self.user_encoder = LabelEncoder()
        self.contexts = self.user_encoder.fit_transform(reads["user"].values)

    @property
    def number_of_topics(self) -> int:
        return len(self.topic_encoder.classes_)

    @property
    def number_of_users(self) -> int:
        return len(self.user_encoder.classes_)

    def __len__(self) -> int:
        return len(self.contexts)

    def __getitem__(self, idx: int) -> tuple[int, int]:
        return self.topics[idx], self.contexts[idx]
