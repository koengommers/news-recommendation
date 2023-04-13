from typing import Callable, Tuple, Union

from torch.utils.data import Dataset

from utils.data import load_news
from utils.encode import CategoricalEncoder

TokenizerOutput = Union[list[int], dict[str, list[int]]]
NewsItem = dict[str, Union[TokenizerOutput, int]]


class NewsDataset(Dataset):
    """
    Dataset for encoding news for evaluation.
    """

    def __init__(
        self,
        mind_variant: str,
        split: str,
        tokenizer: Callable[[str, int], TokenizerOutput],
        num_words_title: int = 20,
        num_words_abstract: int = 50,
        categorical_encoders: dict[str, CategoricalEncoder] = {},
        news_features: list[str] = ["title"],
    ):
        self.mind_variant = mind_variant
        self.split = split
        self.tokenizer = tokenizer
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.news_features = news_features
        self.categorical_encoders = categorical_encoders

        self.news = self.prepare_news()

    def prepare_news(self) -> list[Tuple[str, NewsItem]]:
        categorical_features = ["category", "subcategory"]
        textual_features = ["title", "abstract"]

        news = load_news(
            self.mind_variant, splits=[self.split], columns=self.news_features
        )

        parsed_news: list[Tuple[str, NewsItem]] = []
        for index, row in news.iterrows():
            article: NewsItem = {}
            for feature in self.news_features:
                if feature in textual_features:
                    article[feature] = self.tokenizer(
                        row[feature].lower(),
                        getattr(self, f"num_words_{feature}"),
                    )
                if feature in categorical_features:
                    if feature not in self.categorical_encoders:
                        self.categorical_encoders[feature] = CategoricalEncoder()
                    article[feature] = self.categorical_encoders[feature].encode(
                        row[feature]
                    )
            parsed_news.append((str(index), article))

        return parsed_news

    def __len__(self) -> int:
        return len(self.news)

    def __getitem__(self, idx: int) -> Tuple[str, NewsItem]:
        return self.news[idx]
