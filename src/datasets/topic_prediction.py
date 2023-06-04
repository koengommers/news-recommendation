from collections import defaultdict
from typing import Union

from torch.utils.data import Dataset

from src.utils.data import load_news
from src.utils.encode import CategoricalEncoder
from src.utils.tokenize import NltkTokenizer, PLMTokenizer

Tokenizer = Union[NltkTokenizer, PLMTokenizer]
TokenizerOutput = Union[list[int], dict[str, list[int]]]
NewsItem = dict[str, Union[TokenizerOutput, int]]


class TopicPredictionDataset(Dataset):
    """
    Dataset for encoding news for evaluation.
    """

    def __init__(
        self,
        mind_variant: str,
        tokenizer: Tokenizer,
        num_words_title: int = 20,
        num_words_abstract: int = 50,
        news_features: list[str] = ["title"],
    ):
        self.mind_variant = mind_variant
        self.tokenizer = tokenizer
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.news_features = news_features
        self.categorical_encoders: dict[str, CategoricalEncoder] = defaultdict(CategoricalEncoder)

        self.news = self.prepare_news()

    @property
    def num_words(self) -> int:
        return self.tokenizer.vocab_size + 1

    @property
    def num_categories(self) -> int:
        if "category" not in self.categorical_encoders:
            return 0
        return self.categorical_encoders["category"].n_categories + 1

    def prepare_news(self) -> list[NewsItem]:
        textual_features = ["title", "abstract"]

        if "category" not in self.news_features:
            columns = self.news_features + ["category"]
        else:
            columns = self.news_features

        news = load_news(self.mind_variant, columns=columns)

        parsed_news: list[NewsItem] = []
        for _, row in news.iterrows():
            article: NewsItem = {}
            for feature in columns:
                if feature in textual_features:
                    article[feature] = self.tokenizer(
                        row[feature].lower(),
                        getattr(self, f"num_words_{feature}"),
                    )
                if feature == "category":
                    article[feature] = self.categorical_encoders[feature].encode(
                        row["category"]
                    )
                if feature == "subcategory":
                    article[feature] = self.categorical_encoders[feature].encode(
                        (row["category"], row["subcategory"])
                    )
            parsed_news.append(article)

        return parsed_news

    def __len__(self) -> int:
        return len(self.news)

    def __getitem__(self, idx: int):
        news_item = self.news[idx]
        features = {feature: news_item[feature] for feature in self.news_features}
        return features, news_item["category"]
