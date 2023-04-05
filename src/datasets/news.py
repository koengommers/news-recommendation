from torch.utils.data import Dataset

from utils.data import load_news
from utils.encode import CategoricalEncoder


class NewsDataset(Dataset):
    """
    Dataset for encoding news for evaluation.
    """

    def __init__(
        self,
        mind_variant: str,
        split: str,
        tokenizer,
        num_words_title: int = 20,
        num_words_abstract: int = 50,
        categorical_encoders={},
        news_features=["title"],
    ):
        self.mind_variant = mind_variant
        self.split = split
        self.tokenizer = tokenizer
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.news_features = news_features
        self.categorical_encoders = categorical_encoders

        self.news = self.prepare_news()

    def prepare_news(self):
        categorical_features = ["category", "subcategory"]
        textual_features = ["title", "abstract"]

        news = load_news(
            self.mind_variant, splits=[self.split], columns=self.news_features
        )

        parsed_news = []
        for index, row in news.iterrows():
            article = {}
            for feature in self.news_features:
                if feature in textual_features:
                    article[feature] = self.tokenizer(
                        row[feature].lower(),
                        length=getattr(self, f"num_words_{feature}"),
                    )
                if feature in categorical_features:
                    if feature not in self.categorical_encoders:
                        self.categorical_encoders[feature] = CategoricalEncoder()
                    article[feature] = self.categorical_encoders[feature].encode(
                        row[feature]
                    )
            parsed_news.append((index, article))

        return parsed_news

    def __len__(self):
        return len(self.news)

    def __getitem__(self, idx):
        return self.news[idx]
