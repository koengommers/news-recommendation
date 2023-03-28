import random

from torch.utils.data import Dataset

from utils.data import load_behaviors, load_news
from utils.encode import CategoricalEncoder


def filter_positive_samples(impressions):
    return [sample[:-2] for sample in impressions if sample.endswith("-1")]


def filter_negative_samples(impressions):
    return [sample[:-2] for sample in impressions if sample.endswith("-0")]


class BehaviorsDataset(Dataset):
    def __init__(
        self,
        mind_variant: str,
        split,
        tokenizer,
        negative_sampling_ratio: int = 4,
        num_words_title: int = 20,
        num_words_abstract: int = 50,
        history_length: int = 50,
        news_features=["title"],
    ):
        self.mind_variant = mind_variant
        self.split = split
        self.tokenizer = tokenizer
        self.negative_sampling_ratio = negative_sampling_ratio
        self.num_words_title = num_words_title
        self.num_words_abstract = num_words_abstract
        self.history_length = history_length
        self.news_features = news_features
        self.categorical_encoders = {}

        print("Loading news...")
        self.news = self.prepare_news()

        print("Loading logs...")
        self.logs = self.prepare_logs()

    def prepare_logs(self):
        behaviors = load_behaviors(self.mind_variant, splits=[self.split])

        # Split impressions into positive and negative samples
        behaviors.impressions = behaviors.impressions.str.split()
        behaviors["positive_samples"] = behaviors.impressions.apply(
            filter_positive_samples
        )
        behaviors["negative_samples"] = behaviors.impressions.apply(
            filter_negative_samples
        )

        # Filter out entries with too few negative samples
        behaviors = behaviors[
            behaviors.negative_samples.map(len) >= self.negative_sampling_ratio
        ]

        # Create one datapoint for every positive sample
        behaviors = behaviors.explode("positive_samples").rename(
            columns={"positive_samples": "positive_sample"}
        )
        behaviors.negative_samples = behaviors.negative_samples.apply(
            lambda x: random.sample(x, k=self.negative_sampling_ratio)
        )
        behaviors["candidate_news"] = (
            behaviors.positive_sample.apply(lambda x: [x]) + behaviors.negative_samples
        )

        behaviors = behaviors.reset_index(drop=True).drop(
            columns=[
                "time",
                "impression_id",
                "impressions",
                "positive_sample",
                "negative_samples",
            ]
        )

        return behaviors

    def prepare_news(self):
        categorical_features = ["category", "subcategory"]
        textual_features = ["title", "abstract"]

        news = load_news(
            self.mind_variant, splits=[self.split], columns=self.news_features
        )

        parsed_news = {}
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
            parsed_news[index] = article

        return parsed_news

    def pad_history(self, history):
        padding_all = {
            "title": self.tokenizer("", length=self.num_words_title),
            "abstract": self.tokenizer("", length=self.num_words_abstract),
            "category": 0,
            "subcategory": 0,
        }
        padding = {feature: padding_all[feature] for feature in self.news_features}
        padding_length = self.history_length - len(history)
        return [padding] * padding_length + history

    def __len__(self):
        return len(self.logs)

    def __getitem__(self, idx):
        row = self.logs.iloc[idx]
        history = self.pad_history(
            [self.news[id] for id in row.history[: self.history_length]]
        )
        candidate_news = [self.news[id] for id in row.candidate_news]
        return history, candidate_news
