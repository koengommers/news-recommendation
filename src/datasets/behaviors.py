from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.utils.collate import collate_fn
from src.utils.data import load_behaviors


def behaviors_collate_fn(batch):
    log_ids = [x[0] for x in batch]
    clicked_news_vectors = collate_fn([x[1] for x in batch])
    mask = torch.Tensor([x[2] for x in batch])
    impression_ids = [x[3] for x in batch]
    clicked = [x[4] for x in batch]

    return log_ids, clicked_news_vectors, mask, impression_ids, clicked


class BehaviorsDataset(Dataset):
    """
    Dataset for with logs for evaluationg.
    Returns logs of history and full list of impressions.
    """

    def __init__(
        self,
        mind_variant: str,
        split: str,
        news_vectors,
        history_length: int = 50,
    ):
        self.mind_variant = mind_variant
        self.split = split
        self.history_length = history_length

        self.news_vectors = news_vectors
        news_repr_example = next(iter(self.news_vectors.values()))
        if isinstance(news_repr_example, dict):
            self.news_vectors["<PAD>"] = {
                key: torch.zeros(value.size(), dtype=value.dtype) for key, value in news_repr_example.items()
            }
        else:
            self.news_vectors["<PAD>"] = torch.zeros(news_repr_example.size())

        self.behaviors = load_behaviors(
            mind_variant,
            splits=[self.split],
            columns=["log_id", "history", "impressions"],
        )

    def pad_history_ids(self, history_ids):
        padding_length = self.history_length - len(history_ids)
        padded_history = ["<PAD>"] * padding_length + history_ids
        mask = [0] * padding_length + [1] * len(history_ids)
        return padded_history, mask

    def __len__(self) -> int:
        return len(self.behaviors)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, list[int], list[str], list[int]]:
        row = self.behaviors.iloc[idx]
        padded_history, mask = self.pad_history_ids(row.history[-self.history_length :])
        clicked_news_vectors = collate_fn(
            [self.news_vectors[id] for id in padded_history]
        )

        impression_ids: list[str]
        clicked_str: list[int]
        impression_ids, clicked_str = zip(
            *map(lambda impression: impression.split("-"), row.impressions)
        )
        clicked: list[int] = [int(y) for y in clicked_str]

        return row.log_id, clicked_news_vectors, mask, impression_ids, clicked
