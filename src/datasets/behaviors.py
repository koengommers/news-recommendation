from typing import Tuple

from torch.utils.data import Dataset

from utils.data import load_behaviors


class BehaviorsDataset(Dataset):
    """
    Dataset for with logs for evaluationg.
    Returns logs of history and full list of impressions.
    """

    def __init__(
        self,
        mind_variant: str,
        split: str,
        history_length: int = 50,
    ):
        self.mind_variant = mind_variant
        self.split = split
        self.history_length = history_length

        self.behaviors = load_behaviors(
            mind_variant, splits=[self.split], columns=["history", "impressions"]
        )

    def __len__(self) -> int:
        return len(self.behaviors)

    def __getitem__(self, idx: int) -> Tuple[list[str], list[str], list[int]]:
        row = self.behaviors.iloc[idx]

        impressions: list[str]
        clicked_str: list[int]
        impressions, clicked_str = zip(
            *map(lambda impression: impression.split("-"), row.impressions)
        )
        clicked: list[int] = [int(y) for y in clicked_str]

        return row.history[: self.history_length], impressions, clicked
