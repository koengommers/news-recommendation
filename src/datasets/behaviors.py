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
        padding: str = '<PAD>'
    ):
        self.mind_variant = mind_variant
        self.split = split
        self.history_length = history_length
        self.padding = padding

        self.behaviors = load_behaviors(mind_variant, splits=[self.split], columns=["history", "impressions"])

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx):
        row = self.behaviors.iloc[idx]

        padding_length = self.history_length - len(row.history)
        padded_history = [self.padding] * padding_length + row.history[:self.history_length]

        impressions, clicked = zip(*map(lambda impression: impression.split('-'), row.impressions))
        clicked = [int(y) for y in clicked]

        return row.history[:self.history_length], impressions, clicked
